"""Causal backbone plus a pointer head.

Production backbones are pretrained causal LMs (Qwen, Gemma, LFM). The scratch
`CausalBackbone` is the offline stand-in used by the unit tests.

The head scores each option's closing-token state against the decide token.
Softmax is taken inside the question. Temperature divides logits at eval time
only, so it never changes which option wins and training always sees T = 1.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from decision_model.render import PAD, Branch


class PointerHead(nn.Module):
    def __init__(self, hidden_size: int, proj_size: int = 256) -> None:
        super().__init__()
        self.q = nn.Linear(hidden_size, proj_size)
        self.k = nn.Linear(hidden_size, proj_size)
        self.scale = 1 / math.sqrt(proj_size)
        self.temperature = 1.0

    def forward(self, h_decide: torch.Tensor, h_opts: torch.Tensor) -> torch.Tensor:
        """h_decide [d] or [B, d]; h_opts [K, d] or [B, K, d] -> logits [K] or [B, K]."""
        if h_decide.dim() == 1:
            logits = (self.k(h_opts) @ self.q(h_decide)) * self.scale
        else:
            logits = (self.k(h_opts) * self.q(h_decide)[:, None, :]).sum(-1) * self.scale
        if self.training or self.temperature == 1.0:
            return logits
        return logits / self.temperature


class CausalBackbone(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        n_layers: int,
        n_heads: int,
        max_len: int,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.embed = nn.Embedding(vocab_size, hidden_size, padding_idx=PAD)
        self.position = nn.Embedding(max_len, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.size(1) > self.max_len:
            raise ValueError(f"sequence length {ids.size(1)} exceeds max_len {self.max_len}")
        positions = torch.arange(ids.size(1), device=ids.device)
        hidden = self.embed(ids) + self.position(positions)
        causal = torch.triu(
            torch.ones(ids.size(1), ids.size(1), device=ids.device, dtype=torch.bool),
            diagonal=1,
        )
        padding = ids.eq(PAD)
        # A row of all padding would make every attention weight illegal.
        padding = padding & ~padding.all(dim=1, keepdim=True)
        return self.norm(self.encoder(hidden, mask=causal, src_key_padding_mask=padding))


class PointerDecisionModel(nn.Module):
    def __init__(
        self,
        vocab_size: int | None = None,
        hidden_size: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        proj_size: int = 32,
        max_len: int = 256,
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if backbone is None:
            if vocab_size is None:
                raise TypeError("vocab_size is required when no backbone is passed")
            backbone = CausalBackbone(vocab_size, hidden_size, n_layers, n_heads, max_len)
            pad_id = PAD
            width = hidden_size
        else:
            pad_id = int(backbone.pad_id)
            width = int(backbone.hidden_size)
            max_len = int(backbone.max_len)
        self.backbone = backbone
        self.head = PointerHead(width, proj_size)
        self.pad_id = pad_id
        self.max_len = max_len

    def forward(
        self,
        ids: torch.Tensor,
        decide_index: torch.Tensor,
        opt_index: torch.Tensor,
        opt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits [B, K]. Padded options are filled with a finite minimum."""
        frozen = not any(parameter.requires_grad for parameter in self.backbone.parameters())
        if frozen:
            self.backbone.eval()
            with torch.no_grad():
                hidden = self.backbone(ids)
            hidden = hidden.detach()
        else:
            hidden = self.backbone(ids)
        if hidden.dtype != self.head.q.weight.dtype:
            hidden = hidden.to(dtype=self.head.q.weight.dtype)
        batch = torch.arange(ids.size(0), device=ids.device)
        h_decide = hidden[batch, decide_index]
        h_opts = hidden[batch[:, None], opt_index.clamp(min=0)]
        logits = self.head(h_decide, h_opts)
        return logits.masked_fill(~opt_mask, torch.finfo(logits.dtype).min)


def collate(branches: list[Branch], device: torch.device, pad_id: int = PAD) -> tuple[torch.Tensor, ...]:
    width = max(len(branch.ids) for branch in branches)
    options = max(len(branch.opt_index) for branch in branches)
    ids = torch.full((len(branches), width), pad_id, dtype=torch.long, device=device)
    opt_index = torch.zeros((len(branches), options), dtype=torch.long, device=device)
    opt_mask = torch.zeros((len(branches), options), dtype=torch.bool, device=device)
    decide_index = torch.zeros(len(branches), dtype=torch.long, device=device)
    gold = torch.zeros(len(branches), dtype=torch.long, device=device)
    for row, branch in enumerate(branches):
        ids[row, : len(branch.ids)] = torch.tensor(branch.ids, device=device)
        opt_index[row, : len(branch.opt_index)] = torch.tensor(branch.opt_index, device=device)
        opt_mask[row, : len(branch.opt_index)] = True
        decide_index[row] = branch.decide_index
        if branch.gold is not None:
            gold[row] = branch.gold
    return ids, decide_index, opt_index, opt_mask, gold


def _batches(branches: list[Branch], batch_size: int):
    step = max(1, batch_size)
    for start in range(0, len(branches), step):
        yield branches[start : start + step]


def branch_probabilities(
    model: PointerDecisionModel,
    branches: list[Branch],
    batch_size: int = 8,
) -> list[list[float]]:
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    collected: list[list[float]] = []
    try:
        with torch.inference_mode():
            for batch in _batches(branches, batch_size):
                ids, decide_index, opt_index, opt_mask, _gold = collate(batch, device, model.pad_id)
                logits = model(ids, decide_index, opt_index, opt_mask)
                probs = F.softmax(logits, dim=-1)
                collected.extend(
                    probs[row, : len(branch.opt_index)].tolist()
                    for row, branch in enumerate(batch)
                )
        return collected
    finally:
        model.train(was_training)


def fit_temperature(
    model: PointerDecisionModel,
    branches: list[Branch],
    lows: float = 0.05,
    highs: float = 5.0,
    steps: int = 40,
    batch_size: int = 8,
) -> float:
    """Pick T minimizing held-out log loss. The winning option stays put."""
    if not branches:
        return 1.0
    device = next(model.parameters()).device
    model.eval()
    model.head.temperature = 1.0
    # Questions in one run have different option counts, so logits cannot be
    # stacked into a single [N, K] tensor. Group by the real option count.
    grouped_logits: dict[int, list[torch.Tensor]] = {}
    grouped_gold: dict[int, list[torch.Tensor]] = {}
    with torch.inference_mode():
        for batch in _batches(branches, batch_size):
            ids, decide_index, opt_index, opt_mask, gold = collate(batch, device, model.pad_id)
            logits = model(ids, decide_index, opt_index, opt_mask).detach().cpu()
            gold = gold.detach().cpu()
            for row, branch in enumerate(batch):
                width = len(branch.opt_index)
                grouped_logits.setdefault(width, []).append(logits[row, :width])
                grouped_gold.setdefault(width, []).append(gold[row])
    tables = {
        width: (torch.stack(grouped_logits[width]), torch.stack(grouped_gold[width]))
        for width in grouped_logits
    }
    best_t, best_loss = 1.0, float("inf")
    count = sum(int(gold.shape[0]) for _logits, gold in tables.values())
    for step in range(steps):
        temperature = lows * (highs / lows) ** (step / (steps - 1))
        total = 0.0
        for logits, gold in tables.values():
            total += F.cross_entropy(logits / temperature, gold, reduction="sum").item()
        loss = total / count
        if loss < best_loss:
            best_t, best_loss = temperature, loss
    model.head.temperature = best_t
    return best_t
