"""Pretrained causal backbones. The pointer head reads their hidden states.

The language-model head is not loaded. Qwen, Gemma, and LFM stay frozen;
only the pointer head is trained. That keeps a 0.6B forward pass from also
running a 0.6B backward pass.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from decision_model.render import Markers

MODELS = {
    "qwen": "Qwen/Qwen3-0.6B-Base",
    "qwen3": "Qwen/Qwen3-0.6B-Base",
    "qwen-0.6b": "Qwen/Qwen3-0.6B-Base",
    "qwen-instruct": "Qwen/Qwen3-0.6B",
    "gemma": "google/gemma-3-270m",
    "gemma-270m": "google/gemma-3-270m",
    "lfm": "LiquidAI/LFM2-350M",
    "lfm2": "LiquidAI/LFM2-350M",
}

# Qwen already has these as single tokens. Reusing them avoids a new embedding row.
QWEN_DELIMITERS = (
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|fim_suffix|>",
)
OWN_DELIMITERS = ("<state>", "<q>", "<opt>", "</opt>", "<decide>")


def resolve_model_id(name: str) -> str:
    return MODELS.get(name, name)


def preferred_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class PretrainedEncoder:
    """Tokenizer plus the five delimiter ids the pointer layout needs."""

    def __init__(self, tokenizer: object, markers: Markers, delimiters: tuple[str, ...]) -> None:
        self.tokenizer = tokenizer
        self.markers = markers
        self.delimiters = delimiters

    def encode(self, text: str) -> list[int]:
        for token in self.delimiters:
            text = text.replace(token, token.replace("<", "< ", 1))
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return list(ids)


class HFCausalBackbone(nn.Module):
    """A frozen causal transformer. `forward` returns last-layer hidden states."""

    def __init__(self, model: nn.Module, pad_id: int, max_len: int, model_id: str) -> None:
        super().__init__()
        self.model = model
        self.hidden_size = int(model.config.hidden_size)
        self.pad_id = pad_id
        self.max_len = max_len
        self.model_id = model_id
        self.model.config.use_cache = False

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.size(1) > self.max_len:
            raise ValueError(f"sequence length {ids.size(1)} exceeds max_len {self.max_len}")
        attention = ids.ne(self.pad_id).long()
        empty = attention.sum(dim=1).eq(0)
        if bool(empty.any()):
            attention[empty, 0] = 1
        output = self.model(
            input_ids=ids,
            attention_mask=attention,
            use_cache=False,
            return_dict=True,
        )
        hidden = output.last_hidden_state
        if hidden is None:
            raise RuntimeError(f"{self.model_id} did not return last_hidden_state")
        return hidden


def _single_token_id(tokenizer: object, token: str) -> int | None:
    vocab = tokenizer.get_vocab()
    if token not in vocab:
        return None
    index = tokenizer.convert_tokens_to_ids(token)
    unk = getattr(tokenizer, "unk_token_id", None)
    if unk is not None and index == unk:
        return None
    return int(index)


def _markers_from_existing(tokenizer: object) -> tuple[Markers, tuple[str, ...]] | None:
    ids = [_single_token_id(tokenizer, token) for token in QWEN_DELIMITERS]
    if any(index is None for index in ids):
        return None
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    markers = Markers(
        pad=int(tokenizer.pad_token_id),
        state=int(ids[0]),
        q=int(ids[1]),
        opt=int(ids[2]),
        opt_end=int(ids[3]),
        decide=int(ids[4]),
    )
    return markers, QWEN_DELIMITERS


def _markers_by_adding(tokenizer: object, model: nn.Module) -> tuple[Markers, tuple[str, ...]]:
    added = tokenizer.add_special_tokens({"additional_special_tokens": list(OWN_DELIMITERS)})
    if added:
        model.resize_token_embeddings(len(tokenizer))
        embedding = model.get_input_embeddings()
        with torch.no_grad():
            rows = embedding.weight
            rows[-added:] = rows[:-added].mean(dim=0)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    ids = [int(tokenizer.convert_tokens_to_ids(token)) for token in OWN_DELIMITERS]
    markers = Markers(
        pad=int(tokenizer.pad_token_id),
        state=ids[0],
        q=ids[1],
        opt=ids[2],
        opt_end=ids[3],
        decide=ids[4],
    )
    return markers, OWN_DELIMITERS


def load_causal_lm(model_name: str, max_len: int, device: torch.device) -> tuple[HFCausalBackbone, PretrainedEncoder]:
    from transformers import AutoModel, AutoTokenizer

    model_id = resolve_model_id(model_name)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        causal = AutoModel.from_pretrained(model_id, dtype=torch.float32)
    except OSError as exc:
        raise OSError(
            f"Could not load {model_id}. Gemma weights are gated: accept the license on the "
            f"model page and run `hf auth login`. Original error: {exc}"
        ) from exc
    attached = _markers_from_existing(tokenizer)
    if attached is None:
        markers, delimiters = _markers_by_adding(tokenizer, causal)
    else:
        markers, delimiters = attached
    if markers.pad in {markers.state, markers.q, markers.opt, markers.opt_end, markers.decide}:
        raise RuntimeError(f"{model_id} uses a delimiter id as its pad token")
    backbone = HFCausalBackbone(causal, markers.pad, max_len, model_id)
    backbone.to(device)
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    encoder = PretrainedEncoder(tokenizer, markers, delimiters)
    print(f"backbone {model_id} on {device}; delimiters {delimiters[0]} .. {delimiters[-1]}")
    return backbone, encoder
