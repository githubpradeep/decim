"""Train the pointer head with cross-entropy on the correct option index."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, replace
from pathlib import Path

import torch
import torch.nn.functional as F

from decision_model.config import TrainConfig
from decision_model.data import SAMPLE_REQUEST, load_jsonl, make_rows
from decision_model.model import PointerDecisionModel, branch_probabilities, collate, fit_temperature
from decision_model.render import Vocab, render_question, texts_for_vocab
from decision_model.schema import DecisionRequest, parse_request


def branches_of(request: DecisionRequest, vocab: Vocab, max_len: int, *, labeled: bool):
    rendered = []
    for question in request.questions.values():
        branch = render_question(vocab, request.state, question, max_len)
        if labeled:
            if question.gold is None:
                raise ValueError(f"{question.name}: missing label")
            branch = replace(branch, gold=question.gold)
        rendered.append((question, branch))
    return rendered


def encode_rows(rows: list[dict], vocab: Vocab, max_len: int, *, labeled: bool):
    encoded = []
    for row in rows:
        request = parse_request(row, labeled=labeled)
        encoded.append(branches_of(request, vocab, max_len, labeled=labeled))
    return encoded


def accuracy(
    model: PointerDecisionModel,
    rows: list[dict],
    vocab: Vocab,
    max_len: int,
    batch_size: int = 8,
) -> dict[str, float]:
    encoded = encode_rows(rows, vocab, max_len, labeled=True)
    flat_questions = []
    flat_branches = []
    for request_branches in encoded:
        for question, branch in request_branches:
            flat_questions.append(question)
            flat_branches.append(branch)
    probs = branch_probabilities(model, flat_branches, batch_size)
    correct = {kind: 0 for kind in ("choice", "noul", "score")}
    total = {kind: 0 for kind in ("choice", "noul", "score")}
    for question, distribution in zip(flat_questions, probs, strict=True):
        total[question.kind] += 1
        if distribution.index(max(distribution)) == question.gold:
            correct[question.kind] += 1
    scores = {kind: correct[kind] / total[kind] for kind in correct if total[kind]}
    scores["all"] = sum(correct.values()) / sum(total.values())
    return scores


def train_model(
    train_rows: list[dict],
    eval_rows: list[dict],
    config: TrainConfig,
    checkpoint_path: Path | None = None,
    init_checkpoint: Path | None = None,
) -> tuple[PointerDecisionModel, object]:
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    if config.model == "scratch":
        vocab_text: list[str] = []
        for row in train_rows:
            request = parse_request(row, labeled=True)
            vocab_text.extend(texts_for_vocab(request.state, list(request.questions.values())))
        encoder: object = Vocab.build(vocab_text)
        model = PointerDecisionModel(
            vocab_size=encoder.size,
            hidden_size=config.hidden_size,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            proj_size=config.proj_size,
            max_len=config.max_len,
        )
        device = torch.device("cpu")
    else:
        from decision_model.pretrained import load_causal_lm, preferred_device

        device = preferred_device()
        print(f"loading {config.model_id} on {device}")
        backbone, encoder = load_causal_lm(config.model, config.max_len, device)
        model = PointerDecisionModel(backbone=backbone, proj_size=config.proj_size)
        if init_checkpoint is not None:
            blob = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
            model.head.load_state_dict(blob["head"])
            print(f"loaded head from {init_checkpoint}", flush=True)
        model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.lr, weight_decay=0.01)
    train_branches = [
        branch
        for request_branches in encode_rows(train_rows, encoder, config.max_len, labeled=True)
        for _question, branch in request_branches
    ]
    eval_branches = [
        branch
        for request_branches in encode_rows(eval_rows, encoder, config.max_len, labeled=True)
        for _question, branch in request_branches
    ]
    for epoch in range(config.epochs):
        random.shuffle(train_branches)
        model.train()
        total_loss = 0.0
        steps = 0
        for start in range(0, len(train_branches), config.batch_size):
            batch = train_branches[start : start + config.batch_size]
            ids, decide_index, opt_index, opt_mask, gold = collate(batch, device, model.pad_id)
            optimizer.zero_grad()
            logits = model(ids, decide_index, opt_index, opt_mask)
            loss = F.cross_entropy(logits, gold)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            total_loss += float(loss.detach())
            steps += 1
            if steps % 100 == 0:
                print(f"  step {steps}", flush=True)
        if config.model != "scratch":
            print(f"epoch {epoch + 1} loss {total_loss / max(steps, 1):.4f}", flush=True)
        if checkpoint_path is not None:
            save_checkpoint(checkpoint_path, model, encoder, config)
            print(f"saved {checkpoint_path}", flush=True)
    fit_temperature(model, eval_branches, batch_size=config.batch_size)
    if checkpoint_path is not None:
        save_checkpoint(checkpoint_path, model, encoder, config)
        print(f"saved {checkpoint_path}", flush=True)
    return model, encoder


def save_checkpoint(
    path: Path,
    model: PointerDecisionModel,
    encoder: object,
    config: TrainConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {"config": asdict(config), "temperature": model.head.temperature}
    model_id = getattr(model.backbone, "model_id", None)
    if model_id:
        payload["kind"] = "pretrained"
        payload["model_id"] = model_id
        payload["head"] = {key: value.detach().cpu() for key, value in model.head.state_dict().items()}
    else:
        payload["kind"] = "scratch"
        payload["state_dict"] = model.state_dict()
        payload["vocab"] = encoder.dump()
    torch.save(payload, path)


def main() -> None:
    from decision_model.pretrained import resolve_model_id

    parser = argparse.ArgumentParser(
        description="Train the pointer head on a frozen causal LM. Default backbone is Qwen3-0.6B-Base.",
    )
    parser.add_argument(
        "--model",
        default="qwen",
        help="qwen, qwen-instruct, gemma, lfm, scratch, or a Hugging Face model id",
    )
    parser.add_argument("--data", type=Path, help="JSONL of labeled state/question rows")
    parser.add_argument(
        "--eval",
        type=Path,
        help="Held-out JSONL. When set, --data is used entirely for training.",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rows", type=int, default=None, help="synthetic rows when --data is omitted")
    args = parser.parse_args()
    pretrained = args.model != "scratch"
    epochs = args.epochs if args.epochs is not None else (8 if pretrained else 12)
    rows_n = args.rows if args.rows is not None else (120 if pretrained else 240)
    model_id = "" if not pretrained else resolve_model_id(args.model)
    config = TrainConfig(
        epochs=epochs,
        seed=args.seed,
        model=args.model,
        model_id=model_id,
        name=model_id or "pointer-tiny",
        proj_size=256 if pretrained else 32,
        max_len=args.max_len if args.max_len is not None else (512 if pretrained else 256),
        batch_size=args.batch_size if args.batch_size is not None else (4 if pretrained else 32),
        lr=args.lr if args.lr is not None else (3e-4 if pretrained else 2e-3),
    )
    out = args.out or Path("runs") / ("scratch" if not pretrained else args.model) / "checkpoint.pt"
    if args.eval and not args.data:
        parser.error("--eval requires --data")
    if args.data and args.eval:
        train_rows = load_jsonl(args.data)
        eval_rows = load_jsonl(args.eval)
    elif args.data:
        rows = load_jsonl(args.data)
        split = max(1, int(len(rows) * 0.8))
        train_rows, eval_rows = rows[:split], rows[split:] or rows[:split]
    else:
        rows = make_rows(rows_n, seed=args.seed)
        split = int(len(rows) * 0.8)
        train_rows, eval_rows = rows[:split], rows[split:]
    print(f"train rows {len(train_rows)}  eval rows {len(eval_rows)}", flush=True)
    model, encoder = train_model(train_rows, eval_rows, config, checkpoint_path=out)
    scores = accuracy(model, eval_rows, encoder, config.max_len, config.batch_size)
    save_checkpoint(out, model, encoder, config)
    rendered = ", ".join(f"{kind} {score:.3f}" for kind, score in scores.items())
    if args.data and len(train_rows) <= 20000:
        train_scores = accuracy(model, train_rows, encoder, config.max_len, config.batch_size)
        train_rendered = ", ".join(f"{kind} {score:.3f}" for kind, score in train_scores.items())
        print(f"train accuracy: {train_rendered}")
    print(f"saved {out}")
    print(f"backbone: {model_id or 'scratch'}")
    print(f"held-out accuracy: {rendered}")
    print(f"temperature: {model.head.temperature:.3f}")
    if not args.data:
        from decision_model.decide import decide

        sample = decide(model, encoder, SAMPLE_REQUEST, config.max_len, config.name)
        print(json.dumps(sample, indent=2))


if __name__ == "__main__":
    main()
