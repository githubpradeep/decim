"""Score a request and turn distributions into typed answers."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from decision_model.config import TrainConfig
from decision_model.confidence import choice_confidence, expected_score, score_confidence
from decision_model.model import PointerDecisionModel, branch_probabilities
from decision_model.render import Vocab, render_question
from decision_model.schema import Question, parse_request


def format_answer(question: Question, probs: list[float]) -> dict:
    total = sum(probs)
    probs = [prob / total for prob in probs]
    if question.kind == "noul":
        return {"type": "noul", "noul": round(probs[1], 6)}
    if question.kind == "choice":
        winner = max(range(len(probs)), key=lambda index: probs[index])
        return {
            "type": "choice",
            "choice": question.keys[winner],
            "probabilities": {
                key: round(probs[index], 6) for index, key in enumerate(question.keys)
            },
            "confidence": round(choice_confidence(probs), 6),
        }
    assert question.levels is not None
    return {
        "type": "score",
        "score": round(expected_score(probs), 6),
        "legend": {str(index): level for index, level in enumerate(question.levels)},
        "probabilities": {str(index): round(prob, 6) for index, prob in enumerate(probs)},
        "confidence": round(score_confidence(probs), 6),
    }


def decide(model: PointerDecisionModel, vocab: Vocab, payload: dict, max_len: int, name: str) -> dict:
    started = time.perf_counter()
    request = parse_request(payload)
    pairs = [
        (question, render_question(vocab, request.state, question, max_len))
        for question in request.questions.values()
    ]
    distributions = branch_probabilities(model, [branch for _question, branch in pairs])
    answers = {
        question.name: format_answer(question, distribution)
        for (question, _branch), distribution in zip(pairs, distributions, strict=True)
    }
    state_tokens = pairs[0][1].state_tokens
    input_tokens = state_tokens + sum(branch.branch_tokens for _question, branch in pairs)
    return {
        "model": name,
        "answers": answers,
        "truncated": any(branch.truncated for _question, branch in pairs),
        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def load_model(path: Path) -> tuple[PointerDecisionModel, object, TrainConfig]:
    import torch

    blob = torch.load(path, map_location="cpu", weights_only=False)
    config = TrainConfig(**blob["config"])
    if blob.get("kind") == "pretrained":
        from decision_model.pretrained import load_causal_lm, preferred_device

        device = preferred_device()
        backbone, encoder = load_causal_lm(blob["model_id"], config.max_len, device)
        model = PointerDecisionModel(backbone=backbone, proj_size=config.proj_size)
        model.head.load_state_dict(blob["head"])
        model.head.temperature = float(blob["temperature"])
        model.to(device)
        model.eval()
        return model, encoder, config
    vocab = Vocab.load(blob["vocab"])
    model = PointerDecisionModel(
        vocab_size=vocab.size,
        hidden_size=config.hidden_size,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        proj_size=config.proj_size,
        max_len=config.max_len,
    )
    model.load_state_dict(blob["state_dict"])
    model.head.temperature = float(blob["temperature"])
    model.eval()
    return model, vocab, config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one decision request through a pointer-head checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/qwen/checkpoint.pt"))
    parser.add_argument("request", type=Path, nargs="?", help="JSON file. Reads stdin when omitted.")
    args = parser.parse_args()
    model, vocab, config = load_model(args.checkpoint)
    if args.request:
        payload = json.loads(args.request.read_text())
    else:
        import sys

        payload = json.load(sys.stdin)
    print(json.dumps(decide(model, vocab, payload, config.max_len, config.name), indent=2))


if __name__ == "__main__":
    main()
