"""Score the unlabeled support-desk tickets and compare them with the expected calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from decision_model.decide import decide, load_model
from decision_model.schema import parse_request


def score_directory(model, encoder, config, directory: Path) -> int:
    manifest = json.loads((directory / "expected.json").read_text())
    misses = 0
    checks = 0
    print(f"checkpoint temperature {model.head.temperature:.3f}")
    for item in manifest:
        payload = json.loads((directory / item["file"]).read_text())
        request = parse_request(payload)
        result = decide(model, encoder, payload, config.max_len, config.name)
        answers = result["answers"]
        subject = payload["state"]["subject"]
        print(f"\n{item['file']}  {subject}")
        print(f"  {request.state.splitlines()[-1][:160]}")
        queue = answers["queue"]
        probabilities = queue["probabilities"]
        spread = "  ".join(f"{name} {probabilities[name]:.2f}" for name in probabilities)
        queue_ok = queue["choice"] == item["queue"]
        checks += 1
        misses += not queue_ok
        print(f"  queue     {spread}")
        print(f"            choice {queue['choice']}  expected {item['queue']}  {'ok' if queue_ok else 'MISS'}")
        escalate = answers["escalate"]["noul"]
        escalate_yes = escalate >= 0.5
        escalate_ok = escalate_yes == item["escalate"]
        checks += 1
        misses += not escalate_ok
        print(
            f"  escalate  {escalate:.2f} {'yes' if escalate_yes else 'no'}"
            f"  expected {'yes' if item['escalate'] else 'no'}  {'ok' if escalate_ok else 'MISS'}"
        )
        priority = answers["priority"]
        predicted = max(range(len(priority["probabilities"])), key=lambda index: float(priority["probabilities"][str(index)]))
        legend = priority["legend"]
        priority_ok = predicted == item["priority"]
        checks += 1
        misses += not priority_ok
        print(
            f"  priority  score {priority['score']:.2f}  mode {legend[str(predicted)]}"
            f"  expected {legend[str(item['priority'])]}  {'ok' if priority_ok else 'MISS'}"
        )
    print(f"\n{checks - misses}/{checks} answers match the expected call")
    return misses


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the support-desk demo tickets.")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/support/checkpoint.pt"))
    parser.add_argument("--dir", type=Path, default=Path("data/support/demo"))
    args = parser.parse_args()
    model, encoder, config = load_model(args.checkpoint)
    score_directory(model, encoder, config, args.dir)


if __name__ == "__main__":
    main()
