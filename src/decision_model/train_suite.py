"""Train one pointer head on every public benchmark task.

The allowed answers stay inside each row, so choice, yes/no, and score share
one checkpoint. Each official test file is scored on its own after training.

    python -m decision_model.train_suite --dry-run
    python -m decision_model.train_suite
    python -m decision_model.train_suite --profile full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from decision_model.config import TrainConfig
from decision_model.data import load_jsonl
from decision_model.datasets.public import (
    JOINT_BATCH,
    JOINT_EPOCHS,
    JOINT_LR,
    JOINT_MAX_LEN,
    JOINT_STEP_SECONDS,
    steps_for,
)
from decision_model.train import accuracy, save_checkpoint, train_model


def _train_file(task: dict, profile: str) -> str:
    if profile == "laptop" and task.get("laptop_file"):
        return task["laptop_file"]
    return "train.jsonl"


def _select(manifest: list[dict], names: list[str] | None) -> list[dict]:
    if not names:
        return manifest
    wanted = set(names)
    selected = [task for task in manifest if task["name"] in wanted]
    missing = wanted - {task["name"] for task in selected}
    if missing:
        raise SystemExit(f"unknown task: {', '.join(sorted(missing))}")
    return selected


def joint_plan(manifest: list[dict], profile: str) -> dict:
    rows = sum(task["laptop_rows"] if profile == "laptop" else task["train_rows"] for task in manifest)
    epochs = JOINT_EPOCHS[profile]
    steps = steps_for(rows, JOINT_BATCH, epochs)
    return {
        "rows": rows,
        "epochs": epochs,
        "steps": steps,
        "hours": steps * JOINT_STEP_SECONDS / 3600,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one pointer head on all public benchmark tasks.")
    parser.add_argument("--profile", choices=("laptop", "full"), default="laptop")
    parser.add_argument("--task", action="append", help="Use only these tasks. Repeatable. Default is all of them.")
    parser.add_argument("--root", type=Path, default=Path("data/public"))
    parser.add_argument("--out", type=Path, default=Path("runs/public/joint/checkpoint.pt"))
    parser.add_argument("--dry-run", action="store_true", help="Print the step and time estimate and do not train.")
    parser.add_argument("--model", default="qwen")
    args = parser.parse_args()
    manifest_path = args.root / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit("missing data/public/manifest.json. Run: python -m decision_model.datasets.public")
    manifest = _select(json.loads(manifest_path.read_text()), args.task)
    plan = joint_plan(manifest, args.profile)
    print(f"one checkpoint  profile {args.profile}  max_len {JOINT_MAX_LEN}  batch {JOINT_BATCH}")
    for task in manifest:
        count = task["laptop_rows"] if args.profile == "laptop" else task["train_rows"]
        print(f"  {task['name']:20} {task['kind']:6} train {count:6}  eval {task['eval_rows']:5}")
    print(
        f"mixed train rows {plan['rows']}  epochs {plan['epochs']}"
        f"  steps {plan['steps']}  about {plan['hours']:.1f} hours"
    )
    if args.dry_run:
        return
    from decision_model.pretrained import resolve_model_id

    train_rows = []
    eval_rows = []
    per_task_eval = []
    for task in manifest:
        folder = args.root / task["name"]
        task_train = load_jsonl(folder / _train_file(task, args.profile))
        task_eval = load_jsonl(folder / "eval.jsonl")
        train_rows.extend(task_train)
        eval_rows.extend(task_eval)
        per_task_eval.append((task, task_eval))
    model_id = resolve_model_id(args.model)
    config = TrainConfig(
        epochs=plan["epochs"],
        seed=0,
        model=args.model,
        model_id=model_id,
        name=model_id,
        proj_size=256,
        max_len=JOINT_MAX_LEN,
        batch_size=JOINT_BATCH,
        lr=JOINT_LR,
    )
    model, encoder = train_model(train_rows, eval_rows, config, checkpoint_path=args.out)
    save_checkpoint(args.out, model, encoder, config)
    print(f"saved {args.out}", flush=True)
    print(f"temperature: {model.head.temperature:.3f}", flush=True)
    for task, rows in per_task_eval:
        scores = accuracy(model, rows, encoder, config.max_len, config.batch_size)
        print(f"{task['name']:20} {task['kind']:6} {scores[task['kind']]:.3f}", flush=True)


if __name__ == "__main__":
    main()
