"""Download public benchmarks and write one JSONL file per dataset.

The files stay separate so each official test can be scored on its own.
Training mixes them into one pointer head. The head reads whatever options
are in the row, so one checkpoint covers choice, yes/no, and score. Official
test or validation splits are kept intact. Laptop training files are a fixed
stratified subset of the two largest training splits.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

# Seconds per optimizer step on the Mac that trained the support head
# (Apple MPS, frozen Qwen3-0.6B, October 2026). Short steps measured about
# 0.33s at batch 4. The estimates below use a slower rate so a longer
# sequence is not treated as free.
STEP_SECONDS = {"short": 0.5, "long": 1.0}
# One joint run pads every batch to the longest row, including 60-option
# MASSIVE questions. 0.8s is slower than the measured 0.33s short step.
JOINT_MAX_LEN = 1024
JOINT_BATCH = 4
JOINT_LR = 1e-4
JOINT_EPOCHS = {"laptop": 4, "full": 1}
JOINT_STEP_SECONDS = 0.8

# name, source, config, train split, eval split, kind, max_len, batch, epochs,
# laptop cap (None means the full train split), step class
TASKS: list[dict] = [
    {
        "name": "trec6",
        "source": "SetFit/TREC-QC",
        "config": None,
        "train_split": "train",
        "eval_split": "test",
        "kind": "choice",
        "question": "What type of question is this?",
        "max_len": 512,
        "batch_size": 8,
        "epochs": 8,
        "laptop_train": None,
        "step": "short",
        "text": "Official TREC test of 500 questions. Coarse type, six classes, Li and Roth.",
    },
    {
        "name": "emotion",
        "source": "mteb/emotion",
        "config": None,
        "train_split": "train",
        "eval_split": "test",
        "kind": "choice",
        "question": "Which emotion does this message express?",
        "max_len": 512,
        "batch_size": 8,
        "epochs": 6,
        "laptop_train": None,
        "step": "short",
        "text": "English tweets. Six emotions from Saravia et al. 2018. Test split.",
    },
    {
        "name": "sst2",
        "source": "stanfordnlp/sst2",
        "config": None,
        "train_split": "train",
        "eval_split": "validation",
        "kind": "noul",
        "question": "Is the sentiment of this sentence positive?",
        "max_len": 256,
        "batch_size": 8,
        "epochs": 3,
        "laptop_epochs": 6,
        "laptop_train": 8000,
        "step": "short",
        "text": "GLUE SST-2. Validation is the local score. The 1,821 test labels are hidden.",
    },
    {
        "name": "sst5",
        "source": "SetFit/sst5",
        "config": None,
        "train_split": "train",
        "eval_split": "test",
        "kind": "score",
        "question": "How positive is this sentence?",
        "max_len": 256,
        "batch_size": 8,
        "epochs": 8,
        "laptop_train": None,
        "step": "short",
        "text": "Five ordered sentiment levels, very negative through very positive. Test split.",
    },
    {
        "name": "phrasebank",
        "source": "mteb/financial_phrasebank",
        "config": "sentences_allagree",
        "train_split": "train",
        "eval_split": "test",
        "kind": "score",
        "question": "What is the financial sentiment, from negative to positive?",
        "max_len": 256,
        "batch_size": 16,
        "epochs": 20,
        "laptop_train": None,
        "step": "short",
        "text": "Financial PhraseBank sentences where every annotator agreed. Levels are negative, neutral, positive.",
    },
    {
        "name": "amazon_reviews_en",
        "source": "mteb/AmazonReviewsClassification",
        "config": "en",
        "train_split": "train",
        "eval_split": "test",
        "kind": "score",
        "question": "What star rating does this review give?",
        "max_len": 512,
        "batch_size": 8,
        "epochs": 1,
        "laptop_epochs": 6,
        "laptop_train": 8000,
        "step": "short",
        "text": "English Amazon reviews. Label 0 is 1 star and label 4 is 5 stars. Official test has 5,000 reviews.",
    },
    {
        "name": "massive_intent_en",
        "source": "mteb/amazon_massive_intent",
        "config": "en",
        "train_split": "train",
        "eval_split": "test",
        "kind": "choice",
        "question": "Which assistant intent matches this utterance?",
        "max_len": 1024,
        "batch_size": 4,
        "epochs": 6,
        "laptop_train": None,
        "step": "long",
        "text": "English MASSIVE, 60 intents, FitzGerald et al. 2022. Official test split.",
    },
]


def stratified_cap(rows: list[dict], limit: int | None, seed: int = 0) -> list[dict]:
    if limit is None or limit >= len(rows):
        return list(rows)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[_label_key(row)].append(row)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    weights = {key: len(group) for key, group in groups.items()}
    total = sum(weights.values())
    quota = {key: max(1, int(limit * weight / total)) for key, weight in weights.items()}
    while sum(quota.values()) > limit:
        key = max(quota, key=lambda name: quota[name])
        if quota[key] == 1:
            break
        quota[key] -= 1
    while sum(quota.values()) < limit:
        key = max(weights, key=lambda name: weights[name] - quota[name])
        if quota[key] >= weights[key]:
            break
        quota[key] += 1
    chosen: list[dict] = []
    for key, group in groups.items():
        chosen.extend(group[: quota[key]])
    rng.shuffle(chosen)
    return chosen


def _label_key(row: dict) -> str:
    question = next(iter(row["questions"].values()))
    return str(question["label"])


def choice_row(text: str, question: str, options: list[tuple[str, str]], label: str) -> dict:
    return {
        "state": text,
        "questions": {
            "answer": {
                "type": "choice",
                "instructions": question,
                "criteria": {key: description for key, description in options},
                "label": label,
            }
        },
    }


def noul_row(text: str, question: str, positive: bool) -> dict:
    return {
        "state": text,
        "questions": {
            "answer": {
                "type": "noul",
                "instructions": question,
                "criteria": {"true": "positive", "false": "negative"},
                "label": positive,
            }
        },
    }


def score_row(text: str, question: str, levels: list[str], label: int) -> dict:
    return {
        "state": text,
        "questions": {
            "answer": {
                "type": "score",
                "instructions": question,
                "criteria": list(levels),
                "label": label,
            }
        },
    }


def _load(task: dict, split: str):
    from datasets import load_dataset

    if task["config"]:
        return load_dataset(task["source"], task["config"], split=split)
    return load_dataset(task["source"], split=split)


def _text(row: dict) -> str:
    for key in ("text", "sentence"):
        if key in row and str(row[key]).strip():
            return str(row[key]).strip()
    raise ValueError("row has no text")


def _pairs(mapping: dict[int, tuple[str, str]]) -> list[tuple[str, str]]:
    return [mapping[index] for index in range(len(mapping))]


def _rows_for(task: dict, dataset, options) -> list[dict]:
    kind = task["kind"]
    question = task["question"]
    built = []
    for record in dataset:
        text = _text(record)
        if kind == "noul":
            built.append(noul_row(text, question, int(record["label"]) == 1))
        elif kind == "score":
            built.append(score_row(text, question, options, int(record["label"])))
        else:
            label = options["label_of"](record)
            built.append(choice_row(text, question, options["options"], label))
    return built


def _choice_spec(task: dict, train, evaluation) -> dict:
    if task["name"] == "trec6":
        found: dict[int, tuple[str, str]] = {}
        for row in list(train) + list(evaluation):
            found[int(row["label_coarse"])] = (row["label_coarse_original"], row["label_coarse_text"])
        options = _pairs(found)
        return {
            "options": options,
            "label_of": lambda row: row["label_coarse_original"],
        }
    if task["name"] == "emotion":
        found = {}
        for row in list(train) + list(evaluation):
            found[int(row["label"])] = (row["label_text"], row["label_text"])
        options = _pairs(found)
        return {"options": options, "label_of": lambda row: row["label_text"]}
    labels = sorted({row["label"] for row in list(train) + list(evaluation)})
    options = [(label, label.replace("_", " ")) for label in labels]
    return {"options": options, "label_of": lambda row: row["label"]}


def _score_levels(task: dict, train) -> list[str]:
    if task["name"] == "phrasebank":
        return ["negative", "neutral", "positive"]
    if task["name"] == "amazon_reviews_en":
        return ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
    found = {}
    for row in train:
        found[int(row["label"])] = row["label_text"]
    return [found[index] for index in range(len(found))]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def steps_for(train_rows: int, batch_size: int, epochs: int) -> int:
    return ((train_rows + batch_size - 1) // batch_size) * epochs


def prepare_public(root: Path | None = None) -> Path:
    destination = root or Path("data/public")
    destination.mkdir(parents=True, exist_ok=True)
    manifest = []
    for task in TASKS:
        print(f"loading {task['source']}")
        train = _load(task, task["train_split"])
        evaluation = _load(task, task["eval_split"])
        if task["kind"] == "choice":
            spec = _choice_spec(task, train, evaluation)
            train_rows = _rows_for(task, train, spec)
            eval_rows = _rows_for(task, evaluation, spec)
            option_names = [key for key, _description in spec["options"]]
        elif task["kind"] == "score":
            levels = _score_levels(task, train)
            train_rows = _rows_for(task, train, levels)
            eval_rows = _rows_for(task, evaluation, levels)
            option_names = levels
        else:
            train_rows = _rows_for(task, train, None)
            eval_rows = _rows_for(task, evaluation, None)
            option_names = ["negative", "positive"]
        folder = destination / task["name"]
        _write_jsonl(folder / "train.jsonl", train_rows)
        _write_jsonl(folder / "eval.jsonl", eval_rows)
        laptop_rows = stratified_cap(train_rows, task["laptop_train"])
        laptop_path = None
        if len(laptop_rows) != len(train_rows):
            _write_jsonl(folder / "train.laptop.jsonl", laptop_rows)
            laptop_path = "train.laptop.jsonl"
        record = {
            "name": task["name"],
            "source": task["source"],
            "config": task["config"],
            "kind": task["kind"],
            "question": task["question"],
            "options": option_names,
            "train_split": task["train_split"],
            "eval_split": task["eval_split"],
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "laptop_rows": len(laptop_rows),
            "laptop_file": laptop_path,
            "max_len": task["max_len"],
            "batch_size": task["batch_size"],
            "epochs": task["epochs"],
            "laptop_epochs": task.get("laptop_epochs", task["epochs"]),
            "step": task["step"],
            "note": task["text"],
        }
        manifest.append(record)
        print(f"  {task['name']}: train {len(train_rows)}  laptop {len(laptop_rows)}  eval {len(eval_rows)}")
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return destination


def estimate(manifest: list[dict], profile: str) -> list[dict]:
    rows = []
    for task in manifest:
        if profile == "laptop":
            train_rows = task["laptop_rows"]
            epochs = task["laptop_epochs"]
        else:
            train_rows = task["train_rows"]
            epochs = task["epochs"]
        steps = steps_for(train_rows, task["batch_size"], epochs)
        seconds = steps * STEP_SECONDS[task["step"]]
        rows.append({**task, "profile_rows": train_rows, "profile_epochs": epochs, "steps": steps, "hours": seconds / 3600})
    return rows


def main() -> None:
    root = prepare_public()
    manifest = json.loads((root / "manifest.json").read_text())
    print(f"wrote {root}")
    for profile in ("laptop", "full"):
        planned = estimate(manifest, profile)
        hours = sum(item["hours"] for item in planned)
        print(f"{profile}: about {hours:.1f} hours on the same Apple MPS machine")


if __name__ == "__main__":
    main()
