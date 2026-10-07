"""Synthetic support tickets with a cue the pointer head can learn to point at."""

from __future__ import annotations

import json
import random
from pathlib import Path

TEAMS = {
    "billing": ["invoice", "charged", "refund", "payment"],
    "shipping": ["package", "delayed", "delivery", "tracking"],
    "access": ["password", "login", "locked", "username"],
}
ANGRY = ["furious", "angry", "unacceptable"]
CALM = ["please", "thanks", "wondering"]
URGENCY = {
    0: ["later", "whenever", "eventually"],
    1: ["soon", "today", "waiting"],
    2: ["asap", "immediately", "outage"],
}
TEMPLATES = [
    "Customer is {tone} about {cue} and needs help {urgency}.",
    "Ticket mentions {cue}. Tone is {tone}. Timing is {urgency}.",
]
CHOICE = {
    "billing": "Payments, invoices, refunds",
    "shipping": "Packages, delays, tracking",
    "access": "Login, password, account access",
}
LEVELS = ["Can wait", "Soon", "Immediate"]


def make_rows(n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        team = rng.choice(list(TEAMS))
        cue = rng.choice(TEAMS[team])
        angry = rng.random() < 0.5
        tone = rng.choice(ANGRY if angry else CALM)
        urgency = rng.choice([0, 1, 2])
        when = rng.choice(URGENCY[urgency])
        state = rng.choice(TEMPLATES).format(tone=tone, cue=cue, urgency=when)
        rows.append(
            {
                "state": state,
                "questions": {
                    "team": {
                        "type": "choice",
                        "instructions": "Which team should handle this ticket?",
                        "criteria": dict(CHOICE),
                        "label": team,
                    },
                    "angry": {
                        "type": "noul",
                        "instructions": "Is the customer angry?",
                        "label": angry,
                    },
                    "urgency": {
                        "type": "score",
                        "instructions": "How urgent is this ticket?",
                        "criteria": list(LEVELS),
                        "label": urgency,
                    },
                },
            }
        )
    return rows


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} has no rows")
    return rows


SAMPLE_REQUEST = {
    "state": "Customer is furious about invoice and needs help immediately.",
    "questions": {
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "criteria": dict(CHOICE),
        },
        "angry": {"type": "noul", "instructions": "Is the customer angry?"},
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this ticket?",
            "criteria": list(LEVELS),
        },
    },
}
