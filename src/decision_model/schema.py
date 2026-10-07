"""Request schema for noul, choice, and score questions."""

from __future__ import annotations

from dataclasses import dataclass


class DecisionError(ValueError):
    def __init__(self, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class Question:
    name: str
    kind: str
    instructions: str
    keys: list[str]
    option_text: list[str]
    levels: list[str] | None = None
    gold: int | None = None


@dataclass(frozen=True)
class DecisionRequest:
    state: str
    questions: dict[str, Question]


def flatten(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(flatten(item) for item in value)
    if isinstance(value, dict):
        return "\n".join(f"{key}: {flatten(item)}" for key, item in value.items())
    raise DecisionError(f"unsupported value of type {type(value).__name__}")


def parse_request(payload: object, *, labeled: bool = False) -> DecisionRequest:
    if not isinstance(payload, dict):
        raise DecisionError("request must be an object")
    if "state" not in payload:
        raise DecisionError("state is required")
    state = flatten(payload["state"]).strip()
    raw_questions = payload.get("questions")
    if not isinstance(raw_questions, dict) or not raw_questions:
        raise DecisionError("questions must be a non-empty object")
    questions: dict[str, Question] = {}
    for name, raw in raw_questions.items():
        if not isinstance(name, str) or not name:
            raise DecisionError("question ids must be non-empty strings")
        questions[name] = _parse_question(name, raw, labeled=labeled)
    return DecisionRequest(state=state, questions=questions)


def _parse_question(name: str, raw: object, *, labeled: bool) -> Question:
    if not isinstance(raw, dict):
        raise DecisionError(f"{name}: question must be an object")
    kind = raw.get("type")
    if kind not in {"noul", "choice", "score"}:
        raise DecisionError(f"{name}: type must be noul, choice, or score")
    if "instructions" not in raw:
        raise DecisionError(f"{name}: instructions is required")
    instructions = flatten(raw["instructions"]).strip()
    if not instructions:
        raise DecisionError(f"{name}: instructions is empty")
    keys, option_text, levels = _parse_criteria(name, kind, raw.get("criteria", None))
    gold = _parse_label(name, kind, keys, raw.get("label", None), required=labeled)
    return Question(
        name=name,
        kind=kind,
        instructions=instructions,
        keys=keys,
        option_text=option_text,
        levels=levels,
        gold=gold,
    )


def _parse_criteria(
    name: str, kind: str, criteria: object
) -> tuple[list[str], list[str], list[str] | None]:
    if kind == "noul":
        false_text, true_text = "no", "yes"
        if criteria is not None:
            if not isinstance(criteria, dict):
                raise DecisionError(f"{name}: noul criteria must be an object with true and false")
            unknown = set(criteria) - {"true", "false"}
            if unknown:
                raise DecisionError(f"{name}: noul criteria only allows true and false")
            false_text = _noul_side(name, "false", criteria.get("false"), "no")
            true_text = _noul_side(name, "true", criteria.get("true"), "yes")
        return ["no", "yes"], [false_text, true_text], None
    if kind == "choice":
        if not isinstance(criteria, dict):
            raise DecisionError(f"{name}: choice criteria must be an object of option names")
        if not 2 <= len(criteria) <= 255:
            raise DecisionError(f"{name}: choice accepts 2 to 255 options")
        keys: list[str] = []
        texts: list[str] = []
        for key, description in criteria.items():
            if not isinstance(key, str) or not key.strip():
                raise DecisionError(f"{name}: choice option names must be non-empty strings")
            if description is None:
                text = key
            elif isinstance(description, str) and description.strip():
                text = f"{key}: {description.strip()}"
            else:
                raise DecisionError(f"{name}: option {key!r} description must be a string or null")
            keys.append(key)
            texts.append(text)
        return keys, texts, None
    if not isinstance(criteria, list):
        raise DecisionError(f"{name}: score criteria must be an array of levels, low to high")
    if not 2 <= len(criteria) <= 10:
        raise DecisionError(f"{name}: score accepts 2 to 10 levels")
    levels: list[str] = []
    for level in criteria:
        if not isinstance(level, str) or not level.strip():
            raise DecisionError(f"{name}: each score level must be a non-empty string")
        levels.append(level.strip())
    keys = [str(index) for index in range(len(levels))]
    return keys, list(levels), levels


def _noul_side(name: str, side: str, value: object, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, str) and value.strip():
        return f"{default}: {value.strip()}"
    raise DecisionError(f"{name}: noul {side} description must be a string or null")


def _parse_label(
    name: str, kind: str, keys: list[str], label: object, *, required: bool
) -> int | None:
    if label is None:
        if required:
            raise DecisionError(f"{name}: label is required")
        return None
    if kind == "noul":
        if not isinstance(label, bool):
            raise DecisionError(f"{name}: noul label must be true or false")
        return 1 if label else 0
    if kind == "choice":
        if not isinstance(label, str) or label not in keys:
            raise DecisionError(f"{name}: label must be one of the choice options")
        return keys.index(label)
    if isinstance(label, bool) or not isinstance(label, int):
        raise DecisionError(f"{name}: score label must be the level index, starting at 0")
    if not 0 <= label < len(keys):
        raise DecisionError(f"{name}: score label is outside 0..{len(keys) - 1}")
    return label
