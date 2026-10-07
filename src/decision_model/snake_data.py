"""Snake boards written in the same words the live demo sends.

The gold move is a safe step that most reduces the distance to the food.
A trapped head, with no safe step, is left out of the training file.
"""

from __future__ import annotations

import random
from pathlib import Path

from decision_model.data import load_jsonl

WIDTH = 16
HEIGHT = 16
ORDER = ("UP", "DOWN", "LEFT", "RIGHT")
VECTORS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
NAMES = {"UP": "up", "DOWN": "down", "LEFT": "left", "RIGHT": "right"}


def _inside(x: int, y: int) -> bool:
    return 0 <= x < WIDTH and 0 <= y < HEIGHT


def _look(direction: str, head: tuple[int, int], body: list[tuple[int, int]], food: tuple[int, int]):
    dx, dy = VECTORS[direction]
    x, y = head[0] + dx, head[1] + dy
    if not _inside(x, y):
        return x, y, "wall"
    tail = body[-1]
    if (x, y) in set(body) and (x, y) != tail:
        return x, y, "body"
    if (x, y) == food:
        return x, y, "food"
    return x, y, "empty"


def _reachable(head: tuple[int, int], body: list[tuple[int, int]], food: tuple[int, int]) -> bool:
    blocked = set(body)
    blocked.discard(body[-1])
    seen = {head}
    queue = [head]
    while queue:
        x, y = queue.pop(0)
        for direction in ORDER:
            dx, dy = VECTORS[direction]
            nxt = (x + dx, y + dy)
            if not _inside(*nxt) or nxt in seen or nxt in blocked:
                continue
            if nxt == food:
                return True
            seen.add(nxt)
            queue.append(nxt)
    return False


def describe(body: list[tuple[int, int]], food: tuple[int, int], score: int = 0) -> tuple[str, str | None, bool, bool]:
    """Return state text, gold direction, safe-route label, and food-reachable label."""
    head = body[0]
    views = {direction: _look(direction, head, body, food) for direction in ORDER}
    dx, dy = food[0] - head[0], food[1] - head[1]
    east = f"{dx} columns east" if dx > 0 else f"{-dx} columns west" if dx < 0 else "the same column"
    north = f"{-dy} rows north" if dy < 0 else f"{dy} rows south" if dy > 0 else "the same row"
    blocked, open_moves, closer = [], [], []
    here = abs(dx) + abs(dy)
    for direction in ORDER:
        _x, _y, kind = views[direction]
        if kind in ("wall", "body"):
            blocked.append(NAMES[direction])
            continue
        open_moves.append(NAMES[direction])
        nxt = 0 if kind == "food" else abs(food[0] - _x) + abs(food[1] - _y)
        if kind == "food" or nxt < here:
            closer.append(NAMES[direction])
    cell_line = {
        "wall": "a wall",
        "body": "the snake's body",
        "food": "the food",
        "empty": "empty",
    }
    safe = bool(open_moves)
    food_open = _reachable(head, body, food)
    lines = [
        f"Snake on a {WIDTH} by {HEIGHT} board. Length {len(body)}. Score {score}.",
        f"Head at column {head[0] + 1}, row {head[1] + 1}.",
        f"Food is {east} and {north} of the head.",
        *[
            f"The cell {NAMES[direction]} of the head is {cell_line[views[direction][2]]}."
            for direction in ORDER
        ],
        f"Moves into a wall or the body: {', '.join(blocked) or 'none'}.",
        f"Moves into empty space or food: {', '.join(open_moves) or 'none'}.",
        f"Moves that reduce the distance to the food: {', '.join(closer) or 'none'}.",
        f"A safe route is available: {'yes' if safe else 'no'}.",
        f"Food is reachable through empty cells: {'yes' if food_open else 'no'}.",
    ]
    gold = _gold(views, head, food)
    return "\n".join(lines), gold, safe, food_open


def _gold(views: dict, head: tuple[int, int], food: tuple[int, int]) -> str | None:
    best: str | None = None
    best_distance: int | None = None
    for direction in ORDER:
        x, y, kind = views[direction]
        if kind in ("wall", "body"):
            continue
        distance = 0 if kind == "food" else abs(food[0] - x) + abs(food[1] - y)
        if best is None or distance < best_distance:
            best, best_distance = direction, distance
    return best


def _body(rng: random.Random) -> list[tuple[int, int]] | None:
    length = rng.randint(2, 8)
    for _ in range(40):
        head = (rng.randrange(WIDTH), rng.randrange(HEIGHT))
        body = [head]
        used = {head}
        stuck = False
        for _step in range(length - 1):
            options = []
            x, y = body[-1]
            for dx, dy in VECTORS.values():
                nxt = (x + dx, y + dy)
                if _inside(*nxt) and nxt not in used:
                    options.append(nxt)
            if not options:
                stuck = True
                break
            nxt = rng.choice(options)
            body.append(nxt)
            used.add(nxt)
        if not stuck and len(body) == length:
            return body
    return None


def snake_rows(n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    while len(rows) < n:
        body = _body(rng)
        if body is None:
            continue
        used = set(body)
        empties = [(x, y) for x in range(WIDTH) for y in range(HEIGHT) if (x, y) not in used]
        food = rng.choice(empties)
        state, gold, safe, food_open = describe(body, food)
        if gold is None:
            continue
        rows.append(
            {
                "state": state,
                "questions": {
                    "move": {
                        "type": "choice",
                        "instructions": "Which direction should the snake move?",
                        "criteria": {"UP": "up", "DOWN": "down", "LEFT": "left", "RIGHT": "right"},
                        "label": gold,
                    },
                    "safe": {
                        "type": "noul",
                        "instructions": "Is a safe route available?",
                        "label": safe,
                    },
                    "food": {
                        "type": "noul",
                        "instructions": "Is food reachable through empty cells?",
                        "label": food_open,
                    },
                },
            }
        )
    return rows


def public_sample(root: Path, per_task: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    manifest = root / "manifest.json"
    if not manifest.exists():
        return rows
    import json

    for task in json.loads(manifest.read_text()):
        folder = root / task["name"]
        path = folder / "train.laptop.jsonl"
        if not path.exists():
            path = folder / "train.jsonl"
        if not path.exists():
            continue
        chosen = load_jsonl(path)
        rng.shuffle(chosen)
        rows.extend(chosen[:per_task])
    rng.shuffle(rows)
    return rows
