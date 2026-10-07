"""Confidence is arithmetic on the distribution, not a second model output."""

from __future__ import annotations

from collections.abc import Sequence


def choice_confidence(probs: Sequence[float]) -> float:
    """How far the leading option sits above a uniform distribution.

    For K > 1 this is (p_max - 1/K) / (1 - 1/K). One option is confidence 1.
    """
    k = len(probs)
    if k <= 1:
        return 1.0
    return (max(probs) - 1 / k) / (1 - 1 / k)


def score_confidence(probs: Sequence[float]) -> float:
    """1 when all mass is on one level, 0 when the spread is uniform or wider.

    D is the mean absolute distance of a uniform distribution from the middle
    of the levels. For three levels that middle distance is 2/3.
    """
    n = len(probs)
    mode = max(range(n), key=lambda index: probs[index])
    spread = sum(prob * abs(index - mode) for index, prob in enumerate(probs))
    middle = (n - 1) / 2
    width = sum(abs(index - middle) for index in range(n)) / n
    if width == 0:
        return 1.0
    return max(0.0, 1.0 - spread / width)


def expected_score(probs: Sequence[float]) -> float:
    return sum(index * prob for index, prob in enumerate(probs))
