"""Pointer-head decision model.

A request carries one state and any number of typed questions. Each question is
scored on its own row. A pointer head compares the decide token with each
option, and a softmax over those scores is the answer distribution.
"""

from decision_model.decide import decide, load_model
from decision_model.model import PointerHead

__all__ = ["PointerHead", "decide", "load_model"]
