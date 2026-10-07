"""Turn one question into a causal row the pointer head can read.

Layout, matching the open pointer-head reconstructions:

    <state> state words
    <q> instruction words
    <opt> option words </opt>
    ...
    <decide>

The option vector is the hidden state at </opt>. Under a causal mask that is
the first position that has seen the whole option. The decide token is last,
so it can attend to every option before the head compares them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from decision_model.schema import DecisionError, Question

PAD, UNK, STATE, Q, OPT, OPT_END, DECIDE = range(7)
SPECIAL_TOKENS = {
    "<pad>": PAD,
    "<unk>": UNK,
    "<state>": STATE,
    "<q>": Q,
    "<opt>": OPT,
    "</opt>": OPT_END,
    "<decide>": DECIDE,
}
_WORDS = re.compile(r"\S+")
_FORGED = re.compile(r"</?(?:state|q|opt|decide)>")


def sanitize(text: str) -> str:
    """Break delimiter spellings so caller text cannot invent an option boundary."""
    return _FORGED.sub(lambda match: match.group(0).replace("<", "< ", 1), text)


def words(text: str) -> list[str]:
    return _WORDS.findall(sanitize(text))


@dataclass(frozen=True)
class Markers:
    pad: int
    state: int
    q: int
    opt: int
    opt_end: int
    decide: int


@dataclass
class Vocab:
    stoi: dict[str, int]

    @classmethod
    def build(cls, texts: list[str]) -> Vocab:
        stoi: dict[str, int] = {}
        for text in texts:
            for word in words(text):
                if word not in stoi:
                    stoi[word] = len(SPECIAL_TOKENS) + len(stoi)
        return cls(stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(word, UNK) for word in words(text)]

    @property
    def markers(self) -> Markers:
        return Markers(PAD, STATE, Q, OPT, OPT_END, DECIDE)

    @property
    def size(self) -> int:
        if not self.stoi:
            return len(SPECIAL_TOKENS)
        return max(self.stoi.values()) + 1

    def dump(self) -> dict[str, int]:
        return dict(self.stoi)

    @classmethod
    def load(cls, stoi: dict[str, int]) -> Vocab:
        return cls(dict(stoi))


@dataclass(frozen=True)
class Branch:
    ids: list[int]
    opt_index: list[int]
    decide_index: int
    truncated: bool
    state_tokens: int
    branch_tokens: int
    gold: int | None = None


def texts_for_vocab(state: str, questions: list[Question]) -> list[str]:
    texts = [state]
    for question in questions:
        texts.append(question.instructions)
        texts.extend(question.option_text)
    return texts


def render_question(encoder: object, state: str, question: Question, max_len: int) -> Branch:
    """Encode state plus this question only. Sibling questions are not in the row."""
    markers = encoder.markers
    instruction = encoder.encode(question.instructions)
    spans = [[markers.opt, *encoder.encode(text), markers.opt_end] for text in question.option_text]
    suffix = [markers.q, *instruction]
    opt_from_suffix: list[int] = []
    for span in spans:
        suffix.extend(span)
        opt_from_suffix.append(len(suffix) - 1)
    suffix.append(markers.decide)
    if len(suffix) + 1 > max_len:
        raise DecisionError(
            f"{question.name}: question does not fit in the {max_len}-token context",
        )
    state_ids = encoder.encode(state)
    full_state_tokens = 1 + len(state_ids)
    budget = max_len - len(suffix)
    prefix = [markers.state, *state_ids][:budget]
    ids = prefix + suffix
    offset = len(prefix)
    opt_index = [offset + index for index in opt_from_suffix]
    decide_index = len(ids) - 1
    if ids[decide_index] != markers.decide or any(ids[index] != markers.opt_end for index in opt_index):
        raise RuntimeError("option markers were not placed on the closing token")
    return Branch(
        ids=ids,
        opt_index=opt_index,
        decide_index=decide_index,
        truncated=full_state_tokens > len(prefix),
        state_tokens=len(prefix),
        branch_tokens=len(suffix),
    )
