"""Typed questions about a state, and how they are packed into one encoder pass.

A request is a *state* (any text: a support ticket, a log line, an agent step, serialised JSON) and a list of typed
questions. Each question is one of three kinds:

* ``choice``: pick one of the given options. The answer is a distribution over the options plus a *none fits* slot.
* ``yesno``: a Bernoulli probability.
* ``score``: a position on an ordered scale whose levels are given low to high. The answer is a distribution over
  the levels from a cumulative-link head.

**Tree packing.** The state and every question are read in a single forward pass. The sequence is laid out as

    [CLS] state ... | [Q] <kind> question 1 ... [OPT] option 1.1 ... [OPT] option 1.2 ... | [Q] <kind> question 2 ...

but attention follows a tree, not the order of the sequence:

* state tokens attend to the state only;
* the tokens of question q attend to the state and to question q;
* the tokens of option k of question q attend to the state, to question q and to option k.

Every question block starts at the same position id (just after the state), and every option of a question starts
at the same position id (just after its question). With rotary positions, nothing in the encoder can then tell
which question or option came first: permuting the options of a question permutes their vectors exactly, and a
question's answer does not depend on the other questions packed with it. The state, usually the long part, is
encoded once for all questions. Options meet each other only in the choice head's set layer (``heads.py``), which
has no positions either.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

KINDS = ("choice", "yesno", "score")

#: Group label of the state's tokens, and of padding.
STATE = -1
PAD = -2


@dataclass(frozen=True)
class Question:
    """One typed question. ``options`` holds a choice's options, or a score's levels from low to high."""

    kind: str
    text: str
    options: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, not {self.kind!r}")
        if self.kind == "yesno" and self.options:
            raise ValueError("a yes/no question takes no options")
        if self.kind == "choice" and len(self.options) < 2:
            raise ValueError("a choice needs at least two options")
        if self.kind == "score" and len(self.options) < 2:
            raise ValueError("a score needs at least two levels, low to high")
        if len(set(self.options)) != len(self.options):
            raise ValueError("options must be distinct")


def Choice(text: str, options: Sequence[str]) -> Question:
    return Question("choice", text, tuple(options))


def YesNo(text: str) -> Question:
    return Question("yesno", text)


def Score(text: str, levels: Sequence[str]) -> Question:
    return Question("score", text, tuple(levels))


class Tokenizer(Protocol):
    """What packing needs from a tokenizer: text to ids, and the ids of its markers."""

    cls_id: int
    q_id: int
    opt_id: int
    pad_id: int
    kind_ids: dict[str, int]

    def encode(self, text: str) -> list[int]: ...


@dataclass
class Packed:
    """One state and its questions as a single sequence.

    ``qgroup[i]`` is the index of the question token i belongs to (``STATE`` for the state, ``PAD`` for padding) and
    ``ogroup[i]`` the index of its option within that question (``STATE`` when it is not in an option). Together
    they define the attention tree; ``positions`` are the rotary position ids.
    """

    ids: list[int]
    positions: list[int]
    qgroup: list[int]
    ogroup: list[int]
    questions: list[Question]
    #: Index of each question's [Q] marker.
    q_markers: list[int] = field(default_factory=list)
    #: For each question, the index of each option's [OPT] marker (empty for yes/no).
    opt_markers: list[list[int]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.ids)


def pack(tok: Tokenizer, state: str, questions: Sequence[Question], max_state: int = 4096,
         max_question: int = 64, max_option: int = 32) -> Packed:
    """Lays out a state and its questions for one tree-masked pass. Long texts are cut at the given lengths."""
    if not questions:
        raise ValueError("ask at least one question")
    state_ids = [tok.cls_id] + tok.encode(state)[: max_state - 1]
    ids = list(state_ids)
    positions = list(range(len(state_ids)))
    qgroup = [STATE] * len(state_ids)
    ogroup = [STATE] * len(state_ids)
    start = len(state_ids)
    p = Packed(ids, positions, qgroup, ogroup, list(questions))
    for qi, question in enumerate(questions):
        head = [tok.q_id, tok.kind_ids[question.kind]] + tok.encode(question.text)[: max_question - 2]
        p.q_markers.append(len(ids))
        ids += head
        positions += range(start, start + len(head))
        qgroup += [qi] * len(head)
        ogroup += [STATE] * len(head)
        markers = []
        option_start = start + len(head)
        for oi, option in enumerate(question.options):
            body = [tok.opt_id] + tok.encode(option)[: max_option - 1]
            markers.append(len(ids))
            ids += body
            positions += range(option_start, option_start + len(body))
            qgroup += [qi] * len(body)
            ogroup += [oi] * len(body)
        p.opt_markers.append(markers)
    return p


def allowed(qgroup: Sequence[int], ogroup: Sequence[int]) -> list[list[bool]]:
    """The attention tree as a nested list: ``allowed[i][j]`` is true when token i may attend to token j.

    ``torch_mask`` in ``model.py`` builds the same matrix as a tensor; this version is for reading and testing.
    """
    n = len(qgroup)
    out = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if qgroup[i] == PAD:
                out[i][j] = qgroup[j] in (STATE, PAD)
            elif qgroup[j] == STATE:
                out[i][j] = True
            elif qgroup[j] == qgroup[i] and qgroup[i] != STATE:
                out[i][j] = ogroup[j] == STATE or ogroup[j] == ogroup[i]
    return out
