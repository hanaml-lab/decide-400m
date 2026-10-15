"""The decision model: an encoder read under the tree mask, and the three heads.

``DecisionModel.forward`` takes a batch of packed requests and returns raw logits grouped by kind. ``ask`` answers
the questions about one state with calibrated probabilities, and decides or defers each answer by the thresholds
of a ``Calibration`` (``calibrate.py``, ``risk.py``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .encoders import tree_mask
from .heads import ChoiceHead, ScoreHead, YesNoHead, level_probs
from .packing import PAD, Packed, Question, Tokenizer, pack


@dataclass
class Batch:
    ids: torch.Tensor
    positions: torch.Tensor
    qgroup: torch.Tensor
    ogroup: torch.Tensor
    packs: list[Packed]


def collate(packs: Sequence[Packed], pad_id: int) -> Batch:
    n = max(len(p) for p in packs)
    def grid(fill, rows):
        return torch.tensor([row + [fill] * (n - len(row)) for row in rows], dtype=torch.long)
    return Batch(grid(pad_id, [p.ids for p in packs]), grid(0, [p.positions for p in packs]),
                 grid(PAD, [p.qgroup for p in packs]), grid(PAD, [p.ogroup for p in packs]), list(packs))


@dataclass
class Logits:
    """Raw logits of every question in a batch, by kind. ``where[kind]`` lists (example, question) for each row."""

    choice: torch.Tensor  # (nc, 1 + K): slot 0 is none fits, -inf past a question's own options
    yesno: torch.Tensor  # (ny,)
    eta: torch.Tensor  # (ns,)
    cuts: torch.Tensor  # (ns, K - 1)
    where: dict[str, list[tuple[int, int]]] = field(default_factory=dict)


class DecisionModel(nn.Module):
    def __init__(self, encoder: nn.Module, heads: int = 4):
        super().__init__()
        self.encoder = encoder
        d = encoder.hidden_size
        self.choice_head = ChoiceHead(d, heads)
        self.yesno_head = YesNoHead(d)
        self.score_head = ScoreHead(d)

    def forward(self, batch: Batch) -> Logits:
        mask = tree_mask(batch.qgroup, batch.ogroup)
        h = self.encoder(batch.ids, batch.positions, mask)
        d = h.shape[-1]
        rows: dict[str, list[tuple[int, int]]] = {"choice": [], "yesno": [], "score": []}
        for b, p in enumerate(batch.packs):
            for i, question in enumerate(p.questions):
                rows[question.kind].append((b, i))

        def q_vectors(kind):
            idx = [(b, batch.packs[b].q_markers[i]) for b, i in rows[kind]]
            if not idx:
                return h.new_zeros((0, d))
            bs, ts = zip(*idx)
            return h[list(bs), list(ts)]

        def option_vectors(kind):
            where = rows[kind]
            markers = [batch.packs[b].opt_markers[i] for b, i in where]
            k = max(len(m) for m in markers)
            index = torch.tensor([m + [0] * (k - len(m)) for m in markers], device=h.device)
            valid = torch.tensor([[True] * len(m) + [False] * (k - len(m)) for m in markers], device=h.device)
            rows_b = torch.tensor([b for b, _ in where], device=h.device)[:, None].expand(-1, k)
            return h[rows_b, index] * valid[..., None], valid

        if rows["choice"]:
            options, valid = option_vectors("choice")
            choice = self.choice_head(q_vectors("choice"), options, valid)
        else:
            choice = h.new_zeros((0, 3))
        yesno = self.yesno_head(q_vectors("yesno")) if rows["yesno"] else h.new_zeros((0,))
        if rows["score"]:
            levels, valid = option_vectors("score")
            eta, cuts = self.score_head(q_vectors("score"), levels, valid)
        else:
            eta, cuts = h.new_zeros((0,)), h.new_zeros((0, 1))
        return Logits(choice, yesno, eta, cuts, rows)


def loss(out: Logits, targets: dict[str, torch.Tensor]) -> torch.Tensor:
    """Mean negative log-likelihood over every question in the batch.

    ``targets["choice"]`` holds the slot of the right answer (0 when none fits, k + 1 for option k);
    ``targets["yesno"]`` holds 0 or 1; ``targets["score"]`` the index of the right level.
    """
    terms, count = [], 0
    if len(out.choice):
        terms.append(F.cross_entropy(out.choice, targets["choice"], reduction="sum"))
        count += len(out.choice)
    if len(out.yesno):
        terms.append(F.binary_cross_entropy_with_logits(out.yesno, targets["yesno"].float(), reduction="sum"))
        count += len(out.yesno)
    if len(out.eta):
        p = level_probs(out.eta, out.cuts)
        terms.append(-torch.log(p.gather(1, targets["score"][:, None]).clamp_min(1e-9)).sum())
        count += len(out.eta)
    return torch.stack(terms).sum() / max(count, 1)


@dataclass
class Answer:
    """One typed answer. ``probs`` maps each option (or level) to its probability; for yes/no it holds "yes" and
    "no". ``value`` is the answer the model would give, ``confidence`` the probability it is right, and ``decided``
    whether that clears the threshold for its kind. A choice also reports ``none``, the mass on *none fits*."""

    question: Question
    probs: dict[str, float]
    value: object
    confidence: float
    decided: bool
    none: Optional[float] = None

    def __str__(self) -> str:
        verdict = "decided" if self.decided else "deferred"
        return f"{self.question.kind:6s} {verdict:8s} {self.value!s:24.24s} p={self.confidence:.3f}"


def probabilities(out: Logits, temperatures: dict[str, float]) -> dict[str, torch.Tensor]:
    """Calibrated probabilities by kind: choice (nc, 1 + K) with slot 0 none fits; yes/no (ny,); score (ns, K)."""
    return {
        "choice": torch.softmax(out.choice / temperatures.get("choice", 1.0), dim=-1),
        "yesno": torch.sigmoid(out.yesno / temperatures.get("yesno", 1.0)),
        "score": level_probs(out.eta, out.cuts, temperatures.get("score", 1.0)),
    }


def verdicts(probs: dict[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """The answer each question would get and its confidence, by kind.

    A choice answers its most probable *option*; its confidence is that option's probability, so mass on none fits
    lowers it. A yes/no answers yes when p >= 0.5, with confidence max(p, 1 - p). A score answers its most probable
    level, with that level's probability as confidence.
    """
    c = probs["choice"][:, 1:]
    conf_c, pick_c = c.max(dim=-1)
    y = probs["yesno"]
    s = probs["score"]
    conf_s, pick_s = s.max(dim=-1)
    return {"choice": (pick_c, conf_c), "yesno": ((y >= 0.5).long(), torch.maximum(y, 1 - y)), "score": (pick_s, conf_s)}


@torch.no_grad()
def ask(model: DecisionModel, tok: Tokenizer, state: str, questions: Sequence[Question],
        calibration=None) -> list[Answer]:
    """Answers every question about ``state`` in one forward pass. With no calibration, temperatures are 1 and no
    answer is decided."""
    model.eval()
    packed = pack(tok, state, questions)
    out = model(collate([packed], tok.pad_id))
    temps = calibration.temperatures if calibration else {}
    probs = probabilities(out, temps)
    picks = verdicts(probs)
    answers: list[Answer] = [None] * len(questions)  # type: ignore[list-item]
    for kind, rows in out.where.items():
        for r, (_, i) in enumerate(rows):
            question = questions[i]
            pick, conf = picks[kind][0][r].item(), picks[kind][1][r].item()
            threshold = calibration.thresholds.get(kind, float("inf")) if calibration else float("inf")
            if kind == "choice":
                p = probs["choice"][r]
                table = {o: p[k + 1].item() for k, o in enumerate(question.options)}
                answers[i] = Answer(question, table, question.options[pick], conf, conf >= threshold, p[0].item())
            elif kind == "yesno":
                py = probs["yesno"][r].item()
                answers[i] = Answer(question, {"yes": py, "no": 1 - py}, bool(pick), conf, conf >= threshold)
            else:
                p = probs["score"][r]
                table = {o: p[k].item() for k, o in enumerate(question.options)}
                answers[i] = Answer(question, table, question.options[pick], conf, conf >= threshold)
    return answers
