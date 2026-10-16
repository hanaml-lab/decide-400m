"""The decision model: an encoder read under the tree mask, and the three heads.

``DecisionModel.forward`` takes a batch of packed requests and returns raw logits grouped by kind. ``ask`` answers
the questions about one state with calibrated probabilities, and decides or defers each answer by the thresholds
of a ``Calibration`` (``calibrate.py``, ``risk.py``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .encoders import tree_mask
from .heads import ChoiceHead, ScoreHead, YesNoHead, level_probs
from .packing import PAD, Packed, Question, Separate, Tokenizer, pack


@dataclass
class Batch:
    """Encoder rows and, for every request, where each question's [Q] vector and each option's [OPT] vector are:
    ``q_at[b][i]`` and ``opt_at[b][i][k]`` are (row, token) coordinates. Tree and flat packing give one row per
    request; one pass per option (``collate_separate``) gives several."""

    ids: torch.Tensor
    positions: torch.Tensor
    qgroup: torch.Tensor
    ogroup: torch.Tensor
    packs: list
    q_at: list = field(default_factory=list)
    opt_at: list = field(default_factory=list)
    #: (row, start, end) of each question's text and each option's tokens, for the pooled readout.
    q_span: list = field(default_factory=list)
    opt_span: list = field(default_factory=list)

    @property
    def questions(self) -> list[list[Question]]:
        return [p.questions for p in self.packs]

    def to(self, device) -> "Batch":
        for name in ("ids", "positions", "qgroup", "ogroup"):
            setattr(self, name, getattr(self, name).to(device, non_blocking=True))
        return self


def _grid(rows: Sequence, fill: int, n: int) -> torch.Tensor:
    out = np.full((len(rows), n), fill, dtype=np.int64)
    for r, row in enumerate(rows):
        out[r, : len(row)] = row
    return torch.from_numpy(out)


def collate(packs: Sequence[Packed], pad_id: int) -> Batch:
    n = max(len(p) for p in packs)
    q_at = [[(b, m) for m in p.q_markers] for b, p in enumerate(packs)]
    opt_at = [[[(b, m) for m in ms] for ms in p.opt_markers] for b, p in enumerate(packs)]
    q_span = [[(b, s, e) for s, e in p.q_spans] for b, p in enumerate(packs)]
    opt_span = [[[(b, s, e) for s, e in sp] for sp in p.opt_spans] for b, p in enumerate(packs)]
    return Batch(_grid([p.ids for p in packs], pad_id, n), _grid([p.positions for p in packs], 0, n),
                 _grid([p.qgroup for p in packs], PAD, n), _grid([p.ogroup for p in packs], PAD, n), list(packs),
                 q_at, opt_at, q_span, opt_span)


def collate_separate(requests: Sequence[Separate], pad_id: int) -> Batch:
    """Batches requests read one pass per option: every pass of every request is a row."""
    seqs, q_at, opt_at, q_span, opt_span = [], [], [], [], []
    for r in requests:
        base = len(seqs)
        seqs += r.sequences
        q_at.append([(base + s, t) for s, t in r.q_at])
        opt_at.append([[(base + s, t) for s, t in at] for at in r.opt_at])
        q_span.append([(base + s, a, b) for s, a, b in r.q_span])
        opt_span.append([[(base + s, a, b) for s, a, b in sp] for sp in r.opt_span])
    n = max(len(p) for p in seqs)
    return Batch(_grid([p.ids for p in seqs], pad_id, n), _grid([p.positions for p in seqs], 0, n),
                 _grid([p.qgroup for p in seqs], PAD, n), _grid([p.ogroup for p in seqs], PAD, n), list(requests),
                 q_at, opt_at, q_span, opt_span)


@dataclass
class Logits:
    """Raw logits of every question in a batch, by kind. ``where[kind]`` lists (example, question) for each row."""

    choice: torch.Tensor  # (nc, 1 + K): slot 0 is none fits, -inf past a question's own options
    yesno: torch.Tensor  # (ny,)
    eta: torch.Tensor  # (ns,)
    cuts: torch.Tensor  # (ns, K - 1)
    where: dict[str, list[tuple[int, int]]] = field(default_factory=dict)


class DecisionModel(nn.Module):
    """The encoder and the three heads. ``readout`` says which vectors the heads read: ``"marker"``, each
    question's [Q] vector and each option's [OPT] vector; or ``"pool"`` (decide-400m), each of those plus the mean of
    the block's own tokens (the question's text, the option's words). Both keep every invariance of tree packing:
    a block's mean covers only its own tokens."""

    def __init__(self, encoder: nn.Module, heads: int = 4, none_slot: bool = True, readout: str = "marker"):
        super().__init__()
        if readout not in ("marker", "pool"):
            raise ValueError(f"readout must be 'marker' or 'pool', not {readout!r}")
        self.readout = readout
        self.encoder = encoder
        d = encoder.hidden_size
        # Per-feature standardisation of the vectors the heads read: identity until ``standardize`` sets it.
        self.register_buffer("feature_mean", torch.zeros(d))
        self.register_buffer("feature_std", torch.ones(d))
        self.choice_head = ChoiceHead(d, heads, none_slot=none_slot)
        self.yesno_head = YesNoHead(d)
        self.score_head = ScoreHead(d)

    def forward(self, batch: Batch) -> Logits:
        mask = tree_mask(batch.qgroup, batch.ogroup)
        return self.heads_from(self.encoder(batch.ids, batch.positions, mask), batch)

    @torch.no_grad()
    def standardize(self, vectors: torch.Tensor) -> None:
        """Fixes the per-feature mean and standard deviation the heads' inputs are standardised with, from a sample
        of [Q] and [OPT] vectors of the untrained encoder. A pretrained encoder's output vectors share a large
        common component; read raw, it swamps the freshly initialised heads, whose gradients then carry little
        signal (README, "Training"). The map is the same for every vector, so no invariance changes."""
        self.feature_mean.copy_(vectors.float().mean(0))
        self.feature_std.copy_(vectors.float().std(0).clamp_min(1e-4))

    def heads_from(self, h: torch.Tensor, batch: Batch) -> Logits:
        """The heads' logits from the encoder's output ``h`` for a batch."""
        d = h.shape[-1]
        scale = lambda v: ((v.float() - self.feature_mean) / self.feature_std).to(v.dtype)  # noqa: E731
        rows: dict[str, list[tuple[int, int]]] = {"choice": [], "yesno": [], "score": []}
        for b, questions in enumerate(batch.questions):
            for i, question in enumerate(questions):
                rows[question.kind].append((b, i))

        pool = self.readout == "pool"
        if pool:  # prefix sums: the mean of tokens [s, e) of row r is (cs[r, e] - cs[r, s]) / (e - s)
            cs = F.pad(h.float().cumsum(1), (0, 0, 1, 0))

        def mean_of(spans):
            """spans: a list of (row, start, end); the mean vector of each span, zero for an empty one."""
            r = torch.tensor([x for x, _, _ in spans], device=h.device)
            s = torch.tensor([y for _, y, _ in spans], device=h.device)
            e = torch.tensor([z for _, _, z in spans], device=h.device)
            total = cs[r, e] - cs[r, s]
            return (total / (e - s).clamp_min(1)[:, None].float()).to(h.dtype)

        def q_vectors(kind):
            idx = [batch.q_at[b][i] for b, i in rows[kind]]
            if not idx:
                return h.new_zeros((0, d))
            r, t = zip(*idx)
            v = h[torch.tensor(r, device=h.device), torch.tensor(t, device=h.device)]
            if pool:
                v = v + mean_of([batch.q_span[b][i] for b, i in rows[kind]])
            return scale(v)

        def option_vectors(kind):
            coords = [batch.opt_at[b][i] for b, i in rows[kind]]
            k = max(len(c) for c in coords)
            r = torch.tensor([[x for x, _ in c] + [0] * (k - len(c)) for c in coords], device=h.device)
            t = torch.tensor([[y for _, y in c] + [0] * (k - len(c)) for c in coords], device=h.device)
            valid = torch.tensor([[True] * len(c) + [False] * (k - len(c)) for c in coords], device=h.device)
            v = h[r, t]
            if pool:
                spans = [batch.opt_span[b][i] for b, i in rows[kind]]
                flat = [sp for c in spans for sp in c + [(0, 0, 0)] * (k - len(c))]
                v = v + mean_of(flat).view(len(spans), k, d)
            return scale(v) * valid[..., None], valid

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


def loss_terms(out: Logits, targets: dict[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, int]]:
    """Summed negative log-likelihood and question count of each kind, in fp32.

    ``targets["choice"]`` holds the slot of the right answer (0 when none fits, k + 1 for option k);
    ``targets["yesno"]`` holds 0 or 1; ``targets["score"]`` the index of the right level.
    """
    terms = {}
    if len(out.choice):
        terms["choice"] = (F.cross_entropy(out.choice.float(), targets["choice"], reduction="sum"), len(out.choice))
    if len(out.yesno):
        terms["yesno"] = (F.binary_cross_entropy_with_logits(out.yesno.float(), targets["yesno"].float(),
                                                             reduction="sum"), len(out.yesno))
    if len(out.eta):
        p = level_probs(out.eta.float(), out.cuts.float())
        terms["score"] = (-torch.log(p.gather(1, targets["score"][:, None]).clamp_min(1e-9)).sum(), len(out.eta))
    return terms


def loss(out: Logits, targets: dict[str, torch.Tensor]) -> torch.Tensor:
    """Mean negative log-likelihood over every question in the batch (``loss_terms`` gives it by kind)."""
    terms = loss_terms(out, targets)
    return torch.stack([v for v, _ in terms.values()]).sum() / max(sum(n for _, n in terms.values()), 1)


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
