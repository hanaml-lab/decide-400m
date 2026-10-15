"""The CPU toy: typed questions about requests from CLINC150 (Larson et al., 2019), with requests that no offered
option fits.

Five CLINC150 domains are in scope: banking, credit cards, travel, work, and auto and commute, 75 intents. Every
request (the *state*) gets three questions, packed in one sequence and answered in one pass:

* choice: "Which of these does the user want?" over eight intents of one domain. For a request in scope, the
  options are usually its own intent and seven others of its domain; for 15% of them, eight intents of another
  domain, so no option fits.
* yes/no: "Does the user want: <intent>?", the request's own intent half the time and another intent of its
  domain otherwise.
* score: "How closely does the request match: <intent>?" on the scale unrelated < same area < exact need, with
  the three levels equally often.

A fifth of the requests come from outside the five domains: CLINC150's out-of-scope requests and requests from its
other five domains. For those no choice option fits, every yes/no is no and every score is "unrelated".

Calibration and test questions are drawn from one pool (CLINC150's validation and test splits) and split at
random, so they are exchangeable, as the guarantee in ``risk.py`` assumes.
"""
from __future__ import annotations

import gzip
import json
import random
from dataclasses import dataclass
from pathlib import Path

from .packing import Choice, Question, Score, YesNo

DATA = Path(__file__).resolve().parents[2] / "data" / "clinc150.json.gz"
IN_SCOPE = ("banking", "credit_cards", "travel", "work", "auto_and_commute")
LEVELS = ("unrelated", "same area", "exact need")
#: Options offered with each choice: the menu an application would show.
OPTIONS = 8
CHOICE_TEXT = "Which of these does the user want?"
READABLE = {"apr": "APR", "w2": "W-2 form", "rollover_401k": "401k rollover", "pto_request": "PTO request",
            "pto_balance": "PTO balance", "pto_used": "PTO used", "pto_request_status": "PTO request status",
            "mpg": "miles per gallon", "uber": "ride share", "gas": "fuel level"}


def phrase(intent: str) -> str:
    return READABLE.get(intent, intent.replace("_", " "))


@dataclass
class Example:
    state: str
    questions: list[Question]
    targets: dict[str, int]
    none_fits: bool
    source: str  # "in scope", "out of scope" or "other domain"

    @property
    def answers(self) -> list[int]:
        """The target of each question, in order (as ``data.Record.answers``)."""
        return [self.targets[q.kind] for q in self.questions]


def load(path: Path = DATA) -> dict:
    with gzip.open(path, "rt") as f:
        return json.load(f)


def _questions(rng: random.Random, domains: dict, intent_domain: dict, text: str, intent: str | None) -> Example:
    own = intent_domain.get(intent) if intent else None
    # Choice.
    if own and rng.random() >= 0.15:
        others = [i for i in domains[own] if i != intent]
        options = [intent] + rng.sample(others, OPTIONS - 1)
    else:
        dom = rng.choice([d for d in IN_SCOPE if d != own])
        options = rng.sample(domains[dom], OPTIONS)
    rng.shuffle(options)
    choice_target = options.index(intent) + 1 if intent in options else 0
    # Yes/no.
    if own and rng.random() < 0.5:
        asked, yes = intent, 1
    else:
        pool = [i for i in domains[own] if i != intent] if own else [i for d in IN_SCOPE for i in domains[d]]
        asked, yes = rng.choice(pool), 0
    # Score.
    if own:
        level = rng.randrange(3)
        if level == 2:
            scored = intent
        elif level == 1:
            scored = rng.choice([i for i in domains[own] if i != intent])
        else:
            scored = rng.choice([i for d in IN_SCOPE if d != own for i in domains[d]])
    else:
        level, scored = 0, rng.choice([i for d in IN_SCOPE for i in domains[d]])
    questions = [
        Choice(CHOICE_TEXT, [phrase(o) for o in options]),
        YesNo(f"Does the user want: {phrase(asked)}?"),
        Score(f"How closely does the request match: {phrase(scored)}?", LEVELS),
    ]
    return Example(text, questions, {"choice": choice_target, "yesno": yes, "score": level}, choice_target == 0,
                   "in scope" if own else "")


def build(seed: int = 0, none_share: float = 0.2, calibration_share: float = 0.4, data: dict | None = None):
    """Returns (train, calibration, test) lists of ``Example``."""
    data = data or load()
    domains = {d: data["domains"][d] for d in data["domains"]}
    intent_domain = {i: d for d in IN_SCOPE for i in domains[d]}
    other = {i for d in domains if d not in IN_SCOPE for i in domains[d]}
    splits = data["splits"]
    rng = random.Random(seed)

    def gather(names: list[str], oos_names: list[str]):
        inside = [(t, i) for n in names for t, i in splits[n] if i in intent_domain]
        outside_other = [(t, i) for n in names for t, i in splits[n] if i in other]
        outside_oos = [(t, i) for n in oos_names for t, i in splits[n]]
        k = round(len(inside) * none_share / (1 - none_share))
        n_oos = min(len(outside_oos), k // 2)
        outside = rng.sample(outside_oos, n_oos) + rng.sample(outside_other, k - n_oos)
        rows = [(t, i, "in scope") for t, i in inside]
        rows += [(t, None, "out of scope" if i == "oos" else "other domain") for t, i in outside]
        rng.shuffle(rows)
        return rows

    def make(rows):
        out = []
        for text, intent, source in rows:
            ex = _questions(rng, domains, intent_domain, text, intent)
            ex.source = source
            out.append(ex)
        return out

    train = make(gather(["train"], ["oos_train"]))
    pool = make(gather(["val", "test"], ["oos_val", "oos_test"]))
    cut = round(len(pool) * calibration_share)
    return train, pool[:cut], pool[cut:]
