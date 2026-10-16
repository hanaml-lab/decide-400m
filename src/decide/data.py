"""Decision questions as JSON Lines, one state per line:

    {"state": "...", "questions": [
        {"kind": "choice", "text": "Which team should take this?", "options": ["Billing", "Support"], "answer": "Billing"},
        {"kind": "choice", "text": "...", "options": ["..."], "answer": null},      # no option fits
        {"kind": "yesno", "text": "Is the customer asking for a refund?", "answer": true},
        {"kind": "score", "text": "How urgent is it?", "options": ["can wait", "today", "now"], "answer": "today"}]}

Every question of a state is answered in the same pass, so a state may carry several questions of one kind;
``model.loss`` and ``calibrate`` treat each question on its own.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .packing import Question


@dataclass
class Record:
    state: str
    questions: list[Question]
    #: The right answer of each question, as a target: a choice's slot (0 when none fits, k + 1 for option k), a
    #: yes/no as 0 or 1, a score's level index.
    answers: list[int] = field(default_factory=list)
    #: The task the state comes from, and each question's tag ("natural" or "derived"), when the file gives them.
    task: str = ""
    tags: list[str] = field(default_factory=list)


def target(question: Question, answer) -> int:
    if question.kind == "choice":
        return 0 if answer is None else question.options.index(answer) + 1
    if question.kind == "yesno":
        if not isinstance(answer, bool):
            raise ValueError(f"a yes/no answer is true or false, not {answer!r}")
        return int(answer)
    return question.options.index(answer)


def read(path: str | Path) -> Iterator[Record]:
    with open(path) as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            questions, answers, tags = [], [], []
            for q in row["questions"]:
                tags.append(q.get("tag", ""))
                question = Question(q["kind"], q["text"], tuple(q.get("options", ())))
                questions.append(question)
                try:
                    answers.append(target(question, q.get("answer")))
                except ValueError as err:
                    raise ValueError(f"{path}:{n}: {err}") from err
            yield Record(row["state"], questions, answers, row.get("task", ""), tags)
