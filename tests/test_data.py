from pathlib import Path

import pytest

from decide.data import read, target
from decide.packing import Choice, YesNo

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "data" / "states.jsonl"


def test_the_example_file_reads():
    records = list(read(EXAMPLE))
    assert len(records) == 3
    first = records[0]
    assert [q.kind for q in first.questions] == ["choice", "yesno", "score"]
    assert first.answers == [1, 1, 1]
    assert records[2].answers[0] == 0  # no option fits


def test_targets_are_checked():
    with pytest.raises(ValueError):
        target(YesNo("x"), "yes")
    with pytest.raises(ValueError):
        target(Choice("x", ["a", "b"]), "c")
