import pytest
import torch

from decide.encoders import tree_mask
from decide.model import collate
from decide.packing import PAD, STATE, Choice, Question, Score, YesNo, allowed, pack

STATE_TEXT = "the card was charged twice for one order"
QUESTIONS = [Choice("which team should take this ticket", ["billing", "technical support", "sales"]),
             YesNo("is the customer asking for a refund"),
             Score("how urgent is it", ["can wait", "this week", "today", "now"])]


def test_questions_are_checked():
    with pytest.raises(ValueError):
        Question("maybe", "x")
    with pytest.raises(ValueError):
        Choice("x", ["only one"])
    with pytest.raises(ValueError):
        Choice("x", ["a", "a"])
    with pytest.raises(ValueError):
        Question("yesno", "x", ("a", "b"))


def test_layout(tok):
    p = pack(tok, STATE_TEXT, QUESTIONS)
    n_state = 1 + len(tok.encode(STATE_TEXT))
    assert p.positions[:n_state] == list(range(n_state))
    # Every question block starts just after the state, at the same position.
    assert [p.positions[m] for m in p.q_markers] == [n_state] * 3
    # Every option of a question starts at the same position.
    for q, markers in zip(QUESTIONS, p.opt_markers):
        assert len(markers) == len(q.options)
        assert len({p.positions[m] for m in markers}) == (1 if markers else 0)
    assert p.ids[p.q_markers[1] + 1] == tok.kind_ids["yesno"]
    assert p.qgroup[:n_state] == [STATE] * n_state


def test_tensor_mask_matches_the_reference(tok):
    packs = [pack(tok, STATE_TEXT, QUESTIONS), pack(tok, "need the payment reversed", QUESTIONS[1:])]
    batch = collate(packs, tok.pad_id)
    mask = tree_mask(batch.qgroup, batch.ogroup)
    for b, p in enumerate(packs):
        n = len(batch.ids[b])
        q = p.qgroup + [PAD] * (n - len(p))
        o = p.ogroup + [PAD] * (n - len(p))
        assert mask[b].tolist() == allowed(q, o)


def test_tree_rules(tok):
    p = pack(tok, STATE_TEXT, QUESTIONS)
    a = allowed(p.qgroup, p.ogroup)
    state = [i for i, g in enumerate(p.qgroup) if g == STATE]
    first, second = p.opt_markers[0][0], p.opt_markers[0][1]
    assert all(a[s][j] == (j in state) for s in state for j in range(len(p)))  # the state sees only itself
    assert not a[first][second] and not a[second][first]  # options do not see each other
    assert a[first][p.q_markers[0]] and not a[first][p.q_markers[1]]  # an option sees its own question only
    assert not a[p.q_markers[0]][first]  # a question does not see its options
    assert torch.tensor(a).diagonal().all()
