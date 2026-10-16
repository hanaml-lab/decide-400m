"""The ablation layouts and the trainer's batching, on the small encoder and a small random ModernBERT."""
import random

import numpy as np
import pytest
import torch

from decide.data import Record
from decide.model import collate, collate_separate
from decide.packing import STATE, Choice, Score, YesNo, pack, pack_separate
from decide.trainer import RunConfig, cost, drop_none_fits, micro_batches, steps_of

STATE_TEXT = "the card was charged twice for one order and i need the second payment reversed"
CHOICE = Choice("which team should take this ticket", ["billing", "technical support", "account security", "sales"])
YN = YesNo("is the customer asking for a refund")
SCORE = Score("how urgent is it", ["can wait", "this week", "today", "now"])


def test_flat_layout_is_one_ordinary_sequence(tok):
    tree = pack(tok, STATE_TEXT, [CHOICE, YN])
    flat = pack(tok, STATE_TEXT, [CHOICE, YN], layout="flat")
    assert flat.ids == tree.ids and flat.q_markers == tree.q_markers and flat.opt_markers == tree.opt_markers
    assert flat.positions == list(range(len(flat.ids)))
    assert set(flat.qgroup) == {STATE}


def test_flat_layout_hears_the_order_of_options(model, tok):
    a = model(collate([pack(tok, STATE_TEXT, [CHOICE], layout="flat")], tok.pad_id)).choice[0]
    order = [2, 0, 3, 1]
    shuffled = Choice(CHOICE.text, [CHOICE.options[i] for i in order])
    b = model(collate([pack(tok, STATE_TEXT, [shuffled], layout="flat")], tok.pad_id)).choice[0]
    assert not torch.allclose(a[1:][order], b[1:], atol=1e-4)


def test_one_pass_per_option(model, tok):
    s = pack_separate(tok, STATE_TEXT, [CHOICE, YN, SCORE])
    assert len(s.sequences) == (1 + 4) + 1 + (1 + 4)
    out = model(collate_separate([s], tok.pad_id))
    assert out.choice.shape == (1, 5) and out.yesno.shape == (1,) and out.cuts.shape == (1, 3)
    order = [3, 1, 0, 2]
    s2 = pack_separate(tok, STATE_TEXT, [Choice(CHOICE.text, [CHOICE.options[i] for i in order]), YN, SCORE])
    out2 = model(collate_separate([s2], tok.pad_id))
    assert torch.allclose(out.choice[0, 1:][order], out2.choice[0, 1:], atol=1e-5)
    assert torch.allclose(out.yesno, out2.yesno, atol=1e-5)


def test_separate_batches_mix_requests(model, tok):
    a = pack_separate(tok, STATE_TEXT, [CHOICE])
    b = pack_separate(tok, "can wait", [YN, SCORE])
    both = model(collate_separate([a, b], tok.pad_id))
    alone = model(collate_separate([b], tok.pad_id))
    assert torch.allclose(both.yesno, alone.yesno, atol=1e-5)
    assert both.where["choice"] == [(0, 0)] and both.where["score"] == [(1, 1)]


def test_no_none_slot(tok):
    from decide.encoders import TinyEncoder
    from decide.model import DecisionModel

    torch.manual_seed(0)
    m = DecisionModel(TinyEncoder(len(tok), d=32, layers=1, heads=4, ff=64, dropout=0.0), none_slot=False).eval()
    out = m(collate([pack(tok, STATE_TEXT, [CHOICE])], tok.pad_id))
    assert out.choice[0, 0].item() == float("-inf") and torch.isfinite(out.choice[0, 1:]).all()
    p = torch.softmax(out.choice, -1)
    assert p[0, 0].item() == 0.0 and abs(p.sum().item() - 1) < 1e-6


def test_drop_none_fits():
    recs = [Record("s", [CHOICE, YN], [0, 1], "t", ["natural", "derived"]),
            Record("s2", [CHOICE], [0], "t", ["natural"]),
            Record("s3", [CHOICE], [2], "t", ["natural"])]
    out = drop_none_fits(recs)
    assert [len(r.questions) for r in out] == [1, 1] and out[0].questions[0] is YN and out[0].tags == ["derived"]


def test_micro_batches_fit_the_budget(tok):
    rng = random.Random(0)
    items = [pack(tok, " ".join(["payment"] * rng.randint(1, 60)), [CHOICE, YN]) for _ in range(200)]
    for p in items:
        for name in ("ids", "positions", "qgroup", "ogroup"):
            setattr(p, name, np.asarray(getattr(p, name), dtype=np.int32))
    order = sorted(range(200), key=lambda i: len(items[i]))
    groups = micro_batches(items, order, 600)
    assert sorted(i for g in groups for i in g) == list(range(200))
    for g in groups:
        assert len(g) == 1 or len(g) * max(len(items[i]) for i in g) <= 600
    steps = steps_of(items, RunConfig(batch_size=32, token_budget=600), random.Random(0))
    assert sum(len(g) for s in steps for g in s) == 200 and all(sum(len(g) for g in s) <= 32 for s in steps)
    assert cost(items[0]) == (1, len(items[0]))


transformers = pytest.importorskip("transformers")


def test_modernbert_bool_masks_match_additive():
    from decide.encoders import TreeMaskedModernBert, tree_mask

    torch.manual_seed(0)
    config = transformers.ModernBertConfig(
        vocab_size=200, hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
        global_attn_every_n_layers=3, local_attention=8, max_position_embeddings=256, pad_token_id=0,
        attn_implementation="sdpa")
    bert = transformers.ModernBertModel(config).eval()
    enc = TreeMaskedModernBert(bert)
    ids = torch.randint(10, 200, (2, 24))
    pos = torch.arange(24).expand(2, -1)
    qg = torch.tensor([[-1] * 8 + [0] * 8 + [1] * 8] * 2)
    og = torch.tensor([[-1] * 8 + [-1] * 4 + [0] * 4 + [-1] * 8] * 2)
    mask = tree_mask(qg, og)
    with torch.no_grad():
        a = enc(ids, pos, mask)
        enc.additive = True
        b = enc(ids, pos, mask)
    assert torch.allclose(a, b, atol=1e-5)
    enc.checkpointing = True
    enc.train()
    out = enc(ids, pos, mask)
    out.sum().backward()
    assert bert.embeddings.tok_embeddings.weight.grad is not None


def test_pooled_readout_keeps_the_invariances(tok):
    from decide.encoders import TinyEncoder
    from decide.model import DecisionModel

    torch.manual_seed(0)
    m = DecisionModel(TinyEncoder(len(tok), d=32, layers=2, heads=4, ff=64, dropout=0.0), readout="pool").eval()
    a = m(collate([pack(tok, STATE_TEXT, [YN, CHOICE, SCORE])], tok.pad_id))
    order = [2, 0, 3, 1]
    shuffled = Choice(CHOICE.text, [CHOICE.options[i] for i in order])
    b = m(collate([pack(tok, STATE_TEXT, [shuffled, SCORE, YN])], tok.pad_id))
    assert torch.allclose(a.choice[0, 1:][order], b.choice[0, 1:], atol=1e-5)
    assert torch.allclose(a.choice[0, 0], b.choice[0, 0], atol=1e-5)
    assert torch.allclose(a.yesno, b.yesno, atol=1e-5) and torch.allclose(a.cuts, b.cuts, atol=1e-5)
    alone = m(collate([pack(tok, STATE_TEXT, [YN])], tok.pad_id))
    assert torch.allclose(alone.yesno, a.yesno, atol=1e-5)
    sep = m(collate_separate([pack_separate(tok, STATE_TEXT, [CHOICE, YN])], tok.pad_id))
    assert torch.isfinite(sep.choice[0, 1:]).all() and sep.yesno.shape == (1,)
