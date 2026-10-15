"""The tree mask on the real ModernBERT architecture, with a tiny random configuration (no download)."""
import pytest
import torch

transformers = pytest.importorskip("transformers")

from decide.encoders import TreeMaskedModernBert, tree_mask  # noqa: E402
from decide.model import DecisionModel, collate  # noqa: E402
from decide.packing import Choice, YesNo, pack  # noqa: E402


@pytest.fixture
def bert():
    torch.manual_seed(0)
    config = transformers.ModernBertConfig(
        vocab_size=200, hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
        global_attn_every_n_layers=3, local_attention=8, max_position_embeddings=256, pad_token_id=0,
        attn_implementation="sdpa")
    return transformers.ModernBertModel(config).eval()


class Tok:
    cls_id, q_id, opt_id, pad_id = 1, 2, 3, 0
    kind_ids = {"choice": 4, "yesno": 5, "score": 6}

    def encode(self, text):
        return [10 + (hash(w) % 180) for w in text.split()]


def test_a_flat_sequence_matches_the_model(bert):
    ids = torch.randint(10, 200, (2, 30))
    full = torch.ones(2, 30, 30, dtype=torch.bool)
    pos = torch.arange(30).expand(2, -1)
    with torch.no_grad():
        ours = TreeMaskedModernBert(bert)(ids, pos, full)
        theirs = bert(input_ids=ids).last_hidden_state
    assert torch.allclose(ours, theirs, atol=1e-5)


def test_options_are_equivariant_under_modernbert(bert):
    torch.manual_seed(1)
    tok = Tok()
    model = DecisionModel(TreeMaskedModernBert(bert)).eval()
    state = "the order arrived damaged and the customer wants a replacement shipped today " * 2
    options = ["refund", "replacement", "repair visit", "store credit", "escalate to a manager"]
    order = [4, 2, 0, 3, 1]
    with torch.no_grad():
        a = model(collate([pack(tok, state, [YesNo("is it urgent"), Choice("what next", options)])], 0))
        b = model(collate([pack(tok, state, [Choice("what next", [options[i] for i in order]), YesNo("is it urgent")])], 0))
    assert torch.allclose(a.choice[0, 1:][order], b.choice[0, 1:], atol=1e-4)
    assert torch.allclose(a.yesno, b.yesno, atol=1e-4)


def test_local_layers_window_by_position(bert):
    tok = Tok()
    p = pack(tok, "a b c d e f g h i j k l", [Choice("q", ["x", "y"])])
    batch = collate([p], 0)
    mask = tree_mask(batch.qgroup, batch.ogroup)
    with torch.no_grad():
        h = TreeMaskedModernBert(bert)(batch.ids, batch.positions, mask)
    assert torch.isfinite(h).all()
