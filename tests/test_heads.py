import torch

from decide.heads import ChoiceHead, ScoreHead, level_probs


def test_cutpoints_increase_and_levels_sum_to_one():
    torch.manual_seed(0)
    head = ScoreHead(16)
    levels = torch.randn(5, 6, 16)
    valid = torch.ones(5, 6, dtype=torch.bool)
    valid[0, 3:] = False  # a three-level scale padded to six
    eta, cuts = head(torch.randn(5, 16), levels, valid)
    finite = cuts[1:]
    assert (finite[:, 1:] > finite[:, :-1]).all()
    assert torch.isinf(cuts[0, 2:]).all()
    p = level_probs(eta, cuts)
    assert torch.allclose(p.sum(-1), torch.ones(5), atol=1e-6)
    assert (p[0, 3:] == 0).all()


def test_temperature_flattens_the_levels():
    eta, cuts = torch.tensor([0.0]), torch.tensor([[-1.0, 1.0]])
    sharp, flat = level_probs(eta, cuts, 0.5), level_probs(eta, cuts, 4.0)
    assert sharp.max() > flat.max()


def test_choice_head_masks_missing_options():
    torch.manual_seed(0)
    head = ChoiceHead(16).eval()
    options = torch.randn(2, 4, 16)
    valid = torch.tensor([[True, True, False, False], [True] * 4])
    logits = head(torch.randn(2, 16), options, valid)
    assert logits.shape == (2, 5)
    assert torch.isinf(logits[0, 3:]).all() and torch.isfinite(logits[1]).all()
    # Changing a masked option changes nothing.
    options2 = options.clone()
    options2[0, 2:] = 7.0
    q = torch.zeros(2, 16)
    assert torch.allclose(head(q, options, valid)[0, :3], head(q, options2, valid)[0, :3], atol=1e-6)
