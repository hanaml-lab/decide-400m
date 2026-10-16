import torch

from decide.model import Answer, ask, collate, loss, probabilities, verdicts
from decide.packing import Choice, Question, Score, YesNo, pack

STATE_TEXT = "the card was charged twice for one order and i need the second payment reversed"
CHOICE = Choice("which team should take this ticket", ["billing", "technical support", "account security", "sales"])
YN = YesNo("is the customer asking for a refund")
SCORE = Score("how urgent is it", ["can wait", "this week", "today", "now"])


def run(model, tok, questions, state=STATE_TEXT):
    return model(collate([pack(tok, state, questions)], tok.pad_id))


def test_options_are_permutation_equivariant(model, tok):
    a = run(model, tok, [CHOICE]).choice[0]
    order = [2, 0, 3, 1]
    shuffled = Choice(CHOICE.text, [CHOICE.options[i] for i in order])
    b = run(model, tok, [shuffled]).choice[0]
    assert torch.allclose(a[0], b[0], atol=1e-5)  # none fits
    assert torch.allclose(a[1:][order], b[1:], atol=1e-5)


def test_answers_do_not_depend_on_other_questions(model, tok):
    alone = run(model, tok, [YN])
    packed = run(model, tok, [SCORE, CHOICE, YN])
    assert torch.allclose(alone.yesno, packed.yesno, atol=1e-5)
    reordered = run(model, tok, [YN, CHOICE, SCORE])
    assert torch.allclose(packed.choice, reordered.choice, atol=1e-5)
    assert torch.allclose(packed.cuts, reordered.cuts, atol=1e-5)


def test_the_state_matters(model, tok):
    a = run(model, tok, [YN]).yesno
    b = run(model, tok, [YN], state="can wait this week").yesno
    assert not torch.allclose(a, b)


def test_probabilities_are_distributions(model, tok):
    out = run(model, tok, [CHOICE, YN, SCORE])
    p = probabilities(out, {"choice": 1.3, "yesno": 0.8, "score": 2.0})
    assert torch.allclose(p["choice"].sum(-1), torch.ones(1))
    assert torch.allclose(p["score"].sum(-1), torch.ones(1))
    assert 0 < p["yesno"].item() < 1
    picks = verdicts(p)
    assert picks["choice"][1].item() <= 1 - p["choice"][0, 0].item() + 1e-6


def test_loss_trains(model, tok):
    model.train()
    batch = collate([pack(tok, STATE_TEXT, [CHOICE, YN, SCORE])], tok.pad_id)
    targets = {"choice": torch.tensor([1]), "yesno": torch.tensor([1]), "score": torch.tensor([2])}
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for _ in range(30):
        value = loss(model(batch), targets)
        first = first or value.item()
        opt.zero_grad()
        value.backward()
        opt.step()
    assert value.item() < 0.2 * first


def test_ask(model, tok):
    answers = ask(model, tok, STATE_TEXT, [SCORE, CHOICE, YN])
    assert [a.question for a in answers] == [SCORE, CHOICE, YN]
    assert all(isinstance(a, Answer) and not a.decided for a in answers)  # nothing is decided uncalibrated
    assert abs(sum(answers[1].probs.values()) + answers[1].none - 1) < 1e-5
    assert answers[0].value in SCORE.options
    assert isinstance(answers[2].value, bool)


def test_question_kinds_are_validated():
    assert Question("yesno", "x").options == ()


def test_ask_moves_the_request_to_the_model(model, tok):
    import copy

    if not torch.cuda.is_available():
        return
    gpu = copy.deepcopy(model).cuda()
    answers = ask(gpu, tok, STATE_TEXT, [CHOICE, YN])
    assert len(answers) == 2 and abs(sum(answers[0].probs.values()) + answers[0].none - 1) < 1e-4
