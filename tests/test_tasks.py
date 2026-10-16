"""The casting of public datasets into typed questions, checked offline on made-up examples."""
import random

from decide.tasks import (AG_LABELS, MENU, NONE_SHARE, TASKS, TREC_LABELS, Item, cast_clinc, cast_items, cast_menu,
                          cast_mnli, cast_topic, cast_yelp, select)


def test_topic_offers_every_label_or_all_but_the_right_one():
    cast = cast_topic(AG_LABELS, ("What is this about?",), ("Is this about {}?",))
    none = 0
    for i in range(2000):
        rows = cast(Item(f"text {i}", i % 4), random.Random(i), False)
        (choice, answer, tag), (yesno, yes, tag2) = rows
        assert choice.kind == "choice" and tag == "natural" and yesno.kind == "yesno" and tag2 == "derived"
        if answer is None:
            none += 1
            assert len(choice.options) == 3 and AG_LABELS[i % 4] not in choice.options
        else:
            assert answer == AG_LABELS[i % 4] and sorted(choice.options) == sorted(AG_LABELS)
        assert yes == (yesno.text == f"Is this about {AG_LABELS[i % 4]}?")
    assert abs(none / 2000 - NONE_SHARE) < 0.03


def test_a_large_label_set_offers_a_menu():
    labels = [f"intent {k}" for k in range(40)]
    cast = cast_menu(labels, "Which?", "Is it {}?")
    for i in range(300):
        (choice, answer, _), _ = cast(Item(f"t{i}", labels[i % 40]), random.Random(i), False)
        assert len(choice.options) == MENU and len(set(choice.options)) == MENU
        assert (answer is None) == (labels[i % 40] not in choice.options)


def test_trec_uses_its_six_labels():
    cast = cast_menu(TREC_LABELS, "What kind of answer?", "Is it {}?")
    (choice, answer, _), _ = cast(Item("Who wrote Hamlet ?", 3), random.Random(0), False)
    assert set(choice.options) <= set(TREC_LABELS)


def test_clinc_draws_distractors_from_the_domain():
    domains = {"banking": [f"b{k}" for k in range(15)], "travel": [f"t{k}" for k in range(15)]}
    cast = cast_clinc(domains)
    for i in range(200):
        rows = cast(Item(f"req {i}", "b3"), random.Random(i), False)
        choice, answer, _ = rows[0]
        assert [q.kind for q, _, _ in rows] == ["choice", "yesno", "score"]
        assert all(o.startswith("b") for o in choice.options)
    oos = cast(Item("tell me a joke", None), random.Random(1), False)
    assert oos[0][1] is None and oos[1][1] is False and oos[2][1] == "unrelated"


def test_mnli_and_yelp_ask_two_kinds():
    rows = cast_mnli(Item("A man sleeps.", 2, "A man runs."), random.Random(0), False)
    assert [(q.kind, a) for q, a, _ in rows] == [("yesno", False), ("choice", "false")]
    rows = cast_yelp(Item("Great food.", 4), random.Random(0), False)
    assert [(q.kind, a) for q, a, _ in rows] == [("score", "5 stars"), ("yesno", True)]
    assert len(cast_yelp(Item("Fine.", 2), random.Random(0), False)) == 1  # three stars: no yes/no


def test_select_and_cast_are_deterministic_and_order_free():
    items = [Item(f"text {i}", i % 4) for i in range(100)]
    a = select("ag_news", items, 30)
    b = select("ag_news", list(reversed(items)), 30)
    assert [x.text for x in a] == [x.text for x in b]
    ra = list(cast_items("ag_news", a, training=False))
    rb = list(cast_items("ag_news", b, training=False))
    assert ra == rb


def test_held_out_tasks_are_not_training_tasks():
    train = {k for k, t in TASKS.items() if t.train_cap}
    held = {k for k, t in TASKS.items() if not t.train_cap}
    assert held == {"trec", "banking77", "newsgroups", "rte", "scitail", "mrpc", "sst5", "stsb", "tweet_sentiment"}
    assert {TASKS[k].source for k in train}.isdisjoint({TASKS[k].source for k in held} - {"clinc"})
