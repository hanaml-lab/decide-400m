from decide.toy import IN_SCOPE, LEVELS, OPTIONS, build, load, phrase


def test_build():
    data = load()
    train, cal, test = build(0, data=data)
    assert len(train) > 9000 and len(cal) > 1000 and len(test) > 1000
    for split in (train, cal, test):
        none = sum(ex.source != "in scope" for ex in split) / len(split)
        assert abs(none - 0.2) < 0.01
    intent = {t: i for name in ("val", "test") for t, i in data["splits"][name]}
    for ex in test[:300]:
        choice, yesno, score = ex.questions
        assert [q.kind for q in ex.questions] == ["choice", "yesno", "score"]
        assert len(choice.options) == OPTIONS and score.options == LEVELS
        if ex.targets["choice"]:
            assert choice.options[ex.targets["choice"] - 1] == phrase(intent[ex.state])
        else:
            assert ex.none_fits
        if ex.source != "in scope":
            assert ex.targets == {"choice": 0, "yesno": 0, "score": 0}


def test_build_is_seeded():
    a, b = build(3)[1][:50], build(3)[1][:50]
    assert [(x.state, x.targets) for x in a] == [(x.state, x.targets) for x in b]
    assert len(IN_SCOPE) == 5
