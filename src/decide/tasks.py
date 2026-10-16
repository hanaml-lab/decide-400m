"""The training mixture and the evaluation suites of decide-400m: public datasets cast as typed questions.

Every source is a file on the Hugging Face Hub pinned to a commit (``SOURCES``), except CLINC150, which ships with
this repository (``data/clinc150.json.gz``). ``scripts/prepare_data.py`` downloads them, casts every example into a
state and its questions, and writes JSON Lines in the format of ``decide.data``, with a manifest of the files'
SHA-256 digests.

**Tasks.** A task is one dataset asked one way. Training tasks (``TRAIN_TASKS``) give the training mixture and the
in-domain test suite (from their own test or validation splits). Held-out tasks (``HELD_OUT_TASKS``) are whole
datasets that no training example comes from; they give the zero-shot suite.

* choice. Training: CLINC150 intents (with its out-of-scope requests), AG News topics, DBpedia ontology classes,
  Yahoo Answers topics, and MultiNLI as a three-way choice. Held out: TREC question types, Banking77 intents and
  20 Newsgroups.
* yes/no. Training: MultiNLI (entailment against the rest) and BoolQ. Held out: RTE, SciTail and MRPC.
* score. Training: Yelp review stars and CLINC150 request-to-intent closeness. Held out: SST-5, STS-B and
  TweetEval sentiment.

**None fits.** A choice question offers a menu of options. For 15% of questions whose answer is among the labels,
the right label is left off the menu, so no option fits; CLINC150's out-of-scope requests never fit. A choice
task with more than ten labels offers eight (training: four to ten), the right one and the rest drawn at random;
CLINC150 draws them from the request's own domain, which is harder.

**Derived questions.** A state is asked more than one question when its label supports it, so that the questions
packed into one pass differ in kind: a choice task's state is also asked a yes/no about one label (the right one
half the time), MultiNLI's premise is asked a yes/no and a three-way choice about the same hypothesis, a Yelp review
is also asked whether it is positive (not for three stars), and a CLINC150 request is also asked how closely it
matches an intent. Questions are tagged ``natural`` or ``derived``.

**Sampling.** Which examples are used is decided by a hash of the task and the text, never by file order, and every
random choice about an example (its options, its derived question, its wording) is seeded by the same hash. The
same inputs always give the same questions. Evaluation states whose text also occurs in training are dropped.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from .packing import Question

ROOT = Path(__file__).resolve().parents[2]
CLINC = ROOT / "data" / "clinc150.json.gz"

#: Hub files, pinned to a commit. (repo id, commit, {split name: file}).
SOURCES: dict[str, tuple[str, str, dict[str, str]]] = {
    "ag_news": ("fancyzhx/ag_news", "eb185aade064a813bc0b7f42de02595523103ca4",
                {"train": "data/train-00000-of-00001.parquet", "test": "data/test-00000-of-00001.parquet"}),
    "dbpedia": ("fancyzhx/dbpedia_14", "9abd46cf7fc8b4c64290f26993c540b92aa145ac",
                {"train": "dbpedia_14/train-00000-of-00001.parquet", "test": "dbpedia_14/test-00000-of-00001.parquet"}),
    "yahoo": ("community-datasets/yahoo_answers_topics", "6652a1e7c94f7260a0bfd0c9092dd48e2d536ea1",
              {"train": "yahoo_answers_topics/train-00000-of-00002.parquet",
               "test": "yahoo_answers_topics/test-00000-of-00001.parquet"}),
    "yelp": ("Yelp/yelp_review_full", "c1f9ee939b7d05667af864ee1cb066393154bf85",
             {"train": "yelp_review_full/train-00000-of-00001.parquet",
              "test": "yelp_review_full/test-00000-of-00001.parquet"}),
    "mnli": ("nyu-mll/multi_nli", "da70db2af9d09693783c3320c4249840212ee221",
             {"train": "data/train-00000-of-00001.parquet", "test": "data/validation_matched-00000-of-00001.parquet"}),
    "boolq": ("google/boolq", "35b264d03638db9f4ce671b711558bf7ff0f80d5",
              {"train": "data/train-00000-of-00001.parquet", "test": "data/validation-00000-of-00001.parquet"}),
    "trec": ("CogComp/trec", "65752bf53af25bc935a0dce92fb5b6c930728450",  # the Hub's parquet conversion
             {"train": "default/train/0000.parquet", "test": "default/test/0000.parquet"}),
    "banking77": ("mteb/banking77", "18072d2685ea682290f7b8924d94c62acc19c0b2",
                  {"test": "data/test-00000-of-00001.parquet"}),
    "newsgroups": ("SetFit/20_newsgroups", "f1b91292074e7cfb69be58b642d583ec262f30ed", {"test": "test.jsonl"}),
    "rte": ("nyu-mll/glue", "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c",
            {"train": "rte/train-00000-of-00001.parquet", "test": "rte/validation-00000-of-00001.parquet"}),
    "mrpc": ("nyu-mll/glue", "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c",
             {"train": "mrpc/train-00000-of-00001.parquet", "test": "mrpc/test-00000-of-00001.parquet"}),
    "scitail": ("allenai/scitail", "0cc4353235b289165dfde1c7c5d1be983f99ce44",
                {"test": "snli_format/test-00000-of-00001.parquet"}),
    "sst5": ("SetFit/sst5", "e51bdcd8cd3a30da231967c1a249ba59361279a3", {"test": "test.jsonl"}),
    "stsb": ("mteb/stsbenchmark-sts", "96943a16ea6a35129e253c659081cb59daf81b30", {"test": "test.jsonl.gz"}),
    "tweet_sentiment": ("cardiffnlp/tweet_eval", "b3a375baf0f409c77e6bc7aa35102b7b3534f8be",
                        {"test": "sentiment/test-00000-of-00001.parquet"}),
}

AG_LABELS = ["world news", "sports", "business", "science and technology"]
DBPEDIA_LABELS = ["a company", "an educational institution", "an artist", "an athlete", "an office holder",
                  "a means of transportation", "a building", "a natural place", "a village", "an animal", "a plant",
                  "an album", "a film", "a written work"]
YAHOO_LABELS = ["society and culture", "science and mathematics", "health", "education and reference",
                "computers and internet", "sports", "business and finance", "entertainment and music",
                "family and relationships", "politics and government"]
TREC_LABELS = ["an abbreviation", "an entity or thing", "a description or explanation", "a person or group",
               "a location", "a number or quantity"]
NEWSGROUPS = {
    "alt.atheism": "atheism", "comp.graphics": "computer graphics",
    "comp.os.ms-windows.misc": "Microsoft Windows", "comp.sys.ibm.pc.hardware": "PC hardware",
    "comp.sys.mac.hardware": "Mac hardware", "comp.windows.x": "the X Window System", "misc.forsale": "items for sale",
    "rec.autos": "cars", "rec.motorcycles": "motorcycles", "rec.sport.baseball": "baseball",
    "rec.sport.hockey": "hockey", "sci.crypt": "cryptography", "sci.electronics": "electronics",
    "sci.med": "medicine", "sci.space": "space", "soc.religion.christian": "Christianity", "talk.politics.guns": "guns",
    "talk.politics.mideast": "Middle East politics", "talk.politics.misc": "politics",
    "talk.religion.misc": "religion"}
STARS = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
SENTIMENT5 = ["very negative", "negative", "neutral", "positive", "very positive"]
SENTIMENT3 = ["negative", "neutral", "positive"]
SIMILARITY = ["unrelated", "different but on the same topic", "somewhat related", "roughly equivalent",
              "mostly the same", "the same meaning"]
CLOSENESS = ["unrelated", "same area", "exact need"]
NLI3 = ["true", "false", "impossible to tell"]
#: MultiNLI's labels (0 entailment, 1 neutral, 2 contradiction) as answers to the three-way choice.
MNLI_ANSWER = {0: "true", 1: "impossible to tell", 2: "false"}
CLINC_READABLE = {"apr": "APR", "w2": "W-2 form", "rollover_401k": "401k rollover", "pto_request": "PTO request",
                  "pto_balance": "PTO balance", "pto_used": "PTO used", "pto_request_status": "PTO request status",
                  "mpg": "miles per gallon", "uber": "ride share", "gas": "fuel level"}

NONE_SHARE = 0.15  # choice questions whose right label is left off the menu
MENU = 8  # options offered by a choice task with more than ten labels, at evaluation
MAX_LABELS_ALL = 10  # a task with at most this many labels offers all of them


@dataclass
class Item:
    """One example before it is cast: the state's text and its label(s)."""

    text: str
    label: object
    extra: str = ""  # a second text: a hypothesis, a question, a second sentence


def _rng(*parts) -> random.Random:
    return random.Random(int(hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()[:16], 16))


def _key(task: str, text: str) -> str:
    return hashlib.sha256(f"{task}\x1f{text}".encode()).hexdigest()


def _clean(text: str) -> str:
    text = text.replace("\\n", " ").replace("\\\"", "\"")
    return re.sub(r"\s+", " ", text).strip()


def clinc_phrase(intent: str) -> str:
    return CLINC_READABLE.get(intent, intent.replace("_", " "))


def _menu(rng: random.Random, gold: str, labels: list[str], training: bool, fits: bool = True) -> tuple[list[str], str | None]:
    """The options offered and the answer (None when no option fits). A task with at most ten labels offers all of
    them (all but the right one when none fits); a larger one offers eight (training: four to ten)."""
    others = [x for x in labels if x != gold]
    if len(labels) <= MAX_LABELS_ALL:
        options = list(labels) if fits else others
    else:
        k = rng.randint(4, 10) if training else MENU
        options = [gold] + rng.sample(others, k - 1) if fits else rng.sample(others, k)
    rng.shuffle(options)
    return options, (gold if fits else None)


def _pick(rng: random.Random, training: bool, *wordings: str) -> str:
    """Training draws a wording at random; evaluation always uses the first."""
    return rng.choice(wordings) if training else wordings[0]


# ----------------------------------------------------------------------------------------------- casting functions
# Each takes (item, rng, training) and returns a list of (question, answer, tag); answers are as in decide.data.

def cast_topic(labels: list[str], choice_texts: tuple[str, ...], yes_texts: tuple[str, ...]):
    def cast(item: Item, rng: random.Random, training: bool):
        gold = labels[int(item.label)]
        fits = rng.random() >= NONE_SHARE
        options, answer = _menu(rng, gold, labels, training, fits=fits)
        out = [(Question("choice", _pick(rng, training, *choice_texts), tuple(options)), answer, "natural")]
        asked = gold if rng.random() < 0.5 else rng.choice([x for x in labels if x != gold])
        out.append((Question("yesno", _pick(rng, training, *yes_texts).format(asked)), asked == gold, "derived"))
        return out
    return cast


def cast_clinc(domains: dict[str, list[str]]):
    intent_domain = {i: d for d, xs in domains.items() for i in xs}
    every = [i for xs in domains.values() for i in xs]

    def cast(item: Item, rng: random.Random, training: bool):
        intent = item.label  # None for an out-of-scope request
        own = intent_domain.get(intent)
        if own is None:
            dom = rng.choice(sorted(domains))
            options, answer = rng.sample(domains[dom], MENU if not training else rng.randint(4, 10)), None
        else:
            fits = rng.random() >= NONE_SHARE
            k = MENU if not training else rng.randint(4, 10)
            others = [i for i in domains[own] if i != intent]
            options = ([intent] + rng.sample(others, k - 1)) if fits else rng.sample(others, k)
            answer = intent if fits else None
        rng.shuffle(options)
        text = _pick(rng, training, "Which of these does the user want?", "What is the user asking for?",
                     "Which request is this?")
        out = [(Question("choice", text, tuple(clinc_phrase(o) for o in options)),
                clinc_phrase(answer) if answer else None, "natural")]
        if own and rng.random() < 0.5:
            asked, yes = intent, True
        else:
            pool = [i for i in domains[own] if i != intent] if own else every
            asked, yes = rng.choice(pool), False
        out.append((Question("yesno", _pick(rng, training, "Does the user want: {}?", "Is the user asking about: {}?")
                             .format(clinc_phrase(asked))), yes, "derived"))
        if own:
            level = rng.randrange(3)
            scored = intent if level == 2 else (rng.choice([i for i in domains[own] if i != intent]) if level == 1
                                                else rng.choice([i for d in domains if d != own for i in domains[d]]))
        else:
            level, scored = 0, rng.choice(every)
        out.append((Question("score", _pick(rng, training, "How closely does the request match: {}?")
                             .format(clinc_phrase(scored)), CLOSENESS), CLOSENESS[level], "derived"))
        return out
    return cast


def cast_mnli(item: Item, rng: random.Random, training: bool):
    hyp = item.extra
    label = int(item.label)  # 0 entailment, 1 neutral, 2 contradiction
    yes_text = _pick(rng, training, "Does it follow from the text that: {}", "Is this implied by the text: {}")
    choice_text = _pick(rng, training, "Given the text, is this statement true, false or impossible to tell: {}",
                        "Is the following true, false or impossible to tell from the text: {}")
    options = list(NLI3)
    rng.shuffle(options)
    return [(Question("yesno", yes_text.format(hyp)), label == 0, "natural"),
            (Question("choice", choice_text.format(hyp), tuple(options)), MNLI_ANSWER[label], "derived")]


def cast_boolq(item: Item, rng: random.Random, training: bool):
    q = item.extra.strip()
    return [(Question("yesno", q[0].upper() + q[1:] + "?"), bool(item.label), "natural")]


def cast_yelp(item: Item, rng: random.Random, training: bool):
    stars = int(item.label)
    text = _pick(rng, training, "How many stars did the reviewer give?", "What star rating goes with this review?")
    out = [(Question("score", text, tuple(STARS)), STARS[stars], "natural")]
    if stars != 2:
        out.append((Question("yesno", _pick(rng, training, "Is the review positive?", "Did the reviewer like it?")),
                    stars > 2, "derived"))
    return out


def cast_yesno(template: str):
    def cast(item: Item, rng: random.Random, training: bool):
        return [(Question("yesno", template.format(item.extra.strip())), bool(item.label), "natural")]
    return cast


def cast_score(levels: list[str], template: str):
    def cast(item: Item, rng: random.Random, training: bool):
        return [(Question("score", template.format(item.extra.strip()), tuple(levels)), levels[int(item.label)],
                 "natural")]
    return cast


def cast_menu(labels: list[str], choice_text: str, yes_text: str):
    """A held-out choice task with more labels than fit a menu (Banking77, 20 Newsgroups), or few (TREC)."""
    def cast(item: Item, rng: random.Random, training: bool):
        gold = item.label if isinstance(item.label, str) else labels[int(item.label)]
        fits = rng.random() >= NONE_SHARE
        options, answer = _menu(rng, gold, labels, training, fits=fits)
        out = [(Question("choice", choice_text, tuple(options)), answer, "natural")]
        asked = gold if rng.random() < 0.5 else rng.choice([x for x in labels if x != gold])
        out.append((Question("yesno", yes_text.format(asked)), asked == gold, "derived"))
        return out
    return cast


# ------------------------------------------------------------------------------------------------------ readers

def _frame(path: str):
    import pandas as pd

    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".jsonl.gz"):
        return pd.read_json(path, lines=True, compression="gzip")
    return pd.read_json(path, lines=True)


def read_items(task: str, split: str, fetch: Callable[[str, str], str]) -> list[Item]:
    """The examples of a task's split. ``fetch(source, split)`` returns a local path to the source file."""
    if task == "clinc":
        data = json.load(gzip.open(CLINC, "rt"))
        name = {"train": ("train", "oos_train"), "test": ("test", "oos_test")}[split]
        rows = [Item(t, i) for t, i in data["splits"][name[0]]] + [Item(t, None) for t, _ in data["splits"][name[1]]]
        return rows
    src = TASKS[task].source
    df = _frame(fetch(src, split))
    if src == "ag_news":
        return [Item(_clean(t), int(y)) for t, y in zip(df.text, df.label)]
    if src == "dbpedia":
        return [Item(_clean(f"{a}. {b}"), int(y)) for a, b, y in zip(df.title, df.content, df.label)]
    if src == "yahoo":
        return [Item(_clean(f"{a} {b} Answer: {c}"), int(y))
                for a, b, c, y in zip(df.question_title, df.question_content.fillna(""), df.best_answer.fillna(""),
                                      df.topic)]
    if src == "yelp":
        return [Item(_clean(t), int(y)) for t, y in zip(df.text, df.label)]
    if src == "mnli":
        return [Item(_clean(p), int(y), _clean(h)) for p, h, y in zip(df.premise, df.hypothesis, df.label) if y >= 0]
    if src == "boolq":
        return [Item(_clean(p), bool(y), _clean(q)) for p, q, y in zip(df.passage, df.question, df.answer)]
    if src == "trec":
        return [Item(_clean(t), int(y)) for t, y in zip(df.text, df.coarse_label)]
    if src == "banking77":
        rows = [Item(_clean(t), str(y).replace("_", " ")) for t, y in zip(df.text, df.label_text)]
        TASKS["banking77"].cast = cast_menu(sorted({it.label for it in rows}), "Which of these does the customer want?",
                                            "Is the customer asking about: {}?")
        return rows
    if src == "newsgroups":
        return [Item(_clean(t), NEWSGROUPS[y]) for t, y in zip(df.text, df.label_text) if len(_clean(t)) > 20]
    if src == "rte":  # 0 entailment, 1 not entailment
        return [Item(_clean(a), int(y) == 0, _clean(b)) for a, b, y in zip(df.sentence1, df.sentence2, df.label)]
    if src == "mrpc":  # 1 equivalent
        return [Item(_clean(a), int(y) == 1, _clean(b)) for a, b, y in zip(df.sentence1, df.sentence2, df.label)]
    if src == "scitail":
        return [Item(_clean(a), y == "entailment", _clean(b))
                for a, b, y in zip(df.sentence1, df.sentence2, df.gold_label) if y in ("entailment", "neutral")]
    if src == "sst5":
        return [Item(_clean(t), int(y)) for t, y in zip(df.text, df.label)]
    if src == "stsb":
        return [Item(_clean(a), min(5, int(float(s) + 0.5)), _clean(b)) for a, b, s in zip(df.sentence1, df.sentence2, df.score)]
    if src == "tweet_sentiment":
        return [Item(_clean(t), int(y)) for t, y in zip(df.text, df.label)]
    raise KeyError(task)


@dataclass
class Task:
    name: str  # as printed in tables
    kind: str  # the kind of its natural question
    source: str  # key of SOURCES, or "clinc"
    cast: Callable
    train_cap: int = 0  # training states (0: not a training task)
    eval_cap: int = 2000  # evaluation states
    eval_splits: tuple[str, ...] = ("test",)
    licence: str = ""  # as the dataset's Hub card (or, for Banking77, PolyAI's original card) states it


def _clinc_domains() -> dict[str, list[str]]:
    return json.load(gzip.open(CLINC, "rt"))["domains"]


TASKS: dict[str, Task] = {
    # Training tasks.
    "clinc": Task("CLINC150", "choice", "clinc", cast_clinc(_clinc_domains()), 15100, 2000, licence="CC BY 3.0"),
    "ag_news": Task("AG News", "choice", "ag_news", cast_topic(
        AG_LABELS, ("What is this news article about?", "Which section does this article belong in?"),
        ("Is this article about {}?", "Is this a {} story?")), 15000, licence="not stated"),
    "dbpedia": Task("DBpedia-14", "choice", "dbpedia", cast_topic(
        DBPEDIA_LABELS, ("What kind of thing does this text describe?", "What is the subject of this entry?"),
        ("Is this text about {}?", "Does this entry describe {}?")), 15000, licence="CC BY-SA 3.0"),
    "yahoo": Task("Yahoo Answers", "choice", "yahoo", cast_topic(
        YAHOO_LABELS, ("What is the topic of this question?", "Which category was this question asked in?"),
        ("Is this question about {}?", "Was this asked under {}?")), 15000, licence="not stated"),
    "mnli": Task("MultiNLI", "yesno", "mnli", cast_mnli, 30000, licence="OANC, CC BY-SA 3.0 and others"),
    "boolq": Task("BoolQ", "yesno", "boolq", cast_boolq, 9427, licence="CC BY-SA 3.0"),
    "yelp": Task("Yelp-5", "score", "yelp", cast_yelp, 15000, licence="Yelp dataset agreement"),
    # Held-out tasks: no training example comes from these datasets.
    "trec": Task("TREC-6", "choice", "trec", cast_menu(
        TREC_LABELS, "What kind of answer is this question asking for?", "Is this question asking for {}?"),
        eval_splits=("test", "train"), licence="not stated"),
    "banking77": Task("Banking77", "choice", "banking77", None, licence="CC BY 4.0"),  # cast set once labels are read
    "newsgroups": Task("20 Newsgroups", "choice", "newsgroups", cast_menu(
        sorted(NEWSGROUPS.values()), "Which discussion group was this posted to?", "Was this posted to a group about {}?"),
        licence="not stated"),
    "rte": Task("RTE", "yesno", "rte", cast_yesno("Based on the text, is it true that: {}"),
                eval_splits=("test", "train"), licence="GLUE: the source's terms"),
    "scitail": Task("SciTail", "yesno", "scitail", cast_yesno("Does the text support this claim: {}"),
                    licence="not stated"),
    "mrpc": Task("MRPC", "yesno", "mrpc", cast_yesno("Does it say the same thing as this sentence: {}"),
                 eval_splits=("test", "train"), licence="GLUE: the source's terms"),
    "sst5": Task("SST-5", "score", "sst5", cast_score(SENTIMENT5, "How positive is this movie review?{}"),
                 licence="not stated"),
    "stsb": Task("STS-B", "score", "stsb", cast_score(SIMILARITY, "How close in meaning is it to this sentence: {}"),
                 licence="not stated"),
    "tweet_sentiment": Task("TweetEval sentiment", "score", "tweet_sentiment", cast_score(
        SENTIMENT3, "What is the sentiment of this tweet?{}"), licence="per subset, not stated"),
}

TRAIN_TASKS = [k for k, t in TASKS.items() if t.train_cap]
HELD_OUT_TASKS = [k for k, t in TASKS.items() if not t.train_cap]


def select(task: str, items: list[Item], cap: int) -> list[Item]:
    """The ``cap`` items with the smallest hash of (task, text): a sample that does not depend on file order."""
    seen, unique = set(), []
    for it in items:
        k = (it.text, it.extra)
        if k not in seen:
            seen.add(k)
            unique.append(it)
    unique.sort(key=lambda it: _key(task, it.text + "\x1f" + it.extra))
    return unique[:cap] if cap else unique


def cast_items(task: str, items: list[Item], training: bool, seed: int = 0) -> Iterator[dict]:
    """JSON-ready records: state, questions with answers and tags, the task, and whether the state is held out."""
    t = TASKS[task]
    for it in items:
        rng = _rng(seed, task, it.text, it.extra, "train" if training else "eval")
        rows = t.cast(it, rng, training)
        qs = []
        for q, answer, tag in rows:
            row = {"kind": q.kind, "text": q.text, "answer": answer, "tag": tag}
            if q.options:
                row["options"] = list(q.options)
            qs.append(row)
        yield {"task": task, "state": it.text, "questions": qs}
