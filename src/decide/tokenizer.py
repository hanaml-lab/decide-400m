"""Tokenizers for packing. ``WordTokenizer`` is the toy's: lower-cased words and punctuation, built from the
training text. ``HFTokenizer`` wraps a Hugging Face tokenizer (ModernBERT's in decide-400m) and adds the markers.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

from .packing import KINDS

WORD = re.compile(r"[a-z0-9]+|[^a-z0-9\s]")
SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[Q]", "[OPT]"] + [f"[{k.upper()}]" for k in KINDS]


class WordTokenizer:
    def __init__(self, vocab: list[str]):
        self.vocab = list(vocab)
        self.index = {w: i for i, w in enumerate(self.vocab)}
        self.pad_id, self.unk_id, self.cls_id, self.q_id, self.opt_id = (self.index[s] for s in SPECIALS[:5])
        self.kind_ids = {k: self.index[f"[{k.upper()}]"] for k in KINDS}

    @classmethod
    def build(cls, texts: Iterable[str], min_count: int = 1) -> "WordTokenizer":
        counts = Counter(w for text in texts for w in WORD.findall(text.lower()))
        words = sorted(w for w, c in counts.items() if c >= min_count)
        return cls(SPECIALS + words)

    def encode(self, text: str) -> list[int]:
        return [self.index.get(w, self.unk_id) for w in WORD.findall(text.lower())]

    def __len__(self) -> int:
        return len(self.vocab)


class HFTokenizer:
    """A Hugging Face tokenizer with the packing markers added as special tokens. Resize the encoder's embeddings
    to ``len(tokenizer)`` after making one."""

    def __init__(self, name: str = "answerdotai/ModernBERT-large"):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(name)
        markers = SPECIALS[3:]
        self.tok.add_special_tokens({"additional_special_tokens": markers})
        self.cls_id = self.tok.cls_token_id
        self.pad_id = self.tok.pad_token_id
        self.q_id, self.opt_id = self.tok.convert_tokens_to_ids(["[Q]", "[OPT]"])
        self.kind_ids = {k: self.tok.convert_tokens_to_ids(f"[{k.upper()}]") for k in KINDS}

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def __len__(self) -> int:
        return len(self.tok)
