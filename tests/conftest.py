import pytest
import torch

from decide.encoders import TinyEncoder
from decide.model import DecisionModel
from decide.tokenizer import WordTokenizer

WORDS = ("the card was charged twice for one order and i need the second payment reversed before friday "
         "which team should take this ticket billing technical support account security sales is the customer "
         "asking for a refund how urgent is it can wait this week today now does user want")


@pytest.fixture
def tok():
    return WordTokenizer.build([WORDS])


@pytest.fixture
def model(tok):
    torch.manual_seed(0)
    m = DecisionModel(TinyEncoder(len(tok), d=32, layers=2, heads=4, ff=64, dropout=0.0))
    return m.eval()
