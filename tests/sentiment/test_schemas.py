import pytest
from pydantic import ValidationError

from ticker_news.sentiment.schemas import Verdict


def test_verdict_valid():
    v = Verdict(action="buy", confidence=0.8, reasoning="strong guidance")
    assert v.action == "buy"


def test_verdict_rejects_unknown_action():
    with pytest.raises(ValidationError):
        Verdict(action="short", confidence=0.5)


def test_verdict_confidence_bounds():
    with pytest.raises(ValidationError):
        Verdict(action="hold", confidence=1.5)
