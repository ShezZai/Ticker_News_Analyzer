from typing import Literal

from pydantic import BaseModel, Field

Action = Literal["buy", "sell", "hold"]


class Verdict(BaseModel):
    """The structured output of the sentiment synthesizer."""

    action: Action
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""
