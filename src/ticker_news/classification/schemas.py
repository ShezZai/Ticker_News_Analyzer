from typing import Literal

from pydantic import BaseModel

CATEGORIES = [
    "conference-PR",
    "marketing fluff",
    "real news",
    "recap/review",
    "market speculation",
    "legal solicitation",
    "regulatory filing",
    "book PR",
    "politics/macro",
    "other",
]

Category = Literal[
    "conference-PR",
    "marketing fluff",
    "real news",
    "recap/review",
    "market speculation",
    "legal solicitation",
    "regulatory filing",
    "book PR",
    "politics/macro",
    "other",
]


class Classification(BaseModel):
    """The structured verdict every classifier call must return."""

    category: Category
    reason: str = ""
