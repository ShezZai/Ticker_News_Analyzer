from typing import Literal, get_args

from pydantic import BaseModel

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

CATEGORIES: list[str] = list(get_args(Category))


class Classification(BaseModel):
    """The structured verdict every classifier call must return."""

    category: Category
    reason: str = ""
