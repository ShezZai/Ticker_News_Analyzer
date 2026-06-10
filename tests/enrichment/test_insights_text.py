import json

from ticker_news.enrichment.insights_text import (
    fuzzy_find_in_source,
    parse_boxes,
    split_box,
    verbatimize_quotes,
    with_headline,
)

BOX = "TOPIC: Data center demand\nINSIGHT: Hyperscaler capex is accelerating.\nQUOTES:\n- \"capex will grow 40%\""


def test_with_headline_prefixes_once():
    out = with_headline(BOX, "NVDA beats")
    assert out.startswith("ARTICLE_HEADLINE:")
    assert with_headline(out, "NVDA beats") == out  # idempotent


def test_split_box_roundtrip():
    topic, insight, quotes = split_box(with_headline(BOX, "NVDA beats"))
    assert topic == "Data center demand"
    assert insight == "Hyperscaler capex is accelerating."
    # split_box preserves the full line text including any '- ' prefix
    assert quotes == ['- "capex will grow 40%"']


def test_fuzzy_find_exact_match():
    article = "The CEO said capex will grow 40% next year."
    assert fuzzy_find_in_source("capex will grow 40%", article) == "capex will grow 40%"


def test_fuzzy_find_tolerates_small_differences():
    article = "The CEO said capital expenditure will grow forty percent next year."
    assert fuzzy_find_in_source("zzz qqq xxx", article) is None


def test_verbatimize_drops_unmatched_quotes():
    article = "Revenue rose 12% on cloud strength."
    quotes = ["Revenue rose 12%", "completely fabricated quote about llamas"]
    verbatim, dropped = verbatimize_quotes(quotes, article, 0.75)
    assert dropped == 1
    assert any("Revenue rose 12%" in q for q in verbatim)


def test_parse_boxes_strips_markdown_fences():
    # json.dumps produces properly-escaped JSON (\\n inside strings, not literal newlines)
    raw = "```json\n" + json.dumps({"boxes": ["TOPIC: X\nINSIGHT: Y\nQUOTES:\n- \"q\""]}) + "\n```"
    boxes = parse_boxes(raw)
    assert len(boxes) == 1
    assert boxes[0].startswith("TOPIC:")
