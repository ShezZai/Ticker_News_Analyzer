from ticker_news.enrichment.tagging import build_annotator, build_matcher, compute_row

DATA = {
    "NVDA": ("NVIDIA Corporation", "GPUs"),
    "AMD": ("Advanced Micro Devices", "GPUs"),
    "AI": ("C3.ai", "AI Software"),
}


def test_matcher_finds_company_name():
    find = build_matcher(DATA)
    assert "NVDA" in find("NVIDIA Corporation announced record data center revenue.")


def test_matcher_finds_cashtag_symbol():
    find = build_matcher(DATA)
    assert "AMD" in find("Shares of $AMD rallied after the report.")


def test_ambiguous_symbol_needs_strict_context():
    find = build_matcher(DATA)
    assert "AI" not in find("AI is transforming everything, analysts say.")
    assert "AI" in find("C3.ai (NYSE: AI) reported earnings.")
    assert "AI" in find("$AI jumped 10%.")


def test_compute_row_prefers_row_tickers():
    find = build_matcher(DATA)
    primary, segment, more_t, more_s = compute_row(
        ["NVDA"], "NVIDIA Corporation and Advanced Micro Devices compete.", DATA, find
    )
    assert primary == "NVDA"
    assert segment == "GPUs"
    assert "AMD" in (more_t or [])


def test_annotator_is_idempotent():
    annotate = build_annotator(DATA)
    once = annotate("NVDA reported strong results.")
    assert annotate(once) == once


def test_compute_row_array_remainder_ordered_before_text_finds():
    find = build_matcher(DATA)
    primary, _segment, more_t, _more_s = compute_row(
        ["NVDA", "AMD"],
        "NVIDIA Corporation results. C3.ai (NYSE: AI) also reported.",
        DATA,
        find,
    )
    assert primary == "NVDA"
    assert more_t[0] == "AMD"          # array remainder first
    assert more_t.index("AMD") < more_t.index("AI")  # text-only finds after
