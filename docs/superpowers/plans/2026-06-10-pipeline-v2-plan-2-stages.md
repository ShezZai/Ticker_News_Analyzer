# Pipeline v2 — Plan 2: Stage Migration to LangChain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the embedding, classification, and enrichment stages from standalone `scripts/` into `src/ticker_news/` as LangChain-based modules — structured-output chains for every LLM call, one instrumentable model factory, pure logic preserved verbatim with new offline tests — each stage gaining a `ticker-news` CLI command.

**Architecture:** Phase 2 of `docs/superpowers/specs/2026-06-10-pipeline-v2-design.md`. All LLM construction goes through `shared/llm.py` (the future Langfuse choke point, Plan 5). Every LLM response is a validated Pydantic object. Legacy `scripts/` remain untouched and functional (the search tools import `scripts/embedding/embed_articles.py` until Plan 6); new package code runs in parallel against the same tables with the same resumability semantics (NULL-driven selection). Per-run token-cost readouts are intentionally dropped — Langfuse provides cost observability in Plan 5. CLI commands use **lazy imports** (established convention from Plan 1's final review) so `ticker-news --help` stays fast.

**Tech Stack:** langchain ≥1.3, langchain-google-genai ≥4.2 (Gemini via consolidated google-genai SDK), langchain-openai ≥1.1 (embeddings), pydantic, psycopg 3 + pgvector, typer, pytest.

**Branch:** `refactor/pipeline-v2` (continues from Plan 1). Baseline at start: **48 passed, 1 skipped** offline.

---

### Task 1: LangChain deps, `shared/llm.py`, `shared/db.py`, lazy CLI imports

**Files:**
- Modify: `pyproject.toml` (dependencies)
- Modify: `requirements.txt` (remove conflicting stale langchain 0.3 pins ONLY)
- Create: `src/ticker_news/shared/llm.py`
- Create: `src/ticker_news/shared/db.py`
- Create: `tests/shared/test_llm.py`
- Modify: `src/ticker_news/cli.py` (lazy imports)
- Modify: `tests/test_root_cli.py` (monkeypatch target moves)

- [ ] **Step 1: Add dependencies to pyproject.toml**

In `[project] dependencies`, add these lines (keep existing ones):

```toml
    "langchain>=1.3,<2",
    "langchain-google-genai>=4.2,<5",
    "langchain-openai>=1.1,<2",
    "numpy",
    "tiktoken",
    "tqdm",
    "yfinance",
```

- [ ] **Step 2: Remove the stale langchain 0.3 pins from requirements.txt**

Plan 1 said requirements.txt stays untouched until the final phase — this is the one sanctioned exception: the old pins `langchain>=0.3,<1`, `langchain-community>=0.3,<1`, `langchain-core>=0.3,<1`, `langchain-text-splitters>=0.3,<1`, and `langsmith` directly conflict with langchain 1.3 (their `langchain-core<1` bound cannot coexist with langchain 1.x), and grep proves nothing in the repo imports any of them. Delete exactly those 5 lines and the comment line above them (`# LangChain packages (pinned to 0.3.x — langgraph forces 1.x which breaks retrievers)` — the retrievers concern is vestigial; nothing imports langchain). Leave every other line alone.

- [ ] **Step 3: Install and verify**

Run: `pip install -e ".[dev]"`
Expected: installs langchain 1.3.x, langchain-google-genai 4.2.x, langchain-openai 1.1.x without resolver conflicts.
Run: `python -c "import langchain, langchain_google_genai, langchain_openai; print('ok')"` → `ok`

- [ ] **Step 4: Write failing tests for the model factory**

Create `tests/shared/test_llm.py`:

```python
import pytest

from ticker_news.shared import llm


def test_gemini_chat_requires_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        llm.gemini_chat(llm.GEMINI_FLASH_LITE)


def test_openai_embeddings_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        llm.openai_embeddings()


def test_gemini_chat_builds_with_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    model = llm.gemini_chat(llm.GEMINI_FLASH_LITE, timeout_s=30.0)
    assert model.model is not None
    assert llm.GEMINI_FLASH_LITE in model.model


def test_rate_limiter_is_shared(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    a = llm.gemini_chat(llm.GEMINI_FLASH_LITE)
    b = llm.gemini_chat(llm.GEMINI_FLASH)
    assert a.rate_limiter is b.rate_limiter
```

Run: `pytest tests/shared/test_llm.py -q` → FAIL (`No module named 'ticker_news.shared.llm'`)

- [ ] **Step 5: Implement `shared/llm.py`**

```python
"""Single factory for every LLM client in the pipeline.

All chat models and embedding models are built here so retries, rate limits,
and (in a later phase) Langfuse instrumentation live in exactly one place.
"""

from functools import lru_cache

from langchain_core.rate_limiters import InMemoryRateLimiter

from ticker_news.shared.config import get_settings

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
GEMINI_FLASH_LITE = "gemini-2.5-flash-lite"
GEMINI_FLASH = "gemini-2.5-flash"

# One limiter shared by every Gemini model instance: concurrent stages must not
# blow the per-project quota between them.
_GEMINI_RPS = 8.0


@lru_cache(maxsize=1)
def gemini_rate_limiter() -> InMemoryRateLimiter:
    return InMemoryRateLimiter(
        requests_per_second=_GEMINI_RPS, max_bucket_size=_GEMINI_RPS
    )


def gemini_chat(model: str, *, timeout_s: float = 60.0):
    """A deterministic (temperature 0) Gemini chat model with shared rate limit.

    google-genai has no default request timeout — without one a single stuck
    call freezes a whole worker pool, so timeout_s is mandatory here.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    s = get_settings()
    if not s.google_api_key:
        raise SystemExit("GOOGLE_API_KEY is not set (put it in .env).")
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=0.0,
        timeout=timeout_s,
        google_api_key=s.google_api_key,
        rate_limiter=gemini_rate_limiter(),
    )


def openai_embeddings(*, batch_size: int = 256):
    """text-embedding-3-small client. Inputs must be pre-truncated to the model
    window by the caller (ticker_news.embedding); LangChain's own
    chunk-and-average path for long inputs is disabled because it changes
    vector semantics versus plain truncation."""
    from langchain_openai import OpenAIEmbeddings

    s = get_settings()
    if not s.openai_api_key:
        raise SystemExit("OPENAI_API_KEY is not set (put it in .env).")
    return OpenAIEmbeddings(
        model=EMBED_MODEL,
        api_key=s.openai_api_key,
        chunk_size=batch_size,
        check_embedding_ctx_length=False,
    )
```

Implementer note: if `ChatGoogleGenerativeAI` in the installed langchain-google-genai version supports a `thinking_budget` parameter, add `thinking_budget=0` to the constructor call (the legacy classify script disabled thinking for cost parity); if the constructor rejects it (TypeError/validation error), omit it and record that in your report.

- [ ] **Step 6: Implement `shared/db.py`**

```python
import psycopg

from ticker_news.shared.config import get_settings


def connect(*, vector: bool = False) -> psycopg.Connection:
    """One connection convention for every stage (single DATABASE_URL).

    vector=True registers the pgvector adapter (needed to bind numpy arrays
    to vector columns).
    """
    conn = psycopg.connect(get_settings().database_url)
    if vector:
        from pgvector.psycopg import register_vector

        register_vector(conn)
    return conn
```

- [ ] **Step 7: Lazy imports in the CLI**

In `src/ticker_news/cli.py`: delete the two top-level imports `from ticker_news.scraping.config import Settings` and `from ticker_news.scraping.pipeline import run as pipeline_run` (keep `asyncio`, `replace`, `typer`). Change the `scrape` body to import lazily:

```python
def scrape(
    # ... options unchanged ...
) -> None:
    """Scrape article bodies for every URL in the CSV into the articles table."""
    from ticker_news.scraping import pipeline
    from ticker_news.scraping.config import Settings

    settings = Settings()
    if ignore_robots:
        settings = replace(settings, respect_robots=False)
    if concurrency is not None:
        settings = replace(settings, concurrency=concurrency)
    asyncio.run(pipeline.run(csv, settings, limit=limit, retry_errors=retry_errors))
```

In `tests/test_root_cli.py`, the monkeypatch target moves from `cli.pipeline_run` to the pipeline module. In all three tests that patch, replace `monkeypatch.setattr(cli, "pipeline_run", fake_run)` with:

```python
    monkeypatch.setattr("ticker_news.scraping.pipeline.run", fake_run)
```

(The lazy `from ticker_news.scraping import pipeline` + `pipeline.run(...)` call resolves the attribute at call time, so patching the module attribute works.)

- [ ] **Step 8: Run suites**

Run: `pytest tests/shared/test_llm.py tests/test_root_cli.py -q` → 8 passed.
Run: `pytest -m "not db and not integration" -q` → 52 passed, 1 skipped.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml requirements.txt src/ticker_news/shared/llm.py src/ticker_news/shared/db.py tests/shared/test_llm.py src/ticker_news/cli.py tests/test_root_cli.py
git commit -m "feat: LangChain model factory, shared db helper, lazy CLI imports"
```
(No Co-Authored-By trailers, no AI signatures — standing rule for every commit in this plan.)

---

### Task 2: embedding stage → `ticker_news.embedding` + `ticker-news embed`

Source being ported: `scripts/embedding/embed_articles.py` (226 lines; read it in full first). The OpenAI SDK call is replaced by `shared.llm.openai_embeddings()`; everything else (truncation, text building, NULL-driven selection, HNSW index) ports with minimal change. Token-cost postfix is dropped (Langfuse later). The legacy script stays in place untouched.

**Files:**
- Create: `src/ticker_news/embedding/__init__.py`
- Create: `src/ticker_news/embedding/embedder.py`
- Create: `src/ticker_news/embedding/pipeline.py`
- Create: `tests/embedding/__init__.py`, `tests/embedding/test_embedder.py`
- Modify: `src/ticker_news/cli.py` (add `embed` command)
- Modify: `tests/test_root_cli.py` (add embed test)

- [ ] **Step 1: Write failing embedder tests**

Create empty `tests/embedding/__init__.py`, then `tests/embedding/test_embedder.py`:

```python
import numpy as np
import pytest

from ticker_news.embedding.embedder import (
    MAX_INPUT_TOKENS,
    build_text,
    embed_query,
    embed_texts,
    truncate_tokens,
)


class FakeEmbeddings:
    """Stands in for OpenAIEmbeddings: returns a constant unit vector per text."""

    def __init__(self):
        self.seen = []

    def embed_documents(self, texts):
        self.seen.extend(texts)
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        self.seen.append(text)
        return [0.0, 1.0, 0.0]


def test_truncate_passes_short_text_through():
    assert truncate_tokens("hello world") == "hello world"


def test_truncate_empty_becomes_single_space():
    assert truncate_tokens("") == " "
    assert truncate_tokens("   ") == " "


def test_truncate_caps_long_text():
    long = "word " * (MAX_INPUT_TOKENS * 2)
    out = truncate_tokens(long)
    assert len(out) < len(long)


def test_build_text_joins_title_and_content():
    assert build_text("Title", "Body") == "Title\n\nBody"
    assert build_text(None, "Body") == "Body"
    assert build_text("Title", None) == "Title"
    assert build_text(None, None) == ""


def test_embed_texts_returns_float32_arrays_aligned():
    fake = FakeEmbeddings()
    out = embed_texts(["a", "", "c"], embeddings=fake)
    assert len(out) == 3
    assert all(isinstance(v, np.ndarray) and v.dtype == np.float32 for v in out)
    assert fake.seen[1] == " "  # empty input replaced, alignment preserved


def test_embed_query_rejects_empty():
    with pytest.raises(ValueError):
        embed_query("   ", embeddings=FakeEmbeddings())


def test_embed_query_returns_vector():
    v = embed_query("nvidia datacenter", embeddings=FakeEmbeddings())
    assert isinstance(v, np.ndarray) and v.dtype == np.float32
```

Run: `pytest tests/embedding -q` → FAIL (module missing).

- [ ] **Step 2: Implement `embedding/embedder.py`**

Port from `scripts/embedding/embed_articles.py`: copy `_truncate_tokens` (rename to `truncate_tokens`, keep the module-level `_encoder` cache and tiktoken-fallback logic verbatim from lines 54-73) and `build_text` (verbatim from lines 144-151). Then:

```python
"""Text → vector for articles and queries.

Same model and truncation for stored vectors and search queries on purpose —
do not fork the config. (text-embedding-3-small, unit-normalized, 1536 dims.)
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from ticker_news.shared.llm import EMBED_DIM, EMBED_MODEL, openai_embeddings

MAX_INPUT_TOKENS = 8000  # model hard-caps at 8192; trim just under

_encoder = None


# truncate_tokens(text: str) -> str   ← ported verbatim (see above)
# build_text(title, content) -> str   ← ported verbatim


def embed_texts(texts: Sequence[str], *, embeddings=None) -> List[np.ndarray]:
    """One float32 vector per input, index-aligned (empty inputs sent as ' ')."""
    client = embeddings if embeddings is not None else openai_embeddings()
    cleaned = [truncate_tokens(t) for t in texts]
    vectors = client.embed_documents(cleaned)
    return [np.asarray(v, dtype=np.float32) for v in vectors]


def embed_query(text: str, *, embeddings=None) -> np.ndarray:
    """Embedding for a single search query (same code path as stored vectors)."""
    if not text or not text.strip():
        raise ValueError("query text is empty")
    return embed_texts([text], embeddings=embeddings)[0]
```

Create `src/ticker_news/embedding/__init__.py`:

```python
from ticker_news.embedding.embedder import build_text, embed_query, embed_texts

__all__ = ["build_text", "embed_query", "embed_texts"]
```

Run: `pytest tests/embedding -q` → 7 passed.

- [ ] **Step 3: Implement `embedding/pipeline.py`**

Port from `scripts/embedding/embed_articles.py` with these exact substitutions — otherwise copy function bodies verbatim:
- `get_conn()` is replaced by `from ticker_news.shared.db import connect` … `connect(vector=True)`.
- `ensure_schema`, `ids_to_process`, `fetch_rows`, `create_index`: verbatim (lines 127-184), including the pgvector-missing SystemExit message.
- `embed_all(batch_size=256, limit=None, reembed=False, build_index=True)`: same loop as lines 187-241 but call the new `embed_texts(texts)` (no `return_tokens` — drop all `tokens`/cost accounting and the tqdm postfix; keep tqdm over batches and the final `print(f"Embedded {done} article(s).")`).
- Module constant `DEFAULT_BATCH = 256` stays.

No new tests for pipeline.py in this plan (it is glue over tested parts + DB; db-marked integration tests come with the service phase).

- [ ] **Step 4: Add the CLI command (lazy import) + test**

Append to `tests/test_root_cli.py`:

```python
def test_embed_command_passes_args(monkeypatch):
    captured = {}

    def fake_embed_all(*, batch_size, limit, reembed, build_index):
        captured.update(batch_size=batch_size, limit=limit, reembed=reembed,
                        build_index=build_index)
        return 0

    monkeypatch.setattr("ticker_news.embedding.pipeline.embed_all", fake_embed_all)
    result = runner.invoke(
        cli.app, ["embed", "--batch-size", "64", "--limit", "5", "--reembed", "--no-index"]
    )
    assert result.exit_code == 0, result.output
    assert captured == {"batch_size": 64, "limit": 5, "reembed": True, "build_index": False}
```

Run it → FAIL (no such command). Then add to `src/ticker_news/cli.py`:

```python
@app.command()
def embed(
    batch_size: int = typer.Option(256, min=1, help="Inputs per embeddings API request."),
    limit: int | None = typer.Option(None, help="Only process the first N pending rows."),
    reembed: bool = typer.Option(False, "--reembed", help="Recompute embeddings for every row."),
    no_index: bool = typer.Option(False, "--no-index", help="Skip building the HNSW index."),
) -> None:
    """Embed articles missing an embedding into pgvector (resumable)."""
    from ticker_news.embedding import pipeline

    pipeline.embed_all(
        batch_size=batch_size, limit=limit, reembed=reembed, build_index=not no_index
    )
```

Run: `pytest tests/test_root_cli.py -q` → 5 passed.

- [ ] **Step 5: Full suite + commit**

Run: `pytest -m "not db and not integration" -q` → 60 passed, 1 skipped.

```bash
git add src/ticker_news/embedding tests/embedding src/ticker_news/cli.py tests/test_root_cli.py
git commit -m "feat: embedding stage in ticker_news.embedding with ticker-news embed"
```

---

### Task 3: classification stage → `ticker_news.classification` + `ticker-news classify`

Source: `scripts/classify/classify_articles.py` (462 lines; read in full first). The raw google-genai client + JSON parsing + manual retry/enum-check is replaced by a LangChain structured-output chain; the two-pass logic (flash-lite labels everything, flash re-confirms only "real news") is preserved exactly, including "confirmation failed → keep lite verdict". `reclassify_real_news.py` is NOT ported (one-off, archived in Plan 6).

**Files:**
- Create: `src/ticker_news/classification/__init__.py` (empty)
- Create: `src/ticker_news/classification/schemas.py`
- Create: `src/ticker_news/classification/chain.py`
- Create: `src/ticker_news/classification/pipeline.py`
- Create: `tests/classification/__init__.py`, `tests/classification/test_chain.py`
- Modify: `src/ticker_news/cli.py`, `tests/test_root_cli.py`

- [ ] **Step 1: schemas.py**

```python
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
```

- [ ] **Step 2: Write failing two-pass tests**

Create empty `tests/classification/__init__.py`, then `tests/classification/test_chain.py`:

```python
import pytest
from langchain_core.runnables import RunnableLambda

from ticker_news.classification.chain import MAX_ARTICLE_CHARS, classify_article
from ticker_news.classification.schemas import Classification


def _const(category, reason=""):
    return RunnableLambda(lambda _x: Classification(category=category, reason=reason))


def test_non_real_news_skips_confirmation():
    def explode(_x):
        raise AssertionError("confirmation must not run for non-real-news")

    result, confirmed = classify_article(
        "T", "body", lite=_const("marketing fluff", "ad"), confirm=RunnableLambda(explode)
    )
    assert result.category == "marketing fluff"
    assert confirmed is False


def test_real_news_goes_to_confirmation_and_can_be_overturned():
    result, confirmed = classify_article(
        "T", "body", lite=_const("real news"), confirm=_const("recap/review", "post-hoc")
    )
    assert result.category == "recap/review"
    assert confirmed is True


def test_confirmation_failure_keeps_lite_verdict():
    def boom(_x):
        raise RuntimeError("api down")

    result, confirmed = classify_article(
        "T", "body", lite=_const("real news", "from lite"), confirm=RunnableLambda(boom)
    )
    assert result.category == "real news"
    assert result.reason == "from lite"
    assert confirmed is True


def test_inputs_are_truncated():
    seen = {}

    def capture(x):
        seen.update(x)
        return Classification(category="other")

    classify_article("t" * 1000, "b" * 100_000, lite=RunnableLambda(capture))
    assert len(seen["title"]) == 300
    assert len(seen["body"]) == MAX_ARTICLE_CHARS


def test_invalid_category_rejected_by_schema():
    with pytest.raises(Exception):
        Classification(category="not a real category")
```

Run: `pytest tests/classification -q` → FAIL (module missing).

- [ ] **Step 3: Implement chain.py**

Copy `PROMPT_TEMPLATE` **verbatim** from `scripts/classify/classify_articles.py` lines 108-156 (it already uses `{{...}}` escaping for the JSON example and `{title}`/`{body}` placeholders — both str.format and ChatPromptTemplate read it identically). Then:

```python
"""Two-pass article classifier as LangChain chains.

Pass 1: gemini-2.5-flash-lite labels every article.
Pass 2: gemini-2.5-flash re-runs the same prompt only when pass 1 said
"real news" — it confirms or overturns. If the confirmation call fails
entirely, the lite verdict stands (same behavior as the legacy script).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate

from ticker_news.classification.schemas import Classification
from ticker_news.shared.llm import GEMINI_FLASH, GEMINI_FLASH_LITE, gemini_chat

GEMINI_TIMEOUT_S = 60.0
MAX_ARTICLE_CHARS = 6_000  # classification needs the lede, not the whole article
RETRIES = 4

PROMPT_TEMPLATE = """..."""  # ← verbatim copy, lines 108-156 of the legacy script


def build_classifier(model_name: str):
    """prompt | structured-output Gemini, with retry on transient failures."""
    llm = gemini_chat(model_name, timeout_s=GEMINI_TIMEOUT_S)
    structured = llm.with_structured_output(Classification).with_retry(
        stop_after_attempt=RETRIES, wait_exponential_jitter=True
    )
    return ChatPromptTemplate.from_template(PROMPT_TEMPLATE) | structured


@lru_cache(maxsize=1)
def _default_lite():
    return build_classifier(GEMINI_FLASH_LITE)


@lru_cache(maxsize=1)
def _default_confirm():
    return build_classifier(GEMINI_FLASH)


def classify_article(
    title: Optional[str],
    content: str,
    *,
    lite=None,
    confirm=None,
) -> Tuple[Classification, bool]:
    """Returns (verdict, confirmation_ran). Two-pass semantics preserved."""
    lite = lite if lite is not None else _default_lite()
    inputs = {
        "title": (title or "").strip()[:300],
        "body": (content or "")[:MAX_ARTICLE_CHARS],
    }
    first = lite.invoke(inputs)
    if first.category != "real news":
        return first, False

    confirm = confirm if confirm is not None else _default_confirm()
    try:
        return confirm.invoke(inputs), True
    except Exception:
        # confirmation exhausted its retries: keep the lite "real news" verdict
        return first, True
```

Run: `pytest tests/classification -q` → 5 passed.

- [ ] **Step 4: Implement pipeline.py**

Port from the legacy script with substitutions, bodies otherwise verbatim:
- `get_conn` → `ticker_news.shared.db.connect()` (no vector needed).
- `ensure_schema` (lines 166-178) and `articles_to_process` (lines 184-209): verbatim.
- `classify_all(reprocess=False, limit=None, ids=None, workers=8, batch_size=200)`: same ThreadPoolExecutor + pending-batch-flush structure as lines 351-438, but each future calls `classify_article(title, content or "")` from chain.py and unpacks `(verdict, _confirmed)`; drop all token/cost accounting (lite_in/lite_out/conf_in/conf_out, the cost prints, and the tqdm cost postfix — keep the `{n_failed} failed` postfix); keep the per-category counts print. `pending.append((verdict.category, verdict.reason or None, aid))` feeds the same batched UPDATE.

- [ ] **Step 5: CLI command + test**

Append to `tests/test_root_cli.py`:

```python
def test_classify_command_passes_args(monkeypatch):
    captured = {}

    def fake_classify_all(*, reprocess, limit, ids, workers):
        captured.update(reprocess=reprocess, limit=limit, ids=ids, workers=workers)
        return 0

    monkeypatch.setattr(
        "ticker_news.classification.pipeline.classify_all", fake_classify_all
    )
    result = runner.invoke(
        cli.app, ["classify", "--limit", "20", "--ids", "78,79,80", "--workers", "4"]
    )
    assert result.exit_code == 0, result.output
    assert captured == {"reprocess": False, "limit": 20, "ids": [78, 79, 80], "workers": 4}
```

Run it → FAIL. Then add to cli.py:

```python
@app.command()
def classify(
    limit: int | None = typer.Option(None, help="Only process the first N pending articles."),
    reprocess: bool = typer.Option(False, "--reprocess", help="Re-classify articles that already have a category."),
    ids: str | None = typer.Option(None, help="Comma-separated article ids to (re)classify."),
    workers: int = typer.Option(8, min=1, help="Concurrent Gemini requests."),
) -> None:
    """Classify articles into content categories (two-pass Gemini, resumable)."""
    from ticker_news.classification import pipeline

    id_list = [int(x) for x in ids.split(",") if x.strip()] if ids else None
    pipeline.classify_all(reprocess=reprocess, limit=limit, ids=id_list, workers=workers)
```

Run: `pytest tests/test_root_cli.py -q` → 6 passed.

- [ ] **Step 6: Full suite + commit**

Run: `pytest -m "not db and not integration" -q` → 66 passed, 1 skipped.

```bash
git add src/ticker_news/classification tests/classification src/ticker_news/cli.py tests/test_root_cli.py
git commit -m "feat: two-pass classification stage as LangChain structured-output chains"
```

---

### Task 4: enrichment (no-LLM half) → `tagging.py`, `reference_data.py` + CLI commands

Sources: `scripts/enrichment/tag_segments.py` (367 lines), `load_ticker_data.py` (92), `load_ticker_overview.py` (136) — read all three first. Pure ports (no LLM): only the DSN convention changes. The tagging matcher logic is currently untested — this task adds offline tests for it.

**Files:**
- Create: `src/ticker_news/enrichment/__init__.py` (empty)
- Create: `src/ticker_news/enrichment/tagging.py`
- Create: `src/ticker_news/enrichment/reference_data.py`
- Create: `tests/enrichment/__init__.py`, `tests/enrichment/test_tagging.py`
- Move+modify: `tests/test_load_ticker_overview.py` → `tests/enrichment/test_reference_data.py`
- Modify: `src/ticker_news/cli.py`, `tests/test_root_cli.py`

- [ ] **Step 1: Port tagging.py**

Copy from `tag_segments.py` verbatim: the constant sets (`_AMBIGUOUS_SYMBOLS`, `_CATEGORY_WORDS`, `_GENERIC_FIRST`), `_short_alias`, `load_ticker_data`, `_build_patterns`, `build_matcher`, `build_annotator`, `ensure_schema`, `compute_row`, `create_indexes`, `tag_all`, `BATCH_SIZE`. Substitutions: module-level `DB_DSN`/`get_conn` replaced by `ticker_news.shared.db.connect()`; drop `argparse`/`main()` (the CLI command replaces it); drop `load_dotenv()` (AppSettings owns env loading).

- [ ] **Step 2: Write failing tagging tests (new coverage for previously-untested pure logic)**

`tests/enrichment/test_tagging.py`:

```python
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
    assert "AI" in find("C3.ai (NYSE: AI) reported earnings.") or "AI" in find("$AI jumped 10%.")


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
```

Run → FAIL, then confirm the port makes them pass: `pytest tests/enrichment/test_tagging.py -q` → 5 passed. **If a test fails because the real matcher behaves differently than assumed** (e.g. context rules differ), adjust the TEST to match actual legacy behavior — the port must not change behavior; report what you adjusted.

- [ ] **Step 3: Port reference_data.py (merging the two loaders)**

One module, two halves, function bodies verbatim from the sources:
- From `load_ticker_data.py`: `ensure_schema` → rename `ensure_universe_schema`, `read_rows` (verbatim), `load` → rename `load_universe(csv_path: Path) -> int`. `DEFAULT_CSV` constant: resolve `ai_compute_us_market_universe_consolidated_segments_min5.csv` against the repo root (`Path(__file__).resolve().parents[3]` from `src/ticker_news/enrichment/` — verify the level count lands on the repo root).
- From `load_ticker_overview.py`: `ensure_schema` → rename `ensure_overview_schema`, `fetch_description`, `list_tickers`, `existing_tickers`, `select_pending`, `upsert`, `load` → rename `load_overviews(tickers=None, refresh=False, delay=0.5)`. Keep `yfinance` imported lazily inside `fetch_description` (it's a heavy import).
- All DB access via `shared.db.connect()`.

- [ ] **Step 4: Move and retarget the existing overview tests**

`git mv tests/test_load_ticker_overview.py tests/enrichment/test_reference_data.py`. Update its import/monkeypatch targets from the script module to `ticker_news.enrichment.reference_data` (open the file; it monkeypatches `yf.Ticker` and imports `fetch_description`/`select_pending` — point everything at the new module; if it does `sys.path` manipulation to import the script, delete that). Run: `pytest tests/enrichment -q` → 12 passed (5 tagging + 7 moved).

- [ ] **Step 5: CLI commands + tests**

Append to `tests/test_root_cli.py`:

```python
def test_tag_command(monkeypatch):
    captured = {}

    def fake_tag_all(*, only_missing, build_index):
        captured.update(only_missing=only_missing, build_index=build_index)
        return 0

    monkeypatch.setattr("ticker_news.enrichment.tagging.tag_all", fake_tag_all)
    result = runner.invoke(cli.app, ["tag", "--not-only-missing", "--no-index"])
    assert result.exit_code == 0, result.output
    assert captured == {"only_missing": False, "build_index": False}


def test_load_universe_command(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "ticker_news.enrichment.reference_data.load_universe",
        lambda csv_path: captured.update(csv=str(csv_path)) or 7,
    )
    result = runner.invoke(cli.app, ["load-universe", "--csv", "u.csv"])
    assert result.exit_code == 0, result.output
    assert captured["csv"].endswith("u.csv")


def test_load_overviews_command(monkeypatch):
    captured = {}

    def fake_load(*, tickers, refresh, delay):
        captured.update(tickers=tickers, refresh=refresh, delay=delay)

    monkeypatch.setattr("ticker_news.enrichment.reference_data.load_overviews", fake_load)
    result = runner.invoke(cli.app, ["load-overviews", "--tickers", "NVDA,AMD", "--refresh"])
    assert result.exit_code == 0, result.output
    assert captured == {"tickers": ["NVDA", "AMD"], "refresh": True, "delay": 0.5}
```

Then add three commands to cli.py (lazy imports, mirroring legacy argparse semantics):

```python
@app.command()
def tag(
    not_only_missing: bool = typer.Option(False, "--not-only-missing", help="Recompute every row, not just untagged ones."),
    no_index: bool = typer.Option(False, "--no-index", help="Skip building indexes."),
) -> None:
    """Tag articles with primary/secondary tickers and segments."""
    from ticker_news.enrichment import tagging

    tagging.tag_all(only_missing=not not_only_missing, build_index=not no_index)


@app.command(name="load-universe")
def load_universe(
    csv: Path = typer.Option(None, help="Universe CSV (default: repo-root consolidated CSV)."),
) -> None:
    """Load the ticker → company/segment universe into ticker_data."""
    from ticker_news.enrichment import reference_data

    n = reference_data.load_universe(csv_path=csv or reference_data.DEFAULT_CSV)
    typer.echo(f"Upserted {n} tickers.")


@app.command(name="load-overviews")
def load_overviews(
    tickers: str | None = typer.Option(None, help="Comma-separated tickers (default: all in ticker_data)."),
    refresh: bool = typer.Option(False, "--refresh", help="Re-fetch tickers that already have a row."),
    delay: float = typer.Option(0.5, help="Seconds between Yahoo requests."),
) -> None:
    """Fetch Yahoo Finance company descriptions into ticker_overview."""
    from ticker_news.enrichment import reference_data

    t = [x.strip() for x in tickers.split(",") if x.strip()] if tickers else None
    reference_data.load_overviews(tickers=t, refresh=refresh, delay=delay)
```

(Add `from pathlib import Path` to cli.py's imports.)

Run: `pytest tests/test_root_cli.py -q` → 9 passed.

- [ ] **Step 6: Full suite + commit**

Run: `pytest -m "not db and not integration" -q` → 74 passed, 1 skipped.

```bash
git add src/ticker_news/enrichment tests/enrichment src/ticker_news/cli.py tests/test_root_cli.py
git status --short  # the test_load_ticker_overview.py move (git mv) must show as a rename, nothing unstaged
git commit -m "feat: enrichment tagging and reference-data loaders in the package"
```

---

### Task 5: insights stage → `ticker_news.enrichment.insights` + `ticker-news insights`

Source: `scripts/enrichment/extract_insights.py` (769 lines; read it in full — twice). The largest port. Split into two modules: pure text processing (`insights_text.py`, fully unit-tested) and LLM+DB pipeline (`insights.py`). The Gemini call becomes a structured-output chain returning `InsightBoxes(boxes: list[str])` with a flash fallback; the `sys.path`-hack import of tag_segments becomes a clean package import; insight embedding reuses `ticker_news.embedding.embed_texts`.

**Files:**
- Create: `src/ticker_news/enrichment/insights_text.py`
- Create: `src/ticker_news/enrichment/insights.py`
- Create: `tests/enrichment/test_insights_text.py`
- Modify: `src/ticker_news/cli.py`, `tests/test_root_cli.py`

- [ ] **Step 1: Port the pure text helpers to insights_text.py**

Copy verbatim from `extract_insights.py`: `HEADLINE_PREFIX`, `DEFAULT_QUOTE_THRESHOLD`, `_WRAP_CHARS`, `with_headline`, `split_box`, `fuzzy_find_in_source`, `verbatimize_quotes`, `_box_dict_to_text`, `_first_json_object`, `_extract_boxes`, `parse_boxes`. No LLM, no DB in this module. (`_extract_boxes`/`parse_boxes` stay because the structured-output chain can still return slightly-off box objects inside the JSON; the repair logic is battle-tested — keep it.)

- [ ] **Step 2: Write failing tests for the text helpers**

`tests/enrichment/test_insights_text.py`:

```python
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
    assert quotes == ['"capex will grow 40%"'] or quotes == ["capex will grow 40%"]


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
    raw = '```json\n{"boxes": ["TOPIC: X\\nINSIGHT: Y\\nQUOTES:\\n- \\"q\\""]}\n```'
    boxes = parse_boxes(raw)
    assert len(boxes) == 1
    assert boxes[0].startswith("TOPIC:")
```

Run → FAIL, implement Step 1's module, run again: `pytest tests/enrichment/test_insights_text.py -q` → 6 passed. **Same rule as tagging: if a test's exact expectation mismatches real legacy behavior, fix the TEST, not the ported code** (split_box quote formatting, fuzzy thresholds); report adjustments.

- [ ] **Step 3: Implement insights.py (LLM + DB pipeline)**

Structure:

```python
"""Insight-box extraction: chunk each article into decision-useful boxes
(Gemini, structured output), verbatimize quotes against the source, store in
public.article_insights, embed with the shared embedding stage."""

from pydantic import BaseModel

class InsightBoxes(BaseModel):
    boxes: list[str]
```

- `PROMPT_TEMPLATE`: verbatim copy from `extract_insights.py` (the analyst prompt). Check its braces: if the template contains literal `{`/`}` (JSON examples), escape them as `{{`/`}}` for ChatPromptTemplate OR build the chain with a plain formatted string via `llm.invoke(prompt_str)` — choose the simpler: keep str.format semantics identical to legacy by formatting the article text into the template yourself and invoking the structured model directly (no ChatPromptTemplate needed):

```python
MAX_ARTICLE_CHARS = 48_000
GEMINI_TIMEOUT_S = 120.0
RETRIES = 5


@lru_cache(maxsize=1)
def _box_chain():
    lite = gemini_chat(GEMINI_FLASH_LITE, timeout_s=GEMINI_TIMEOUT_S)
    flash = gemini_chat(GEMINI_FLASH, timeout_s=GEMINI_TIMEOUT_S)
    structured = lite.with_structured_output(InsightBoxes).with_retry(
        stop_after_attempt=RETRIES, wait_exponential_jitter=True
    )
    fallback = flash.with_structured_output(InsightBoxes).with_retry(
        stop_after_attempt=2, wait_exponential_jitter=True
    )
    return structured.with_fallbacks([fallback])


def generate_boxes(article_text: str, *, chain=None) -> list[str]:
    chain = chain if chain is not None else _box_chain()
    result = chain.invoke(PROMPT_TEMPLATE.format(article=article_text[:MAX_ARTICLE_CHARS]))
    return result.boxes
```

(Adapt the `.format(...)` placeholder name to whatever the legacy template actually uses — read it. If the legacy template embeds the article via a different mechanism, mirror it exactly.)

- DB half, ported with bodies verbatim where possible: `ensure_schema` (the CREATE TABLE article_insights + ALTER articles + indexes), `articles_to_process`, `_store_article_boxes`, `extract_all` (ThreadPoolExecutor pattern, workers=8, drop token/cost accounting), `fix_quotes`, `embed_missing` — for `embed_missing`, replace the script's private `_embed_texts`/`_truncate_embed` with `from ticker_news.embedding.embedder import embed_texts` (same model, same truncation — that is the point of the shared module) and keep the HNSW index creation on `article_insights`.
- `_load_box_annotator`: replace the sys.path hack with `from ticker_news.enrichment.tagging import build_annotator, load_ticker_data`.
- All connections via `shared.db.connect(vector=True)`.

- [ ] **Step 4: CLI command + test**

Append to `tests/test_root_cli.py`:

```python
def test_insights_command_passes_args(monkeypatch):
    captured = {}

    def fake_extract_all(*, reprocess, limit, quote_threshold, ids, workers):
        captured.update(reprocess=reprocess, limit=limit,
                        quote_threshold=quote_threshold, ids=ids, workers=workers)
        return 0

    monkeypatch.setattr("ticker_news.enrichment.insights.extract_all", fake_extract_all)
    monkeypatch.setattr("ticker_news.enrichment.insights.embed_missing", lambda **kw: 0)
    result = runner.invoke(cli.app, ["insights", "--limit", "3", "--workers", "2"])
    assert result.exit_code == 0, result.output
    assert captured["limit"] == 3 and captured["workers"] == 2
```

Add to cli.py (flags mirror the legacy argparse exactly):

```python
@app.command()
def insights(
    limit: int | None = typer.Option(None, help="Only process the first N pending articles."),
    reprocess: bool = typer.Option(False, "--reprocess", help="Re-extract articles that already have insights."),
    ids: str | None = typer.Option(None, help="Comma-separated article ids (implies reprocess for those)."),
    workers: int = typer.Option(8, min=1, help="Concurrent Gemini requests."),
    no_embed: bool = typer.Option(False, "--no-embed", help="Extract boxes only; skip embedding."),
    embed_only: bool = typer.Option(False, "--embed-only", help="Skip extraction; only fill missing embeddings."),
    fix_quotes: bool = typer.Option(False, "--fix-quotes", help="Re-verbatimize existing rows (no LLM)."),
    quote_threshold: float = typer.Option(0.75, help="Min similarity for fuzzy quote match."),
    no_index: bool = typer.Option(False, "--no-index", help="Skip building the HNSW index."),
) -> None:
    """Extract embedded insight boxes from articles (resumable)."""
    from ticker_news.enrichment import insights as mod

    if fix_quotes:
        mod.fix_quotes(quote_threshold=quote_threshold)
        return
    id_list = [int(x) for x in ids.split(",") if x.strip()] if ids else None
    if not embed_only:
        mod.extract_all(reprocess=reprocess, limit=limit,
                        quote_threshold=quote_threshold, ids=id_list, workers=workers)
    if not no_embed:
        mod.embed_missing(build_index=not no_index)
```

Run: `pytest tests/test_root_cli.py -q` → 10 passed.

- [ ] **Step 5: Full suite + commit**

Run: `pytest -m "not db and not integration" -q` → 81 passed, 1 skipped.

```bash
git add src/ticker_news/enrichment tests/enrichment src/ticker_news/cli.py tests/test_root_cli.py
git commit -m "feat: insight extraction stage with structured-output chain and shared embeddings"
```

---

### Task 6: verification sweep + push

**Files:** none — verification only.

- [ ] **Step 1: Full offline suite** — `pytest -m "not db and not integration" -q` → 81 passed, 1 skipped (or the corrected count if behavior-matching test adjustments were reported in Tasks 4-5 — the number must match what the task reports said).
- [ ] **Step 2: CLI surface smoke** — `ticker-news --help` lists: scrape, embed, classify, tag, load-universe, load-overviews, insights. Each `<cmd> --help` exits 0.
- [ ] **Step 3: Legacy paths still alive** — `python run_scrape.py --help` exits 0; `python scripts/classify/classify_articles.py --help` and `python scripts/embedding/embed_articles.py --help` exit 0 (legacy scripts untouched and runnable — they need the old SDKs which are still in requirements.txt/installed).
- [ ] **Step 4: Import-time budget** — `python -X importtime -c "import ticker_news.cli" 2>&1 | Select-Object -Last 5`: confirm langchain/google-genai do NOT appear in the import tree (lazy imports working).
- [ ] **Step 5: Push** — `git push` (branch already tracks origin). Do NOT open a PR.

---

## Out of scope (later plans)

- Plan 3: `pipeline_jobs` queue, worker service, `NewsFeedSource` + Massive poller + CSV backfill.
- Plan 4: sentiment LangGraph (`langgraph` dep lands there).
- Plan 5: Langfuse (compose, tracing, prompt management — and restoring cost visibility lost in this plan).
- Plan 6: research/ port (search CLIs switch off `scripts/embedding`), deletion of `scripts/` + `run_scrape.py` + legacy `scraping/cli.py`, requirements.txt reconciliation, CLAUDE.md update, final PR.
