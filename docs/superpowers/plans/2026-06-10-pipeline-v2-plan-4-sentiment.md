# Pipeline v2 — Plan 4: Sentiment LangGraph Stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the sixth pipeline stage: a LangGraph orchestrator that fans out (Send API) to three fixed-role analyst sub-agents — fundamentals, market context, historical precedent — and synthesizes a structured buy/sell/hold `Verdict` per article, written to a new `article_sentiment` table. Also persist the Massive feed's per-ticker provider sentiment end-to-end (user decision), giving future evals a ground-truth comparison column.

**Architecture:** Phase 4 of `docs/superpowers/specs/2026-06-10-pipeline-v2-design.md`. The graph uses `Send` fan-out (orchestrator-worker), NOT a supervisor — fixed roles, always all run, deterministic. Analyst nodes are single Gemini flash-lite calls; the synthesizer is flash with `with_structured_output(Verdict)`. Nodes are SYNC (LangGraph's Pregel engine runs a superstep's parallel tasks on a background executor, and the stage itself already runs inside `asyncio.to_thread`); the shared rate limiter in `shared/llm.py` bounds total Gemini pressure. Historical precedent comes from a plain pgvector SQL lookup (reusing the article's own stored embedding — no extra OpenAI call) computed by the stage and injected into the article payload, so the graph stays DB-free and trivially testable.

**Policy decision (user-visible):** the sentiment stage runs ONLY for articles with `category = 'real news'` AND a tagged `primary_ticker`; all others skip the stage (cheap, idempotent skip). Judging promo/recap content would waste LLM quota and pollute eval data.

**Tech Stack:** langgraph ≥1.0,<2 (new dep), langchain structured output, pgvector cosine search, psycopg Jsonb.

**Branch:** `refactor/pipeline-v2`. Baseline at start: **109 passed, 1 skipped** offline; **11 passed** db.

---

### Task 1: persist provider sentiment end-to-end

FeedItem.source_meta (the Massive poller already fills `{"sentiments": {ticker: {sentiment, sentiment_reasoning}}, "provider": "massive"}`) currently dies at enqueue. Carry it: `pipeline_jobs.source_meta jsonb` → `Job.source_meta` → scrape stage writes `articles.provider_sentiments jsonb`.

**Files:**
- Modify: `src/ticker_news/service/jobs.py` (schema, enqueue, Job, claim)
- Modify: `src/ticker_news/scraping/store/schema.sql` (articles column)
- Modify: `src/ticker_news/service/stages.py` (scrape_stage writes it)
- Modify: `tests/service/test_jobs_db.py`, `tests/service/test_stages.py`

- [ ] **Step 1: jobs.py — column, dataclass, enqueue, claim**

In `_SCHEMA`, append after the CREATE INDEX statement (ALTER is idempotent and upgrades existing tables):

```sql
ALTER TABLE pipeline_jobs ADD COLUMN IF NOT EXISTS source_meta jsonb NOT NULL DEFAULT '{}'::jsonb;
```

`Job` gains a field (with default so existing constructions keep working):

```python
    source_meta: dict = field(default_factory=dict)
```

(add `field` to the dataclasses import). `enqueue` binds it (psycopg needs the Jsonb wrapper):

```python
from psycopg.types.json import Jsonb
...
    cur = conn.execute(
        "INSERT INTO pipeline_jobs (article_url, tickers, published_utc, publisher, source_meta) "
        "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (article_url) DO NOTHING",
        (item.url, item.tickers, item.published_utc, item.publisher,
         Jsonb(item.source_meta or {})),
    )
```

`claim`'s RETURNING adds `source_meta` (last position, matching the dataclass field order — psycopg returns jsonb as a dict already).

- [ ] **Step 2: articles column**

Append to `src/ticker_news/scraping/store/schema.sql` (the file is split on `;` and executed statement-by-statement at Store init — ALTER IF NOT EXISTS is fine on pg16):

```sql
ALTER TABLE articles ADD COLUMN IF NOT EXISTS provider_sentiments JSONB;
```

- [ ] **Step 3: scrape_stage writes provider sentiment**

In `stages.py` `scrape_stage`, AFTER the exists_ok skip check resolves and AFTER a successful `process_job` (both the skip path and the ok/empty path — the article row exists in both), write the sentiments once:

```python
def _save_provider_sentiments(store, url: str, source_meta: dict) -> None:
    sentiments = (source_meta or {}).get("sentiments")
    if not sentiments:
        return
    store.conn.execute(
        "UPDATE articles SET provider_sentiments = %s "
        "WHERE url = %s AND provider_sentiments IS NULL",
        (Jsonb(sentiments), url),
    )
```

(import Jsonb; `store.conn` is autocommit). Call it via `await asyncio.to_thread(_save_provider_sentiments, resources.store, job.article_url, job.source_meta)` in both paths before returning.

- [ ] **Step 4: tests**

`tests/service/test_jobs_db.py` — extend `test_claim_marks_running_and_returns_payload`: build the item as `FeedItem(url=..., tickers=["NVDA"], publisher="Benzinga", source_meta={"sentiments": {"NVDA": {"sentiment": "positive"}}})` and assert `job.source_meta["sentiments"]["NVDA"]["sentiment"] == "positive"`.

`tests/service/test_stages.py` — add:

```python
async def test_scrape_persists_provider_sentiments(monkeypatch):
    writes = {}

    class FakeConn:
        def execute(self, sql, params):
            writes["params"] = params

            class R:
                def fetchone(self):
                    return None
            return R()

    class FakeStore:
        conn = FakeConn()

        def exists_ok(self, url):
            return True  # skip path must STILL persist sentiments

    res = _Resources()
    res.store = FakeStore()
    job = _job()
    job.source_meta = {"sentiments": {"NVDA": {"sentiment": "positive"}}}
    assert await stages.scrape_stage(job, res) == "ok"
    assert writes["params"][1] == "https://example.com/a"
```

(Adapt `_job()` if Job's constructor ordering makes source_meta awkward — it has a default, so set it after construction as shown. If the existing `_Resources`/FakeStore stubs need a `conn` with this shape, reconcile minimally; report.)

- [ ] **Step 5: run + commit**

Offline: `pytest -m "not db and not integration" -q` → 110 passed, 1 skipped. Db: `pytest -m db -q` → 11 passed (the extended claim test still counts as one).

```bash
git add -A -- src tests
git commit -m "feat: persist provider sentiment from feed to articles"
```
(Standing rule every commit: no Co-Authored-By, no AI signatures.)

---

### Task 2: the sentiment graph (`ticker_news.sentiment`)

**Files:**
- Modify: `pyproject.toml` (add `"langgraph>=1.0,<2",`)
- Create: `src/ticker_news/sentiment/__init__.py` (empty)
- Create: `src/ticker_news/sentiment/schemas.py`
- Create: `src/ticker_news/sentiment/analysts.py`
- Create: `src/ticker_news/sentiment/graph.py`
- Create: `tests/sentiment/__init__.py`, `tests/sentiment/test_schemas.py`, `tests/sentiment/test_graph.py`

- [ ] **Step 1: dep**

Add `"langgraph>=1.0,<2",` to pyproject dependencies; `pip install -e ".[dev]"`; verify `python -c "import langgraph; print(langgraph.__version__)"` → 1.0.x.

- [ ] **Step 2: schemas (TDD)**

`tests/sentiment/test_schemas.py`:

```python
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
```

`src/ticker_news/sentiment/schemas.py`:

```python
from typing import Literal

from pydantic import BaseModel, Field

Action = Literal["buy", "sell", "hold"]


class Verdict(BaseModel):
    """The structured output of the sentiment synthesizer."""

    action: Action
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""
```

- [ ] **Step 3: analyst registry + prompts**

`src/ticker_news/sentiment/analysts.py` — the role registry; adding an analyst is one dict entry. Prompts are in-repo for now (they migrate to Langfuse Prompts in Plan 5 with these as fallbacks). All prompts receive the same rendered article block.

```python
"""Fixed-role analyst registry for the sentiment graph.

Each analyst examines one aspect of a news article for a specific ticker at
its publication moment. Roles are intentionally diverse — the synthesizer
weighs their (possibly conflicting) takes into one verdict.
"""

MAX_CONTENT_CHARS = 24_000

ARTICLE_BLOCK = """\
TICKER UNDER ANALYSIS: {ticker}
PUBLISHED (UTC): {published_utc}
HEADLINE: {title}

ARTICLE BODY:
\"\"\"
{content}
\"\"\"

NEWS-PROVIDER SENTIMENT FOR {ticker} (third-party, may be wrong): {provider_sentiment}
"""

ANALYST_PROMPTS: dict[str, str] = {
    "fundamentals": ARTICLE_BLOCK + """
You are an equity fundamentals analyst. Assess ONLY the impact of this news on
{ticker}'s business fundamentals: revenue, margins, guidance, demand/supply,
competitive position, capital structure. Quantify magnitude where the article
allows it; say explicitly when the article carries no fundamental information.
3-6 sentences, no preamble.""",
    "market_context": ARTICLE_BLOCK + """
You are a market/sector strategist. Assess this news in its market context at
publication time: sector backdrop, what is likely already priced in, how
similar names trade on such news, crowd positioning, and whether the headline
overstates or understates the substance. 3-6 sentences, no preamble.""",
    "historical_precedent": ARTICLE_BLOCK + """
SIMILAR PAST ARTICLES (same corpus, published earlier; cosine-nearest first):
{precedents}

You are a quantitative analyst of historical precedent. Judge how distinctive
this news actually is versus the precedents above: is it a recurring news
pattern for this name/sector or genuinely new information? If the precedent
list is empty or weak, say so. 3-6 sentences, no preamble.""",
}

SYNTHESIS_PROMPT = ARTICLE_BLOCK + """
Three analysts examined this article for {ticker}:

{analyses}

You are the senior portfolio manager. Weigh the analyses (they may conflict),
decide buy / sell / hold for {ticker} at the moment of publication, and give a
confidence in [0,1] proportional to the strength and agreement of the evidence.
Cite which analyst arguments drove your decision in the reasoning.
"""


def render_article(article: dict) -> dict:
    """Common .format kwargs for every prompt."""
    return {
        "ticker": article.get("ticker") or "?",
        "published_utc": str(article.get("published_utc") or "unknown"),
        "title": (article.get("title") or "").strip(),
        "content": (article.get("content") or "")[:MAX_CONTENT_CHARS],
        "provider_sentiment": article.get("provider_sentiment") or "none given",
    }


def render_analyst(role: str, article: dict) -> str:
    kwargs = render_article(article)
    if role == "historical_precedent":
        precedents = article.get("precedents") or []
        kwargs["precedents"] = (
            "\n".join(f"- {p}" for p in precedents) if precedents else "(none found)"
        )
    return ANALYST_PROMPTS[role].format(**kwargs)


def render_synthesis(article: dict, analyses: list[dict]) -> str:
    blocks = "\n\n".join(
        f"[{a['role']}]\n{a['analysis']}" for a in analyses
    )
    return SYNTHESIS_PROMPT.format(analyses=blocks, **render_article(article))
```

- [ ] **Step 4: the graph (TDD)**

`tests/sentiment/test_graph.py`:

```python
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from ticker_news.sentiment.analysts import ANALYST_PROMPTS
from ticker_news.sentiment.graph import build_graph, judge_article
from ticker_news.sentiment.schemas import Verdict

ARTICLE = {
    "ticker": "NVDA",
    "title": "NVDA beats and raises",
    "content": "Data center revenue grew 90%.",
    "published_utc": "2026-06-09T10:00:00Z",
    "provider_sentiment": "positive",
    "precedents": ["2026-05-01 NVDA: prior beat"],
}


def _fakes():
    prompts = []
    analyst = RunnableLambda(
        lambda p: (prompts.append(p) or AIMessage(content=f"take #{len(prompts)}"))
    )
    judge_prompts = []
    judge = RunnableLambda(
        lambda p: (judge_prompts.append(p)
                   or Verdict(action="buy", confidence=0.9, reasoning="beat"))
    )
    return analyst, judge, prompts, judge_prompts


def test_fans_out_to_every_role_and_synthesizes():
    analyst, judge, prompts, judge_prompts = _fakes()
    graph = build_graph(analyst_llm=analyst, judge=judge)
    result = graph.invoke({"article": ARTICLE, "analyses": [], "verdict": None})
    assert len(result["analyses"]) == len(ANALYST_PROMPTS)
    assert {a["role"] for a in result["analyses"]} == set(ANALYST_PROMPTS)
    assert result["verdict"].action == "buy"
    # every analyst saw the article; the judge saw every analysis
    assert all("NVDA beats and raises" in p for p in prompts)
    assert len(judge_prompts) == 1
    for a in result["analyses"]:
        assert a["analysis"] in judge_prompts[0]


def test_precedents_reach_only_the_historical_analyst():
    analyst, judge, prompts, _ = _fakes()
    graph = build_graph(analyst_llm=analyst, judge=judge)
    graph.invoke({"article": ARTICLE, "analyses": [], "verdict": None})
    with_precedent = [p for p in prompts if "prior beat" in p]
    assert len(with_precedent) == 1


def test_judge_article_returns_verdict_and_analyses():
    analyst, judge, _, _ = _fakes()
    graph = build_graph(analyst_llm=analyst, judge=judge)
    verdict, analyses = judge_article(ARTICLE, graph=graph)
    assert isinstance(verdict, Verdict)
    assert len(analyses) == len(ANALYST_PROMPTS)
```

`src/ticker_news/sentiment/graph.py`:

```python
"""Sentiment orchestration: Send fan-out to fixed-role analysts, then a
structured-output synthesis. No supervisor — roles are static, always all run.

Nodes are sync on purpose: the stage runs inside asyncio.to_thread, and
LangGraph executes a superstep's parallel Send tasks on its background
executor, so the three analysts still run concurrently.
"""

from __future__ import annotations

import operator
from functools import lru_cache
from typing import Annotated, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from ticker_news.sentiment.analysts import (
    ANALYST_PROMPTS,
    render_analyst,
    render_synthesis,
)
from ticker_news.sentiment.schemas import Verdict
from ticker_news.shared.llm import GEMINI_FLASH, GEMINI_FLASH_LITE, gemini_chat

GEMINI_TIMEOUT_S = 90.0
RETRIES = 3


class SentimentState(TypedDict):
    article: dict
    analyses: Annotated[list[dict], operator.add]
    verdict: Optional[Verdict]


def _default_analyst_llm():
    return gemini_chat(GEMINI_FLASH_LITE, timeout_s=GEMINI_TIMEOUT_S).with_retry(
        stop_after_attempt=RETRIES, wait_exponential_jitter=True
    )


def _default_judge():
    return (
        gemini_chat(GEMINI_FLASH, timeout_s=GEMINI_TIMEOUT_S)
        .with_structured_output(Verdict)
        .with_retry(stop_after_attempt=RETRIES, wait_exponential_jitter=True)
    )


def build_graph(*, analyst_llm=None, judge=None):
    analyst_llm = analyst_llm if analyst_llm is not None else _default_analyst_llm()
    judge = judge if judge is not None else _default_judge()

    def fan_out(state: SentimentState):
        return [
            Send("analyst", {"article": state["article"], "role": role})
            for role in ANALYST_PROMPTS
        ]

    def analyst(payload: dict) -> dict:
        prompt = render_analyst(payload["role"], payload["article"])
        message = analyst_llm.invoke(prompt)
        text = getattr(message, "content", None) or str(message)
        return {"analyses": [{"role": payload["role"], "analysis": text}]}

    def synthesize(state: SentimentState) -> dict:
        prompt = render_synthesis(state["article"], state["analyses"])
        return {"verdict": judge.invoke(prompt)}

    g = StateGraph(SentimentState)
    g.add_node("analyst", analyst)
    g.add_node("synthesize", synthesize)
    g.add_conditional_edges(START, fan_out, ["analyst"])
    g.add_edge("analyst", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()


@lru_cache(maxsize=1)
def _default_graph():
    return build_graph()


def judge_article(article: dict, *, graph=None) -> tuple[Verdict, list[dict]]:
    """Run the full analyst panel + synthesis for one article/ticker."""
    graph = graph if graph is not None else _default_graph()
    result = graph.invoke({"article": article, "analyses": [], "verdict": None})
    return result["verdict"], result["analyses"]
```

Implementer note: if langgraph 1.0.x's `add_conditional_edges(START, fan_out, ["analyst"])` signature differs (e.g. path map form), adapt minimally so the three tests pass — the tests are the contract. Same for `Send` import location (`langgraph.types` vs `langgraph.constants` across versions).

- [ ] **Step 5: run + commit**

`pytest tests/sentiment -q` → 6 passed. Full offline → 116 passed, 1 skipped.

```bash
git add pyproject.toml src/ticker_news/sentiment tests/sentiment
git commit -m "feat: sentiment LangGraph with Send fan-out analysts and structured verdict"
```

---

### Task 3: sentiment store + precedent retriever + stage adapter

**Files:**
- Create: `src/ticker_news/sentiment/store.py`
- Modify: `src/ticker_news/service/stages.py` (sentiment_stage + precedent lookup)
- Create: `tests/sentiment/test_store_db.py` (db-marked)
- Modify: `tests/service/test_stages.py` (offline skip-logic tests)

- [ ] **Step 1: sentiment/store.py**

```python
"""Persistence for sentiment verdicts: public.article_sentiment."""

from __future__ import annotations

import psycopg
from psycopg.types.json import Jsonb

from ticker_news.sentiment.schemas import Verdict

_SCHEMA = """
CREATE TABLE IF NOT EXISTS article_sentiment (
    article_id  bigint NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    ticker      text NOT NULL,
    action      text NOT NULL,
    confidence  real NOT NULL,
    reasoning   text,
    analyses    jsonb NOT NULL DEFAULT '[]'::jsonb,
    model       text,
    created_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (article_id, ticker)
);
CREATE INDEX IF NOT EXISTS article_sentiment_ticker_idx ON article_sentiment (ticker)
"""


def ensure_schema(conn: psycopg.Connection) -> None:
    for statement in (s.strip() for s in _SCHEMA.split(";")):
        if statement:
            conn.execute(statement)
    conn.commit()


def save_verdict(
    conn: psycopg.Connection,
    article_id: int,
    ticker: str,
    verdict: Verdict,
    analyses: list[dict],
    model: str,
) -> None:
    conn.execute(
        "INSERT INTO article_sentiment "
        "(article_id, ticker, action, confidence, reasoning, analyses, model) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (article_id, ticker) DO NOTHING",
        (article_id, ticker, verdict.action, verdict.confidence,
         verdict.reasoning or None, Jsonb(analyses), model),
    )
    conn.commit()


def has_verdict(conn: psycopg.Connection, article_id: int, ticker: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM article_sentiment WHERE article_id = %s AND ticker = %s",
        (article_id, ticker),
    ).fetchone()
    return row is not None
```

- [ ] **Step 2: precedent retriever + sentiment_stage in stages.py**

Append to `stages.py` (and add imports: `from ticker_news.sentiment.graph import judge_article`, `from ticker_news.sentiment import store as sentiment_store`, `from ticker_news.shared.llm import GEMINI_FLASH`):

```python
def similar_past_articles(conn: psycopg.Connection, article_id: int, k: int = 5) -> list[str]:
    """Cosine-nearest earlier real-news articles, using the stored embedding.

    No new embedding call — the article was embedded in the embed stage.
    Returns display lines for the historical-precedent analyst.
    """
    rows = conn.execute(
        """
        SELECT to_char(b.published_utc, 'YYYY-MM-DD'), b.primary_ticker, b.title
        FROM public.articles a
        JOIN public.articles b
          ON b.id != a.id
         AND b.embedding IS NOT NULL
         AND b.category = 'real news'
         AND b.published_utc < a.published_utc
        WHERE a.id = %s AND a.embedding IS NOT NULL
        ORDER BY b.embedding <=> a.embedding
        LIMIT %s
        """,
        (article_id, k),
    ).fetchall()
    return [f"{d} [{t or '?'}] {title}" for d, t, title in rows]


def sentiment_stage(conn: psycopg.Connection, url: str) -> None:
    """Judge buy/sell/hold for the article's primary ticker.

    Policy: only 'real news' articles with a tagged primary_ticker are judged;
    everything else skips (cheap, idempotent).
    """
    row = conn.execute(
        "SELECT id, title, content, category, primary_ticker, published_utc, "
        "provider_sentiments FROM public.articles WHERE url = %s", (url,),
    ).fetchone()
    if row is None:
        raise StageError(f"article row missing for {url}")
    aid, title, content, category, ticker, published, provider = row
    if category != "real news" or not ticker or not (content or "").strip():
        conn.rollback()
        return
    if sentiment_store.has_verdict(conn, aid, ticker):
        conn.rollback()
        return
    precedents = similar_past_articles(conn, aid)
    provider_sentiment = ""
    if provider and isinstance(provider, dict):
        entry = provider.get(ticker) or {}
        provider_sentiment = entry.get("sentiment") or ""
    article = {
        "ticker": ticker,
        "title": title,
        "content": content,
        "published_utc": published,
        "provider_sentiment": provider_sentiment,
        "precedents": precedents,
    }
    verdict, analyses = judge_article(article)
    sentiment_store.save_verdict(conn, aid, ticker, verdict, analyses, GEMINI_FLASH)
```

- [ ] **Step 3: offline stage tests**

Append to `tests/service/test_stages.py` (stub conn pattern — keep it minimal):

```python
class _Row:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row

    def fetchall(self):
        return self._row


class _StubConn:
    """Returns queued rows in order; records rollbacks."""

    def __init__(self, rows):
        self.rows = list(rows)
        self.rolled_back = 0

    def execute(self, sql, params=None):
        return _Row(self.rows.pop(0))

    def rollback(self):
        self.rolled_back += 1

    def commit(self):
        pass


async def test_sentiment_skips_non_real_news(monkeypatch):
    conn = _StubConn([(1, "T", "body", "marketing fluff", "NVDA", None, None)])
    called = {}
    monkeypatch.setattr(stages, "judge_article", lambda a: called.setdefault("x", True))
    stages.sentiment_stage(conn, "https://example.com/a")
    assert "x" not in called
    assert conn.rolled_back == 1


async def test_sentiment_skips_untagged(monkeypatch):
    conn = _StubConn([(1, "T", "body", "real news", None, None, None)])
    monkeypatch.setattr(stages, "judge_article", lambda a: (_ for _ in ()).throw(AssertionError))
    stages.sentiment_stage(conn, "https://example.com/a")
    assert conn.rolled_back == 1


async def test_sentiment_judges_real_news(monkeypatch):
    from ticker_news.sentiment.schemas import Verdict

    conn = _StubConn([
        (1, "T", "body", "real news", "NVDA", None, {"NVDA": {"sentiment": "positive"}}),
    ])
    seen = {}
    monkeypatch.setattr(stages.sentiment_store, "has_verdict", lambda c, a, t: False)
    monkeypatch.setattr(stages, "similar_past_articles", lambda c, a, k=5: ["p1"])
    monkeypatch.setattr(
        stages, "judge_article",
        lambda article: (seen.update(article) or
                         (Verdict(action="hold", confidence=0.5, reasoning=""), [])),
    )
    saved = {}
    monkeypatch.setattr(
        stages.sentiment_store, "save_verdict",
        lambda c, aid, t, v, an, m: saved.update(aid=aid, ticker=t, action=v.action),
    )
    stages.sentiment_stage(conn, "https://example.com/a")
    assert seen["provider_sentiment"] == "positive"
    assert seen["precedents"] == ["p1"]
    assert saved == {"aid": 1, "ticker": "NVDA", "action": "hold"}
```

- [ ] **Step 4: db-marked store test**

`tests/sentiment/test_store_db.py`:

```python
import psycopg
import pytest

from ticker_news.sentiment import store
from ticker_news.sentiment.schemas import Verdict

pytestmark = pytest.mark.db

from tests.scraping.conftest import _connect_test_db


@pytest.fixture
def conn():
    try:
        c = _connect_test_db()
    except psycopg.OperationalError:
        pytest.skip("Postgres not reachable")
    # articles table must exist for the FK
    from ticker_news.scraping.store.db import Store
    from tests.scraping.conftest import TEST_DSN

    s = Store(TEST_DSN)
    s.init_schema()
    s.close()
    store.ensure_schema(c)
    c.execute("TRUNCATE article_sentiment, articles RESTART IDENTITY CASCADE")
    c.commit()
    yield c
    c.execute("TRUNCATE article_sentiment, articles RESTART IDENTITY CASCADE")
    c.commit()
    c.close()


def _seed_article(conn) -> int:
    row = conn.execute(
        "INSERT INTO articles (url, source_domain, status) "
        "VALUES ('https://example.com/s', 'example.com', 'ok') RETURNING id"
    ).fetchone()
    conn.commit()
    return row[0]


def test_save_and_has_verdict_roundtrip(conn):
    aid = _seed_article(conn)
    v = Verdict(action="buy", confidence=0.7, reasoning="r")
    assert store.has_verdict(conn, aid, "NVDA") is False
    store.save_verdict(conn, aid, "NVDA", v, [{"role": "x", "analysis": "y"}], "m")
    assert store.has_verdict(conn, aid, "NVDA") is True
    # idempotent: second save is a no-op, not an error
    store.save_verdict(conn, aid, "NVDA", v, [], "m")
```

- [ ] **Step 5: run + commit**

Offline → 119 passed, 1 skipped. Db (`pytest -m db -q`) → 12 passed.

```bash
git add src/ticker_news/sentiment src/ticker_news/service/stages.py tests/sentiment tests/service/test_stages.py
git commit -m "feat: sentiment stage adapter with precedent retrieval and verdict store"
```

---

### Task 4: wire into the service + batch CLI

**Files:**
- Modify: `src/ticker_news/service/jobs.py` (STAGES)
- Modify: `src/ticker_news/service/worker.py` (runner + schema ensure)
- Modify: `tests/service/test_jobs_unit.py`, `tests/service/test_worker.py`, `tests/service/test_serve_db.py` (the three chain-pinning tests — expected churn, budgeted)
- Modify: `src/ticker_news/cli.py` (`sentiment` batch command), `tests/test_root_cli.py`

- [ ] **Step 1: chain + runner**

`jobs.py`: `STAGES = ["scrape", "embed", "classify", "tag", "insights", "sentiment"]`.
`worker.py` serve(): add `"sentiment": lambda job: stages.sentiment_stage(conn, job.article_url)` to each worker's runners dict, and call `sentiment_store.ensure_schema(setup_conn)` during startup (import `from ticker_news.sentiment import store as sentiment_store`).

- [ ] **Step 2: update the three pinned tests**

- `test_jobs_unit.py::test_stage_chain_order` → expect the 6-stage list; `test_next_stage_walks_the_chain_then_done` → `next_stage("insights") == "sentiment"`, `next_stage("sentiment") == DONE`.
- `test_worker.py::test_process_article_runs_stages_in_order_and_advances` → add `"sentiment": sync_stage("sentiment")` to runners; expected ran/advanced lists gain the stage.
- `test_worker.py::test_process_article_resumes_mid_chain` → runners gain `"sentiment"`; expectations updated.
- `test_serve_db.py` → monkeypatch `sentiment_stage` like the others; expected stage order gains "sentiment"; also monkeypatch `worker.sentiment_store.ensure_schema` if serve's startup call would hit the FakeConn-less path (it runs against real news_test — real ensure_schema is fine; verify).

- [ ] **Step 3: batch CLI command (TDD)**

Test in `tests/test_root_cli.py`:

```python
def test_sentiment_command(monkeypatch):
    captured = {}

    def fake_run(*, limit, reprocess):
        captured.update(limit=limit, reprocess=reprocess)
        return 3

    monkeypatch.setattr("ticker_news.sentiment.batch.run_batch", fake_run)
    result = runner.invoke(cli.app, ["sentiment", "--limit", "10"])
    assert result.exit_code == 0, result.output
    assert captured == {"limit": 10, "reprocess": False}
```

Create `src/ticker_news/sentiment/batch.py` — judge existing real-news articles that have no verdict yet (backfill over the historical corpus):

```python
"""Batch sentiment over already-ingested articles (no service required)."""

from __future__ import annotations

from ticker_news.sentiment import store
from ticker_news.service.stages import sentiment_stage
from ticker_news.shared.db import connect

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def run_batch(*, limit: int | None = None, reprocess: bool = False) -> int:
    conn = connect(vector=True)
    try:
        store.ensure_schema(conn)
        not_done = "" if reprocess else (
            "AND NOT EXISTS (SELECT 1 FROM article_sentiment s "
            "WHERE s.article_id = a.id AND s.ticker = a.primary_ticker)"
        )
        lim = f"LIMIT {int(limit)}" if limit else ""
        urls = [r[0] for r in conn.execute(
            f"SELECT a.url FROM public.articles a "
            f"WHERE a.category = 'real news' AND a.primary_ticker IS NOT NULL "
            f"AND a.content IS NOT NULL {not_done} ORDER BY a.published_utc {lim}"
        ).fetchall()]
        conn.commit()
        done = 0
        bar = tqdm(urls, unit="article") if tqdm else urls
        for url in bar:
            try:
                sentiment_stage(conn, url)
                done += 1
            except Exception as exc:
                print(f"  {url}: {exc!r}")
        print(f"Judged {done}/{len(urls)} article(s).")
        return done
    finally:
        conn.close()
```

(Note: `reprocess=True` re-runs the graph but `save_verdict` is DO NOTHING — for a true re-judge the row must be deleted first. Keep the flag semantics honest: with reprocess, DELETE the existing row before judging — add `if reprocess: conn.execute("DELETE FROM article_sentiment WHERE article_id = %s AND ticker = %s", ...)` — simplest correct form: when reprocess is set, delete matching rows for the selected articles up front in one statement. Implement it that way and keep the test contract.)

CLI command (lazy import):

```python
@app.command()
def sentiment(
    limit: int | None = typer.Option(None, help="Judge at most N pending articles."),
    reprocess: bool = typer.Option(False, "--reprocess", help="Re-judge articles that already have a verdict."),
) -> None:
    """Run the analyst-panel sentiment over real-news articles missing a verdict."""
    from ticker_news.sentiment import batch

    n = batch.run_batch(limit=limit, reprocess=reprocess)
    typer.echo(f"judged {n} article(s)")
```

- [ ] **Step 4: run + commit**

Offline → 121 passed, 1 skipped (119 + 1 CLI + adjusted worker tests net +1 — verify real arithmetic and report). Db → 12 passed.

```bash
git add -A -- src tests
git commit -m "feat: sentiment as sixth pipeline stage with batch CLI"
```

---

### Task 5: verification sweep + push

- [ ] Full offline suite + db suite — report real counts.
- [ ] `ticker-news --help` lists all 11 commands (incl. sentiment); each `--help` exits 0.
- [ ] Lazy-import check: `python -X importtime -c "import ticker_news.cli" 2>&1 | Select-String "langchain|langgraph|google"` → no output.
- [ ] Legacy alive: `python run_scrape.py --help`, `python scripts/search/insight_sentiment.py --help` exit 0 (legacy sentiment script untouched — it ports/retires in Plan 6).
- [ ] `git push`. Do NOT open a PR.

---

## Out of scope (later plans)

- Plan 5: Langfuse — traces around process_article/judge_article, analyst:<role> observation naming, prompts to Langfuse, cost visibility; converging chain construction styles.
- Plan 6: research/ port (incl. legacy insight_sentiment/backtest scripts), deletions, CLAUDE.md, final PR. Backtest-driven scoring of `article_sentiment` rows against realized returns (the eval ground-truth loop) is designed in the spec and lands with the eval milestone.
