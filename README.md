# filings-analyst

An agent that answers questions about SEC filings (10-K, 10-Q, 8-K, earnings releases) with cited, machine-checked answers. It is built for financial research, where a wrong number is worse than no answer.

It runs as a CLI, as a web app with a source viewer, and as an MCP server, so Claude, ChatGPT, Cursor or any other MCP client can use the same filing library.

## The problem

Point a generic "chat with your PDFs" pipeline at a 10-K and ask *"What was 3M's FY2018 capital expenditure?"*. Three things usually go wrong:

1. **Wrong filing.** A library of 360 filings holds the same company in five fiscal years. Plain vector search does not know that "FY2018" has to match one filing, so it returns similar-looking pages from 2016, 2021 and 2022.
2. **Broken tables.** Standard PDF text extraction emits a financial statement one cell per line, so `Purchases of property, plant and equipment` and `(1,577)` end up far apart. Neither keyword nor embedding search can connect them.
3. **Unchecked numbers.** The model rounds, mixes up years, or does arithmetic in its head, and nothing catches it.

On [FinanceBench](https://github.com/patronus-ai/financebench) (150 analyst-written questions over 360 public filings), keyword search over the whole library puts the right page in the top 5 only **23%** of the time. This project gets that to **58%** without any model calls, and to **70%** once the filing is known. The agent then reads full pages and must pass a deterministic check before it answers.

## How it works

```
question
   │
   ▼
FilingResolver ─── company / ticker / alias, fiscal year, quarter, form  →  candidate filings
   │
   ▼
Hybrid retrieval ── SQLite FTS5 BM25 (header ×2)  +  dense vectors  →  reciprocal rank fusion
   │
   ▼
SectionRouter ──── "capex" → cash flow statement, "quick ratio" → balance sheet, "risks" → Item 1A
   │
   ▼
Analyst agent ──── search_filings · read_page · calculate · list_filings · submit_answer
   │                every passage gets a source id [S1], [S2] … in an evidence ledger
   ▼
Verifier ───────── citations exist, and every figure matches a cited page,
   │                a calculator result or operand, or the question (rounding- and unit-aware)
   │                fail → one revision with specific feedback
   ▼
cited answer + sources + calculations + verification report
```

**Ingestion** (`ingest/`)
- PDF pages are rebuilt from word positions, so each table row reads as one line: `Purchases of property, plant and equipment (PP&E) (1,577) (1,373) (1,420)`.
- Each page is tagged with its 10-K section (`Item 7. MD&A`) and any statement title (`Consolidated Statements of Cash Flows`). Table-of-contents rows and passing mentions in prose are ignored.
- EDGAR HTML filings become pages at the filing's own page breaks, with tables as `|`-separated rows.
- Chunks never cross a page boundary, so every citation is a real page number. Each chunk carries a context header: `3M | 10-K | FY2018 | Item 8 | Consolidated Statement of Cash Flows | page 60`.
- `EdgarClient` downloads filings straight from SEC EDGAR. It enforces the SEC's User-Agent rule and rate limit, and can read XBRL company facts.

**Retrieval** (`index/`)
- One SQLite file holds documents, pages, chunks and an FTS5 index. There is no external database.
- Dense vectors come from a pluggable embedder:
  - `lsa` (TF-IDF + SVD) is offline and the default;
  - `fastembed` (bge-small, ONNX) and `openai` are optional.
  - Vectors are stored as `.npy` and JSON, never pickle.
- `FilingResolver` maps names, tickers and aliases ("JnJ", "MMM"), and periods ("FY22", "Q2 2023", "second fiscal quarter") to filings. When the question gives no year, the most recent filing wins.
- `SectionRouter` boosts pages whose section or statement matches what the question asks for.

**Agent** (`agent/`)
- The tool loop is provider-neutral. Adapters for the Anthropic Messages API and OpenAI-compatible chat (OpenAI, Azure, vLLM, Ollama) are plain `httpx`, with retries on 429/5xx/529.
- `calculate` is a safe AST evaluator, not `eval`. It understands `$1,577`, `(1,577)`, `12%`, `×` and `÷`.
- The verifier is deterministic, so the model cannot argue its way past it. It catches invented numbers, figures from uncited pages, and citations that were never retrieved.
- Filing text is treated as data. The system prompt tells the model to ignore instructions found inside tool results.

**Serving**
- `filings-analyst serve` starts the web UI, a JSON API and an MCP endpoint at `/mcp`, on spec 2026-07-28 (stateless Streamable HTTP):
  - `server/discover`;
  - `tools/list` with `ttlMs` and `cacheScope`, and read-only annotations;
  - `Mcp-Method` / `Mcp-Name` header validation.
  - Tools: `search_filings`, `read_page`, `calculate`, `list_filings`, plus `ask_analyst` when a model is configured.
- The UI shows the answer with citation chips. Clicking one opens the source page with the answer's numbers highlighted. Without a model it runs in search-only mode.

## Results

Retrieval, measured on FinanceBench with no model involved. *Hit@k* means a page the FinanceBench annotator used as evidence is in the top k retrieved pages.

Questions are split by **company** (crc32 of the name), so no company appears in both halves:
- **dev:** 93 questions, used for every tuning decision (fusion weights, router boosts, title rules);
- **test:** 57 questions, reported only after those decisions were made.

**Test split** (`reports/retrieval-test.md`)

| config | scope | hit@1 | hit@5 | hit@10 | recall@10 | MRR@20 | filing found | p50 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bm25, no filing filter | corpus | 7.0% | 22.8% | 26.3% | 25.4% | 0.134 | n/a | 282 |
| bm25 | resolved | 19.3% | 38.6% | 50.9% | 43.9% | 0.279 | 96% | 136 |
| dense | resolved | 10.5% | 35.1% | 43.9% | 41.2% | 0.209 | 96% | 24 |
| hybrid | resolved | 17.5% | 40.4% | 49.1% | 43.0% | 0.265 | 96% | 154 |
| hybrid + section router | resolved | 31.6% | 57.9% | 64.9% | 60.5% | 0.428 | 96% | 163 |
| hybrid + section router, filing given | oracle | 40.4% | 70.2% | 80.7% | 75.4% | 0.545 | given | 154 |

**Dev split** (`reports/retrieval-dev.md`)

| config | scope | hit@1 | hit@5 | hit@10 | recall@10 | MRR@20 | filing found | p50 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bm25, no filing filter | corpus | 7.5% | 18.3% | 22.6% | 22.0% | 0.125 | n/a | 254 |
| bm25 | resolved | 9.7% | 24.7% | 39.8% | 36.9% | 0.193 | 97% | 120 |
| dense | resolved | 8.6% | 17.2% | 26.9% | 25.3% | 0.140 | 97% | 20 |
| hybrid | resolved | 6.5% | 29.0% | 43.0% | 39.6% | 0.174 | 97% | 136 |
| hybrid + section router | resolved | 28.0% | 53.8% | 61.3% | 57.9% | 0.392 | 97% | 149 |
| hybrid + section router, filing given | oracle | 34.4% | 68.8% | 75.3% | 71.9% | 0.493 | given | 141 |

What the numbers say:
- **Choosing the filing first matters most.** It is worth more than any change to the ranking model.
- **Structure beats similarity on financial questions.** The section router adds 17 to 25 points of hit@5 over plain hybrid search.
- **The remaining gap is mostly narrative questions** ("Does the company have a shareholder rights plan?"). There the evidence page has no structural signature, and the offline LSA embedder is weak.

### What is not measured here

- **End-to-end answer accuracy needs a model API key**, so it was not run in this build. The harness is complete and tested:

  ```bash
  export ANTHROPIC_API_KEY=...
  filings-analyst eval-answers --financebench ../financebench --split test \
      --provider anthropic --model claude-sonnet-5-5 --judge --price-in 3 --price-out 15
  ```

  It reports:
  - numeric accuracy (unit- and rounding-aware, 1% tolerance);
  - LLM-judge accuracy and refusal rate;
  - the share of answers that pass verification;
  - the share that cite an annotator's evidence page;
  - steps, p50 latency and cost per question.
- **Dense retrieval uses LSA** because the build environment could not download model weights. `--embedder fastembed` (bge-small) or `--embedder openai` should lift the dense and hybrid rows. Re-run `eval-retrieval` to measure the change.

## Quickstart

Requires Python 3.10+.

```bash
pip install -e ".[dev]"            # add ".[fastembed]" for neural embeddings
pytest -q                          # 71 tests, fully offline, about 2 s
```

**FinanceBench**

```bash
git clone https://github.com/patronus-ai/financebench ../financebench
filings-analyst index --financebench ../financebench --workers 4                # 360 filings, ~6 min on 4 cores
filings-analyst index --financebench ../financebench --questions-only           # just the 84 filings the questions use
filings-analyst eval-retrieval --financebench ../financebench --split test      # writes reports/retrieval-test.md
filings-analyst search "What is the FY2018 capital expenditure amount for 3M? Use the cash flow statement."
```

**Your own filings**

```bash
export SEC_USER_AGENT="Your Name you@example.com"
filings-analyst edgar NVDA AMD --forms 10-K 10-Q --limit 4                # download from EDGAR and index
filings-analyst index --folder ~/filings      # files named COMPANY_PERIOD_FORM.pdf, e.g. ACME_2024Q2_10Q.pdf
```

**Ask and serve**

```bash
export ANTHROPIC_API_KEY=...                  # or OPENAI_API_KEY with --provider openai
filings-analyst ask "How did Netflix's operating margin change from FY2021 to FY2022?" --provider anthropic
filings-analyst serve --provider anthropic    # http://127.0.0.1:8700, MCP at /mcp
```

**Docker** (expects an index in `./data/index`)

```bash
docker build -t filings-analyst .
docker run -p 8700:8700 -v "$PWD/data:/data" -e ANTHROPIC_API_KEY filings-analyst serve --host 0.0.0.0 --provider anthropic
```

### Connect an MCP client

Point any MCP client that speaks Streamable HTTP at `http://127.0.0.1:8700/mcp`. A raw call looks like this:

```bash
curl -s localhost:8700/mcp -H 'content-type: application/json' -H 'Mcp-Method: tools/call' -H 'Mcp-Name: read_page' -d '{
  "jsonrpc": "2.0", "id": 1, "method": "tools/call",
  "params": {"name": "read_page", "arguments": {"doc_id": "3M_2018_10K", "page": 60},
             "_meta": {"io.modelcontextprotocol/protocolVersion": "2026-07-28"}}}'
```

Results include `structuredContent.sources` (doc id, page, header), so a client can show the citations.

## Layout

```
src/filings_analyst/
  ingest/     pdf.py (layout-aware text, sections, titles) · html.py · chunking.py · edgar.py
  index/      store.py (SQLite + FTS5) · embed.py · resolver.py · retrieve.py (RRF, router) · indexer.py
  agent/      llm.py (Anthropic / OpenAI adapters) · tools.py (ledger) · numbers.py · verify.py · analyst.py
  eval/       financebench.py · retrieval.py · answers.py
  mcp_server.py · web/ (Starlette app + single-file UI) · cli.py
tests/        synthetic 10-K PDFs built with PyMuPDF; scripted models; mocked EDGAR and model APIs
reports/      retrieval-dev.md · retrieval-test.md
```

## Design choices

- **Page-level citations, not chunk ids.** Analysts check sources by page, and FinanceBench grades evidence by page.
- **A deterministic verifier instead of an LLM judge at answer time.** It is cheap, cannot be persuaded, and its feedback names the exact figure to fix.
- **SQLite for everything.** One file per index, nothing to run. FTS5 BM25 is competitive with dedicated search engines at this scale (167k chunks, ~130 ms p50).
- **No SDKs in the model adapters.** The request shapes are visible and tested against mocked transports.

## Limits

- Scanned PDFs without a text layer are not OCR'd.
- The resolver depends on company names and aliases. A filing library with many similarly named companies needs a ticker or CIK in the question.
- Quarter detection uses calendar-style labels. Unusual fiscal calendars resolve by year.

## License

MIT
