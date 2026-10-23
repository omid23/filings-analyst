"""MCP server, web app, EDGAR client, FinanceBench loader and both eval harnesses."""

from __future__ import annotations

import json
import shutil

import httpx
import pytest
from starlette.testclient import TestClient

from filings_analyst.agent.analyst import Analyst
from filings_analyst.agent.llm import LLMResponse, ScriptedLLM, ToolCall
from filings_analyst.eval import answers
from filings_analyst.eval.financebench import load_documents, load_questions, period_from, split_for
from filings_analyst.eval.retrieval import Config, evaluate, to_markdown
from filings_analyst.index.retrieve import SectionRouter
from filings_analyst.ingest.edgar import EdgarClient, EdgarError, xbrl_fact
from filings_analyst.web.app import create_app

V = "2026-07-28"


def rpc(method: str, params: dict | None = None, req_id: int | None = 1) -> dict:
    params = dict(params or {})
    params.setdefault("_meta", {"io.modelcontextprotocol/protocolVersion": V,
                                "io.modelcontextprotocol/clientInfo": {"name": "pytest", "version": "1"}})
    body = {"jsonrpc": "2.0", "method": method, "params": params}
    if req_id is not None:
        body["id"] = req_id
    return body


def answering_analyst(searcher):
    """An analyst whose model reads the cash flow page and answers from it."""
    def factory() -> Analyst:
        return Analyst(ScriptedLLM([
            LLMResponse("", [ToolCall("a", "read_page", {"doc_id": "ACME_2023_10K", "page": 5})]),
            LLMResponse("", [ToolCall("b", "submit_answer", {"answer": "Capex was $1,577 million [S1].",
                                                             "confidence": "high"})], input_tokens=50,
                        output_tokens=20),
        ]), searcher)
    return factory


@pytest.fixture
def client(searcher):
    return TestClient(create_app(searcher, answering_analyst(searcher)))


@pytest.fixture
def search_only(searcher):
    return TestClient(create_app(searcher, None))


# -- MCP ------------------------------------------------------------------------------------------

def test_mcp_discover_and_list(client):
    r = client.post("/mcp", json=rpc("server/discover"), headers={"Mcp-Method": "server/discover"}).json()
    assert r["result"]["supportedVersions"] == [V] and r["result"]["resultType"] == "complete"
    tools = client.post("/mcp", json=rpc("tools/list")).json()["result"]
    names = [t["name"] for t in tools["tools"]]
    assert names == sorted(names) == ["ask_analyst", "calculate", "list_filings", "read_page", "search_filings"]
    assert tools["ttlMs"] == 300_000 and tools["cacheScope"] == "public"
    assert all(t["annotations"]["readOnlyHint"] for t in tools["tools"])


def test_mcp_list_without_model_hides_ask(search_only):
    names = [t["name"] for t in search_only.post("/mcp", json=rpc("tools/list")).json()["result"]["tools"]]
    assert "ask_analyst" not in names and "submit_answer" not in names


def test_mcp_tool_calls(client):
    r = client.post("/mcp", json=rpc("tools/call", {"name": "read_page", "arguments": {"doc_id": "ACME_2023_10K",
                                                                                     "page": 5}}),
                    headers={"Mcp-Method": "tools/call", "Mcp-Name": "read_page"}).json()["result"]
    assert not r["isError"] and "(1,577)" in r["content"][0]["text"]
    assert r["structuredContent"]["sources"][0] == {"id": "S1", "doc_id": "ACME_2023_10K", "page": 5,
                                                    "header": r["structuredContent"]["sources"][0]["header"]}
    calc = client.post("/mcp", json=rpc("tools/call", {"name": "calculate", "arguments": {"expression": "2**"}}))
    assert calc.json()["result"]["isError"] is True
    ask = client.post("/mcp", json=rpc("tools/call", {"name": "ask_analyst", "arguments": {"question": "capex?"}}))
    res = ask.json()["result"]
    assert res["structuredContent"]["verification"]["passed"] and "[S1] ACME_2023_10K, p. 5" in res["content"][0]["text"]


def test_mcp_protocol_errors(client):
    def err(body, headers=None, raw=None):
        r = client.post("/mcp", content=raw, headers=headers) if raw is not None else \
            client.post("/mcp", json=body, headers=headers)
        return r.status_code, (r.json().get("error") or {}).get("code") if r.content else None

    assert err(None, raw=b"{nope") == (400, -32700)
    assert err({"jsonrpc": "1.0", "id": 1, "method": "x"}) == (400, -32600)
    assert err(rpc("tools/list"), {"Mcp-Method": "tools/call"}) == (400, -32020)
    assert err(rpc("tools/call", {"name": "calculate", "arguments": {}}), {"Mcp-Name": "read_page"}) == (400, -32020)
    old = rpc("tools/list", {"_meta": {"io.modelcontextprotocol/protocolVersion": "2024-11-05"}})
    assert err(old) == (200, -32022)
    assert err(rpc("resources/list")) == (200, -32601)
    assert err(rpc("tools/call", {"name": "submit_answer", "arguments": {}})) == (200, -32602)
    assert err(rpc("tools/call", {"name": "ask_analyst", "arguments": {"question": " "}})) == (200, -32602)
    assert client.post("/mcp", json=rpc("notifications/cancelled", req_id=None)).status_code == 202


# -- web app --------------------------------------------------------------------------------------

def test_web_pages_and_api(client):
    home = client.get("/")
    assert home.status_code == 200 and "<title>" in home.text
    info = client.get("/api/info").json()
    assert info["documents"] == 3 and info["can_ask"] is True and info["dense"] == "lsa"
    assert {d["doc_id"] for d in client.get("/api/filings").json()} == {"ACME_2022_10K", "ACME_2023_10K", "GBX_2023_10K"}
    assert client.get("/healthz").json() == {"status": "ok"}


def test_web_search_page_ask(client, search_only):
    s = client.post("/api/search", json={"query": "Acme FY2023 cash flow statement capital expenditure", "k": 3}).json()
    assert s["hits"][0]["doc_id"] == "ACME_2023_10K" and s["hits"][0]["page"] == 5 and len(s["hits"]) <= 3
    assert len({(h["doc_id"], h["page"]) for h in s["hits"]}) == len(s["hits"])  # one hit per page
    assert client.post("/api/search", json={"query": "  "}).status_code == 400

    p = client.get("/api/page", params={"doc_id": "ACME_2023_10K", "page": 5}).json()
    assert p["pages"] == 7 and p["title"] == "Consolidated Statements of Cash Flows"
    assert client.get("/api/page", params={"doc_id": "ACME_2023_10K", "page": 50}).status_code == 404
    assert client.get("/api/page", params={"doc_id": "ACME_2023_10K", "page": "x"}).status_code == 400

    a = client.post("/api/ask", json={"question": "What was Acme's FY2023 capex?"}).json()
    assert a["verification"]["passed"] and a["sources"][0]["page"] == 5
    assert client.post("/api/ask", json={"question": ""}).status_code == 400
    r = search_only.post("/api/ask", json={"question": "q"})
    assert r.status_code == 503 and "--provider" in r.json()["error"]


# -- EDGAR ----------------------------------------------------------------------------------------

SUBMISSIONS = {"name": "ACME INDUSTRIAL CORP", "filings": {"recent": {
    "form": ["8-K", "10-Q", "10-K", "10-K"],
    "accessionNumber": ["0000000001-24-000003", "0000000001-24-000002", "0000000001-24-000001", "0000000001-23-000001"],
    "filingDate": ["2024-05-01", "2024-04-30", "2024-02-10", "2023-02-11"],
    "reportDate": ["2024-05-01", "2024-03-31", "2023-12-31", "2022-12-31"],
    "primaryDocument": ["a8k.htm", "a10q.htm", "a10k2023.htm", "a10k2022.htm"],
}}}


def edgar_transport(seen: list):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        url = str(request.url)
        if url.endswith("company_tickers.json"):
            return httpx.Response(200, json={"0": {"cik_str": 1234, "ticker": "ACME", "title": "Acme Industrial Corp"}})
        if "submissions/CIK0000001234.json" in url:
            return httpx.Response(200, json=SUBMISSIONS)
        if url.endswith("a10k2023.htm"):
            return httpx.Response(200, text="<p>ITEM 8. FINANCIAL STATEMENTS</p><table><tr><td>Revenue</td>"
                                            "<td>32,765</td></tr></table>")
        return httpx.Response(404)
    return httpx.MockTransport(handler)


def test_edgar_requires_contact_user_agent(monkeypatch):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(EdgarError, match="User-Agent"):
        EdgarClient()
    with pytest.raises(EdgarError):
        EdgarClient("research bot")


def test_edgar_filings_and_download(tmp_path):
    seen: list = []
    client = EdgarClient("Test Person test@example.com", client=httpx.Client(transport=edgar_transport(seen)),
                         max_rps=1000)
    filings = client.filings("acme", ("10-K", "10-Q"), limit=3)
    assert [f.doc_id for f in filings] == ["ACME_2024Q1_10Q_0000000001-24-000002", "ACME_2023_10K_0000000001-24-000001",
                                           "ACME_2022_10K_0000000001-23-000001"]
    assert filings[1].url == "https://www.sec.gov/Archives/edgar/data/1234/000000000124000001/a10k2023.htm"
    doc = client.download(filings[1], tmp_path)
    assert doc.company == "Acme Industrial Corp" and doc.period == "2023" and doc.meta["aliases"] == ["ACME"]
    assert "32,765" in (tmp_path / f"{filings[1].doc_id}.htm").read_text()
    assert all(r.headers["user-agent"] == "Test Person test@example.com" for r in seen)
    n = len(seen)
    client.download(filings[1], tmp_path)  # cached on disk
    assert len(seen) == n
    with pytest.raises(EdgarError, match="unknown ticker"):
        client.filings("ZZZZ")
    with pytest.raises(EdgarError, match="404"):
        client.download(filings[0], tmp_path)


def test_xbrl_fact_picks_annual_period_and_restatement():
    rows = [
        {"form": "10-K", "start": "2023-01-01", "end": "2023-12-31", "val": 100, "filed": "2024-02-10"},
        {"form": "10-K", "start": "2023-10-01", "end": "2023-12-31", "val": 30, "filed": "2024-02-10"},  # a quarter
        {"form": "10-K", "start": "2023-01-01", "end": "2023-12-31", "val": 104, "filed": "2025-02-10"},  # restated
        {"form": "10-Q", "start": "2023-01-01", "end": "2023-12-31", "val": 1, "filed": "2024-01-01"},
    ]
    facts = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": rows}},
                                   "Cash": {"units": {"USD": [{"form": "10-K", "end": "2023-12-31", "val": 7,
                                                               "filed": "2024-02-10"}]}}}}}
    assert xbrl_fact(facts, "Revenues", 2023) == 104
    assert xbrl_fact(facts, "Cash", 2023) == 7
    assert xbrl_fact(facts, "Revenues", 2021) is None and xbrl_fact(facts, "Missing", 2023) is None


# -- FinanceBench loader and evals ----------------------------------------------------------------

@pytest.fixture
def financebench(tmp_path, corpus):
    """A miniature FinanceBench checkout built from the synthetic filings."""
    (tmp_path / "data").mkdir()
    (tmp_path / "pdfs").mkdir()
    docs = []
    for d in corpus:
        shutil.copy(d.source, tmp_path / "pdfs" / f"{d.doc_id}.pdf")
        docs.append({"doc_name": d.doc_id, "company": d.company, "gics_sector": "Industrials", "doc_type": "10k",
                     "doc_period": int(d.period), "doc_link": "https://example.com"})
    docs.append(dict(docs[0], doc_name="MISSING_2020_10K"))  # listed but no PDF
    questions = [
        {"financebench_id": "q1", "company": "Acme Industrial", "doc_name": "ACME_2023_10K",
         "question_type": "metrics-generated", "question_reasoning": "Information extraction",
         "question": "What is the FY2023 capital expenditure for Acme Industrial? Use the cash flow statement.",
         "answer": "$1577.00", "justification": "Purchases of PP&E",
         "evidence": [{"evidence_text": "...", "doc_name": "ACME_2023_10K", "evidence_page_num": 4}]},
        {"financebench_id": "q2", "company": "Globex Corporation", "doc_name": "GBX_2023_10K",
         "question_type": "domain-relevant", "question_reasoning": None,
         "question": "What does Globex sell, according to its FY2023 10-K?",
         "answer": "Logistics software and hardware for retailers.", "justification": None,
         "evidence": [{"evidence_text": "...", "doc_name": "GBX_2023_10K", "evidence_page_num": 1}]},
    ]
    for name, rows in (("financebench_document_information", docs), ("financebench_open_source", questions)):
        (tmp_path / "data" / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return tmp_path


def test_financebench_loader(financebench):
    docs = load_documents(financebench)
    assert [d.doc_id for d in docs] == ["ACME_2022_10K", "ACME_2023_10K", "GBX_2023_10K"]
    assert docs[0].form == "10-K" and docs[0].sector == "Industrials"
    assert [d.doc_id for d in load_documents(financebench, {"GBX_2023_10K"})] == ["GBX_2023_10K"]
    qs = load_questions(financebench)
    assert qs[0].gold_pages == {("ACME_2023_10K", 4)} and qs[0].split in ("dev", "test")
    assert period_from("AMCOR_2023Q4_EARNINGS", 2023) == "2023Q4" and period_from("3M_2018_10K", 2018) == "2018"
    from filings_analyst.eval.financebench import ALIASES

    splits = [split_for(c) for c in ALIASES]
    assert splits.count("dev") >= 10 and splits.count("test") >= 10  # both halves populated, deterministic


def test_retrieval_eval(searcher, financebench):
    qs = load_questions(financebench)
    configs = [Config("bm25", "bm25", "corpus"), Config("hybrid + router", "hybrid", "resolved", rerank=True),
               Config("oracle", "hybrid", "oracle", rerank=True)]
    report = evaluate(searcher, qs, configs, SectionRouter())
    routed = report["configs"]["hybrid + router"]["summary"]
    assert routed["n"] == 2 and routed["hit@5"] == 1.0 and routed["filing_top1"] == 1.0
    assert report["configs"]["oracle"]["summary"]["hit@1"] == 1.0
    assert "filing_in_candidates" not in report["configs"]["bm25"]["summary"]
    md = to_markdown(report, "mini")
    assert md.count("\n|") == 5 and "| bm25 | corpus |" in md and "| n/a |" in md


def test_numeric_correct():
    assert answers.numeric_correct("Capex was $1,577 million [S1].", "$1577.00")
    assert answers.numeric_correct("It was $1.58 billion.", "$1577.00")
    assert answers.numeric_correct("Margin was 22.4%.", "0.224")
    assert answers.numeric_correct("Capex was $1,700 million.", "$1577.00") is False
    assert answers.numeric_correct("anything", "Yes, it is cyclical.") is None


def test_answer_eval_harness(searcher, financebench):
    qs = load_questions(financebench)
    judge = ScriptedLLM(lambda s, m: LLMResponse("Correct."))
    report = answers.run(qs, answering_analyst(searcher), judge_llm=judge, workers=2, price_per_mtok=(3.0, 15.0))
    s = report["summary"]
    rows = {r["id"]: r for r in report["rows"]}
    assert rows["q1"]["numeric"] is True and rows["q1"]["cited_gold_page"] is True and rows["q1"]["grounded"]
    assert rows["q2"]["numeric"] is None and rows["q2"]["cited_gold_page"] is False
    assert s["numeric_accuracy"] == 1.0 and s["judge_correct"] == 1.0 and s["grounded"] == 1.0
    assert s["cost_per_question_usd"] == pytest.approx((50 * 3 + 20 * 15) / 1e6)
    assert "| numeric answers correct (1) | 100.0% |" in answers.to_markdown(report, "mini")
    assert "JUDGE" not in answers.JUDGE_PROMPT and answers.judge(ScriptedLLM([]), qs[0], "x") == "error"
    assert s["cache_hit_rate"] == 0.0


def test_cache_aware_pricing():
    row = answers.AnswerRow("q", "t", "test", "a", "g", None, None, True, 1.0, False, 3, input_tokens=1_000,
                            output_tokens=500, latency_ms=1.0, stopped="answered", cache_read_tokens=8_000,
                            cache_write_tokens=2_000)
    # Defaults: reads at 0.1x and writes at 1.25x the input price.
    assert answers.Pricing(3.0, 15.0).cost(row) == pytest.approx((1_000 * 3 + 500 * 15 + 8_000 * 0.3 + 2_000 * 3.75) / 1e6)
    assert answers.Pricing(4.0, 20.0, cache_read=0.2, cache_write=5.0).cost(row) == pytest.approx(
        (1_000 * 4 + 500 * 20 + 8_000 * 0.2 + 2_000 * 5) / 1e6)
    s = answers.summarize([row], (3.0, 15.0))  # a plain (input, output) tuple still works
    assert s["cache_hit_rate"] == pytest.approx(8_000 / 11_000, abs=1e-4)
    assert s["cost_per_question_usd"] == pytest.approx(answers.Pricing(3.0, 15.0).cost(row), abs=1e-5)
