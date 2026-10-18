from __future__ import annotations

import json

import httpx
import pytest

from filings_analyst.agent.analyst import Analyst
from filings_analyst.agent.llm import (
    AnthropicLLM,
    LLMError,
    LLMResponse,
    OpenAICompatLLM,
    ScriptedLLM,
    ToolCall,
    ToolSpec,
    make_llm,
)
from filings_analyst.agent.numbers import CalcError, calculate, extract_numbers, is_trivial, matches
from filings_analyst.agent.tools import Calculation, Ledger, Toolbox
from filings_analyst.agent.verify import cited_ids, verify


def call(name: str, i: int = 1, **args) -> ToolCall:
    return ToolCall(f"call_{name}_{i}", name, args)


# -- calculator and number matching ---------------------------------------------------------------

@pytest.mark.parametrize("expr,expected", [
    ("(1577 - 1373) / 1373 * 100", 14.8579752),
    ("$1,577 + $1,373", 2950),
    ("(1,577)", -1577),
    ("12% * 200", 24),
    ("avg(1, 2, 3) + max(4, 5) - abs(-1)", 6),
    ("2 × 3 − 1 ÷ 2", 5.5),
    ("round(14.8579, 1)", 14.9),
])
def test_calculate(expr, expected):
    assert calculate(expr) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("expr", [
    "__import__('os').system('id')", "open('x')", "(lambda: 1)()", "[1, 2]", "2 ** 1000", "1 / 0",
    "a + 1", "x" * 600, "True + 1",
])
def test_calculate_rejects(expr):
    with pytest.raises(CalcError):
        calculate(expr)


def test_extract_numbers():
    nums = {n.text: n for n in extract_numbers("Capex was $1.58 billion (1,577) in FY2018, margin 22.5%, 3M said")}
    assert nums["$1.58 billion"].absolute == pytest.approx(1.58e9)
    assert nums["(1,577)"].value == -1577
    assert nums["22.5%"].percent
    assert "2018" not in nums  # part of FY2018
    assert nums["3M"].scale == 1  # "3M" is the company, not $3 million
    assert is_trivial(extract_numbers("in 2023")[0]) and is_trivial(extract_numbers("3 segments")[0])


@pytest.mark.parametrize("answer,candidate,ok", [
    ("$1.58 billion", 1577, True),      # table in millions, answer in billions, rounded
    ("$1,577 million", 1577, True),
    ("1,577", -1577, True),             # sign from parentheses
    ("14.9%", 14.857, True),
    ("14.9%", 0.14857, True),           # ratio vs percent
    ("15.2%", 14.857, False),
    ("$1.62 billion", 1577, False),
    ("$2,000", 1577, False),
])
def test_matches(answer, candidate, ok):
    assert matches(extract_numbers(answer)[0], [candidate]) is ok


# -- verifier -------------------------------------------------------------------------------------

def _ledger() -> Ledger:
    ledger = Ledger()
    ledger.add_source("ACME_2023_10K", 4, "h", "Purchases of PP&E (1,577) (1,373) (1,420)")
    ledger.add_source("ACME_2023_10K", 6, "h", "Net sales 32,765 31,657 30,109")
    return ledger


def test_verify_passes_grounded_answer():
    ledger = _ledger()
    ledger.calculations.append(Calculation("(1577 - 1373) / 1373 * 100", 14.8579))
    v = verify("Capex rose from $1,373M to $1,577M [S1], or 14.9% [S1].", ledger)
    assert v.passed and v.citations == ["S1"] and v.numbers_checked == 3 and v.grounded_ratio == 1
    assert verify("Up 14.9% [S1] = (1577 - 1373) / 1373 * 100.", ledger).passed  # formula operands are grounded


def test_verify_catches_problems():
    ledger = _ledger()
    no_cite = verify("Capex was $1,577 million.", ledger)
    assert not no_cite.passed and "cites no sources" in no_cite.feedback()
    unknown = verify("Capex was $1,577 million [S9].", ledger)
    assert unknown.unknown_citations == ["S9"]
    wrong = verify("Capex was $1,600 million and net sales 32,765 [S1, S2].", ledger)
    assert wrong.ungrounded == ["$1,600 million"] and wrong.grounded_ratio == 0.5
    # a number on an uncited page does not count
    assert verify("Net sales were 32,765 [S1].", ledger).ungrounded == ["32,765"]
    assert cited_ids("a [S1] b [S2, S3] c [S1]") == ["S1", "S2", "S3"]


# -- tools ----------------------------------------------------------------------------------------

def test_toolbox(searcher):
    tb = Toolbox(searcher)
    out, err = tb.call("search_filings", {"query": "Acme FY2023 cash flow statement capital expenditure"})
    assert not err and out.startswith("(filings searched:") and "[S1]" in out
    out, err = tb.call("read_page", {"doc_id": "ACME_2023_10K", "page": 5})
    assert not err and "(1,577)" in out and "Cash Flows" in out
    again, _ = tb.call("read_page", {"doc_id": "ACME_2023_10K", "page": 5})
    assert again.split("]")[0] == out.split("]")[0]  # same page, same source id
    assert tb.call("read_page", {"doc_id": "ACME_2023_10K", "page": 99}) == \
        ("Error: ACME_2023_10K has no page 99 (it has 7 pages)", True)
    assert tb.call("read_page", {"doc_id": "NOPE", "page": 1})[1]
    out, err = tb.call("calculate", {"expression": "1577 - 1373"})
    assert json.loads(out)["result"] == 204 and tb.ledger.calculations[0].result == 204
    assert tb.call("calculate", {"expression": "import os"})[1]
    assert tb.call("teleport", {}) == ("Unknown tool 'teleport'.", True)
    assert "GBX_2023_10K | Globex Corporation | 10-K | 2023 | 7 pages" in tb.call("list_filings", {"company": "globex"})[0]
    assert tb.call("list_filings", {"company": "initech"})[0] == "No filings match."
    assert {s.name for s in tb.specs} == {"list_filings", "search_filings", "read_page", "calculate", "submit_answer"}


# -- analyst loop ---------------------------------------------------------------------------------

GOOD = ("Acme's FY2023 capital expenditure (purchases of PP&E) was $1,577 million, up from $1,373 million in "
        "FY2022 [S1]. That is an increase of 14.9% (calculate: (1577 - 1373) / 1373 * 100).")


def test_analyst_researches_calculates_and_cites(searcher):
    llm = ScriptedLLM([
        LLMResponse("", [call("search_filings", query="Acme FY2023 cash flow statement capital expenditure")],
                    input_tokens=100, output_tokens=10),
        LLMResponse("", [call("read_page", doc_id="ACME_2023_10K", page=5)]),
        LLMResponse("", [call("calculate", expression="(1577 - 1373) / 1373 * 100")]),
        LLMResponse("", [call("submit_answer", answer=GOOD.replace("[S1]", "[{sid}]"), confidence="high")]),
    ])

    # The page source id depends on how many search passages came first; fill it in from the transcript.
    def policy(system, messages):
        resp = llm._script[llm._i]
        llm._i += 1
        if resp.tool_calls and resp.tool_calls[0].name == "submit_answer":
            page_out = next(m["content"] for m in messages if m.get("name") == "read_page")
            sid = page_out[1:page_out.index("]")]
            args = dict(resp.tool_calls[0].arguments, answer=resp.tool_calls[0].arguments["answer"].format(sid=sid))
            return LLMResponse("", [ToolCall("call_submit", "submit_answer", args)])
        return resp

    result = Analyst(ScriptedLLM(policy), searcher).run("What was Acme's FY2023 capex and how did it change?")
    assert result.stopped == "answered" and result.verification.passed, result.verification.to_dict()
    assert result.confidence == "high"
    assert [s.kind for s in result.steps].count("tool") == 3
    page = result.cited_sources()[0]
    assert (page["doc_id"], page["page"]) == ("ACME_2023_10K", 5)
    d = result.to_dict()
    assert d["calculations"][0]["result"] == pytest.approx(14.858, abs=1e-3)
    assert d["usage"]["input_tokens"] == 100 and json.dumps(d)


def test_analyst_revises_after_failed_verification(searcher):
    llm = ScriptedLLM([
        LLMResponse("", [call("read_page", doc_id="ACME_2023_10K", page=5)]),
        LLMResponse("", [call("submit_answer", answer="Capex was $1,600 million [S1].")]),
        LLMResponse("", [call("submit_answer", 2, answer="Capex was $1,577 million [S1].")]),
    ])
    result = Analyst(llm, searcher).run("Acme FY2023 capex?")
    assert result.verification.passed and result.answer == "Capex was $1,577 million [S1]."
    feedback = llm.calls[2][-1]
    assert feedback["role"] == "tool" and feedback["is_error"] and "$1,600 million" in feedback["content"]
    assert [s.detail["passed"] for s in result.steps if s.kind == "verify"] == [False, True]


def test_analyst_gives_up_revising_after_budget(searcher):
    bad = LLMResponse("Capex was $9,999 million [S1].")  # plain text answer, no tool call
    llm = ScriptedLLM([LLMResponse("", [call("read_page", doc_id="ACME_2023_10K", page=5)]), bad, bad])
    result = Analyst(llm, searcher, max_revisions=1).run("Acme FY2023 capex?")
    assert result.stopped == "answered" and not result.verification.passed
    assert result.verification.ungrounded == ["$9,999 million"]
    assert llm.calls[2][-1] == {"role": "user", "content": llm.calls[2][-1]["content"]}  # feedback as a user turn


def test_analyst_answers_unanswered_calls_in_same_turn(searcher):
    llm = ScriptedLLM([
        LLMResponse("", [call("calculate", expression="1+1"), call("submit_answer", answer="No sources.")]),
        LLMResponse("The filings do not say."),
    ])
    Analyst(llm, searcher).run("q")
    tool_msgs = [m for m in llm.calls[1] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["call_calculate_1", "call_submit_answer_1"]
    assert "Not executed" in tool_msgs[0]["content"]


def test_analyst_step_budget_and_errors(searcher):
    loop = ScriptedLLM(lambda s, m: LLMResponse("", [call("list_filings")]))
    result = Analyst(loop, searcher, max_steps=3).run("q")
    assert result.stopped == "max_steps" and len(loop.calls) == 3 and "step budget" in result.answer

    result = Analyst(ScriptedLLM([]), searcher).run("q")
    assert result.stopped == "error" and result.error == "script exhausted"


def test_doc_scope_in_task(searcher):
    llm = ScriptedLLM([LLMResponse("Nothing found.")])
    Analyst(llm, searcher, max_revisions=0).run("q", ["ACME_2023_10K"])
    assert llm.calls[0][0]["content"].startswith("Filings to use: ACME_2023_10K")


# -- model adapters -------------------------------------------------------------------------------

TOOLS = [ToolSpec("calculate", "calc", {"type": "object", "properties": {"expression": {"type": "string"}}})]
TRANSCRIPT = [
    {"role": "user", "content": "q"},
    {"role": "assistant", "content": "thinking", "tool_calls": [ToolCall("t1", "calculate", {"expression": "1+1"}),
                                                                ToolCall("t2", "calculate", {"expression": "2"})]},
    {"role": "tool", "tool_call_id": "t1", "name": "calculate", "content": "2", "is_error": False},
    {"role": "tool", "tool_call_id": "t2", "name": "calculate", "content": "bad", "is_error": True},
]


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_anthropic_wire_format(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"], seen["body"] = request.headers, json.loads(request.content)
        return httpx.Response(200, json={"content": [{"type": "text", "text": "ok "},
                                                     {"type": "tool_use", "id": "t3", "name": "calculate",
                                                      "input": {"expression": "3"}}],
                                         "stop_reason": "tool_use", "usage": {"input_tokens": 11, "output_tokens": 4}})

    resp = AnthropicLLM("claude-sonnet-5-5", client=_client(handler)).complete("sys", TRANSCRIPT, TOOLS)
    assert seen["headers"]["x-api-key"] == "sk-test" and seen["headers"]["anthropic-version"] == "2023-06-01"
    body = seen["body"]
    assert body["system"] == "sys" and body["tools"][0]["input_schema"]["type"] == "object"
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    results = body["messages"][2]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"] and results[1]["is_error"] is True
    assert body["messages"][1]["content"][1] == {"type": "tool_use", "id": "t1", "name": "calculate",
                                                 "input": {"expression": "1+1"}}
    assert resp.text == "ok " and resp.tool_calls[0].arguments == {"expression": "3"}
    assert (resp.input_tokens, resp.output_tokens, resp.stop_reason) == (11, 4, "tool_use")


def test_openai_wire_format(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oa")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"], seen["body"] = request.headers["authorization"], json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"finish_reason": "tool_calls", "message": {
            "content": None, "tool_calls": [{"id": "c1", "type": "function",
                                             "function": {"name": "calculate", "arguments": '{"expression": "4"}'}},
                                            {"id": "c2", "type": "function",
                                             "function": {"name": "calculate", "arguments": "{oops"}}]}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 2}})

    resp = OpenAICompatLLM(client=_client(handler)).complete("sys", TRANSCRIPT, TOOLS)
    body = seen["body"]
    assert seen["auth"] == "Bearer sk-oa" and body["model"] == "gpt-6-luna" and "max_completion_tokens" in body
    assert [m["role"] for m in body["messages"]] == ["system", "user", "assistant", "tool", "tool"]
    assert json.loads(body["messages"][2]["tool_calls"][0]["function"]["arguments"]) == {"expression": "1+1"}
    assert resp.tool_calls[0].arguments == {"expression": "4"} and resp.tool_calls[1].arguments == {"_raw": "{oops"}
    assert (resp.input_tokens, resp.output_tokens) == (7, 2)


def test_retries_then_fails(monkeypatch):
    monkeypatch.setattr("filings_analyst.agent.llm.time.sleep", lambda s: None)
    statuses = iter([529, 429, 200])

    def flaky(request):
        status = next(statuses)
        return httpx.Response(status, json={"content": [{"type": "text", "text": "done"}]} if status == 200 else {})

    assert AnthropicLLM(client=_client(flaky), retries=3).complete("s", [{"role": "user", "content": "q"}], []).text == "done"

    attempts = []

    def bad_request(request):
        attempts.append(1)
        return httpx.Response(400, text="tool_choice not supported")

    with pytest.raises(LLMError, match="HTTP 400"):
        AnthropicLLM(client=_client(bad_request)).complete("s", [{"role": "user", "content": "q"}], [])
    assert len(attempts) == 1  # 4xx other than 408/409/429 is not retried

    with pytest.raises(LLMError, match="no choices"):
        OpenAICompatLLM(client=_client(lambda r: httpx.Response(200, json={}))).complete("s", [], [])
    with pytest.raises(ValueError):
        make_llm("cohere")
