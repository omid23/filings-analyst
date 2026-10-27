from __future__ import annotations

import json

import httpx
import pytest

from filings_analyst.agent.analyst import Analyst
from filings_analyst.agent.llm import LLMError, ToolCall, ToolSpec, make_llm
from filings_analyst.agent.vertex import GeminiLLM, GoogleCredentials, VertexClaudeLLM

TOOLS = [ToolSpec("calculate", "calc", {"type": "object", "properties": {"expression": {"type": "string"}}})]


class FakeCredentials:
    def token(self) -> str:
        return "ya29.test"

    def project(self) -> str:
        return "acme-research"


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _recorder(response: dict):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["headers"], seen["body"] = str(request.url), request.headers, json.loads(request.content)
        return httpx.Response(200, json=response)

    return seen, handler


def test_gemini_wire_format():
    signed = ToolCall("g1", "calculate", {"expression": "1+1"}, {"id": "g1", "thoughtSignature": "sig-A"})
    unsigned = ToolCall("call_x", "calculate", {"expression": "2"})  # parallel call: no signature, no Gemini id
    transcript = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [signed, unsigned]},
        {"role": "tool", "tool_call_id": "g1", "name": "calculate", "content": "2", "is_error": False},
        {"role": "tool", "tool_call_id": "call_x", "name": "calculate", "content": "bad", "is_error": True},
    ]
    seen, handler = _recorder({
        "candidates": [{"finishReason": "STOP", "content": {"role": "model", "parts": [
            {"text": "weighing it", "thought": True},
            {"text": "Let me compute. "},
            {"functionCall": {"id": "g2", "name": "calculate", "args": {"expression": "3"}}, "thoughtSignature": "sig-B"},
            {"functionCall": {"name": "submit_answer", "args": {"answer": "3"}}},
        ]}}],
        "usageMetadata": {"promptTokenCount": 1200, "cachedContentTokenCount": 1000,
                          "candidatesTokenCount": 30, "thoughtsTokenCount": 70},
        "modelVersion": "gemini-3.5-flash",
    })

    llm = GeminiLLM(credentials=FakeCredentials(), client=_client(handler))
    resp = llm.complete("sys", transcript, TOOLS)
    assert seen["url"] == ("https://aiplatform.googleapis.com/v1/projects/acme-research/locations/global"
                           "/publishers/google/models/gemini-3.5-flash:generateContent")
    assert seen["headers"]["authorization"] == "Bearer ya29.test"
    body = seen["body"]
    assert body["systemInstruction"] == {"parts": [{"text": "sys"}]}
    assert body["tools"][0]["functionDeclarations"][0]["parameters"]["type"] == "object"
    # All calls in one model turn, all responses in the next user turn, in the same order.
    assert [c["role"] for c in body["contents"]] == ["user", "model", "user"]
    model_parts, responses = body["contents"][1]["parts"], body["contents"][2]["parts"]
    assert model_parts[0] == {"functionCall": {"name": "calculate", "args": {"expression": "1+1"}, "id": "g1"},
                              "thoughtSignature": "sig-A"}
    assert model_parts[1] == {"functionCall": {"name": "calculate", "args": {"expression": "2"}}}
    assert responses[0]["functionResponse"] == {"name": "calculate", "response": {"content": "2"}, "id": "g1"}
    assert responses[1]["functionResponse"] == {"name": "calculate", "response": {"error": "bad"}}  # made-up id stays local

    assert resp.text == "Let me compute. "  # thought summaries are not part of the answer
    assert [(c.id, c.name, c.extra) for c in resp.tool_calls][0] == ("g2", "calculate", {"id": "g2", "thoughtSignature": "sig-B"})
    assert resp.tool_calls[1].id and resp.tool_calls[1].extra == {}
    # cachedContentTokenCount is inside promptTokenCount; thinking is billed as output.
    assert (resp.input_tokens, resp.cache_read_tokens, resp.output_tokens) == (200, 1000, 100)
    assert (resp.stop_reason, resp.model) == ("STOP", "gemini-3.5-flash")


def test_gemini_signature_survives_the_agent_loop(searcher):
    bodies = []
    replies = iter([
        {"parts": [{"functionCall": {"name": "calculate", "args": {"expression": "6 * 7"}}, "thoughtSignature": "sig"}]},
        {"parts": [{"functionCall": {"name": "submit_answer", "args": {"answer": "The product is 42."}}}]},
    ])

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"candidates": [{"content": next(replies), "finishReason": "STOP"}]})

    llm = GeminiLLM(credentials=FakeCredentials(), client=_client(handler))
    result = Analyst(llm, searcher, max_revisions=0).run("What is 6 times 7?")  # uncited: skip the revision
    assert result.answer == "The product is 42." and result.stopped == "answered"
    second = bodies[1]["contents"]
    assert second[1]["parts"][0]["thoughtSignature"] == "sig"
    assert json.loads(second[2]["parts"][0]["functionResponse"]["response"]["content"])["result"] == 42


def test_gemini_errors():
    blocked = _client(lambda r: httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}}))
    with pytest.raises(LLMError, match="blocked: SAFETY"):
        GeminiLLM(credentials=FakeCredentials(), client=blocked).complete("s", [{"role": "user", "content": "q"}], [])


def test_vertex_claude_request():
    seen, handler = _recorder({"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
                               "usage": {"input_tokens": 5, "output_tokens": 2, "cache_read_input_tokens": 40}})
    llm = VertexClaudeLLM("claude-sonnet-5-5", location="us-east5", credentials=FakeCredentials(),
                          client=_client(handler))
    resp = llm.complete("sys", [{"role": "user", "content": "q"}], TOOLS)
    assert seen["url"] == ("https://us-east5-aiplatform.googleapis.com/v1/projects/acme-research/locations/us-east5"
                           "/publishers/anthropic/models/claude-sonnet-5-5:rawPredict")
    assert seen["headers"]["authorization"] == "Bearer ya29.test" and "x-api-key" not in seen["headers"]
    body = seen["body"]
    assert "model" not in body and body["anthropic_version"] == "vertex-2023-10-16"
    # Same caching layout as the first-party adapter.
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"} and body["cache_control"] == {"type": "ephemeral"}
    assert (resp.text, resp.input_tokens, resp.cache_read_tokens) == ("ok", 5, 40)


def test_make_llm_vertex_routing(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "europe-west4")
    gemini = make_llm("vertex", cache=False)  # --no-prompt-cache is meaningless for Gemini and dropped
    assert isinstance(gemini, GeminiLLM) and gemini.model == "gemini-3.5-flash" and gemini.location == "europe-west4"
    claude = make_llm("vertex", "claude-opus-5-5", cache=False, project="p1")
    assert isinstance(claude, VertexClaudeLLM) and claude.cache is False and claude.project == "p1"


def test_google_credentials_from_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_OAUTH_ACCESS_TOKEN", "ya29.env")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-project")
    creds = GoogleCredentials()
    assert (creds.token(), creds.project()) == ("ya29.env", "env-project")


def test_google_credentials_cache_and_missing_tools(monkeypatch):
    for var in ("GOOGLE_OAUTH_ACCESS_TOKEN", "GOOGLE_CLOUD_PROJECT", "CLOUDSDK_CORE_PROJECT"):
        monkeypatch.delenv(var, raising=False)
    fetched = []
    creds = GoogleCredentials()
    monkeypatch.setattr(creds, "_fetch_token", lambda: fetched.append(1) or f"tok{len(fetched)}")
    assert creds.token() == creds.token() == "tok1" and len(fetched) == 1  # cached within the TTL
    creds._expires = 0
    assert creds.token() == "tok2"

    monkeypatch.setattr(GoogleCredentials, "_adc", staticmethod(lambda: None))
    monkeypatch.setattr("filings_analyst.agent.vertex.shutil.which", lambda name: None)
    with pytest.raises(LLMError, match="no Google credentials"):
        GoogleCredentials().project()
