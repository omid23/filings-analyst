"""Google Cloud Vertex AI adapters: Gemini (``generateContent``) and Claude (``rawPredict``).

Same provider-neutral transcript as ``llm.py``, same plain-httpx style. Vertex authenticates with
short-lived OAuth tokens instead of an API key and puts the project, region and model in the URL.

Configuration comes from the usual Google environment variables:

* ``GOOGLE_CLOUD_PROJECT``: project id (falls back to the ADC or gcloud default project)
* ``GOOGLE_CLOUD_LOCATION``: region, default ``global``
* ``GOOGLE_OAUTH_ACCESS_TOKEN``: optional fixed token, otherwise Application Default Credentials
  (``pip install 'filings-analyst[gcp]'``) or the gcloud CLI
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
import uuid
from collections.abc import Sequence
from typing import Any, Protocol

import httpx

from .llm import AnthropicLLM, LLMError, LLMResponse, ToolCall, ToolSpec, _post_with_retry

SCOPE = "https://www.googleapis.com/auth/cloud-platform"
DEFAULT_GEMINI = "gemini-3.5-flash"
NO_CREDENTIALS = ("no Google credentials: run `gcloud auth application-default login` and "
                  "`pip install 'filings-analyst[gcp]'`, install the gcloud CLI, or set GOOGLE_OAUTH_ACCESS_TOKEN")


class Credentials(Protocol):
    def token(self) -> str: ...

    def project(self) -> str: ...


class GoogleCredentials:
    """OAuth access tokens and the default project for Vertex AI.

    Tokens come from, in order: ``GOOGLE_OAUTH_ACCESS_TOKEN`` (CI, short jobs), Application Default
    Credentials through ``google-auth`` when it is installed (service accounts, workload identity,
    user logins), then ``gcloud auth print-access-token``. Tokens live an hour and are renewed after
    45 minutes. One instance is shared by every adapter, so parallel eval workers don't each shell out.
    """

    TTL = 45 * 60

    def __init__(self):
        self._lock = threading.Lock()
        self._token, self._expires = "", 0.0
        self._project: str | None = None

    def token(self) -> str:
        if os.environ.get("GOOGLE_OAUTH_ACCESS_TOKEN"):
            return os.environ["GOOGLE_OAUTH_ACCESS_TOKEN"]
        with self._lock:
            if not self._token or time.monotonic() >= self._expires:
                self._token = self._fetch_token()
                self._expires = time.monotonic() + self.TTL
            return self._token

    def project(self) -> str:
        for var in ("GOOGLE_CLOUD_PROJECT", "CLOUDSDK_CORE_PROJECT"):
            if os.environ.get(var):
                return os.environ[var]
        with self._lock:
            if self._project is None:
                adc = self._adc()
                self._project = (adc[1] if adc else None) or self._gcloud("config", "get-value", "project")
        if not self._project:
            raise LLMError("no Google Cloud project: set GOOGLE_CLOUD_PROJECT")
        return self._project

    def _fetch_token(self) -> str:
        adc = self._adc()
        if adc is None:
            return self._gcloud("auth", "print-access-token")
        creds, project = adc
        try:
            from google.auth.transport.requests import Request

            creds.refresh(Request())
        except Exception as exc:  # RefreshError, TransportError
            raise LLMError(f"Google credentials: {exc}") from exc
        self._project = self._project or project
        return creds.token

    @staticmethod
    def _adc():
        """(credentials, project) from google-auth, or None when it isn't installed or finds nothing."""
        try:
            import google.auth
            import google.auth.exceptions
            import google.auth.transport.requests  # noqa: F401  (needs the requests extra)
        except ImportError:
            return None
        try:
            return google.auth.default(scopes=[SCOPE])
        except google.auth.exceptions.DefaultCredentialsError:
            return None

    @staticmethod
    def _gcloud(*args: str) -> str:
        exe = shutil.which("gcloud")  # resolves gcloud.cmd on Windows
        if exe is None:
            raise LLMError(NO_CREDENTIALS)
        try:
            out = subprocess.run([exe, *args], capture_output=True, text=True, timeout=60, check=True)
        except subprocess.CalledProcessError as exc:
            raise LLMError(f"gcloud {' '.join(args)}: {exc.stderr.strip()[:300]}") from exc
        except (subprocess.SubprocessError, OSError) as exc:
            raise LLMError(f"gcloud {' '.join(args)}: {exc}") from exc
        return out.stdout.strip()


DEFAULT_CREDENTIALS = GoogleCredentials()


class _VertexEndpoint:
    model: str
    project: str | None
    location: str
    endpoint: str | None
    credentials: Credentials

    def _setup(self, project: str | None, location: str | None, endpoint: str | None,
               credentials: Credentials | None) -> None:
        self.project, self.endpoint = project, endpoint
        self.location = location or os.environ.get("GOOGLE_CLOUD_LOCATION") or "global"
        self.credentials = credentials or DEFAULT_CREDENTIALS

    def _url(self, publisher: str, method: str) -> str:
        host = self.endpoint or ("https://aiplatform.googleapis.com" if self.location == "global"
                                 else f"https://{self.location}-aiplatform.googleapis.com")
        project = self.project or self.credentials.project()
        return (f"{host.rstrip('/')}/v1/projects/{project}/locations/{self.location}"
                f"/publishers/{publisher}/models/{self.model}:{method}")

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.credentials.token()}", "content-type": "application/json"}


class VertexClaudeLLM(_VertexEndpoint, AnthropicLLM):
    """Claude on Vertex AI. The body is the Messages API body with the model moved into the URL and
    ``anthropic_version`` in the body, so prompt caching works exactly as in ``AnthropicLLM``."""

    name = "vertex"

    def __init__(self, model: str = "claude-sonnet-5-5", *, project: str | None = None, location: str | None = None,
                 endpoint: str | None = None, credentials: Credentials | None = None, **kwargs):
        super().__init__(model, **kwargs)
        self._setup(project, location, endpoint, credentials)

    def _request(self, body: dict) -> tuple[str, dict, dict]:
        body = {k: v for k, v in body.items() if k != "model"}
        body["anthropic_version"] = "vertex-2023-10-16"
        return self._url("anthropic", "rawPredict"), self._headers(), body


class GeminiLLM(_VertexEndpoint):
    """Gemini through Vertex AI ``generateContent``.

    Gemini 3 models attach a thought signature to the first function call of each step and reject the
    next request (400) unless it comes back verbatim, so it rides along in ``ToolCall.extra``. Calls
    and their responses are grouped (all calls in one model turn, all responses in the next), which
    is the order Vertex requires for parallel calls. Gemini caches repeated prefixes implicitly;
    ``cachedContentTokenCount`` is part of ``promptTokenCount`` and is split out like OpenAI's.
    """

    name = "vertex"

    def __init__(self, model: str = DEFAULT_GEMINI, *, project: str | None = None, location: str | None = None,
                 endpoint: str | None = None, credentials: Credentials | None = None,
                 max_tokens: int = 8192, retries: int = 3, client: httpx.Client | None = None):
        # Thinking tokens count against maxOutputTokens, hence a larger budget than the other adapters.
        self.model, self.max_tokens, self.retries = model, max_tokens, retries
        self._setup(project, location, endpoint, credentials)
        self._client = client or httpx.Client(timeout=120)

    @staticmethod
    def to_wire(messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        call_ids: set[str] = set()  # ids Gemini assigned; ids we made up for it are never sent

        def add(role: str, parts: list[dict]) -> None:
            if out and out[-1]["role"] == role:
                out[-1]["parts"].extend(parts)
            else:
                out.append({"role": role, "parts": parts})

        for m in messages:
            if m["role"] == "user":
                add("user", [{"text": m["content"]}])
            elif m["role"] == "assistant":
                parts: list[dict] = []
                if m.get("content"):
                    parts.append({"text": m["content"]})
                for tc in m.get("tool_calls") or []:
                    call: dict[str, Any] = {"name": tc.name, "args": tc.arguments}
                    if tc.extra.get("id"):
                        call["id"] = tc.extra["id"]
                        call_ids.add(tc.extra["id"])
                    part: dict[str, Any] = {"functionCall": call}
                    if tc.extra.get("thoughtSignature"):
                        part["thoughtSignature"] = tc.extra["thoughtSignature"]
                    parts.append(part)
                add("model", parts or [{"text": ""}])
            elif m["role"] == "tool":
                result = {"name": m["name"], "response": {"error" if m.get("is_error") else "content": m["content"]}}
                if m["tool_call_id"] in call_ids:
                    result["id"] = m["tool_call_id"]
                add("user", [{"functionResponse": result}])
        return out

    def complete(self, system: str, messages: list[dict], tools: Sequence[ToolSpec]) -> LLMResponse:
        body: dict[str, Any] = {
            "contents": self.to_wire(messages),
            "systemInstruction": {"parts": [{"text": system}]},
            "generationConfig": {"maxOutputTokens": self.max_tokens},
        }
        if tools:
            body["tools"] = [{"functionDeclarations": [{"name": t.name, "description": t.description,
                                                        "parameters": t.parameters} for t in tools]}]
        url, headers = self._url("google", "generateContent"), self._headers()
        started = time.perf_counter()
        data = _post_with_retry(self._client, url, headers, body, self.retries)
        candidates = data.get("candidates") or []
        if not candidates:
            reason = (data.get("promptFeedback") or {}).get("blockReason", "")
            raise LLMError(f"response has no candidates{f' (blocked: {reason})' if reason else ''}")
        cand = candidates[0]
        text, calls = [], []
        for part in (cand.get("content") or {}).get("parts") or []:
            if "functionCall" in part:
                fc = part["functionCall"]
                extra = {k: v for k, v in (("id", fc.get("id")), ("thoughtSignature", part.get("thoughtSignature"))) if v}
                args = fc.get("args")
                calls.append(ToolCall(fc.get("id") or f"call_{uuid.uuid4().hex[:8]}", fc.get("name", ""),
                                      args if isinstance(args, dict) else {}, extra))
            elif "text" in part and not part.get("thought"):
                text.append(part["text"])
        usage = data.get("usageMetadata") or {}
        cached = int(usage.get("cachedContentTokenCount") or 0)
        return LLMResponse("".join(text), calls, cand.get("finishReason", ""),
                           int(usage.get("promptTokenCount") or 0) - cached,
                           int(usage.get("candidatesTokenCount") or 0) + int(usage.get("thoughtsTokenCount") or 0),
                           data.get("modelVersion", self.model), (time.perf_counter() - started) * 1000,
                           cache_read_tokens=cached)


def make_vertex_llm(model: str | None = None, **kwargs):
    """Claude models (``claude-*``) go to the Anthropic publisher on Vertex, everything else to Gemini."""
    if model and model.startswith("claude"):
        return VertexClaudeLLM(model, **kwargs)
    kwargs.pop("cache", None)  # Gemini's implicit caching has no request switch
    return GeminiLLM(model or DEFAULT_GEMINI, **kwargs)
