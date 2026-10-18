"""MCP server (spec 2026-07-28, stateless Streamable HTTP) exposing the filings tools.

Any MCP client (Claude, ChatGPT, Cursor, VS Code, an agent framework) can
search filings, read pages, compute, and, when a model is configured, ask the
full analyst agent for a cited, verified answer.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from . import __version__
from .agent.analyst import Analyst
from .agent.tools import Toolbox

PROTOCOL_VERSION = "2026-07-28"
SUPPORTED = [PROTOCOL_VERSION]
HEADER_MISMATCH = -32020
UNSUPPORTED_VERSION = -32022

READ_ONLY = {"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False}


def _err(req_id: Any, code: int, message: str, status: int = 200) -> JSONResponse:
    return JSONResponse({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}, status)


def _ok(req_id: Any, result: dict) -> JSONResponse:
    result.setdefault("resultType", "complete")
    result.setdefault("_meta", {})["io.modelcontextprotocol/serverInfo"] = {"name": "filings-analyst",
                                                                            "version": __version__}
    return JSONResponse({"jsonrpc": "2.0", "id": req_id, "result": result})


class FilingsMCP:
    def __init__(self, toolbox_factory: Callable[[], Toolbox], analyst_factory: Callable[[], Analyst] | None = None):
        self.toolbox_factory = toolbox_factory
        self.analyst_factory = analyst_factory

    def tools(self) -> list[dict]:
        specs = [s for s in self.toolbox_factory().specs if s.name != "submit_answer"]
        tools = [{"name": s.name, "description": s.description, "inputSchema": s.parameters,
                  "annotations": READ_ONLY} for s in specs]
        if self.analyst_factory is not None:
            tools.append({
                "name": "ask_analyst",
                "description": "Research a question across the filings with the analyst agent and return a cited "
                               "answer whose figures were checked against the cited pages.",
                "inputSchema": {"type": "object", "properties": {
                    "question": {"type": "string"},
                    "doc_ids": {"type": "array", "items": {"type": "string"}},
                }, "required": ["question"]},
                "annotations": {**READ_ONLY, "openWorldHint": True},
            })
        return sorted(tools, key=lambda t: t["name"])  # deterministic order keeps client caches stable

    async def handle(self, request: Request) -> Response:
        try:
            body = json.loads(await request.body())
        except (json.JSONDecodeError, UnicodeDecodeError):
            return _err(None, -32700, "Parse error", 400)
        if not isinstance(body, dict) or body.get("jsonrpc") != "2.0" or not isinstance(body.get("method"), str):
            return _err(None, -32600, "Invalid Request", 400)
        req_id, method = body.get("id"), body["method"]
        params = body.get("params") if isinstance(body.get("params"), dict) else {}
        if "id" not in body:
            return Response(status_code=202)

        h_method = request.headers.get("mcp-method")
        if h_method is not None and h_method != method:
            return _err(req_id, HEADER_MISMATCH, "Mcp-Method header does not match body", 400)
        if method == "tools/call":
            h_name = request.headers.get("mcp-name")
            if h_name is not None and h_name != params.get("name"):
                return _err(req_id, HEADER_MISMATCH, "Mcp-Name header does not match body", 400)
        version = (params.get("_meta") or {}).get("io.modelcontextprotocol/protocolVersion") \
            or request.headers.get("mcp-protocol-version")
        if version and version not in SUPPORTED:
            return _err(req_id, UNSUPPORTED_VERSION, f"Unsupported protocol version {version}")

        if method == "server/discover":
            return _ok(req_id, {"supportedVersions": SUPPORTED,
                                "capabilities": {"tools": {"listChanged": False}},
                                "serverInfo": {"name": "filings-analyst", "version": __version__},
                                "instructions": "Search and read SEC filings; cite results by doc_id and page."})
        if method == "tools/list":
            return _ok(req_id, {"tools": self.tools(), "ttlMs": 300_000, "cacheScope": "public"})
        if method == "tools/call":
            return await self._call(req_id, params)
        return _err(req_id, -32601, f"Method not found: {method}")

    async def _call(self, req_id: Any, params: dict) -> Response:
        name = params.get("name")
        args = params.get("arguments") or {}
        if name == "ask_analyst" and self.analyst_factory is not None:
            if not isinstance(args.get("question"), str) or not args["question"].strip():
                return _err(req_id, -32602, "ask_analyst requires a question")
            result = await asyncio.to_thread(self.analyst_factory().run, args["question"], args.get("doc_ids"))
            data = result.to_dict()
            text = result.answer + "\n\nSources:\n" + "\n".join(
                f"[{s['id']}] {s['doc_id']}, p. {s['page']}" for s in data["sources"])
            return _ok(req_id, {"content": [{"type": "text", "text": text}],
                                "structuredContent": {k: data[k] for k in ("answer", "sources", "verification",
                                                                           "calculations", "usage", "latency_ms")},
                                "isError": result.stopped == "error"})
        names = {t["name"] for t in self.tools()}
        if name not in names:
            return _err(req_id, -32602, f"Unknown tool: {name}")
        toolbox = self.toolbox_factory()
        output, is_error = await asyncio.to_thread(toolbox.call, name, args)
        structured: dict = {}
        if toolbox.ledger.sources:
            structured["sources"] = [{"id": s.id, "doc_id": s.doc_id, "page": s.page_no + 1, "header": s.header}
                                     for s in toolbox.ledger.sources.values()]
        result: dict = {"content": [{"type": "text", "text": output}], "isError": is_error}
        if structured:
            result["structuredContent"] = structured
        return _ok(req_id, result)
