"""HTTP app: a small research UI, a JSON API and the MCP endpoint, served together."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from importlib import resources

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Route

from .. import __version__
from ..agent.analyst import Analyst
from ..agent.tools import Toolbox
from ..index.retrieve import Searcher
from ..mcp_server import FilingsMCP


def create_app(searcher: Searcher, analyst_factory: Callable[[], Analyst] | None = None) -> Starlette:
    mcp = FilingsMCP(lambda: Toolbox(searcher), analyst_factory)
    index_html = (resources.files("filings_analyst.web") / "index.html").read_text(encoding="utf-8")

    async def home(request: Request) -> Response:
        return HTMLResponse(index_html)

    async def info(request: Request) -> Response:
        stats = await asyncio.to_thread(searcher.store.stats)
        return JSONResponse({"version": __version__, **stats, "can_ask": analyst_factory is not None,
                             "dense": searcher.dense.embedder.name if searcher.dense else None})

    async def filings(request: Request) -> Response:
        docs = await asyncio.to_thread(searcher.store.documents)
        return JSONResponse([{"doc_id": d.doc_id, "company": d.company, "form": d.form, "period": d.period}
                             for d in docs])

    async def search(request: Request) -> Response:
        body = await request.json()
        query = str(body.get("query") or "").strip()
        if not query:
            return JSONResponse({"error": "Type a question or keywords to search."}, 400)
        k = max(1, min(int(body.get("k") or 8), 30))
        result = await asyncio.to_thread(searcher.search, query, k, mode=body.get("mode") or "hybrid",
                                         doc_ids=body.get("doc_ids") or None)
        hits, seen = [], set()
        for h in result.hits:
            key = (h.chunk.doc_id, h.chunk.page_no)
            if key in seen:
                continue
            seen.add(key)
            hits.append({"doc_id": h.chunk.doc_id, "page": h.chunk.page_no + 1, "header": h.chunk.header,
                         "text": h.chunk.text[:900], "score": round(h.score, 5), "retrievers": h.sources})
            if len(hits) >= k:
                break
        return JSONResponse({"query": query, "filter": result.resolution.reason or "all filings",
                             "filings": result.resolution.doc_ids[:8], "hits": hits,
                             "timings_ms": {k2: round(v, 1) for k2, v in result.timings_ms.items()}})

    async def page(request: Request) -> Response:
        doc_id = request.query_params.get("doc_id", "")
        try:
            number = int(request.query_params.get("page", "0"))
        except ValueError:
            return JSONResponse({"error": "page must be a number"}, 400)
        p = await asyncio.to_thread(searcher.store.page, doc_id, number - 1)
        if p is None:
            return JSONResponse({"error": f"{doc_id} has no page {number}"}, 404)
        doc = await asyncio.to_thread(searcher.store.document, doc_id)
        total = await asyncio.to_thread(searcher.store.page_count, doc_id)
        return JSONResponse({"doc_id": doc_id, "page": number, "pages": total, "title": p.title, "section": p.section,
                             "label": doc.label if doc else doc_id, "text": p.text})

    async def ask(request: Request) -> Response:
        if analyst_factory is None:
            return JSONResponse({"error": "No model is configured. Start the server with --provider anthropic or "
                                          "--provider openai and the matching API key to ask questions. Search works "
                                          "without one."}, 503)
        body = await request.json()
        question = str(body.get("question") or "").strip()
        if not question:
            return JSONResponse({"error": "Type a question to ask."}, 400)
        result = await asyncio.to_thread(analyst_factory().run, question, body.get("doc_ids") or None)
        return JSONResponse(result.to_dict())

    async def healthz(request: Request) -> Response:
        return JSONResponse({"status": "ok"})

    return Starlette(routes=[
        Route("/", home), Route("/api/info", info), Route("/api/filings", filings),
        Route("/api/search", search, methods=["POST"]), Route("/api/page", page),
        Route("/api/ask", ask, methods=["POST"]), Route("/mcp", mcp.handle, methods=["POST"]),
        Route("/healthz", healthz),
    ])
