"""Command line: index, search, ask, serve, eval, edgar."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

DEFAULT_INDEX = "data/index"


def _llm(args, provider: str | None = None, model: str | None = None):
    from .agent.llm import make_llm

    provider = provider or args.provider
    if model is None and provider == args.provider:
        model = args.model  # another provider gets its own default model, not this one's name
    kwargs = {"cache": False} if provider == "anthropic" and getattr(args, "no_prompt_cache", False) else {}
    return make_llm(provider, model, **kwargs)


def _analyst_factory(args, searcher):
    if not getattr(args, "provider", None):
        return None
    from .agent.analyst import Analyst

    def factory() -> Analyst:
        return Analyst(_llm(args), searcher, max_steps=args.max_steps)

    return factory


def _searcher(args):
    from .index.indexer import open_searcher
    from .index.retrieve import SectionRouter

    return open_searcher(args.index, rerankers=[] if getattr(args, "no_router", False) else [SectionRouter()])


def _documents_from_folder(folder: Path):
    from .models import Document

    docs = []
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() not in (".pdf", ".htm", ".html"):
            continue
        # Convention: COMPANY_PERIOD_FORM.pdf, e.g. ACME_2023_10K.pdf or ACME_2024Q2_10Q.pdf
        parts = path.stem.split("_")
        company = parts[0].replace("-", " ").title() if parts else path.stem
        period = parts[1] if len(parts) > 1 else "unknown"
        form = parts[2].upper().replace("10K", "10-K").replace("10Q", "10-Q").replace("8K", "8-K") if len(parts) > 2 else "FILING"
        docs.append(Document(path.stem, company, form, period, str(path)))
    return docs


def cmd_index(args) -> None:
    from .index.indexer import build_index

    if args.financebench:
        from .eval.financebench import load_documents, load_questions

        only = {q.doc_id for q in load_questions(args.financebench)} if args.questions_only else None
        docs = load_documents(args.financebench, only)
    elif args.folder:
        docs = _documents_from_folder(Path(args.folder))
    else:
        sys.exit("give --financebench PATH or --folder PATH (or use `edgar` to download filings)")
    stats = build_index(docs, args.index, embedder=None if args.embedder == "none" else args.embedder,
                        workers=args.workers)
    print(json.dumps(stats, indent=1))


def cmd_edgar(args) -> None:
    from .index.indexer import build_index
    from .ingest.edgar import EdgarClient

    client = EdgarClient(args.user_agent)
    docs = []
    for ticker in args.tickers:
        for filing in client.filings(ticker, tuple(args.forms), limit=args.limit):
            docs.append(client.download(filing, Path(args.index) / "raw"))
            print(f"downloaded {filing.doc_id}")
    print(json.dumps(build_index(docs, args.index, embedder=args.embedder, workers=args.workers), indent=1))


def cmd_search(args) -> None:
    searcher = _searcher(args)
    result = searcher.search(args.query, k=args.k, mode=args.mode)
    print(f"filter: {result.resolution.reason or 'all filings'}")
    for i, (doc_id, page) in enumerate(result.pages(args.k), start=1):
        hit = next(h for h in result.hits if (h.chunk.doc_id, h.chunk.page_no) == (doc_id, page))
        snippet = " ".join(hit.chunk.text.split())[:160]
        print(f"{i:>2}. {doc_id} p.{page + 1}  {hit.chunk.title or hit.chunk.section or ''}\n    {snippet}")


def cmd_ask(args) -> None:
    searcher = _searcher(args)
    factory = _analyst_factory(args, searcher)
    if factory is None:
        sys.exit("--provider is required (anthropic or openai)")
    result = factory().run(args.question, args.doc or None)
    if args.json:
        print(json.dumps(result.to_dict(), indent=1))
        return
    print(result.answer, "\n")
    for s in result.cited_sources():
        print(f"[{s['id']}] {s['doc_id']}, p. {s['page']}")
    v = result.verification
    print(f"\nverified: {v.passed} ({v.numbers_checked} figures checked; ungrounded: {v.ungrounded or 'none'})")


def cmd_serve(args) -> None:
    import uvicorn

    from .web.app import create_app

    searcher = _searcher(args)
    app = create_app(searcher, _analyst_factory(args, searcher))
    print(f"Filings Analyst on http://{args.host}:{args.port}  (MCP endpoint: /mcp)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


def cmd_eval_retrieval(args) -> None:
    from .eval.financebench import load_questions
    from .eval.retrieval import Config, evaluate, to_markdown
    from .index.indexer import open_searcher
    from .index.retrieve import SectionRouter

    questions = [q for q in load_questions(args.financebench) if args.split == "all" or q.split == args.split]
    searcher = open_searcher(args.index)
    configs = [
        Config("bm25, no filing filter", "bm25", "corpus"),
        Config("bm25", "bm25", "resolved"),
        Config("dense", "dense", "resolved"),
        Config("hybrid", "hybrid", "resolved"),
        Config("hybrid + section router", "hybrid", "resolved", rerank=True),
        Config("hybrid + section router, filing given", "hybrid", "oracle", rerank=True),
    ]
    if searcher.dense is None:
        configs = [c for c in configs if c.mode == "bm25"]
    report = evaluate(searcher, questions, configs, SectionRouter())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md = to_markdown(report, f"FinanceBench retrieval, {args.split} split")
    (out / f"retrieval-{args.split}.md").write_text(md, encoding="utf-8")
    if args.save_json:
        (out / f"retrieval-{args.split}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(md)


def cmd_eval_answers(args) -> None:
    from .agent.analyst import Analyst
    from .eval import answers
    from .eval.financebench import load_questions

    questions = [q for q in load_questions(args.financebench) if args.split == "all" or q.split == args.split]
    if args.limit:
        questions = questions[: args.limit]
    searcher = _searcher(args)
    judge = _llm(args, args.judge_provider, args.judge_model) if args.judge else None
    pricing = answers.Pricing(args.price_in, args.price_out, args.price_cache_read, args.price_cache_write)
    report = answers.run(questions, lambda: Analyst(_llm(args), searcher, max_steps=args.max_steps),
                         judge_llm=judge, oracle_filing=args.filing_given, workers=args.workers,
                         price_per_mtok=pricing)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"answers-{args.provider}-{args.split}"
    answers.dump(report, out / f"{name}.json")
    md = answers.to_markdown(report, f"FinanceBench answers, {args.provider} {args.model or ''}, {args.split} split")
    (out / f"{name}.md").write_text(md, encoding="utf-8")
    print(md)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="filings-analyst", description="Agentic research over SEC filings")
    p.add_argument("--index", default=DEFAULT_INDEX, help=f"index directory (default {DEFAULT_INDEX})")
    sub = p.add_subparsers(dest="cmd", required=True)

    def llm_args(sp, required=False):
        sp.add_argument("--provider", choices=["anthropic", "openai"], required=required)
        sp.add_argument("--model", default=None)
        sp.add_argument("--max-steps", type=int, default=12)
        sp.add_argument("--no-prompt-cache", action="store_true",
                        help="don't send cache_control (for gateways that reject it); Anthropic only")

    s = sub.add_parser("index", help="index filings from a FinanceBench clone or a folder of PDFs/HTML")
    s.add_argument("--financebench", help="path to a clone of github.com/patronus-ai/financebench")
    s.add_argument("--questions-only", action="store_true", help="only the filings the 150 questions use")
    s.add_argument("--folder", help="folder of COMPANY_PERIOD_FORM.pdf files")
    s.add_argument("--embedder", default="lsa", choices=["lsa", "fastembed", "openai", "none"])
    s.add_argument("--workers", type=int, default=4)
    s.set_defaults(func=cmd_index)

    s = sub.add_parser("edgar", help="download filings from SEC EDGAR and index them")
    s.add_argument("tickers", nargs="+")
    s.add_argument("--forms", nargs="+", default=["10-K", "10-Q"])
    s.add_argument("--limit", type=int, default=4, help="filings per ticker")
    s.add_argument("--user-agent", default=None, help="'Name email' (or set SEC_USER_AGENT)")
    s.add_argument("--embedder", default="lsa", choices=["lsa", "fastembed", "openai"])
    s.add_argument("--workers", type=int, default=2)
    s.set_defaults(func=cmd_edgar)

    s = sub.add_parser("search", help="hybrid search; prints the best pages")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=8)
    s.add_argument("--mode", default="hybrid", choices=["bm25", "dense", "hybrid"])
    s.add_argument("--no-router", action="store_true")
    s.set_defaults(func=cmd_search)

    s = sub.add_parser("ask", help="research a question with the agent")
    s.add_argument("question")
    s.add_argument("--doc", action="append", help="restrict to a filing id (repeatable)")
    s.add_argument("--json", action="store_true")
    llm_args(s, required=True)
    s.set_defaults(func=cmd_ask)

    s = sub.add_parser("serve", help="web UI + JSON API + MCP endpoint")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8700)
    llm_args(s)
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("eval-retrieval", help="page-level retrieval eval on FinanceBench (no LLM)")
    s.add_argument("--financebench", required=True)
    s.add_argument("--split", default="test", choices=["dev", "test", "all"])
    s.add_argument("--out", default="reports")
    s.add_argument("--save-json", action="store_true")
    s.set_defaults(func=cmd_eval_retrieval)

    s = sub.add_parser("eval-answers", help="end-to-end answer eval on FinanceBench (needs a model)")
    s.add_argument("--financebench", required=True)
    s.add_argument("--split", default="test", choices=["dev", "test", "all"])
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--filing-given", action="store_true", help="tell the agent which filing to use")
    s.add_argument("--judge", action="store_true", help="also grade with an LLM judge")
    s.add_argument("--judge-provider", choices=["anthropic", "openai"])
    s.add_argument("--judge-model")
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--price-in", type=float, default=0.0, help="USD per million input tokens")
    s.add_argument("--price-out", type=float, default=0.0, help="USD per million output tokens")
    s.add_argument("--price-cache-read", type=float, default=None,
                   help="USD per million cached input tokens read (default 0.1x --price-in)")
    s.add_argument("--price-cache-write", type=float, default=None,
                   help="USD per million input tokens written to the cache (default 1.25x --price-in)")
    s.add_argument("--out", default="reports")
    llm_args(s, required=True)
    s.set_defaults(func=cmd_eval_answers)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
