"""Build an index directory from filings: store + optional dense vectors."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from ..ingest.chunking import chunk_document
from ..ingest.html import extract_html_pages
from ..ingest.pdf import extract_pages
from ..models import Document, Page
from .embed import make_embedder
from .retrieve import DenseIndex, Searcher
from .store import Store

log = logging.getLogger("filings_analyst")


@dataclass
class IndexPaths:
    root: Path

    @property
    def db(self) -> Path:
        return self.root / "index.sqlite"

    @property
    def dense(self) -> Path:
        return self.root / "dense"


def _extract(doc: Document) -> tuple[Document, list[Page]]:
    src = Path(doc.source)
    if src.suffix.lower() == ".pdf":
        return doc, extract_pages(src, doc.doc_id)
    if src.suffix.lower() in (".htm", ".html"):
        return doc, extract_html_pages(src.read_text(encoding="utf-8", errors="replace"), doc.doc_id)
    raise ValueError(f"unsupported source type: {src}")


def build_index(documents: Iterable[Document], root: str | Path, *, embedder: str | None = "lsa",
                workers: int = 2, skip_existing: bool = True) -> dict:
    paths = IndexPaths(Path(root))
    store = Store(paths.db)
    existing = {d.doc_id for d in store.documents()} if skip_existing else set()
    todo = [d for d in documents if d.doc_id not in existing]
    started = time.perf_counter()
    next_id = store.next_chunk_id()
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        for i, (doc, pages) in enumerate(pool.map(_extract, todo, chunksize=1), start=1):
            chunks = chunk_document(doc, pages, start_id=next_id)
            next_id += len(chunks)
            store.add_document(doc, pages, chunks)
            if i % 25 == 0 or i == len(todo):
                log.info("indexed %d/%d filings", i, len(todo))
    stats = store.stats()
    stats["extract_s"] = round(time.perf_counter() - started, 1)
    if embedder:
        t = time.perf_counter()
        DenseIndex.build(store, make_embedder(embedder), paths.dense)
        stats["dense_s"] = round(time.perf_counter() - t, 1)
        stats["embedder"] = embedder
    store.close()
    return stats


def open_searcher(root: str | Path, rerankers: Iterable = ()) -> Searcher:
    paths = IndexPaths(Path(root))
    if not paths.db.exists():
        raise FileNotFoundError(f"no index at {paths.root}; run `filings-analyst index` first")
    store = Store(paths.db)
    dense = DenseIndex.load(store, paths.dense) if (paths.dense / "dense.json").exists() else None
    return Searcher(store, dense, list(rerankers))
