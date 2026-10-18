"""SQLite store for documents, pages and chunks, with an FTS5 index for BM25."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path

from ..models import Chunk, Document, Page

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  doc_id TEXT PRIMARY KEY, company TEXT NOT NULL, form TEXT NOT NULL, period TEXT NOT NULL,
  source TEXT NOT NULL, ticker TEXT, sector TEXT, meta TEXT
);
CREATE TABLE IF NOT EXISTS pages (
  doc_id TEXT NOT NULL, page_no INTEGER NOT NULL, text TEXT NOT NULL, section TEXT, title TEXT,
  PRIMARY KEY (doc_id, page_no)
);
CREATE TABLE IF NOT EXISTS chunks (
  chunk_id INTEGER PRIMARY KEY, doc_id TEXT NOT NULL, page_no INTEGER NOT NULL,
  header TEXT NOT NULL, text TEXT NOT NULL, section TEXT, title TEXT
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  header, text, content='chunks', content_rowid='chunk_id', tokenize='porter unicode61'
);
"""

STOPWORDS = frozenset("""
a an and are as at be by for from has have how in is it its of on or that the this to was were what when which who
why will with do does did their there these those than then so such any all can could would should may might must
our your his her they them we you i me my give response question details shown based using please provide also
""".split())
_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[&'][A-Za-z0-9]+)*")


def fts_query(text: str, max_terms: int = 40) -> str:
    """Turn free text into a safe FTS5 OR-query of quoted terms."""
    seen: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        tok = tok.replace("'", "")
        if tok in STOPWORDS or (len(tok) < 2 and not tok.isdigit()):
            continue
        if tok not in seen:
            seen.append(tok)
    return " OR ".join(f'"{t}"' for t in seen[:max_terms])


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._lock = threading.RLock()  # one connection shared by web/MCP worker threads

    def _all(self, sql: str, params: Sequence = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(sql, params).fetchall()

    def _one(self, sql: str, params: Sequence = ()) -> sqlite3.Row | None:
        with self._lock:
            return self.conn.execute(sql, params).fetchone()

    # -- writes ---------------------------------------------------------------------------

    def add_document(self, doc: Document, pages: Sequence[Page], chunks: Sequence[Chunk]) -> None:
        with self._lock, self.conn:
            self.delete_document(doc.doc_id, commit=False)
            self.conn.execute(
                "INSERT INTO documents VALUES (?,?,?,?,?,?,?,?)",
                (doc.doc_id, doc.company, doc.form, doc.period, doc.source, doc.ticker, doc.sector,
                 json.dumps(doc.meta)),
            )
            self.conn.executemany(
                "INSERT INTO pages VALUES (?,?,?,?,?)",
                [(p.doc_id, p.page_no, p.text, p.section, p.title) for p in pages],
            )
            self.conn.executemany(
                "INSERT INTO chunks VALUES (?,?,?,?,?,?,?)",
                [(c.chunk_id, c.doc_id, c.page_no, c.header, c.text, c.section, c.title) for c in chunks],
            )
            self.conn.executemany(
                "INSERT INTO chunks_fts(rowid, header, text) VALUES (?,?,?)",
                [(c.chunk_id, c.header, c.text) for c in chunks],
            )

    def delete_document(self, doc_id: str, commit: bool = True) -> None:
        with self._lock:
            rows = self.conn.execute("SELECT chunk_id, header, text FROM chunks WHERE doc_id=?", (doc_id,)).fetchall()
            self.conn.executemany(
                "INSERT INTO chunks_fts(chunks_fts, rowid, header, text) VALUES ('delete', ?, ?, ?)",
                [(r["chunk_id"], r["header"], r["text"]) for r in rows],
            )
            for table in ("chunks", "pages", "documents"):
                self.conn.execute(f"DELETE FROM {table} WHERE doc_id=?", (doc_id,))
            if commit:
                self.conn.commit()

    def next_chunk_id(self) -> int:
        row = self._one("SELECT COALESCE(MAX(chunk_id), -1) + 1 FROM chunks")
        return int(row[0])

    # -- reads ----------------------------------------------------------------------------

    @staticmethod
    def _doc(row: sqlite3.Row) -> Document:
        return Document(row["doc_id"], row["company"], row["form"], row["period"], row["source"],
                        row["ticker"], row["sector"], json.loads(row["meta"] or "{}"))

    def documents(self) -> list[Document]:
        return [self._doc(r) for r in self._all("SELECT * FROM documents ORDER BY doc_id")]

    def document(self, doc_id: str) -> Document | None:
        row = self._one("SELECT * FROM documents WHERE doc_id=?", (doc_id,))
        return self._doc(row) if row else None

    def page(self, doc_id: str, page_no: int) -> Page | None:
        row = self._one("SELECT * FROM pages WHERE doc_id=? AND page_no=?", (doc_id, page_no))
        return Page(row["doc_id"], row["page_no"], row["text"], row["section"], row["title"]) if row else None

    def page_count(self, doc_id: str) -> int:
        return int(self._one("SELECT COUNT(*) FROM pages WHERE doc_id=?", (doc_id,))[0])

    @staticmethod
    def _chunk(row: sqlite3.Row) -> Chunk:
        return Chunk(row["chunk_id"], row["doc_id"], row["page_no"], row["header"], row["text"], row["section"],
                     row["title"])

    def chunks(self, ids: Iterable[int]) -> dict[int, Chunk]:
        ids = list(ids)
        out: dict[int, Chunk] = {}
        for i in range(0, len(ids), 900):
            part = ids[i:i + 900]
            q = f"SELECT * FROM chunks WHERE chunk_id IN ({','.join('?' * len(part))})"
            for row in self._all(q, part):
                out[row["chunk_id"]] = self._chunk(row)
        return out

    def all_chunk_rows(self) -> list[tuple[int, str, str]]:
        return [(r[0], r[1], r[2]) for r in self._all("SELECT chunk_id, header, text FROM chunks ORDER BY chunk_id")]

    def chunk_doc_ids(self) -> dict[int, str]:
        return {r[0]: r[1] for r in self._all("SELECT chunk_id, doc_id FROM chunks")}

    def bm25(self, query: str, k: int = 50, doc_ids: Sequence[str] | None = None,
             weights: tuple[float, float] = (2.0, 1.0)) -> list[tuple[int, float]]:
        match = fts_query(query)
        if not match:
            return []
        sql = (f"SELECT c.chunk_id, bm25(chunks_fts, {weights[0]}, {weights[1]}) AS s "
               "FROM chunks_fts JOIN chunks c ON c.chunk_id = chunks_fts.rowid WHERE chunks_fts MATCH ?")
        params: list = [match]
        if doc_ids:
            sql += f" AND c.doc_id IN ({','.join('?' * len(doc_ids))})"
            params.extend(doc_ids)
        sql += " ORDER BY s LIMIT ?"
        params.append(k)
        # SQLite's bm25() is lower-is-better; negate so higher is better everywhere.
        return [(r[0], -float(r[1])) for r in self._all(sql, params)]

    def stats(self) -> dict[str, int]:
        return {t: int(self._one(f"SELECT COUNT(*) FROM {t}")[0]) for t in ("documents", "pages", "chunks")}

    def close(self) -> None:
        self.conn.close()
