from __future__ import annotations

import numpy as np
import pytest

from filings_analyst.index.embed import LSAEmbedder
from filings_analyst.index.resolver import FilingResolver
from filings_analyst.index.retrieve import SectionRouter, rrf
from filings_analyst.index.store import Store, fts_query
from filings_analyst.ingest.chunking import chunk_page, make_header
from filings_analyst.ingest.html import extract_html_pages, html_to_text
from filings_analyst.ingest.pdf import detect_item, detect_title, extract_pages
from filings_analyst.models import Document, Page


def test_pdf_tables_become_rows(corpus):
    pages = extract_pages(corpus[0].source, corpus[0].doc_id)
    cash = pages[4]
    assert cash.title == "Consolidated Statements of Cash Flows"
    assert cash.section.startswith("Item 8")
    assert "Purchases of property, plant and equipment (PP&E) (1,577) (1,373) (1,420)" in cash.text.split("\n")
    assert pages[0].section is None  # the table of contents line is not a section heading
    assert pages[3].section.startswith("Item 7")


@pytest.mark.parametrize("line,expected", [
    ("CONSOLIDATED STATEMENTS OF OPERATIONS", "Consolidated Statements of Operations"),
    ("Consolidated Balance Sheet", "Consolidated Balance Sheet"),
    ("Consolidated Statement of Shareholders' Equity", "Consolidated Statement of Shareholders' Equity"),
    ("Statements of Comprehensive Income (Loss)", "Statements of Comprehensive Income"),
    ("Revenue by segment", None),
])
def test_detect_title(line, expected):
    assert detect_title(["Acme Corp", line]) == expected


def test_detect_item_skips_toc():
    assert detect_item(["Item 7. Management's Discussion .......... 22"]) is None
    assert detect_item(["ITEM 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURES"]) == \
        "Item 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURES"


def test_html_tables_and_pages():
    html = """<html><head><style>p{}</style></head><body>
      <p>ITEM 8. FINANCIAL STATEMENTS</p><p>Consolidated Balance Sheets</p>
      <table><tr><td>Cash</td><td></td><td>2,853</td><td>3,053</td></tr>
             <tr><td>Inventories</td><td>4,366</td><td>4,034</td></tr></table>
      <div style="page-break-before: always"></div><p>ITEM 9. CHANGES IN AND DISAGREEMENTS</p>
      <script>alert(1)</script></body></html>"""
    text = html_to_text(html)
    assert "Cash | 2,853 | 3,053" in text
    assert "alert" not in text
    pages = extract_html_pages(html, "X")
    assert len(pages) == 2
    assert pages[0].title == "Consolidated Balance Sheets" and pages[1].section.startswith("Item 9")


def test_chunks_stay_on_page_and_carry_context():
    doc = Document("ACME_2023_10K", "Acme", "10-K", "2023", "x.pdf")
    page = Page("ACME_2023_10K", 4, "\n".join(f"Row {i} label value {i * 10}" for i in range(200)),
                section="Item 8. Financial Statements", title="Consolidated Balance Sheets")
    chunks = chunk_page(doc, page, start_id=100, max_words=60, overlap=12)
    assert len(chunks) > 5
    assert all(c.page_no == 4 and c.header == make_header(doc, page) for c in chunks)
    assert "Acme | 10-K | FY2023 | Item 8" in chunks[0].header and "page 5" in chunks[0].header
    assert [c.chunk_id for c in chunks] == list(range(100, 100 + len(chunks)))
    # overlap: the next chunk starts with lines from the end of the previous one
    assert chunks[1].text.split("\n")[0] in chunks[0].text.split("\n")[-4:]
    assert "Row 199" in chunks[-1].text


@pytest.mark.parametrize("text", ['AT&T\'s "Q2" (FY23) capex?', "NEAR(x) OR * :", "", "the of and"])
def test_fts_query_is_always_safe(tmp_path, text):
    store = Store(tmp_path / "i.sqlite")
    q = fts_query(text)
    assert all(part.startswith('"') for part in q.split(" OR ") if part)
    assert store.bm25(text) == []  # empty store, but no syntax error


def test_bm25_filters_by_filing(searcher):
    all_hits = searcher.store.bm25("purchases of property plant equipment", 20)
    acme = searcher.store.bm25("purchases of property plant equipment", 20, ["ACME_2023_10K"])
    docs = {searcher.store.chunks([c])[c].doc_id for c, _ in acme}
    assert docs == {"ACME_2023_10K"}
    assert len(all_hits) > len(acme)


def test_reindex_replaces_document(index_dir, corpus, tmp_path):
    from filings_analyst.index.indexer import build_index
    from filings_analyst.ingest.chunking import chunk_document

    store = Store(tmp_path / "copy.sqlite")
    doc = corpus[2]
    pages = extract_pages(doc.source, doc.doc_id)
    store.add_document(doc, pages, chunk_document(doc, pages))
    before = store.stats()
    store.add_document(doc, pages, chunk_document(doc, pages))
    assert store.stats() == before
    assert store.bm25("logistics software", 5)
    store.delete_document(doc.doc_id)
    assert store.stats()["chunks"] == 0 and store.bm25("logistics software", 5) == []
    assert build_index  # imported for coverage of the module


def test_resolver(corpus):
    docs = corpus + [Document("ACME_2023Q2_10Q", "Acme Industrial", "10-Q", "2023Q2", "x", ticker="ACME")]
    r = FilingResolver(docs)
    assert r.resolve("What was ACME's FY22 capex?").doc_ids[0] == "ACME_2022_10K"
    assert r.resolve("acme industrial capex in 2023").doc_ids[0] == "ACME_2023_10K"
    assert r.resolve("Acme cash at the end of Q2 2023").doc_ids[0] == "ACME_2023Q2_10Q"
    assert r.resolve("How cyclical is Acme?").doc_ids[0] in ("ACME_2023_10K", "ACME_2023Q2_10Q")  # most recent first
    assert r.resolve("Globex revenue FY2023").doc_ids == ["GBX_2023_10K"]
    nothing = r.resolve("What is a good capex ratio?")
    assert nothing.doc_ids == [] and "no company" in nothing.reason


def test_rrf_math():
    fused = rrf({"a": [1, 2, 3], "b": [3, 1]}, k=60)
    ids = [c for c, _, _ in fused]
    assert ids[0] == 1 and set(ids) == {1, 2, 3}
    assert fused[0][2] == {"a": 1, "b": 2}
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)


def test_search_finds_statement_page(searcher):
    res = searcher.search("What was Acme's FY2023 capital expenditure? Use the cash flow statement.", k=5)
    assert res.resolution.doc_ids[0] == "ACME_2023_10K"
    assert res.pages(3)[0] == ("ACME_2023_10K", 4)
    assert all(h.chunk.doc_id in res.resolution.doc_ids for h in res.hits)


def test_router_promotes_balance_sheet(searcher):
    q = "Acme FY2023 quick ratio liquidity"
    searcher.rerankers = []
    plain = searcher.search(q, k=5)
    searcher.rerankers = [SectionRouter()]
    routed = searcher.search(q, k=5)
    assert routed.pages(1)[0] == ("ACME_2023_10K", 5)
    assert [r.name for r in SectionRouter().active(q)] == ["metric:balance"]
    assert plain.hits  # the unrouted search still returns results


def test_modes_and_dense_roundtrip(searcher, index_dir):
    for mode in ("bm25", "dense", "hybrid"):
        assert searcher.search("Globex logistics software retailers", k=3, mode=mode).pages(1)[0][0] == "GBX_2023_10K"
    loaded = LSAEmbedder.load(index_dir / "dense")
    v1 = searcher.dense.embedder.embed_query("net sales")
    v2 = loaded.embed_query("net sales")
    assert np.allclose(v1, v2) and abs(np.linalg.norm(v1) - 1) < 1e-5
