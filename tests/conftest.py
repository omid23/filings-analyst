"""Synthetic filings that look like real 10-Ks: sections, statement titles and column tables."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from filings_analyst.models import Document

COLUMNS = (330, 410, 490)  # x positions of the three year columns


def _write_page(doc: pymupdf.Document, lines: list[str | tuple]) -> None:
    page = doc.new_page(width=612, height=792)
    y = 60
    for line in lines:
        if isinstance(line, tuple):  # table row: label, value, value, value
            label, *values = line
            page.insert_text((50, y), label, fontsize=9)
            for x, v in zip(COLUMNS, values, strict=False):
                page.insert_text((x, y), v, fontsize=9)
        else:
            page.insert_text((50, y), line, fontsize=9)
        y += 14


def make_10k(path: Path, company: str, year: int, capex: tuple[str, str, str], revenue: tuple[str, str, str],
             cash: tuple[str, str, str], business: str) -> None:
    y0, y1, y2 = str(year), str(year - 1), str(year - 2)
    doc = pymupdf.open()
    _write_page(doc, [f"{company} Annual Report on Form 10-K for fiscal year {year}", "Table of Contents",
                      "Item 1. Business ........ 3", "Item 7. Management's Discussion ........ 20"])
    _write_page(doc, ["ITEM 1. BUSINESS", business,
                      "Our primary customers are industrial distributors in North America and Europe."])
    _write_page(doc, ["ITEM 1A. RISK FACTORS", "Our results are cyclical and depend on manufacturing demand.",
                      "Raw material prices may rise."])
    _write_page(doc, ["ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION",
                      f"Net sales increased in fiscal {year} driven by higher pricing and volume growth in Asia.",
                      "Operating margin expanded because of lower freight costs."])
    _write_page(doc, ["ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA", f"{company}",
                      "Consolidated Statements of Cash Flows", ("(Millions)", y0, y1, y2),
                      ("Net income", "5,363", "4,869", "5,058"),
                      ("Depreciation and amortization", "1,488", "1,544", "1,474"),
                      ("Net cash provided by operating activities", "6,439", "6,240", "6,662"),
                      ("Purchases of property, plant and equipment (PP&E)", *capex),
                      ("Net cash used in investing activities", "(1,403)", "(3,086)", "222")])
    _write_page(doc, [f"{company}", "Consolidated Balance Sheets", ("(Millions)", y0, y1),
                      ("Cash and cash equivalents", *cash[:2]), ("Total current assets", "14,277", "13,709"),
                      ("Total current liabilities", "7,244", "7,687")])
    _write_page(doc, [f"{company}", "Consolidated Statements of Income", ("(Millions)", y0, y1, y2),
                      ("Net sales", *revenue), ("Operating income", "7,207", "7,820", "7,223")])
    doc.save(str(path))
    doc.close()


@pytest.fixture(scope="session")
def corpus(tmp_path_factory) -> list[Document]:
    root = tmp_path_factory.mktemp("filings")
    specs = [
        ("Acme Industrial", "ACME", 2023, ("(1,577)", "(1,373)", "(1,420)"), ("32,765", "31,657", "30,109"),
         ("2,853", "3,053", "2,398"), "Acme makes industrial adhesives, abrasives and safety products."),
        ("Acme Industrial", "ACME", 2022, ("(1,373)", "(1,420)", "(1,300)"), ("31,657", "30,109", "29,500"),
         ("3,053", "2,398", "2,100"), "Acme makes industrial adhesives, abrasives and safety products."),
        ("Globex Corporation", "GBX", 2023, ("(812)", "(790)", "(705)"), ("12,400", "11,950", "11,020"),
         ("1,120", "980", "870"), "Globex sells logistics software and hardware to retailers."),
    ]
    docs = []
    for company, ticker, year, capex, revenue, cash, business in specs:
        doc_id = f"{ticker}_{year}_10K"
        path = root / f"{doc_id}.pdf"
        make_10k(path, company, year, capex, revenue, cash, business)
        docs.append(Document(doc_id, company, "10-K", str(year), str(path), ticker=ticker,
                             meta={"aliases": [ticker]}))
    return docs


@pytest.fixture(scope="session")
def index_dir(corpus, tmp_path_factory) -> Path:
    from filings_analyst.index.indexer import build_index

    root = tmp_path_factory.mktemp("index")
    build_index(corpus, root, embedder="lsa", workers=1)
    return root


@pytest.fixture
def searcher(index_dir):
    from filings_analyst.index.indexer import open_searcher
    from filings_analyst.index.retrieve import SectionRouter

    s = open_searcher(index_dir, rerankers=[SectionRouter()])
    yield s
    s.store.close()
