"""SEC EDGAR client: find a company's filings, download them, and read XBRL facts.

EDGAR requires a descriptive User-Agent with contact details and allows at
most 10 requests per second; both are enforced here.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from ..models import Document

SEC = "https://www.sec.gov"
DATA = "https://data.sec.gov"
FORMS = {"10-K": "10-K", "10-Q": "10-Q", "8-K": "8-K", "20-F": "20-F", "40-F": "40-F"}


class EdgarError(Exception):
    pass


@dataclass
class Filing:
    cik: int
    company: str
    ticker: str | None
    form: str
    accession: str
    filing_date: str
    report_date: str
    primary_document: str
    fiscal_year: int | None = None
    fiscal_period: str | None = None  # FY, Q1, Q2, Q3

    @property
    def url(self) -> str:
        return f"{SEC}/Archives/edgar/data/{self.cik}/{self.accession.replace('-', '')}/{self.primary_document}"

    @property
    def doc_id(self) -> str:
        base = (self.ticker or str(self.cik)).upper()
        period = f"{self.fiscal_year}" if self.fiscal_period in (None, "FY") else f"{self.fiscal_year}{self.fiscal_period}"
        return f"{base}_{period}_{self.form.replace('-', '')}_{self.accession}"


class EdgarClient:
    def __init__(self, user_agent: str | None = None, client: httpx.Client | None = None, max_rps: float = 8.0):
        ua = user_agent or os.environ.get("SEC_USER_AGENT")
        if not ua or "@" not in ua:
            raise EdgarError("EDGAR requires a User-Agent with contact email, e.g. "
                             "SEC_USER_AGENT='Your Name your@email.com'")
        self._client = client or httpx.Client(timeout=60, follow_redirects=True)
        self._headers = {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}
        self._min_interval = 1.0 / max_rps
        self._last = 0.0
        self._lock = threading.Lock()
        self._tickers: dict[str, tuple[int, str]] | None = None

    def _get(self, url: str) -> httpx.Response:
        with self._lock:
            wait = self._min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
        resp = self._client.get(url, headers=self._headers)
        if resp.status_code != 200:
            raise EdgarError(f"GET {url} -> HTTP {resp.status_code}")
        return resp

    def ticker_map(self) -> dict[str, tuple[int, str]]:
        if self._tickers is None:
            data = self._get(f"{SEC}/files/company_tickers.json").json()
            self._tickers = {row["ticker"].upper(): (int(row["cik_str"]), row["title"]) for row in data.values()}
        return self._tickers

    def lookup(self, ticker: str) -> tuple[int, str]:
        try:
            return self.ticker_map()[ticker.upper()]
        except KeyError as exc:
            raise EdgarError(f"unknown ticker {ticker}") from exc

    def filings(self, ticker: str, forms: tuple[str, ...] = ("10-K", "10-Q"), limit: int = 8) -> list[Filing]:
        cik, title = self.lookup(ticker)
        data = self._get(f"{DATA}/submissions/CIK{cik:010d}.json").json()
        recent = data["filings"]["recent"]
        out: list[Filing] = []
        for i, form in enumerate(recent["form"]):
            if form not in forms:
                continue
            report = recent["reportDate"][i] or recent["filingDate"][i]
            out.append(Filing(
                cik=cik, company=data.get("name") or title, ticker=ticker.upper(), form=form,
                accession=recent["accessionNumber"][i], filing_date=recent["filingDate"][i], report_date=report,
                primary_document=recent["primaryDocument"][i],
                fiscal_year=int(report[:4]) if report else None,
                fiscal_period="FY" if form in ("10-K", "20-F", "40-F") else _quarter(report),
            ))
            if len(out) >= limit:
                break
        return out

    def download(self, filing: Filing, directory: str | Path) -> Document:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        suffix = Path(filing.primary_document).suffix or ".htm"
        path = directory / f"{filing.doc_id}{suffix}"
        if not path.exists():
            path.write_bytes(self._get(filing.url).content)
        period = str(filing.fiscal_year) if filing.fiscal_period == "FY" else f"{filing.fiscal_year}{filing.fiscal_period}"
        return Document(doc_id=filing.doc_id, company=_clean_company(filing.company), form=filing.form, period=period,
                        source=str(path), ticker=filing.ticker,
                        meta={"url": filing.url, "filed": filing.filing_date, "report_date": filing.report_date,
                              "aliases": [filing.ticker] if filing.ticker else []})

    def company_facts(self, ticker: str) -> dict:
        cik, _ = self.lookup(ticker)
        return self._get(f"{DATA}/api/xbrl/companyfacts/CIK{cik:010d}.json").json()


def _quarter(report_date: str | None) -> str | None:
    if not report_date:
        return None
    month = int(report_date[5:7])
    return f"Q{(month - 1) // 3 + 1}"


def _clean_company(name: str) -> str:
    return " ".join(w.capitalize() if w.isupper() and len(w) > 3 else w for w in name.replace("/DE/", "").split())


def xbrl_fact(facts: dict, concept: str, fiscal_year: int, form: str = "10-K", unit: str = "USD") -> float | None:
    """Value of a us-gaap concept for the fiscal year ending in ``fiscal_year``, from annual filings.

    Each 10-K reports several years, so rows are selected by period end date, not by the
    filing's ``fy`` field. Duration facts must span roughly one year; instant facts (balance
    sheet items) have no start date. The most recently filed value wins, which picks up
    restatements.
    """
    from datetime import date

    try:
        rows = facts["facts"]["us-gaap"][concept]["units"][unit]
    except KeyError:
        return None
    picked = []
    for r in rows:
        if r.get("form") != form or not str(r.get("end", "")).startswith(str(fiscal_year)):
            continue
        if r.get("start"):
            days = (date.fromisoformat(r["end"]) - date.fromisoformat(r["start"])).days
            if not 350 <= days <= 380:
                continue
        picked.append(r)
    if not picked:
        return None
    return float(max(picked, key=lambda r: r.get("filed", ""))["val"])
