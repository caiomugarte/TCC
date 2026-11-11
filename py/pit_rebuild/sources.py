"""URLs of the raw sources and the per-date file plan of the rebuild (SRC-01).

The yearly CVM files, the COTAHIST files and the cadastro are planned per date
by plan_files. The B3 company and cash-dividend pages and the Yahoo chart are
built per ticker by the functions below.
"""

import base64
from collections.abc import Iterable
from datetime import date, datetime, timezone
import json
from urllib.parse import quote

CVM_BASE = "https://dados.cvm.gov.br/dados/CIA_ABERTA"
DFP_LISTING = f"{CVM_BASE}/DOC/DFP/DADOS/"
ITR_LISTING = f"{CVM_BASE}/DOC/ITR/DADOS/"
FCA_LISTING = f"{CVM_BASE}/DOC/FCA/DADOS/"
CAD_URL = f"{CVM_BASE}/CAD/DADOS/cad_cia_aberta.csv"
COTAHIST_BASE = "https://bvmf.bmfbovespa.com.br/InstDados/SerHist"
B3_BASE = "https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall"
YAHOO_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"
YAHOO_PERIOD1 = 946_684_800  # 2000-01-01T00:00:00Z
B3_PAGE_SIZE = 120
FCA_FIRST_YEAR = 2010


def dfp_url(year: int) -> str:
    return f"{CVM_BASE}/DOC/DFP/DADOS/dfp_cia_aberta_{year}.zip"


def itr_url(year: int) -> str:
    return f"{CVM_BASE}/DOC/ITR/DADOS/itr_cia_aberta_{year}.zip"


def fca_url(year: int) -> str:
    return f"{CVM_BASE}/DOC/FCA/DADOS/fca_cia_aberta_{year}.zip"


def cotahist_url(year: int) -> str:
    return f"{COTAHIST_BASE}/COTAHIST_A{year}.ZIP"


def b3_company_url(root: str) -> str:
    payload = {"issuingCompany": root, "language": "pt-br"}
    return f"{B3_BASE}/GetListedSupplementCompany/{_b64_json(payload)}"


def b3_query_name(trading_name: str) -> str:
    """The name GetListedCashDividends is queried with: stripped, every "/" removed, inner whitespace collapsed (A-15, F15)."""
    return " ".join(trading_name.replace("/", "").split())


def b3_cash_url(trading_name: str, page: int) -> str:
    payload = {"language": "pt-br", "pageNumber": page, "pageSize": B3_PAGE_SIZE, "tradingName": b3_query_name(trading_name)}
    return f"{B3_BASE}/GetListedCashDividends/{_b64_json(payload)}"


def yahoo_url(ticker: str, period2: date) -> str:
    """Daily dividend events from 2000-01-01 up to period2 (exclusive); no range parameter (A-15)."""
    return (
        f"{YAHOO_BASE}/{ticker}.SA?period1={YAHOO_PERIOD1}"
        f"&period2={_epoch(period2)}&interval=1d&events=div"
    )


def plan_files(dates: Iterable[date]) -> list[tuple[str, str]]:
    """Return (url, cache name) of every file the dates need, deduplicated and sorted.

    DFP covers year(d) - 7 to year(d); ITR year(d) - 2 to year(d); FCA 2010 to
    year(d); COTAHIST year(d) - 1 to year(d); the cadastro once.
    """
    dfp_years: set[int] = set()
    itr_years: set[int] = set()
    fca_years: set[int] = set()
    cotahist_years: set[int] = set()
    for day in dates:
        dfp_years.update(range(day.year - 7, day.year + 1))
        itr_years.update(range(day.year - 2, day.year + 1))
        fca_years.update(range(FCA_FIRST_YEAR, day.year + 1))
        cotahist_years.update(range(day.year - 1, day.year + 1))
    urls = (
        [dfp_url(y) for y in dfp_years]
        + [itr_url(y) for y in itr_years]
        + [fca_url(y) for y in fca_years]
        + [cotahist_url(y) for y in cotahist_years]
        + [CAD_URL]
    )
    return [(url, url.rsplit("/", 1)[1]) for url in sorted(set(urls))]


def _b64_json(payload: dict) -> str:
    text = json.dumps(payload, separators=(",", ":"), sort_keys=True, ensure_ascii=False)
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return quote(encoded, safe="=")


def _epoch(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())
