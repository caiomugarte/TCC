"""Command line of the PIT rebuild (CLI-01..CLI-05): read the raw sources in the cache, compare, write the report.

Run from the repository root:

    python3 py/pit_rebuild report --date 2026-08-07 --si data/raw/status_invest_fundamentals.csv \\
        --date 2026-11-05 --si snapshots/data/selectors-2026-11-06/stocks.csv --offline

    the committed run takes the collection dates and a filings cutoff as well (A-11, A-22):

    python3 py/pit_rebuild report --date 2026-08-07 --si data/raw/status_invest_fundamentals.csv \\
        --date 2026-11-05 --si snapshots/data/selectors-2026-11-06/stocks.csv \\
        --collected-on 2026-08-07 --collected-on 2026-11-06 --filings-cutoff 2026-11-05:2026-11-02 --offline

``report`` rebuilds every date from the cache and writes pit-rebuild-stocks.json and
pit-rebuild-stocks.md to --out-dir, both only after every computation finished. Without
--offline a missing file is downloaded first. Exit codes:

    0  report written
    1  the report files could not be written
    2  invalid arguments, or an output or cache directory inside data/, snapshots/ or the baselines tree
    3  a given path does not exist (a --si CSV, or the cache directory of report)
    4  integrity violation: cached file hash mismatch, PIT leak, Status Invest CSV missing a
       required column or repeating a ticker
    5  a required download failed, or a required file is absent from the cache in --offline mode
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from . import compare, coverage, cotahist, cvm, dividends, mapping, report, safety, sources, status_invest
from . import shares as share_counts
from .cache import Cache
from .errors import ArgumentError, MissingPathError, OutputError, RebuildError, SourceError
from .fields import ADJUSTED_FIELDS, ALL_FIELDS
from .indicators import STALE_FILING_REASON, Input, TickerInputs, Value, compute, mask_accounting

DEFAULT_CACHE_DIR = safety.REPO_ROOT / ".cache" / "pit-rebuild"
DEFAULT_OUT_DIR = safety.REPO_ROOT / "docs" / "validation"
REPORT_JSON = "pit-rebuild-stocks.json"
REPORT_MD = "pit-rebuild-stocks.md"
PAUSE_SECONDS = 0.5  # pacing between requests to the public servers
PROBE_YEARS = range(2005, 2013)  # COV-01: COTAHIST probe, 2005 to 2012
KINDS = ("DFP", "ITR", "FCA", "CADASTRO", "COTAHIST")
LISTINGS = {"DFP": sources.DFP_LISTING, "ITR": sources.ITR_LISTING, "FCA": sources.FCA_LISTING}
GROUP_KEYS = ("non_financial", "non_financial_without_unclassified", "financial")
# The statement accounts the rebuild reads (design: Indicator formulas); other rows are not loaded.
STATEMENT_ACCOUNTS = frozenset(
    {"1", "1.01", "1.01.01", "1.01.02", "2.01", "2.01.04", "2.02.01", "2.03", "2.03.09", "3.01", "3.05", "3.08", "3.11", "3.11.01"}
)
SOURCES = ("b3", "yahoo", "none")
YAHOO_REASONS = ("b3_failed", "b3_empty", "b3_incomplete", "b3_short_history")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_TICKER = re.compile(r"^[A-Z]{3,4}\d{1,2}$")  # A-9 (r5): a 3- or 4-letter key, as in the FCA string "1 ENG3 e 4 ENGI4"
_NO_FILINGS = "no_usable_filing"
_NO_QUOTE = "no_quote"


# --- names of the cached files -------------------------------------------------------------


def listing_name(source: str) -> str:
    return f"listing_{source.lower()}.html"


def b3_company_name(root: str) -> str:
    return f"b3_company_{root}.json"


def b3_cash_name(trading_name: str, page: int) -> str:
    return f"b3_cash_{_slug(trading_name)}_p{page}.json"


def yahoo_name(ticker: str, period2: date) -> str:
    return f"yahoo_{ticker}_{period2.isoformat()}.json"


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text.strip()).strip("_") or "x"


def _kind_of_url(url: str) -> str:
    if url == sources.CAD_URL:
        return "CADASTRO"
    if url.startswith(f"{sources.CVM_BASE}/DOC/DFP/"):
        return "DFP"
    if url.startswith(f"{sources.CVM_BASE}/DOC/ITR/"):
        return "ITR"
    if url.startswith(f"{sources.CVM_BASE}/DOC/FCA/"):
        return "FCA"
    if url.startswith(sources.COTAHIST_BASE):
        return "COTAHIST"
    raise ValueError(f"not a yearly source file: {url}")


def _full_rebuild_urls(years: dict[str, tuple[int, int]]) -> list[str]:
    """Every yearly file of the full rebuild (COV-03) plus the cadastro, for the Content-Length sizes."""
    builders: dict[str, Callable[[int], str]] = {
        "DFP": sources.dfp_url,
        "ITR": sources.itr_url,
        "FCA": sources.fca_url,
        "COTAHIST": sources.cotahist_url,
    }
    urls = [builders[source](year) for source, (first, last) in years.items() for year in range(first, last + 1)]
    return sorted(set(urls)) + [sources.CAD_URL]


# --- arguments ------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Request:
    dates: tuple[date, ...]  # in the order given, paired with si_paths by position
    si_paths: tuple[Path, ...]
    cache_dir: Path
    out_dir: Path | None
    collected_on: tuple[date, ...] = ()  # the collection date of each CSV, paired by position (A-11, F4)
    filings_cutoffs: dict[date, date] = field(default_factory=dict)  # --filings-cutoff DATE:CUTOFF (A-22 (b))


def _dates(values: Sequence[str]) -> tuple[date, ...]:
    days: list[date] = []
    for text in values:
        if not _DATE.fullmatch(text):
            raise ArgumentError("--date", f"{text!r} is not YYYY-MM-DD")
        try:
            day = date.fromisoformat(text)
        except ValueError as exc:
            raise ArgumentError("--date", f"{text!r} is not a calendar date") from exc
        if day in days:
            raise ArgumentError("--date", f"{text} is given more than once")
        days.append(day)
    return tuple(days)


def _one_date(argument: str, text: str) -> date:
    if not _DATE.fullmatch(text):
        raise ArgumentError(argument, f"{text!r} is not YYYY-MM-DD")
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ArgumentError(argument, f"{text!r} is not a calendar date") from exc


def _collected(values: Sequence[str], dates: tuple[date, ...]) -> tuple[date, ...]:
    """The collection date of each CSV: zero --collected-on (each collected on its --date) or one per --date (CLI-03)."""
    if values and len(values) != len(dates):
        raise ArgumentError("--collected-on", f"{len(values)} given for {len(dates)} --date (zero, or one per --date)")
    collected = tuple(_one_date("--collected-on", text) for text in values) or dates
    for day, when in zip(dates, collected):
        if when < day:
            raise ArgumentError("--collected-on", f"{when} is before its --date {day}")
    return collected


def _filings_cutoffs(values: Sequence[str], dates: tuple[date, ...]) -> dict[date, date]:
    """--filings-cutoff DATE:CUTOFF, each DATE a --date given once, CUTOFF not after DATE (A-22 (b), CLI-03)."""
    cutoffs: dict[date, date] = {}
    for text in values:
        if ":" not in text:
            raise ArgumentError("--filings-cutoff", f"{text!r} is not DATE:CUTOFF")
        day_text, cutoff_text = text.split(":", 1)
        day = _one_date("--filings-cutoff", day_text)
        cutoff = _one_date("--filings-cutoff", cutoff_text)
        if day not in dates:
            raise ArgumentError("--filings-cutoff", f"{day} is not given with --date")
        if day in cutoffs:
            raise ArgumentError("--filings-cutoff", f"{day} is given more than once")
        if cutoff > day:
            raise ArgumentError("--filings-cutoff", f"cutoff {cutoff} is after its date {day}")
        cutoffs[day] = cutoff
    return cutoffs


def _request(args: argparse.Namespace, *, needs_out: bool) -> _Request:
    dates = _dates(args.date)
    collected = _collected(getattr(args, "collected_on", None) or [], dates)
    cutoffs = _filings_cutoffs(getattr(args, "filings_cutoff", None) or [], dates)
    si_paths = tuple(Path(item) for item in args.si)
    if len(si_paths) != len(dates):
        raise ArgumentError("--si", f"{len(si_paths)} given for {len(dates)} --date (one --si per --date)")
    cache_dir = safety.assert_writable_dir(Path(args.cache_dir), "--cache-dir")
    out_dir = safety.assert_writable_dir(Path(args.out_dir), "--out-dir") if needs_out else None
    for path in si_paths:
        if not path.is_file():
            raise MissingPathError("--si", f"{path} does not exist")
    if needs_out and not cache_dir.is_dir():
        raise MissingPathError("--cache-dir", f"{cache_dir} does not exist")
    return _Request(
        dates=dates, si_paths=si_paths, cache_dir=cache_dir, out_dir=out_dir,
        collected_on=collected, filings_cutoffs=cutoffs,
    )


# --- acquisition: the raw files every date needs, in the order of design.md -------------------


@dataclass
class _Acquired:
    si: dict[date, status_invest.SICsv]
    eligible: dict[date, list[str]]
    b3: dict[str, dividends.B3Company | None]
    fca: mapping.FcaIndex
    cadastro: dict
    dfp: cvm.FilingStore
    itr: cvm.FilingStore
    quotes: cotahist.Quotes
    mapped: dict[date, dict[str, mapping.Mapped | mapping.Unmapped]]
    files: dict[str, list[Path]]
    timings: dict[str, tuple[float, int]]  # kind -> (seconds, bytes parsed)
    histories: dict[str, dividends.B3History | None] = field(default_factory=dict)
    yahoo: dict[str, list[dividends.Dividend] | None] = field(default_factory=dict)
    dividend_seconds: float = 0.0
    dividend_files: int = 0


def _acquire(
    cache: Cache, dates: Sequence[date], si_paths: Sequence[Path], clock: Callable[[], float], collected_on: Sequence[date]
) -> _Acquired:
    files: dict[str, list[Path]] = {kind: [] for kind in KINDS}
    for url, name in sources.plan_files(dates):
        files[_kind_of_url(url)].append(cache.get(url, name, required=True))
    si = {day: status_invest.load(path) for day, path in zip(dates, si_paths)}
    eligible = {day: status_invest.eligible(si[day]) for day in dates}
    roots = sorted({ticker[:4] for day in dates for ticker in eligible[day]})
    b3 = {root: _b3_company(cache, root) for root in roots}
    timings: dict[str, tuple[float, int]] = {}
    fca = _timed(clock, timings, "FCA", files["FCA"], lambda: mapping.load_fca(files["FCA"]))
    cadastro = _timed(clock, timings, "CADASTRO", files["CADASTRO"], lambda: cvm.load_cadastro(files["CADASTRO"][0]))
    dfp = _timed(clock, timings, "DFP", files["DFP"], lambda: cvm.load_filings(files["DFP"], cvm.DFP, STATEMENT_ACCOUNTS))
    itr = _timed(clock, timings, "ITR", files["ITR"], lambda: cvm.load_filings(files["ITR"], cvm.ITR, STATEMENT_ACCOUNTS))
    mapped = {day: mapping.map_tickers(eligible[day], fca, cadastro, b3, day) for day in dates}
    si_tickers = {ticker for day in dates for ticker in si[day].rows}
    wanted: set[str] = set()
    for day in dates:
        wanted |= mapping.quote_tickers(eligible[day], mapped[day], fca, si_tickers)
    quotes = _timed(clock, timings, "COTAHIST", files["COTAHIST"], lambda: cotahist.load(files["COTAHIST"], wanted))
    acq = _Acquired(
        si=si, eligible=eligible, b3=b3, fca=fca, cadastro=cadastro, dfp=dfp, itr=itr,
        quotes=quotes, mapped=mapped, files=files, timings=timings,
    )
    needed = [(day, ticker) for day in dates for ticker in eligible[day] if _unmapped_reason(acq, day, ticker) is None]
    names = sorted({b3[ticker[:4]].trading_name for _day, ticker in needed if b3.get(ticker[:4])})
    for name in names:
        acq.histories[name] = _b3_history(cache, name, acq, clock)
    need_yahoo = sorted({ticker for day, ticker in needed if _choice(acq, day, ticker)[0] == "yahoo"})
    period2 = max(collected_on) + timedelta(days=1)  # Yahoo runs up to the last collection date (A-11)
    for ticker in need_yahoo:
        acq.yahoo[ticker] = _yahoo(cache, ticker, period2, acq, clock)
    return acq


def _timed(clock, timings: dict, kind: str, paths: Sequence[Path], load: Callable[[], object]):
    start = clock()
    value = load()
    timings[kind] = (clock() - start, sum(path.stat().st_size for path in paths))
    return value


SAO_PAULO = timezone(timedelta(hours=-3))  # B-3: the retrieval day R is read in São Paulo time (UTC−3, no daylight saving)


def _retrieved_on(retrieved_at: str | None) -> date | None:
    """The São Paulo day of a manifest retrieved_at, a UTC moment written "%Y-%m-%dT%H:%M:%SZ" (B-3, SHR-08)."""
    if retrieved_at is None:
        return None
    moment = datetime.strptime(retrieved_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return moment.astimezone(SAO_PAULO).date()


def _b3_company(cache: Cache, root: str) -> dividends.B3Company | None:
    name = b3_company_name(root)
    path = cache.get(sources.b3_company_url(root), name, required=False)
    if path is None:
        return None
    try:
        company = dividends.parse_b3_company(path.read_bytes())
    except (ValueError, KeyError, TypeError, AttributeError, IndexError):
        return None
    if company is None:
        return None
    return replace(company, retrieved_on=_retrieved_on(cache.retrieved_at(name)))


def _b3_history(cache: Cache, name: str, acq: _Acquired, clock) -> dividends.B3History | None:
    """Every page of GetListedCashDividends under the query name (A-15, F15). Page 1 missing is b3_failed; a missing later page is incomplete."""
    query = sources.b3_query_name(name)
    pages: list[bytes] = []
    total = 1
    page = 1
    while page <= total:
        path = cache.get(sources.b3_cash_url(query, page), b3_cash_name(query, page), required=False)
        if path is None:
            break
        pages.append(path.read_bytes())
        acq.dividend_files += 1
        if page == 1:
            try:
                total = int(json.loads(pages[0])["page"]["totalPages"])
            except (ValueError, KeyError, TypeError):
                return None
        page += 1
    if not pages:
        return None
    start = clock()
    try:
        history = dividends.parse_b3_cash(pages)
    except (ValueError, KeyError, TypeError):
        history = None
    acq.dividend_seconds += clock() - start
    return history


def _yahoo(cache: Cache, ticker: str, period2: date, acq: _Acquired, clock) -> list[dividends.Dividend] | None:
    path = cache.get(sources.yahoo_url(ticker, period2), yahoo_name(ticker, period2), required=False)
    if path is None:
        return None
    acq.dividend_files += 1
    start = clock()
    try:
        result = dividends.parse_yahoo(path.read_bytes())
    except (ValueError, KeyError, TypeError, IndexError):
        result = None
    acq.dividend_seconds += clock() - start
    return result


def _has_filings(acq: _Acquired, cnpj: str, day: date) -> bool:
    return bool(cvm.usable(acq.dfp, cnpj, day) or cvm.usable(acq.itr, cnpj, day))


def _unmapped_reason(acq: _Acquired, day: date, ticker: str, filings_day: date | None = None) -> str | None:
    """Why an eligible ticker is not computed at day (MAP-01), or None when it is. CVM filings are read at filings_day (A-22 (b))."""
    item = acq.mapped[day][ticker]
    if isinstance(item, mapping.Unmapped):
        return item.reason
    if not _has_filings(acq, item.cnpj, filings_day or day):
        return _NO_FILINGS
    if cotahist.quote_at(acq.quotes, ticker, day) is None:
        return _NO_QUOTE
    return None


def _history_of(acq: _Acquired, ticker: str) -> dividends.B3History | None:
    company = acq.b3.get(ticker[:4])
    return None if company is None else acq.histories.get(company.trading_name)


def _choice(acq: _Acquired, day: date, ticker: str) -> tuple[str, str | None]:
    """The dividend source of a ticker at day. A "/" name whose complete history is empty is b3_failed, not b3_empty (F15, A-15)."""
    company = acq.b3.get(ticker[:4])
    history = _history_of(acq, ticker)
    if company is not None and "/" in company.trading_name and history is not None and history.complete and history.total_records == 0:
        return "yahoo", "b3_failed"
    return dividends.choose_source(history, day)


# --- one date: companies, tickers, dividends, statistics ------------------------------------


@dataclass
class _Company:
    cnpj: str
    cd_cvm: str
    n_on: int | None  # issued ON shares, rounded after the A-18 factor
    n_pref: int | None
    scaled: dict | None  # {cd_cvm, factor, residual} for a verified factor other than 1 (A-18)
    unverified: bool  # no B3 count: the counts are used as filed (A-18)
    price_on: Input | None
    price_pref: Input | None
    fallback: dict | None  # vm_price_fallback: a class with shares and no session on or before d (A-17)
    stale: list[dict]  # vm_stale_class_price: a class priced from a session older than 30 days (F14)
    inputs: dict
    dre_desc: str | None
    share_events: list[dict]  # share_events.adjusted: the classes whose B3 events moved their count (AD-033, SHR-03)
    no_isin: list[dict]  # share_events.no_isin: the classes with shares and no ISIN at d (SHR-04)
    suspect: dict | None  # share_count_suspect: a jump of the count with no B3 event between two filings (SHR-05)
    period_end: date | None  # DT_REFER of the newest usable filing, read at filings_day (B-6, STL-01)
    ttm_prior: tuple[str, ...]  # the flows whose prior year-to-date is last year's ITR at d (TTM-04)
    newest: cvm.Filing | None  # the newest usable filing at filings_day: the filing a worst row of the adjusted verdict cites (ADJ-04)


def _dated(item) -> Input | None:
    return None if item is None else Input(item.value, item.dt_receb)


def _sum_inputs(items: list[Input | None]) -> Input | None:
    if any(item is None for item in items):
        return None
    return Input(sum(item.value for item in items), max(item.when for item in items))


def _controllers_equity(pl: Input | None, minority: Input | None) -> Input | None:
    """PL_ctrl = 2.03 − (2.03.09 or 0), None when 2.03 is absent (A-21, F8)."""
    if pl is None:
        return None
    if minority is None:
        return pl
    return Input(pl.value - minority.value, max(pl.when, minority.when))


def _company_inputs(acq: _Acquired, cnpj: str, day: date) -> dict:
    dfp, itr = acq.dfp, acq.itr

    def flow(name: str) -> Input | None:
        return _dated(cvm.ttm(dfp, itr, cnpj, day, name))

    def balance(name: str) -> Input | None:
        return _dated(cvm.balance(dfp, itr, cnpj, day, name))

    return {
        "ll": flow(cvm.NET_INCOME),
        "rec": flow("3.01"),
        "ebit": flow("3.05"),
        "ir": flow("3.08"),
        "ll_1y": _dated(cvm.ttm_year_before(dfp, itr, cnpj, day, cvm.NET_INCOME)),
        "pl": balance("2.03"),
        "pl_ctrl": _controllers_equity(balance("2.03"), balance("2.03.09")),
        "at": balance("1"),
        "ac": balance("1.01"),
        "pc": balance("2.01"),
        "debt": _sum_inputs([balance("2.01.04"), balance("2.02.01")]),
        "cash": _sum_inputs([balance("1.01.01"), balance("1.01.02")]),
        "rec_fy": _dated(cvm.annual(dfp, cnpj, day, "3.01", 0)),
        "rec_fy5": _dated(cvm.annual(dfp, cnpj, day, "3.01", 5)),
        "ll_fy5": _dated(cvm.annual(dfp, cnpj, day, cvm.NET_INCOME, 5)),
    }


def _quote_candidates(acq: _Acquired, cnpj: str, tickers: Sequence[str], day: date) -> set[str]:
    """The tickers whose quotes and ISINs a company may use: its FCA tickers, its eligible tickers and same-root Status Invest tickers (A-17)."""
    roots = {ticker[:4] for ticker in tickers}
    return set(acq.fca.tickers_of(cnpj)) | set(tickers) | {t for t in acq.si[day].rows if t[:4] in roots}


@dataclass(frozen=True)
class _Counts:
    """The share counts of one company at the decision date and the rows of the share rules that moved them (AD-033)."""

    n_on: int | None
    n_pref: int | None
    scaled: dict | None  # {cd_cvm, factor, residual} for a verified factor other than 1 (A-18)
    unverified: bool  # no B3 count: the counts are used as filed (A-18)
    share_events: list[dict]  # share_events.adjusted (SHR-03)
    no_isin: list[dict]  # share_events.no_isin (SHR-04)
    suspect: dict | None  # share_count_suspect (SHR-05)


def _class_isin_sets(quotes: cotahist.Quotes, ticker: str | None, dt_refer: date, day: date) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """I_d and I_all of a share class from its A-17 ticker (B-2, B-3): its ISINs from dt_refer to day, and those it had after day too."""
    if ticker is None:
        return (), ()
    at_day = cotahist.class_isins(quotes, ticker, dt_refer, day)
    return at_day, tuple(dict.fromkeys((*at_day, *cotahist.isins(quotes, ticker, day, None))))


def _counts(
    acq: _Acquired, day: date, cnpj: str, cd_cvm: str, tickers: list[str], filings_day: date,
    shares: cvm.Shares | None, prices: dict[str, tuple[str, float, date] | None],
) -> _Counts:
    """The filed counts times the A-18 factor and the B3 events of each class up to day (shares.adjust), and the suspect check.

    The B3 total and its retrieval day come from the first company record with a total; the events come from every record
    of the company's tickers (AD-033). The previous filing is read at filings_day (B-4).
    """
    if shares is None or shares.on + shares.pref <= 0:
        return _Counts(None, None, None, False, [], [], None)
    b3_total, retrieved_on = None, None
    for ticker in sorted(tickers):
        company = acq.b3.get(ticker[:4])
        if company is not None and company.total_shares:
            b3_total, retrieved_on = company.total_shares, company.retrieved_on
            break
    records = [acq.b3[root] for root in sorted({ticker[:4] for ticker in tickers}) if acq.b3.get(root) is not None]
    events = tuple(event for company in records for event in company.stock_events)

    def next_session(value: date) -> date | None:
        return cotahist.next_session(acq.quotes, value)

    isin_sets = {
        cls: _class_isin_sets(acq.quotes, None if prices[cls] is None else prices[cls][0], shares.dt_refer, day)
        for cls in ("ON", "PREF")
    }
    isins_d = {cls: sets[0] for cls, sets in isin_sets.items()}
    isins_all = {cls: sets[1] for cls, sets in isin_sets.items()}
    adjusted = share_counts.adjust(shares, isins_d, isins_all, events, b3_total, retrieved_on, day, next_session)
    share_events = [
        {
            "cd_cvm": cd_cvm,
            "class": row["class"],
            "isins": row["isins"],
            "dt_refer": shares.dt_refer.isoformat(),
            "factor": row["factor"],
            "events": [
                {"isin": event["isin"], "label": event["label"], "ex_date": event["ex_date"].isoformat(), "multiplier": event["multiplier"]}
                for event in row["events"]
            ],
        }
        for row in adjusted.applied
    ]
    no_isin = [{"cd_cvm": cd_cvm, "class": cls} for cls in adjusted.no_isin]
    suspect = None
    previous = cvm.previous_shares(acq.dfp, acq.itr, cnpj, filings_day)
    if previous is not None:
        previous_factor = share_counts.adjust(previous, isins_d, isins_all, events, b3_total, retrieved_on, day, next_session).factor
        interval = tuple(dict.fromkeys(
            isin
            for cls in ("ON", "PREF") if prices[cls] is not None
            for isin in cotahist.class_isins(acq.quotes, prices[cls][0], previous.dt_refer, day)
        ))
        row = share_counts.suspect(previous, previous_factor, shares, adjusted.factor, events, interval, next_session)
        if row is not None:
            suspect = {
                "cd_cvm": cd_cvm,
                **row,
                "dt_refer_prev": row["dt_refer_prev"].isoformat(),
                "dt_refer": row["dt_refer"].isoformat(),
            }
    if not adjusted.verified:
        return _Counts(adjusted.n_on, adjusted.n_pref, None, True, share_events, no_isin, suspect)
    scaled = {"cd_cvm": cd_cvm, "factor": adjusted.factor, "residual": adjusted.residual} if adjusted.factor != 1 else None
    return _Counts(adjusted.n_on, adjusted.n_pref, scaled, False, share_events, no_isin, suspect)


TTM_FLOWS = (cvm.NET_INCOME, "3.01", "3.05", "3.08")  # the flows that the indicators read as 12-month values (IND-01)


def _flows_from_itr(acq: _Acquired, cnpj: str, day: date) -> tuple[str, ...]:
    """The flows whose prior year-to-date at day is last year's ITR, because the comparative of this year's ITR is absent or 0 (TTM-01, TTM-04)."""
    used: list[str] = []
    for account in TTM_FLOWS:
        flow = cvm.ttm(acq.dfp, acq.itr, cnpj, day, account)
        if flow is not None and flow.prior_from_itr:
            used.append(account)
    return tuple(used)


def _is_stale(period_end: date | None, day: date) -> bool:
    """Whether the newest usable filing ends more than STALE_FILING_DAYS before day, so its accounting fields have no value (STL-01, STL-04, B-6)."""
    return period_end is not None and (day - period_end).days > cvm.STALE_FILING_DAYS


def _without_newest(acq: _Acquired, cnpj: str, filings_day: date) -> _Acquired | None:
    """The reads of acq with the newest usable period of cnpj taken out (B-9), or None when no older usable period is left."""
    newest = cvm.newest_period_end(acq.dfp, acq.itr, cnpj, filings_day)
    if newest is None:
        return None
    older = replace(acq, dfp=acq.dfp.without(cnpj, newest), itr=acq.itr.without(cnpj, newest))
    return older if cvm.newest_period_end(older.dfp, older.itr, cnpj, filings_day) is not None else None


def _company(acq: _Acquired, day: date, cnpj: str, cd_cvm: str, tickers: list[str], filings_day: date) -> _Company:
    """Share counts with the A-18 scale and the B3 events of their classes (AD-033), the A-17 class prices (F14) and the class fallback of one company.

    The CVM reads use filings_day; the quotes, the events and the B3 record use day (A-22 (b)).
    """
    shares = cvm.shares(acq.dfp, acq.itr, cnpj, filings_day)
    prices = cotahist.class_prices(acq.quotes, _quote_candidates(acq, cnpj, tickers, day), day)
    quoted = {cls: None if item is None else Input(item[1], item[2]) for cls, item in prices.items()}
    stale = [
        {"cd_cvm": cd_cvm, "class": cls, "ticker": item[0], "session": item[2].isoformat()}
        for cls, item in sorted(prices.items())
        if item is not None and (day - item[2]).days > cotahist.MAX_QUOTE_AGE_DAYS
    ]
    counts = _counts(acq, day, cnpj, cd_cvm, tickers, filings_day, shares, prices)
    fallback = None
    for cls, other, count in (("ON", "PREF", counts.n_on), ("PREF", "ON", counts.n_pref)):
        if count and count > 0 and prices[cls] is None and prices[other] is not None:
            fallback = {"cd_cvm": cd_cvm, "class_without_quote": cls, "price_from": prices[other][0]}
            break
    return _Company(
        cnpj=cnpj, cd_cvm=cd_cvm, n_on=counts.n_on, n_pref=counts.n_pref, scaled=counts.scaled, unverified=counts.unverified,
        price_on=quoted["ON"], price_pref=quoted["PREF"], fallback=fallback, stale=stale,
        inputs=_company_inputs(acq, cnpj, filings_day), dre_desc=cvm.dre_description(acq.dfp, acq.itr, cnpj, filings_day),
        share_events=counts.share_events, no_isin=counts.no_isin, suspect=counts.suspect,
        period_end=cvm.newest_period_end(acq.dfp, acq.itr, cnpj, filings_day),
        ttm_prior=_flows_from_itr(acq, cnpj, filings_day),
        newest=cvm.newest_filing(acq.dfp, acq.itr, cnpj, filings_day),
    )


def _known_at(dividend_list: list[dividends.Dividend], day: date) -> list[dividends.Dividend]:
    """Dividends whose ex-date is known at day. Later records of the full history are not used (PIT-02)."""
    return [item for item in dividend_list if item.ex_date is None or item.ex_date <= day]


def _class_isin(quotes: cotahist.Quotes, share_class: str, tickers: Iterable[str], d: date) -> str | None:
    """The ISIN of a share class at d: that of the smallest ticker of the company in the class which has one (A-20, ISN-03)."""
    for ticker in sorted(tickers):
        if cotahist.share_class(quotes, ticker, d) == share_class:
            isin = cotahist.isin(quotes, ticker, d)
            if isin is not None:
                return isin
    return None


def _b3_dividends(
    acq: _Acquired, ticker: str, item: mapping.Mapped, history: dividends.B3History, day: date, candidates: set[str]
) -> tuple[list | None, list, int]:
    """(dividends at day, the B3 records the ticker draws on, classes with records and no ISIN) (A-15, A-20).

    The records are the unit's UNT records when it has them, else the records of its composition classes.
    Each record is divided by the split multipliers of its ISIN's company events (A-20). A class without an ISIN
    keeps its amounts as they are, and is counted.
    """
    q = acq.quotes
    company = acq.b3.get(ticker[:4])
    events = company.stock_events if company is not None else ()

    def next_session(value: date) -> date | None:
        return cotahist.next_session(q, value)

    if item.composition is None:
        cls = cotahist.share_class(q, ticker, day)
        if cls is None:
            return None, [], 0
        records = dividends.class_records(history.records, cls)
        isin = cotahist.isin(q, ticker, day)
        missing = 1 if records and isin is None else 0
        return dividends.b3_ex_dates(records, next_session, events=events, isin=isin, d=day), records, missing
    isins: dict[str, str | None] = {"UNT": cotahist.isin(q, ticker, day)}
    composition: dict[str, int] = {}
    for key, count in item.composition.items():
        if _TICKER.match(key):
            cls, isin = cotahist.share_class(q, key, day), cotahist.isin(q, key, day)
        else:
            cls, isin = key, _class_isin(q, key, candidates, day)
        if cls is None:
            return None, [], 0
        composition[cls] = composition.get(cls, 0) + count
        isins.setdefault(cls, isin)
    units = dividends.class_records(history.records, "UNT")
    if units:
        records = units
        missing = 1 if isins["UNT"] is None else 0
    else:
        records = [record for cls in composition for record in dividends.class_records(history.records, cls)]
        missing = sum(1 for cls in composition if dividends.class_records(history.records, cls) and isins.get(cls) is None)
    amounts = dividends.unit_dividends(history.records, composition, next_session, events=events, isins=isins, d=day)
    return amounts, records, missing


@dataclass(frozen=True)
class _Source:
    source: str  # "b3", "yahoo" or "none"
    reason: str | None
    amounts: list | None  # dividends known at d; None when the ticker has no source
    raw_date: date | None  # the earliest raw date of the source the ticker uses (COV-02, F3)
    missing_isin: int  # B3 classes with records and no ISIN, so their split adjustment was skipped


def _dividends_of(acq: _Acquired, day: date, ticker: str, item: mapping.Mapped, candidates: set[str]) -> _Source:
    """The source, reason and dividends of one ticker at day: B3 first, Yahoo for the reasons of A-15 (SRC-05)."""
    choice = _choice(acq, day, ticker)
    if choice[0] == "b3":
        amounts, records, missing = _b3_dividends(acq, ticker, item, _history_of(acq, ticker), day, candidates)
        if amounts is None:
            return _Source("none", "no_dividend_source", None, None, 0)
        return _Source("b3", None, _known_at(amounts, day), dividends.b3_earliest_raw(records), missing)
    yahoo = acq.yahoo.get(ticker)
    source, reason, amounts = dividends.resolve(choice, None, None if yahoo is None else _known_at(yahoo, day))
    if source != "yahoo":
        return _Source(source, reason, amounts, None, 0)  # a confirmed non-payer has no record to date
    raw = [event.ex_date for event in (yahoo or []) if event.ex_date is not None]
    return _Source("yahoo", reason, amounts, min(raw, default=None), 0)


@dataclass
class _DayResult:
    eligible: int
    mapping: dict
    pairs: dict[str, dict[str, list]]
    dividends: dict
    dy_dates: dict[str, dict[str, date | None]]  # per dividend source, each ticker's earliest date (COV-02)
    previous: dict[str, dict[str, float | None]] = field(default_factory=dict)  # B-9: each ticker's values on its previous filing
    filings: dict[str, cvm.Filing | None] = field(default_factory=dict)  # each computed ticker's newest usable filing (ADJ-04)


def _evaluate_day(acq: _Acquired, day: date, filings_day: date | None = None, *, previous: bool = False) -> _DayResult:
    """One decision date. Every CVM read uses filings_day (A-22 (b)); quotes, dividends and eligibility use day.

    previous=True also reads each company on its previous filing (B-9); those values feed the si_lag marks only.
    """
    fday = filings_day or day
    eligible = acq.eligible[day]
    csv = acq.si[day]
    mapped = acq.mapped[day]
    members: dict[str, list[str]] = defaultdict(list)
    for ticker in eligible:
        item = mapped[ticker]
        if isinstance(item, mapping.Mapped) and _unmapped_reason(acq, day, ticker, fday) is None:
            members[item.cnpj].append(ticker)
    companies: dict[str, _Company] = {}
    for cnpj, tickers in members.items():
        companies[cnpj] = _company(acq, day, cnpj, mapped[tickers[0]].cd_cvm, tickers, fday)
    older_companies: dict[str, _Company] = {}
    if previous:
        for cnpj, tickers in members.items():
            older = _without_newest(acq, cnpj, fday)
            if older is not None:
                older_companies[cnpj] = _company(older, day, cnpj, companies[cnpj].cd_cvm, tickers, fday)
    candidates_of = {cnpj: _quote_candidates(acq, cnpj, tickers, day) for cnpj, tickers in members.items()}
    out = {
        "mapped": 0, "unmapped": [], "refiled_after_d": [], "mapped_via_b3": [],
        "vm_price_fallback": {"count": 0, "companies": []},
        "vm_stale_class_price": [],
        "share_scale": {"scaled": [], "unverified": []},
        "share_events": {"adjusted": [], "no_isin": []},
        "share_count_suspect": [],
        "stale_filing": [],
        "ttm_prior_from_itr": [],
        "reference_stale": sorted(ticker for ticker in eligible if cotahist.reference_stale(acq.quotes, ticker, day)),
        "financial": [], "unclassified": [],
    }
    for cnpj in sorted(companies):
        company = companies[cnpj]
        if company.fallback is not None:
            out["vm_price_fallback"]["companies"].append(company.fallback)
        out["vm_stale_class_price"].extend(company.stale)
        if company.scaled is not None:
            out["share_scale"]["scaled"].append(company.scaled)
        if company.unverified:
            out["share_scale"]["unverified"].append(company.cd_cvm)
        out["share_events"]["adjusted"].extend(company.share_events)
        out["share_events"]["no_isin"].extend(company.no_isin)
        if company.suspect is not None:
            out["share_count_suspect"].append(company.suspect)
        if company.ttm_prior:
            out["ttm_prior_from_itr"].append({
                "cd_cvm": company.cd_cvm, "dt_refer": company.period_end.isoformat(), "accounts": sorted(company.ttm_prior),
            })
    out["vm_price_fallback"]["companies"].sort(key=lambda row: row["cd_cvm"])
    out["vm_price_fallback"]["count"] = len(out["vm_price_fallback"]["companies"])
    out["vm_stale_class_price"].sort(key=lambda row: (row["cd_cvm"], row["class"]))
    out["share_scale"]["scaled"].sort(key=lambda row: row["cd_cvm"])
    out["share_scale"]["unverified"].sort()
    out["share_events"]["adjusted"].sort(key=lambda row: (row["cd_cvm"], row["class"]))
    out["share_events"]["no_isin"].sort(key=lambda row: (row["cd_cvm"], row["class"]))
    out["share_count_suspect"].sort(key=lambda row: row["cd_cvm"])
    out["ttm_prior_from_itr"].sort(key=lambda row: row["cd_cvm"])

    values: dict[str, dict[str, Value]] = {}
    group_of: dict[str, str] = {}
    previous_values: dict[str, dict[str, float | None]] = {}
    filings: dict[str, cvm.Filing | None] = {}
    counts = {
        "by_source": {name: 0 for name in SOURCES}, "yahoo_reasons": {r: 0 for r in YAHOO_REASONS},
        "b3_empty_yahoo_failed": 0, "split_adjust_no_isin": 0,
    }
    dy_dates: dict[str, dict[str, date | None]] = defaultdict(dict)
    for ticker in eligible:
        item = mapped[ticker]
        reason = _unmapped_reason(acq, day, ticker, fday)
        dre = cvm.dre_description(acq.dfp, acq.itr, item.cnpj, fday) if isinstance(item, mapping.Mapped) else None
        classified = mapping.classify(item, acq.cadastro, dre, csv.setor.get(ticker))
        group_of[ticker] = classified["group"]
        if classified["group"] == "financial":
            out["financial"].append({"ticker": ticker, "rule": classified["rule"]})
        if reason is not None:
            out["unmapped"].append({"ticker": ticker, "reason": reason})
            if classified["group"] == "unclassified":
                out["unclassified"].append({"ticker": ticker, "si_setor": csv.setor.get(ticker, ""), "reason": reason})
            values[ticker] = {name: Value(None, reason) for name in ALL_FIELDS}
            continue
        out["mapped"] += 1
        if item.via == "b3":
            out["mapped_via_b3"].append(ticker)
        if cvm.refiled_after_d(acq.dfp, acq.itr, item.cnpj, fday):
            out["refiled_after_d"].append(ticker)
        company = companies[item.cnpj]
        found = _dividends_of(acq, day, ticker, item, candidates_of[item.cnpj])
        counts["by_source"][found.source] += 1
        choice = _choice(acq, day, ticker)
        if choice[0] == "yahoo":  # every switch to Yahoo counts with its reason, whether Yahoo answered or not
            counts["yahoo_reasons"][choice[1]] += 1
        if found.reason == "b3_empty_yahoo_failed":
            counts["b3_empty_yahoo_failed"] += 1
        counts["split_adjust_no_isin"] += found.missing_isin
        if found.source in ("b3", "yahoo") and found.amounts is not None:
            dy_dates[found.source][ticker] = found.raw_date  # COV-02: the raw date, neither cut by d nor mapped to sessions
        close_day, close = cotahist.quote_at(acq.quotes, ticker, day)
        last_day = cotahist.last_trade(acq.quotes, ticker, day)[0]
        composition = item.composition
        inputs = TickerInputs(
            ticker=ticker,
            unit_divisor=sum(composition.values()) if composition else 1,
            price=Input(close, close_day),
            price_on=company.price_on,
            price_pref=company.price_pref,
            n_on=company.n_on,
            n_pref=company.n_pref,
            liquidity=Input(cotahist.avg_volume(acq.quotes, ticker, day), last_day),
            dividends=found.amounts,
            **company.inputs,
        )
        values[ticker] = compute(inputs, day)
        if _is_stale(company.period_end, day):  # STL-01: a stale filing gives its accounting fields no value
            values[ticker] = mask_accounting(values[ticker], STALE_FILING_REASON)
            out["stale_filing"].append({
                "ticker": ticker, "cd_cvm": company.cd_cvm, "dt_refer": company.period_end.isoformat(),
                "age_days": (day - company.period_end).days,
            })
        filings[ticker] = company.newest
        older = older_companies.get(item.cnpj)
        if older is not None:  # B-9: the same ticker read on the previous filing, with the same quotes and dividends
            before = compute(replace(inputs, n_on=older.n_on, n_pref=older.n_pref, **older.inputs), day)
            if _is_stale(older.period_end, day):
                before = mask_accounting(before, STALE_FILING_REASON)
            previous_values[ticker] = {name: before[name].value for name in ALL_FIELDS}
    out["stale_filing"].sort(key=lambda row: row["ticker"])
    pairs: dict[str, dict[str, list]] = {name: {group: [] for group in GROUP_KEYS} for name in ALL_FIELDS}
    for ticker in eligible:
        group = group_of[ticker]
        for name in ALL_FIELDS:
            value = values[ticker][name]
            row = (ticker, csv.rows[ticker][name], value.value, value.reason or "no_value")
            if group in ("non_financial", "unclassified"):
                pairs[name]["non_financial"].append(row)
            if group == "non_financial":
                pairs[name]["non_financial_without_unclassified"].append(row)
            if group == "financial":
                pairs[name]["financial"].append(row)
    return _DayResult(
        eligible=len(eligible),
        mapping=out,
        pairs=pairs,
        dividends=counts,
        dy_dates=dy_dates,
        previous=previous_values,
        filings=filings,
    )


# --- report object --------------------------------------------------------------------------


def _close_on(quotes: cotahist.Quotes, ticker: str, day: date) -> float | None:
    """The close of the ticker's record on exactly this session, or None (A-11)."""
    dates = quotes.dates.get(ticker, [])
    index = bisect_left(dates, day)
    if index < len(dates) and dates[index] == day:
        return quotes.rows[ticker][index].close
    return None


def _same_cent(close: float | None, price: float) -> bool:
    return close is not None and round(close, 2) == round(price, 2)


def _as_of_check(acq: _Acquired, day: date, collected_on: date) -> dict:
    """Which session the CSV's PRECO values belong to (A-11, F4). Evidence only: it never chooses d.

    The sessions run from the one before day to the last one on or before collected_on. matches counts the eligible
    tickers whose Status Invest PRECO equals that session's close to the cent; best_session has the most matches,
    the later session on a tie.
    """
    sessions = acq.quotes.sessions
    before = [session for session in sessions if session < day]
    start = before[-1] if before else day
    window = [session for session in sessions if start <= session <= collected_on]
    prices: dict[str, float] = {}
    for ticker in acq.eligible[day]:
        price = acq.si[day].rows[ticker]["PRECO"]
        if price is not None:
            prices[ticker] = price
    matches = {
        session.isoformat(): sum(1 for ticker, price in prices.items() if _same_cent(_close_on(acq.quotes, ticker, session), price))
        for session in window
    }
    best = max(window, key=lambda session: (matches[session.isoformat()], session), default=None)
    return {
        "eligible_with_price": len(prices),
        "matches": matches,
        "best_session": None if best is None else best.isoformat(),
    }


def _sensitivities(acq: _Acquired, req: _Request, results: dict[date, _DayResult]) -> dict:
    """The non-official sensitivities (A-22), computed after the timed official evaluations (A-13).

    D-A: each date's official statistics without the reference-stale tickers, from the same pairs (no second
    evaluation). D-B: one more evaluation per --filings-cutoff date with the filings received by its cutoff.
    """
    stale_excluded: dict[str, list[str]] = {}
    stale_fields: dict[str, dict] = {}
    for day in sorted(req.dates):
        iso = day.isoformat()
        stale = set(results[day].mapping["reference_stale"])
        stale_excluded[iso] = sorted(stale)
        stale_fields[iso] = {
            name: {
                group: compare.summarize(name, [row for row in results[day].pairs[name][group] if row[0] not in stale])
                for group in GROUP_KEYS
            }
            for name in ALL_FIELDS
        }
    cutoff_blocks: dict[str, dict] = {}
    for day in sorted(req.filings_cutoffs):
        cutoff = req.filings_cutoffs[day]
        result = _evaluate_day(acq, day, filings_day=cutoff)
        cutoff_blocks[day.isoformat()] = {
            "cutoff": cutoff,
            "fields": {name: {group: compare.summarize(name, result.pairs[name][group]) for group in GROUP_KEYS} for name in ALL_FIELDS},
        }
    return {
        "without_reference_stale": {"excluded": stale_excluded, "fields": stale_fields},
        "filings_cutoff": cutoff_blocks,
    }


NO_FILING_FIELDS = ("PRECO", "LIQUIDEZ MEDIA DIARIA", "DY")  # B-9: these fields read no filing, so none of them can be si_lag


def _adjusted_input(result: _DayResult, excluded: Sequence[str], stats: dict[str, compare.FieldStats]) -> dict:
    """One date of the adjusted verdict for report.build (ADJ-02, ADJ-04, LAG-01): the stale-free statistics of the non-financial
    group, the si_lag tickers over the same rows, and for each worst row its filing, previous value and si_lag mark.

    A si_lag ticker is in the field's denominator (its raw value is a reference and its rebuilt value is outside tolerance).
    The marks read the previous values of the official evaluation and change no statistic (LAG-04).
    """
    left_out = set(excluded)
    marks: dict[str, dict[str, bool]] = {}
    for name in ADJUSTED_FIELDS:
        marks[name] = {}
        if name in NO_FILING_FIELDS:
            continue
        for ticker, raw, calc, _reason in result.pairs[name]["non_financial"]:
            if ticker not in left_out:
                marks[name][ticker] = compare.is_si_lag(name, raw, calc, result.previous.get(ticker, {}).get(name))
    worst_details: dict[str, dict[str, dict]] = {}
    for name in ADJUSTED_FIELDS:
        worst_details[name] = {}
        for item in stats[name].worst:
            ticker = item["ticker"]
            filing = result.filings.get(ticker)
            cited = None if filing is None or name in NO_FILING_FIELDS else {
                "kind": filing.kind, "dt_refer": filing.dt_refer.isoformat(), "versao": filing.versao,
                "dt_receb": filing.dt_receb.isoformat(),
            }
            worst_details[name][ticker] = {
                "filing": cited,
                "previous_value": result.previous.get(ticker, {}).get(name),
                "si_lag": marks[name].get(ticker, False),
            }
    return {
        "excluded": list(excluded),
        "fields": stats,
        "si_lag": {name: sorted(ticker for ticker, mark in marks[name].items() if mark) for name in ADJUSTED_FIELDS},
        "worst_details": worst_details,
    }


def _report_object(cache: Cache, acq: _Acquired, req: _Request, clock: Callable[[], float], tool_commit: str | None) -> dict:
    compute_seconds = 0.0
    results: dict[date, _DayResult] = {}
    per_date: dict[str, dict] = {}
    dividends_out: dict[str, dict] = {"by_source": {}, "yahoo_reasons": {}, "b3_empty_yahoo_failed": {}, "split_adjust_no_isin": {}}
    dy_dates: dict[str, dict[str, date | None]] = defaultdict(dict)
    for day in sorted(req.dates):
        start = clock()
        result = _evaluate_day(acq, day, previous=True)
        fields_out = {
            name: {group: compare.summarize(name, result.pairs[name][group]) for group in GROUP_KEYS}
            for name in ALL_FIELDS
        }
        compute_seconds += clock() - start
        results[day] = result
        iso = day.isoformat()
        per_date[iso] = {"eligible": result.eligible, "mapping": result.mapping, "fields": fields_out}
        dividends_out["by_source"][iso] = dict(result.dividends["by_source"])
        dividends_out["yahoo_reasons"][iso] = dict(result.dividends["yahoo_reasons"])
        dividends_out["b3_empty_yahoo_failed"][iso] = result.dividends["b3_empty_yahoo_failed"]
        dividends_out["split_adjust_no_isin"][iso] = result.dividends["split_adjust_no_isin"]
        for source, found in result.dy_dates.items():
            for ticker, first in found.items():
                known = [day for day in (dy_dates[source].get(ticker), first) if day is not None]
                dy_dates[source][ticker] = min(known) if known else None
    dy = coverage.dy_coverage({source: dict(dy_dates.get(source, {})) for source in ("b3", "yahoo")})
    coverage_out = _coverage(cache, acq, req, compute_seconds, dy)
    histories = [history for history in acq.histories.values() if history is not None]
    dividends_out["b3_filters"] = {
        "raw_records": sum(history.raw_records for history in histories),
        "duplicates": sum(history.duplicates for history in histories),
        "capital_returns": sum(history.capital_returns for history in histories),
    }
    dividends_out["slash_names"] = sorted(
        (
            {
                "trading_name": name,
                "queried_as": sources.b3_query_name(name),
                "total_records": None if history is None else history.total_records,
            }
            for name, history in acq.histories.items()
            if "/" in name
        ),
        key=lambda row: row["trading_name"],
    )
    raw = list(cache.used)
    collected = dict(zip(req.dates, req.collected_on))
    sensitivities = _sensitivities(acq, req, results)
    stale_free = sensitivities["without_reference_stale"]
    adjusted = {
        day.isoformat(): _adjusted_input(
            results[day],
            stale_free["excluded"][day.isoformat()],
            {name: stale_free["fields"][day.isoformat()][name]["non_financial"] for name in ADJUSTED_FIELDS},
        )
        for day in req.dates
    }
    return report.build(
        dates=list(req.dates),
        tool_commit=tool_commit,
        status_invest=[
            {
                "date": day.isoformat(),
                "collected_on": collected[day].isoformat(),
                "path": str(path),
                "sha256": acq.si[day].sha256,
                "as_of_check": _as_of_check(acq, day, collected[day]),
            }
            for day, path in zip(req.dates, req.si_paths)
        ],
        raw_files=raw,
        per_date=per_date,
        dividends=dividends_out,
        coverage=coverage_out,
        filings_cutoffs=req.filings_cutoffs,
        sensitivities=sensitivities,
        adjusted=adjusted,
    )


def _survey(cache: Cache) -> tuple[dict, dict]:
    """The CVM listings and the COTAHIST probes: each source's earliest year and the full-rebuild years (COV-01)."""
    listings = {
        source: cache.get(url, listing_name(source), required=True).read_bytes() for source, url in LISTINGS.items()
    }
    probes = {year: cache.probe(sources.cotahist_url(year)) for year in PROBE_YEARS}
    earliest = coverage.earliest_years(listings, probes)
    years = coverage.full_rebuild_years({source: info["year"] for source, info in earliest.items()})
    return earliest, years


def _sizes(cache: Cache, years: dict) -> dict[str, int]:
    """The Content-Length of every yearly file of the full rebuild and of the cadastro (COV-03)."""
    sizes: dict[str, int] = {}
    for url in _full_rebuild_urls(years):
        size = cache.head_size(url)
        if size is None:
            raise SourceError(url, "Content-Length is not known; run fetch while online first")
        sizes[url] = size
    return sizes


def _coverage(cache: Cache, acq: _Acquired, req: _Request, compute_seconds: float, dy: dict) -> dict:
    """Earliest years, the field years, the DY coverage and the effort of the full rebuild (COV-01..COV-03)."""
    earliest, years = _survey(cache)
    sizes = _sizes(cache, years)
    effort = coverage.effort(
        {kind: acq.timings[kind] for kind in KINDS},
        compute_seconds,
        len(req.dates),
        sizes,
        _kind_of_url,
    )
    effort["dividend_files"] = acq.dividend_files
    effort["dividends_parse_seconds"] = acq.dividend_seconds
    known = {source: info["year"] for source, info in earliest.items()}
    by_field = {
        name: dy["earliest_year"] if name == "DY" else coverage.field_earliest(name, known)
        for name in ALL_FIELDS
    }
    return {
        "earliest_year_by_source": earliest,
        "earliest_year_by_field": by_field,
        "dy": dy,
        "effort": effort,
        "years": years,
    }


def _tool_commit() -> str | None:
    try:
        done = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=safety.REPO_ROOT, capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = done.stdout.strip()
    return value if done.returncode == 0 and value else None


# --- commands and entry point ---------------------------------------------------------------


def _cache(cache_dir: Path, *, offline: bool, opener) -> Cache:
    if opener is None:
        return Cache(cache_dir, offline=offline, pause_seconds=PAUSE_SECONDS)
    return Cache(cache_dir, offline=offline, opener=opener)


def _command_report(args: argparse.Namespace, *, opener, clock) -> int:
    req = _request(args, needs_out=True)
    cache = _cache(req.cache_dir, offline=args.offline, opener=opener)
    cache.verify()
    acq = _acquire(cache, req.dates, req.si_paths, clock, req.collected_on)
    obj = _report_object(cache, acq, req, clock, _tool_commit())
    _write_report(req.out_dir, obj)
    return 0


def _command_fetch(args: argparse.Namespace, *, opener, clock) -> int:
    """Download every file the rebuild of the dates needs, in the order of design.md, and record the survey (COV-01, SRC-01)."""
    req = _request(args, needs_out=False)
    cache = _cache(req.cache_dir, offline=False, opener=opener)
    cache.verify()
    _acquire(cache, req.dates, req.si_paths, clock, req.collected_on)
    _, years = _survey(cache)
    _sizes(cache, years)
    return 0


def _write_report(out_dir: Path, obj: dict) -> None:
    payload = {
        out_dir / REPORT_JSON: report.encode_json(obj),
        out_dir / REPORT_MD: report.render_markdown(obj).encode("utf-8"),
    }
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        safety.write_files_atomically(payload)
    except OSError as exc:
        raise OutputError(str(out_dir), f"report files could not be written: {exc}") from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pit_rebuild", description="PIT rebuild of the stock indicators (issue #59).")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    report_parser = commands.add_parser("report", help="rebuild the dates from the cache and write the report")
    report_parser.add_argument("--date", action="append", required=True, metavar="YYYY-MM-DD", help="decision date (repeatable)")
    report_parser.add_argument("--si", action="append", required=True, metavar="PATH", help="Status Invest CSV of the date at the same position")
    report_parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="download cache (default .cache/pit-rebuild)")
    report_parser.add_argument("--offline", action="store_true", help="read only the cache: no network request")
    report_parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="where the report files are written")
    report_parser.add_argument(
        "--collected-on", action="append", metavar="YYYY-MM-DD",
        help="collection date of the CSV at the same position (default: its --date; repeatable)",
    )
    report_parser.add_argument(
        "--filings-cutoff", action="append", metavar="DATE:CUTOFF",
        help="a non-official sensitivity: DATE with the filings received by CUTOFF (repeatable)",
    )
    report_parser.set_defaults(handler=_command_report)
    fetch_parser = commands.add_parser("fetch", help="download the raw sources into the cache, without a report")
    fetch_parser.add_argument("--date", action="append", required=True, metavar="YYYY-MM-DD", help="decision date (repeatable)")
    fetch_parser.add_argument("--si", action="append", required=True, metavar="PATH", help="Status Invest CSV of the date at the same position")
    fetch_parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="download cache (default .cache/pit-rebuild)")
    fetch_parser.add_argument(
        "--collected-on", action="append", metavar="YYYY-MM-DD",
        help="collection date of the CSV at the same position (default: its --date; repeatable)",
    )
    fetch_parser.set_defaults(handler=_command_fetch)
    return parser


def main(argv: Sequence[str] | None = None, *, opener=None, clock=None) -> int:
    """Run one command; return its exit code. opener replaces urlopen and clock replaces time.perf_counter (tests)."""
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    try:
        return args.handler(args, opener=opener, clock=clock or time.perf_counter)
    except RebuildError as exc:
        print(f"pit_rebuild: {exc.field_name}: {exc.reason}", file=sys.stderr)
        return exc.exit_code
