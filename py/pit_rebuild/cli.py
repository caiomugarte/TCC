"""Command line of the PIT rebuild (CLI-01..CLI-05): read the raw sources in the cache, compare, write the report.

Run from the repository root:

    python3 py/pit_rebuild report --date 2026-08-07 --si data/raw/status_invest_fundamentals.csv \\
        --date 2026-11-06 --si snapshots/data/selectors-2026-11-06/stocks.csv --offline

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
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from . import compare, coverage, cotahist, cvm, dividends, mapping, report, safety, sources, status_invest
from .cache import Cache
from .errors import ArgumentError, MissingPathError, OutputError, RebuildError, SourceError
from .fields import ALL_FIELDS
from .indicators import Input, TickerInputs, Value, compute

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
_TICKER = re.compile(r"^[A-Z]{4}\d{1,2}$")
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


def _request(args: argparse.Namespace, *, needs_out: bool) -> _Request:
    dates = _dates(args.date)
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
    return _Request(dates=dates, si_paths=si_paths, cache_dir=cache_dir, out_dir=out_dir)


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


def _acquire(cache: Cache, dates: Sequence[date], si_paths: Sequence[Path], clock: Callable[[], float]) -> _Acquired:
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
    period2 = max(dates) + timedelta(days=1)
    for ticker in need_yahoo:
        acq.yahoo[ticker] = _yahoo(cache, ticker, period2, acq, clock)
    return acq


def _timed(clock, timings: dict, kind: str, paths: Sequence[Path], load: Callable[[], object]):
    start = clock()
    value = load()
    timings[kind] = (clock() - start, sum(path.stat().st_size for path in paths))
    return value


def _b3_company(cache: Cache, root: str) -> dividends.B3Company | None:
    path = cache.get(sources.b3_company_url(root), b3_company_name(root), required=False)
    if path is None:
        return None
    try:
        return dividends.parse_b3_company(path.read_bytes())
    except (ValueError, KeyError, TypeError, AttributeError, IndexError):
        return None


def _b3_history(cache: Cache, name: str, acq: _Acquired, clock) -> dividends.B3History | None:
    """Every page of GetListedCashDividends (SRC-05). Page 1 missing is b3_failed; a missing later page is incomplete."""
    pages: list[bytes] = []
    total = 1
    page = 1
    while page <= total:
        path = cache.get(sources.b3_cash_url(name, page), b3_cash_name(name, page), required=False)
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


def _unmapped_reason(acq: _Acquired, day: date, ticker: str) -> str | None:
    """Why an eligible ticker is not computed at day (MAP-01), or None when it is."""
    item = acq.mapped[day][ticker]
    if isinstance(item, mapping.Unmapped):
        return item.reason
    if not _has_filings(acq, item.cnpj, day):
        return _NO_FILINGS
    if cotahist.quote_at(acq.quotes, ticker, day) is None:
        return _NO_QUOTE
    return None


def _history_of(acq: _Acquired, ticker: str) -> dividends.B3History | None:
    company = acq.b3.get(ticker[:4])
    return None if company is None else acq.histories.get(company.trading_name)


def _choice(acq: _Acquired, day: date, ticker: str) -> tuple[str, str | None]:
    return dividends.choose_source(_history_of(acq, ticker), day)


# --- one date: companies, tickers, dividends, statistics ------------------------------------


@dataclass
class _Company:
    cnpj: str
    cd_cvm: str
    n_on: int | None
    n_pref: int | None
    scale: str | None  # "x1000", "unverified" or None
    price_on: Input | None
    price_pref: Input | None
    fallback: dict | None
    inputs: dict
    dre_desc: str | None


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
    account = "3.11.01" if cvm.ttm(dfp, itr, cnpj, day, "3.11.01") is not None else "3.11"

    def flow(name: str) -> Input | None:
        return _dated(cvm.ttm(dfp, itr, cnpj, day, name))

    def balance(name: str) -> Input | None:
        return _dated(cvm.balance(dfp, itr, cnpj, day, name))

    return {
        "ll": flow(account),
        "rec": flow("3.01"),
        "ebit": flow("3.05"),
        "ir": flow("3.08"),
        "ll_1y": _dated(cvm.ttm_year_before(dfp, itr, cnpj, day, account)),
        "pl": balance("2.03"),
        "pl_ctrl": _controllers_equity(balance("2.03"), balance("2.03.09")),
        "at": balance("1"),
        "ac": balance("1.01"),
        "pc": balance("2.01"),
        "debt": _sum_inputs([balance("2.01.04"), balance("2.02.01")]),
        "cash": _sum_inputs([balance("1.01.01"), balance("1.01.02")]),
        "rec_fy": _dated(cvm.annual(dfp, cnpj, day, "3.01", 0)),
        "rec_fy5": _dated(cvm.annual(dfp, cnpj, day, "3.01", 5)),
        "ll_fy5": _dated(cvm.annual(dfp, cnpj, day, account, 5)),
    }


def _r4_class_price(item, day: date):
    """A class price only from a ticker with a session within MAX_QUOTE_AGE_DAYS, the r4 rule.

    cotahist.class_prices also returns a stale class price (F14). Until T29 wires it into
    vm_stale_class_price, the report keeps the r4 rule: a stale class price is no price here.
    """
    if item is None or (day - item[2]).days > cotahist.MAX_QUOTE_AGE_DAYS:
        return None
    return item


def _company(acq: _Acquired, day: date, cnpj: str, cd_cvm: str, tickers: list[str]) -> _Company:
    """Share counts with the A-18 scale, the A-17 class prices and the class fallback of one company."""
    shares = cvm.shares(acq.dfp, acq.itr, cnpj, day)
    n_on = n_pref = None
    scale = None
    if shares is not None and shares.on + shares.pref > 0:
        b3_total = None
        for ticker in sorted(tickers):
            company = acq.b3.get(ticker[:4])
            if company is not None and company.total_shares:
                b3_total = company.total_shares
                break
        factor, verified = cvm.share_scale(shares.on + shares.pref, b3_total)
        n_on, n_pref = shares.on * factor, shares.pref * factor
        if not verified:
            scale = "unverified"
        elif factor == 1000:
            scale = "x1000"
    roots = {ticker[:4] for ticker in tickers}
    candidates = set(acq.fca.tickers_of(cnpj)) | set(tickers) | {t for t in acq.si[day].rows if t[:4] in roots}
    raw_prices = cotahist.class_prices(acq.quotes, candidates, day)
    prices = {cls: _r4_class_price(item, day) for cls, item in raw_prices.items()}
    quoted = {}
    for cls, item in prices.items():
        quoted[cls] = None if item is None else Input(item[1], item[2])
    fallback = None
    for cls, other, count in (("ON", "PREF", n_on), ("PREF", "ON", n_pref)):
        if count and count > 0 and prices[cls] is None and prices[other] is not None:
            fallback = {"cd_cvm": cd_cvm, "class_without_quote": cls, "price_from": prices[other][0]}
            break
    return _Company(
        cnpj=cnpj, cd_cvm=cd_cvm, n_on=n_on, n_pref=n_pref, scale=scale,
        price_on=quoted["ON"], price_pref=quoted["PREF"], fallback=fallback,
        inputs=_company_inputs(acq, cnpj, day), dre_desc=cvm.dre_description(acq.dfp, acq.itr, cnpj, day),
    )


def _known_at(dividend_list: list[dividends.Dividend], day: date) -> list[dividends.Dividend]:
    """Dividends whose ex-date is known at day. Later records of the full history are not used (PIT-02)."""
    return [item for item in dividend_list if item.ex_date is None or item.ex_date <= day]


def _b3_dividends(acq: _Acquired, ticker: str, item: mapping.Mapped, history: dividends.B3History) -> list | None:
    q = acq.quotes

    def next_session(day: date) -> date | None:
        return cotahist.next_session(q, day)

    if item.composition is not None:
        composition: dict[str, int] = {}
        for key, count in item.composition.items():
            cls = key if not _TICKER.match(key) else cotahist.share_class(q, key)
            if cls is None:
                return None
            composition[cls] = composition.get(cls, 0) + count
        return dividends.unit_dividends(history.records, composition, next_session)
    cls = cotahist.share_class(q, ticker)
    if cls is None:
        return None
    return dividends.b3_ex_dates(dividends.class_records(history.records, cls), next_session)


def _dividends_of(acq: _Acquired, day: date, ticker: str, item: mapping.Mapped):
    """(source, reason, dividends) of one ticker at day: B3 first, Yahoo for the reasons of A-15 (SRC-05)."""
    choice = _choice(acq, day, ticker)
    if choice[0] == "b3":
        divs = _b3_dividends(acq, ticker, item, _history_of(acq, ticker))
        if divs is None:
            return "none", "no_dividend_source", None
        return "b3", None, _known_at(divs, day)
    yahoo = acq.yahoo.get(ticker)
    return dividends.resolve(choice, None, None if yahoo is None else _known_at(yahoo, day))


@dataclass
class _DayResult:
    eligible: int
    mapping: dict
    pairs: dict[str, dict[str, list]]
    dividends: dict
    dy_tickers: dict[str, set[str]]
    dy_earliest: dict[str, date]


def _evaluate_day(acq: _Acquired, day: date) -> _DayResult:
    eligible = acq.eligible[day]
    csv = acq.si[day]
    mapped = acq.mapped[day]
    members: dict[str, list[str]] = defaultdict(list)
    for ticker in eligible:
        item = mapped[ticker]
        if isinstance(item, mapping.Mapped) and _unmapped_reason(acq, day, ticker) is None:
            members[item.cnpj].append(ticker)
    companies: dict[str, _Company] = {}
    for cnpj, tickers in members.items():
        companies[cnpj] = _company(acq, day, cnpj, mapped[tickers[0]].cd_cvm, tickers)
    out = {
        "mapped": 0, "unmapped": [], "refiled_after_d": [], "mapped_via_b3": [],
        "vm_price_fallback": {"count": 0, "companies": []},
        "share_scale": {"x1000": [], "unverified": []}, "financial": [], "unclassified": [],
    }
    seen_scale: set[tuple[str, str]] = set()
    for cnpj in sorted(companies):
        company = companies[cnpj]
        if company.fallback is not None:
            out["vm_price_fallback"]["companies"].append(company.fallback)
        if company.scale is not None and (company.scale, company.cd_cvm) not in seen_scale:
            seen_scale.add((company.scale, company.cd_cvm))
            key = "x1000" if company.scale == "x1000" else "unverified"
            out["share_scale"][key].append(company.cd_cvm)
    out["vm_price_fallback"]["companies"].sort(key=lambda row: row["cd_cvm"])
    out["vm_price_fallback"]["count"] = len(out["vm_price_fallback"]["companies"])
    out["share_scale"]["x1000"].sort()
    out["share_scale"]["unverified"].sort()

    values: dict[str, dict[str, Value]] = {}
    group_of: dict[str, str] = {}
    counts = {"by_source": {name: 0 for name in SOURCES}, "yahoo_reasons": {r: 0 for r in YAHOO_REASONS}, "b3_empty_yahoo_failed": 0}
    dy_tickers: dict[str, set[str]] = defaultdict(set)
    dy_earliest: dict[str, date] = {}
    for ticker in eligible:
        item = mapped[ticker]
        reason = _unmapped_reason(acq, day, ticker)
        dre = cvm.dre_description(acq.dfp, acq.itr, item.cnpj, day) if isinstance(item, mapping.Mapped) else None
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
        if cvm.refiled_after_d(acq.dfp, acq.itr, item.cnpj, day):
            out["refiled_after_d"].append(ticker)
        company = companies[item.cnpj]
        source, dy_reason, divs = _dividends_of(acq, day, ticker, item)
        counts["by_source"][source] += 1
        choice = _choice(acq, day, ticker)
        if choice[0] == "yahoo":  # every switch to Yahoo counts with its reason, whether Yahoo answered or not
            counts["yahoo_reasons"][choice[1]] += 1
        if dy_reason == "b3_empty_yahoo_failed":
            counts["b3_empty_yahoo_failed"] += 1
        if source in ("b3", "yahoo") and divs is not None:
            dy_tickers[source].add(ticker)
            known = [item_.ex_date for item_ in divs if item_.ex_date is not None]
            if known:
                dy_earliest[source] = min(dy_earliest.get(source, min(known)), min(known))
        close_day, close = cotahist.quote_at(acq.quotes, ticker, day)
        last_day = cotahist.quote_at(acq.quotes, ticker, day, max_age_days=100_000)[0]
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
            dividends=divs,
            **company.inputs,
        )
        values[ticker] = compute(inputs, day)
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
        dy_tickers=dy_tickers,
        dy_earliest=dy_earliest,
    )


# --- report object --------------------------------------------------------------------------


def _report_object(cache: Cache, acq: _Acquired, req: _Request, clock: Callable[[], float], tool_commit: str | None) -> dict:
    compute_seconds = 0.0
    per_date: dict[str, dict] = {}
    dividends_out: dict[str, dict] = {"by_source": {}, "yahoo_reasons": {}, "b3_empty_yahoo_failed": {}}
    dy_tickers: dict[str, set[str]] = defaultdict(set)
    dy_earliest: dict[str, date] = {}
    for day in sorted(req.dates):
        start = clock()
        result = _evaluate_day(acq, day)
        fields_out = {
            name: {group: compare.summarize(name, result.pairs[name][group]) for group in GROUP_KEYS}
            for name in ALL_FIELDS
        }
        compute_seconds += clock() - start
        iso = day.isoformat()
        per_date[iso] = {"eligible": result.eligible, "mapping": result.mapping, "fields": fields_out}
        dividends_out["by_source"][iso] = dict(result.dividends["by_source"])
        dividends_out["yahoo_reasons"][iso] = dict(result.dividends["yahoo_reasons"])
        dividends_out["b3_empty_yahoo_failed"][iso] = result.dividends["b3_empty_yahoo_failed"]
        for source, tickers in result.dy_tickers.items():
            dy_tickers[source] |= tickers
        for source, day_seen in result.dy_earliest.items():
            dy_earliest[source] = min(dy_earliest.get(source, day_seen), day_seen)
    dy = coverage.dy_coverage({
        source: {"tickers": len(dy_tickers.get(source, ())), "earliest_ex_date": dy_earliest.get(source)}
        for source in ("b3", "yahoo")
    })
    coverage_out = _coverage(cache, acq, req, compute_seconds, dy)
    raw = list(cache.used)
    return report.build(
        dates=list(req.dates),
        tool_commit=tool_commit,
        status_invest=[
            {"date": day.isoformat(), "path": str(path), "sha256": acq.si[day].sha256}
            for day, path in zip(req.dates, req.si_paths)
        ],
        raw_files=raw,
        per_date=per_date,
        dividends=dividends_out,
        coverage=coverage_out,
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
    acq = _acquire(cache, req.dates, req.si_paths, clock)
    obj = _report_object(cache, acq, req, clock, _tool_commit())
    _write_report(req.out_dir, obj)
    return 0


def _command_fetch(args: argparse.Namespace, *, opener, clock) -> int:
    """Download every file the rebuild of the dates needs, in the order of design.md, and record the survey (COV-01, SRC-01)."""
    req = _request(args, needs_out=False)
    cache = _cache(req.cache_dir, offline=False, opener=opener)
    cache.verify()
    _acquire(cache, req.dates, req.si_paths, clock)
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
    report_parser.set_defaults(handler=_command_report)
    fetch_parser = commands.add_parser("fetch", help="download the raw sources into the cache, without a report")
    fetch_parser.add_argument("--date", action="append", required=True, metavar="YYYY-MM-DD", help="decision date (repeatable)")
    fetch_parser.add_argument("--si", action="append", required=True, metavar="PATH", help="Status Invest CSV of the date at the same position")
    fetch_parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="download cache (default .cache/pit-rebuild)")
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
