"""Command line of the FII measurement (issue #60): can the FII selector indicators be rebuilt from CVM and B3?

Run from the repository root:

    python3 py/pit_rebuild fii-report --date 2026-08-10 --si data/raw/status_invest_fii.csv \\
        --date 2026-11-05 --si <checkout>/snapshots/data/selectors-2026-11-06/fiis.csv \\
        --collected-on 2026-08-10 --collected-on 2026-11-06

fii-report reads the CVM FII monthly reports, COTAHIST, the B3 fund pages and, as a marked fallback,
Yahoo from the cache (downloading what is missing unless --offline), rebuilds the 13 indicators of
every eligible FII at each date and writes pit-rebuild-fiis.json and pit-rebuild-fiis.md to
--out-dir, both only after every computation finished. Exit codes, as the stock tool:

    0  report written
    1  the report files could not be written
    2  invalid arguments, or an output or cache directory inside data/, snapshots/ or the baselines tree
    3  a given path does not exist (a --si CSV, or the cache directory with --offline)
    4  integrity violation: cached file hash mismatch, Status Invest CSV missing a column or repeating a ticker
    5  a required download failed, or a required file is absent from the cache in --offline mode
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
import json
from pathlib import Path
import statistics
import sys

from . import cotahist, fii_audit, fii_compare, fii_dividends, fii_indicators, fii_informes, fii_report, fii_sources
from . import fii_status_invest, safety, sources
from .cache import Cache
from .cli import PAUSE_SECONDS, _collected, _dates, _tool_commit, yahoo_name
from .compare import verdict, worst
from .dividends import Dividend, parse_yahoo
from .errors import ArgumentError, MissingPathError, OutputError, RebuildError

SCHEMA = "pit-rebuild-fiis/v1"
DEFAULT_CACHE_DIR = safety.REPO_ROOT / ".cache" / "pit-rebuild"
DEFAULT_OUT_DIR = safety.REPO_ROOT / "docs" / "validation"
REPORT_JSON = "pit-rebuild-fiis.json"
REPORT_MD = "pit-rebuild-fiis.md"
THRESHOLD_PCT = 70
SESSION_CHECK_DAYS = (-7, 3)
COVERAGE = {
    "N COTISTAS": "Total_Numero_Cotistas",
    "PATRIMONIO": "Patrimonio_Liquido",
    "N COTAS": "Cotas_Emitidas",
    "PERCENTUAL EM CAIXA": "Total_Necessidades_Liquidez",
    "P/VP": "Valor_Patrimonial_Cotas",
    "CAGR VALOR COTA 3 ANOS": "Valor_Patrimonial_Cotas",
    "SETOR": "Segmento_Atuacao",
    "GESTAO": "Tipo_Gestao",
}


@dataclass
class _Request:
    dates: tuple[date, ...]
    si_paths: tuple[Path, ...]
    collected_on: tuple[date, ...]
    cache_dir: Path
    out_dir: Path
    offline: bool


def _request(args: argparse.Namespace) -> _Request:
    dates = _dates(args.date)
    collected = _collected(args.collected_on or [], dates)
    si_paths = tuple(Path(item) for item in args.si)
    if len(si_paths) != len(dates):
        raise ArgumentError("--si", f"{len(si_paths)} given for {len(dates)} --date (one --si per --date)")
    cache_dir = safety.assert_writable_dir(Path(args.cache_dir), "--cache-dir")
    out_dir = safety.assert_writable_dir(Path(args.out_dir), "--out-dir")
    for path in si_paths:
        if not path.is_file():
            raise MissingPathError("--si", f"{path} does not exist")
    if args.offline and not cache_dir.is_dir():
        raise MissingPathError("--cache-dir", f"{cache_dir} does not exist")
    return _Request(dates, si_paths, collected, cache_dir, out_dir, args.offline)


@dataclass
class _Sources:
    informes: fii_informes.Informes
    informe_years: list[int]
    quotes: cotahist.Quotes
    cotahist_years: list[int]
    b3: dict[str, fii_dividends.B3Fund | None]
    yahoo: dict[str, list[Dividend] | None]


def _acquire(cache: Cache, req: _Request, tickers: set[str]) -> _Sources:
    listing = cache.get(fii_sources.INFORME_LISTING, fii_sources.informe_listing_name())
    last_year = max(req.dates).year
    years = [year for year in fii_sources.listed_years(listing.read_text(encoding="utf-8", errors="replace")) if year <= last_year]
    informe_paths = [cache.get(fii_sources.informe_url(year), fii_sources.informe_name(year)) for year in years]
    cotahist_years = sorted({year for day in req.dates for year in (day.year - 1, day.year)})
    cotahist_paths = [cache.get(sources.cotahist_url(year), f"COTAHIST_A{year}.ZIP") for year in cotahist_years]
    b3: dict[str, fii_dividends.B3Fund | None] = {}
    for root in sorted({ticker[:4] for ticker in tickers}):
        path = cache.get(fii_sources.b3_fund_url(root), fii_sources.b3_fund_name(root), required=False)
        b3[root] = _parse(path, fii_dividends.parse_b3_fund)
    period2 = max(req.collected_on) + timedelta(days=1)
    yahoo: dict[str, list[Dividend] | None] = {}
    for ticker in sorted(tickers):
        path = cache.get(sources.yahoo_url(ticker, period2), yahoo_name(ticker, period2), required=False)
        yahoo[ticker] = _parse(path, parse_yahoo)
    return _Sources(
        informes=fii_informes.load(informe_paths),
        informe_years=years,
        quotes=cotahist.load(cotahist_paths, tickers, codbdi=cotahist.FII_CODBDI),
        cotahist_years=cotahist_years,
        b3=b3,
        yahoo=yahoo,
    )


def _parse(path: Path | None, parser):
    if path is None:
        return None
    try:
        return parser(path.read_bytes())
    except (ValueError, KeyError, TypeError, IndexError):
        return None  # an answer that is not the expected document counts as a failed request


# --- one date --------------------------------------------------------------------------------


def _measure_date(src: _Sources, si: fii_status_invest.SIFiiCsv, d: date) -> dict:
    eligible = fii_status_invest.eligible(si)
    rebuilt = {
        ticker: fii_indicators.rebuild(ticker, d, src.quotes, src.informes, src.b3.get(ticker[:4]), src.yahoo.get(ticker))
        for ticker in eligible
    }
    contexts = {
        ticker: fii_audit.Context(item, src.informes, src.quotes, src.b3.get(ticker[:4]), src.yahoo.get(ticker))
        for ticker, item in rebuilt.items()
    }
    fields = {}
    for name in fii_status_invest.NUMERIC_FIELDS:
        stats = fii_compare.summarize(
            name, [(t, si.rows[t][name], rebuilt[t].values[name], rebuilt[t].reasons.get(name)) for t in eligible]
        )
        for case in stats.worst:
            case.update(fii_audit.classify(name, case["reference"], contexts[case["ticker"]]))
            case["used"] = _used(rebuilt[case["ticker"]], name)
        fields[name] = {**_plain(stats), "verdict": verdict(stats.pct_within, stats.denominator)}
    categories = {}
    for name in fii_status_invest.CATEGORY_FIELDS:
        stats = fii_compare.summarize_categories(
            name, [(t, si.text[t][name], rebuilt[t].values[name], rebuilt[t].reasons.get(name)) for t in eligible]
        )
        categories[name] = {
            "denominator": stats.denominator,
            "within": stats.within,
            "pct_within": stats.pct_within,
            "reference_blank": stats.reference_blank,
            "verdict": verdict(stats.pct_within, stats.denominator),
            "crosstab": [
                {"cvm": cvm_value, "status_invest": ref, "tickers": count,
                 "same_after_mapping": ref in fii_compare.mapped(name, cvm_value)}
                for (cvm_value, ref), count in sorted(stats.crosstab.items(), key=lambda item: (-item[1], item[0]))
            ],
            "mismatches": [
                {**case, "class": fii_audit.METHOD, "evidence": rebuilt[case["ticker"]].informe.label(),
                 "note": "a classificação da CVM e a do Status Invest diferem depois do mapeamento"}
                for case in stats.mismatches
            ],
            "missing": stats.missing,
        }
    return {
        "eligible": len(eligible),
        "fields": fields,
        "categories": categories,
        "dividends": _dividend_counts(rebuilt, si, eligible),
        "mapping": _mapping_counts(rebuilt),
        "as_of_check": _as_of_check(src.quotes, si, eligible, d),
        "tickers": {ticker: _ticker(item) for ticker, item in rebuilt.items()},
    }


def _used(item: fii_indicators.Rebuilt, name: str) -> str:
    if name in ("PRECO", "LIQUIDEZ MEDIA DIARIA"):
        return f"pregão de {item.quote_day.isoformat()}" if item.quote_day else "sem pregão"
    if name in fii_indicators.DIVIDEND_FIELDS:
        source, reason = item.sources.get(name, ("none", None))
        return f"proventos: {source}" + (f" ({reason})" if reason else "")
    return item.informe.label() if item.informe else "sem informe"


def _plain(stats: fii_compare.FieldStats) -> dict:
    return {
        "denominator": stats.denominator, "within": stats.within, "pct_within": stats.pct_within,
        "reference_blank": stats.reference_blank, "diff_unit": stats.diff_unit,
        "median_diff": stats.median_diff, "p90_diff": stats.p90_diff, "worst": stats.worst, "missing": stats.missing,
    }


def _dividend_counts(rebuilt: dict, si: fii_status_invest.SIFiiCsv, eligible: list[str]) -> dict:
    counts = {}
    for name in fii_indicators.DIVIDEND_FIELDS:
        sources_seen = Counter(item.sources[name][0] for item in rebuilt.values())
        reasons = Counter(item.sources[name][1] for item in rebuilt.values() if item.sources[name][1])
        counts[name] = {"sources": dict(sorted(sources_seen.items())), "reasons": dict(sorted(reasons.items()))}
    b3_only = fii_compare.summarize("DY", [(t, si.rows[t]["DY"], rebuilt[t].b3_only_dy, "no_b3_record") for t in eligible])
    counts["b3_only_dy"] = {"denominator": b3_only.denominator, "within": b3_only.within, "pct_within": b3_only.pct_within}
    return counts


def _mapping_counts(rebuilt: dict) -> dict:
    reasons = Counter(item.reasons.get("PATRIMONIO") for item in rebuilt.values() if item.informe is None)
    return {
        "with_informe": sum(1 for item in rebuilt.values() if item.informe is not None),
        "without_informe": dict(sorted(reasons.items())),
        "delivered_after_d": sum(1 for item in rebuilt.values() if item.delivered_after),
    }


def _as_of_check(quotes: cotahist.Quotes, si: fii_status_invest.SIFiiCsv, eligible: list[str], d: date) -> dict:
    """How many eligible prices equal, to the cent, the close of each session near d (evidence only)."""
    low, high = d + timedelta(days=SESSION_CHECK_DAYS[0]), d + timedelta(days=SESSION_CHECK_DAYS[1])
    matches: dict[str, int] = {}
    for session in [day for day in quotes.sessions if low <= day <= high]:
        count = 0
        for ticker in eligible:
            row = next((item for item in quotes.rows.get(ticker, []) if item.day == session), None)
            price = si.rows[ticker]["PRECO"]
            if row is not None and price is not None and round(row.close, 2) == round(price, 2):
                count += 1
        matches[session.isoformat()] = count
    best = max(matches.items(), key=lambda item: (item[1], item[0]))[0] if matches else None
    return {"sessions": matches, "best": best, "matches_d": best == d.isoformat()}


def _ticker(item: fii_indicators.Rebuilt) -> dict:
    return {
        "isin": item.isin,
        "cnpj": item.cnpj,
        "informe": item.informe.label() if item.informe else None,
        "informe_36_months_before": item.informe_3y.label() if item.informe_3y else None,
        "delivered_after": [day.isoformat() for day in item.delivered_after],
        "quote_day": item.quote_day.isoformat() if item.quote_day else None,
        "values": item.values,
        "reasons": dict(sorted(item.reasons.items())),
        "sources": {name: list(value) for name, value in sorted(item.sources.items())},
        "b3_only_dy": item.b3_only_dy,
    }


# --- the report --------------------------------------------------------------------------------


def _report_object(cache: Cache, req: _Request, src: _Sources, results: dict[date, dict], si: dict[date, fii_status_invest.SIFiiCsv]) -> dict:
    isos = [day.isoformat() for day in req.dates]
    verdicts = {}
    for name in fii_status_invest.FIELDS:
        block = "fields" if name in fii_status_invest.NUMERIC_FIELDS else "categories"
        per_date = {iso: results[day][block][name]["verdict"] for iso, day in zip(isos, req.dates)}
        verdicts[name] = {"dates": per_date, "overall": worst(per_date.values())}
    failing = []
    for name in fii_status_invest.FIELDS:
        block = "fields" if name in fii_status_invest.NUMERIC_FIELDS else "categories"
        pcts = [results[day][block][name]["pct_within"] for day in req.dates]
        if any(pct is None or pct < THRESHOLD_PCT for pct in pcts):
            failing.append(name)
    cases = [case for day in req.dates for stats in results[day]["fields"].values() for case in stats["worst"]]
    cases += [case for day in req.dates for stats in results[day]["categories"].values() for case in stats["mismatches"][:10]]
    open_cases = sum(1 for case in cases if case["class"] == fii_audit.OPEN)
    part1 = {"threshold_pct": THRESHOLD_PCT, "met": not failing, "failing": failing}
    part2 = {"cases": len(cases), "open": open_cases, "met": open_cases == 0,
             "by_class": dict(Counter(case["class"] for case in cases))}
    return {
        "schema": SCHEMA,
        "issue": 60,
        "parent": 75,
        "tool_commit": _tool_commit(),
        "dates": {iso: results[day] for iso, day in zip(isos, req.dates)},
        "decision_dates": isos,
        "inputs": {
            "status_invest": [
                {"date": day.isoformat(), "collected_on": collected.isoformat(), "file": si[day].path.name,
                 "sha256": si[day].sha256, "rows": len(si[day].rows)}
                for day, collected in zip(req.dates, req.collected_on)
            ],
            "files": sorted(cache.used, key=lambda item: item["name"]),
        },
        "eligibility": {"source": fii_status_invest.ELIGIBILITY_SOURCE, "floor": fii_status_invest.FLOOR,
                        "core_positive": list(fii_status_invest.CORE_POSITIVE)},
        "category_maps": {name: {key: sorted(value) for key, value in table.items()} for name, table in fii_compare.CATEGORY_MAPS.items()},
        "verdicts": verdicts,
        "class_verdict": worst(item["overall"] for item in verdicts.values()),
        "criterion": {"part1": part1, "part2": part2, "met": part1["met"] and part2["met"]},
        "dividends": {iso: results[day]["dividends"] for iso, day in zip(isos, req.dates)},
        "coverage": _coverage(src, results[req.dates[-1]]),
    }


def _coverage(src: _Sources, last: dict) -> dict:
    first = fii_informes.first_years(src.informes)
    matched = {item["cnpj"] for item in last["tickers"].values() if item["cnpj"]}
    informes = {}
    for name, column in COVERAGE.items():
        shares = {}
        for year in src.informe_years:
            with_value = fii_informes.funds_with_value(src.informes, column, year) & matched
            shares[str(year)] = round(100 * len(with_value) / len(matched), 2) if matched else None
        first_year = first.get(column)
        if name == "CAGR VALOR COTA 3 ANOS" and first_year is not None:
            first_year += 3
        informes[name] = {"column": column, "first_year": first_year, "pct_of_matched_funds_by_year": shares}
    b3_first = sorted(
        min(payment.last_date_prior for payment in fund.payments)
        for fund in src.b3.values() if fund is not None and fund.payments
    )
    yahoo_first = sorted(min(item.ex_date for item in series) for series in src.yahoo.values() if series)
    return {
        "informes": informes,
        "informe_years_listed": src.informe_years,
        "cotahist_years_loaded": src.cotahist_years,
        "b3_dividends": _spread(b3_first),
        "yahoo_dividends": _spread(yahoo_first),
        "matched_funds_last_date": len(matched),
    }


def _spread(days: list[date]) -> dict:
    if not days:
        return {"funds": 0, "earliest": None, "median": None}
    return {"funds": len(days), "earliest": days[0].isoformat(), "median": statistics.median_low(days).isoformat()}


# --- entry point -------------------------------------------------------------------------------


def _command_report(args: argparse.Namespace, *, opener) -> int:
    req = _request(args)
    si = {day: fii_status_invest.load(path) for day, path in zip(req.dates, req.si_paths)}
    tickers = {ticker for day in req.dates for ticker in fii_status_invest.eligible(si[day])}
    cache = Cache(req.cache_dir, offline=req.offline, pause_seconds=PAUSE_SECONDS) if opener is None else Cache(req.cache_dir, offline=req.offline, opener=opener)
    cache.verify()
    src = _acquire(cache, req, tickers)
    results = {day: _measure_date(src, si[day], day) for day in req.dates}
    obj = _report_object(cache, req, src, results, si)
    payload = {
        req.out_dir / REPORT_JSON: (json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True, default=str) + "\n").encode("utf-8"),
        req.out_dir / REPORT_MD: fii_report.render_markdown(obj).encode("utf-8"),
    }
    try:
        req.out_dir.mkdir(parents=True, exist_ok=True)
        safety.write_files_atomically(payload)
    except OSError as exc:
        raise OutputError(str(req.out_dir), f"report files could not be written: {exc}") from exc
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pit_rebuild", description="PIT rebuild of the FII indicators (issue #60).")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    report = commands.add_parser("fii-report", help="rebuild the FII indicators at each date and write the report")
    report.add_argument("--date", action="append", required=True, metavar="YYYY-MM-DD", help="decision date (repeatable)")
    report.add_argument("--si", action="append", required=True, metavar="PATH", help="Status Invest FII CSV of the date at the same position")
    report.add_argument("--collected-on", action="append", metavar="YYYY-MM-DD", help="collection date of the CSV at the same position")
    report.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="download cache (default .cache/pit-rebuild)")
    report.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="where the report files are written")
    report.add_argument("--offline", action="store_true", help="read only the cache: no network request")
    report.set_defaults(handler=_command_report)
    return parser


def main(argv: Sequence[str] | None = None, *, opener=None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    try:
        return args.handler(args, opener=opener)
    except RebuildError as exc:
        print(f"pit_rebuild: {exc.field_name}: {exc.reason}", file=sys.stderr)
        return exc.exit_code
