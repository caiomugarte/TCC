"""The stock CSV of a snapshot, built from CVM and B3 at a cutoff date d (issue #78, spec #75).

Run from the repository root:

    python3 py/pit_rebuild collect --date 2026-11-05 --out snapshots/data/selectors-2026-11-06/stocks.csv --offline

The collector reuses the point-in-time rebuild with d = the cutoff date: the same raw files, the same
cache, the same evaluation of one date (cli._evaluate_day) and so the same rules of #64 and #77 (count
moved by the B3 events, filing older than 270 days without accounting fields, 12-month flow from last
year's ITR when the comparative is zero, share scale confirmed by B3). Each field has one source: CVM
for the accounting data, B3 for price, liquidity, dividends, events, share count and sector. Dividends
come from B3 only; there is no Yahoo fallback here.

The universe is every cash-market ticker of COTAHIST with a session in the 30 days up to d that maps to
a CVM company with a usable filing. The CSV has the columns of the Status Invest export, in its scale
(percent indicators in percent, money in BRL), plus SUBSETOR, so the selector reads it unchanged.
ROIC and PEG RATIO follow the spec's own formulas and do not try to reproduce Status Invest.

Next to the CSV, <out stem>.provenance.json records the provider "cvm-b3", d, the CSV sha256, the
sha256 of every raw file used, by kind, and the tickers left out with their reason.

Exit codes (the table of the rebuild):

    0  CSV and provenance written
    1  the output files could not be written
    2  invalid arguments, a cache directory inside data/, snapshots/ or baselines/, or an output inside baselines/
    3  the cache directory does not exist in --offline mode
    4  integrity violation: cached file hash mismatch, PIT leak
    5  a required download failed, or a required file is absent from the cache in --offline mode
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import time

from . import cli, cotahist, cvm, mapping, safety, sectors, sources, status_invest
from .cache import Cache
from .errors import ArgumentError, MissingPathError, OutputError, RebuildError
from .fields import ALL_FIELDS
from .indicators import TickerInputs, Value

PROVIDER = "cvm-b3"
# The columns of the Status Invest export (py/fetch_status_invest.py OUTPUT_COLUMNS), then SUBSETOR.
COLUMNS = (
    "TICKER", "PRECO", "DY", "P/L", "P/VP", "P/ATIVOS", "MARGEM BRUTA",
    "MARGEM EBIT", "MARG. LIQUIDA", "P/EBIT", "EV/EBIT",
    "DIVIDA LIQUIDA / EBIT", "DIV. LIQ. / PATRI.", "PSR", "P/CAP. GIRO",
    "P. AT CIR. LIQ.", "LIQ. CORRENTE", "ROE", "ROA", "ROIC",
    "PATRIMONIO / ATIVOS", "PASSIVOS / ATIVOS", "GIRO ATIVOS",
    "CAGR RECEITAS 5 ANOS", "CAGR LUCROS 5 ANOS", "LIQUIDEZ MEDIA DIARIA",
    "VPA", "LPA", "PEG RATIO", "VALOR DE MERCADO", "SETOR", "SUBSETOR",
)
GROSS_PROFIT = "3.03"
TAX_RATE = 0.34  # ROIC = EBIT_12m × (1 − 34%) ÷ (PL_ctrl + dívida bruta − caixa), spec #75
ACCOUNTS = cli.STATEMENT_ACCOUNTS | {GROSS_PROFIT}
_TICKER = re.compile(r"^[A-Z]{3,4}\d{1,2}$")
_RAW_KINDS = (  # cache file name prefix -> kind of raw file in the provenance
    ("dfp_cia_aberta_", "dfp_itr"),
    ("itr_cia_aberta_", "dfp_itr"),
    ("fca_cia_aberta_", "cadastro"),
    ("cad_cia_aberta", "cadastro"),
    ("COTAHIST_", "cotahist"),
    ("b3_cash_", "proventos"),
    ("b3_company_", "eventos"),
    ("b3_detail_", "classificacao_setorial"),
)


class _Everything:
    """A ticker filter that keeps every ticker (cotahist.load reads the whole cash market)."""

    def __contains__(self, item: object) -> bool:
        return True


@dataclass(frozen=True)
class Collected:
    day: date
    rows: list[dict[str, str]]
    excluded: dict[str, str]  # ticker -> reason it has no row
    raw_files: dict[str, dict[str, str]]  # kind -> cache file name -> sha256


def detail_name(cd_cvm: str) -> str:
    return f"b3_detail_{cd_cvm}.json"


def collect(cache: Cache, day: date, clock=time.perf_counter) -> Collected:
    """Read the raw files of d from the cache (downloading what is missing when online) and build the rows."""
    files: dict[str, list[Path]] = {kind: [] for kind in cli.KINDS}
    for url, name in sources.plan_files([day]):
        files[cli._kind_of_url(url)].append(cache.get(url, name, required=True))
    fca = mapping.load_fca(files["FCA"])
    cadastro = cvm.load_cadastro(files["CADASTRO"][0])
    quotes = cotahist.load(files["COTAHIST"], _Everything())
    universe = sorted(
        ticker for ticker in quotes.rows if _TICKER.match(ticker) and cotahist.quote_at(quotes, ticker, day) is not None
    )
    b3 = {root: cli._b3_company(cache, root) for root in sorted({ticker[:4] for ticker in universe})}
    acq = cli._Acquired(
        si={day: _blank_csv(universe)},
        eligible={day: universe},
        b3=b3,
        fca=fca,
        cadastro=cadastro,
        dfp=cvm.load_filings(files["DFP"], cvm.DFP, ACCOUNTS),
        itr=cvm.load_filings(files["ITR"], cvm.ITR, ACCOUNTS),
        quotes=quotes,
        mapped={day: mapping.map_tickers(universe, fca, cadastro, b3, day)},
        files=files,
        timings={},
    )
    excluded = {ticker: reason for ticker in universe if (reason := cli._unmapped_reason(acq, day, ticker)) is not None}
    kept = [ticker for ticker in universe if ticker not in excluded]
    for name in sorted({b3[ticker[:4]].trading_name for ticker in kept if b3.get(ticker[:4])}):
        acq.histories[name] = cli._b3_history(cache, name, acq, clock)
    classes = {cd_cvm: _classification(cache, cd_cvm) for cd_cvm in sorted({acq.mapped[day][t].cd_cvm for t in kept})}
    result = cli._evaluate_day(acq, day)
    stale = {row["ticker"] for row in result.mapping["stale_filing"]}
    rows = []
    for ticker in kept:
        item = acq.mapped[day][ticker]
        extra = _extra_fields(result.inputs[ticker], _gross_profit(acq, item.cnpj, day))
        if ticker in stale:  # STL-01: every extra column reads the filing, so a stale filing leaves them empty too
            extra = dict.fromkeys(extra)
        rows.append(_row(ticker, result.values[ticker], extra, classes[item.cd_cvm]))
    return Collected(day=day, rows=rows, excluded=excluded, raw_files=_raw_files(cache.used))


def _blank_csv(universe: Sequence[str]) -> status_invest.SICsv:
    """The evaluation reads a Status Invest CSV for its comparison pairs; the collector has none, so every cell is empty."""
    return status_invest.SICsv(
        path=Path(), sha256="", rows={ticker: dict.fromkeys(ALL_FIELDS) for ticker in universe},
        setor=dict.fromkeys(universe, ""),
    )


def _classification(cache: Cache, cd_cvm: str) -> sectors.Classification | None:
    path = cache.get(sources.b3_detail_url(cd_cvm), detail_name(cd_cvm), required=False)
    if path is None:
        return None
    try:
        return sectors.parse_b3_detail(path.read_bytes())
    except ValueError:
        return None


def _gross_profit(acq, cnpj: str, day: date) -> float | None:
    flow = cvm.ttm(acq.dfp, acq.itr, cnpj, day, GROSS_PROFIT)
    return None if flow is None else flow.value


def _extra_fields(inputs: TickerInputs, gross: float | None) -> dict[str, float | None]:
    """The Status Invest columns outside the 19 rebuilt fields, from the same inputs."""

    def value(item) -> float | None:
        return None if item is None else item.value

    at, pl, pl_ctrl = value(inputs.at), value(inputs.pl), value(inputs.pl_ctrl)
    ac, pc, rec, ebit, ll = value(inputs.ac), value(inputs.pc), value(inputs.rec), value(inputs.ebit), value(inputs.ll)
    total = None if inputs.n_on is None or inputs.n_pref is None else inputs.n_on + inputs.n_pref
    price = None if inputs.price is None else inputs.price.value / inputs.unit_divisor
    liabilities = None if at is None or pl is None else at - pl
    invested = _minus(_plus(pl_ctrl, value(inputs.debt)), value(inputs.cash))
    return {
        "ROIC": _times(_div(_times(ebit, 1 - TAX_RATE), invested if invested is None or invested > 0 else None), 100),
        "VPA": _div(pl_ctrl, total),
        "LPA": _div(ll, total),
        "P/ATIVOS": _price_over(price, at, total),
        "P/EBIT": _price_over(price, ebit, total),
        "P/CAP. GIRO": _price_over(price, _minus(ac, pc), total),
        "P. AT CIR. LIQ.": _price_over(price, _minus(ac, liabilities), total),
        "MARGEM BRUTA": _times(_div(gross, rec), 100),
        "PATRIMONIO / ATIVOS": _div(pl, at),
        "PASSIVOS / ATIVOS": _div(liabilities, at),
        "GIRO ATIVOS": _div(rec, at),
    }


def _div(num: float | None, den: float | None) -> float | None:
    return None if num is None or den is None or den == 0 else num / den


def _plus(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else left + right


def _minus(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else left - right


def _times(item: float | None, factor: float) -> float | None:
    return None if item is None else item * factor


def _price_over(price: float | None, flow: float | None, total: int | None) -> float | None:
    """p_eq / (flow / N_total), the per-share form of P/L (indicators._price_per)."""
    if total is None or total <= 0:
        return None
    return _div(price, _div(flow, total))


def _peg(values: dict[str, Value]) -> float | None:
    """P/L ÷ CAGR LUCROS 5 ANOS (in %), empty when the CAGR is <= 0 or the P/L is negative (spec #75)."""
    p_l, cagr = values["P/L"].value, values["CAGR LUCROS 5 ANOS"].value
    if p_l is None or cagr is None or p_l < 0 or cagr <= 0:
        return None
    return p_l / cagr


def _row(ticker: str, values: dict[str, Value], extra: dict[str, float | None], classification) -> dict[str, str]:
    numbers = {name: values[name].value for name in ALL_FIELDS} | extra | {"PEG RATIO": _peg(values)}
    row = {"TICKER": ticker}
    for column in COLUMNS[1:-2]:
        row[column] = _cell(numbers[column])
    row["SETOR"] = "" if classification is None else classification.setor
    row["SUBSETOR"] = "" if classification is None else classification.subsetor
    return row


def _cell(number: float | None) -> str:
    if number is None:
        return ""
    text = f"{number:.6f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def _raw_files(used: Sequence[dict]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for entry in used:
        kind = next((kind for prefix, kind in _RAW_KINDS if entry["name"].startswith(prefix)), "outros")
        out.setdefault(kind, {})[entry["name"]] = entry["sha256"]
    return {kind: dict(sorted(names.items())) for kind, names in sorted(out.items())}


def provenance_path(out: Path) -> Path:
    return out.with_name(f"{out.stem}.provenance.json")


def encode(collected: Collected) -> tuple[bytes, bytes]:
    """The CSV bytes and the provenance JSON bytes of a collection."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(collected.rows)
    data = buffer.getvalue().encode("utf-8")
    provenance = {
        "provider": PROVIDER,
        "as_of": collected.day.isoformat(),
        "csv_sha256": hashlib.sha256(data).hexdigest(),
        "raw_files": collected.raw_files,
        "excluded": dict(sorted(collected.excluded.items())),
    }
    text = json.dumps(provenance, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    return data, text.encode("utf-8")


def _out_path(text: str) -> Path:
    resolved = Path(text).resolve()
    baselines = (safety.REPO_ROOT / "baselines").resolve()
    if resolved == baselines or baselines in resolved.parents:
        raise ArgumentError("--out", f"{resolved} is inside baselines/, which this tool never writes")
    return resolved


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pit_rebuild collect", description="Stock CSV of a snapshot from CVM and B3 (issue #78).")
    parser.add_argument("--date", required=True, metavar="YYYY-MM-DD", help="cutoff date of the snapshot")
    parser.add_argument("--out", required=True, metavar="PATH", help="stock CSV to write; the provenance goes next to it")
    parser.add_argument("--cache-dir", default=str(cli.DEFAULT_CACHE_DIR), help="raw file cache (default .cache/pit-rebuild)")
    parser.add_argument("--offline", action="store_true", help="read only the cache: no network request")
    return parser


def main(argv: Sequence[str] | None = None, *, opener=None) -> int:
    """Collect one cutoff date; return the exit code. opener replaces urlopen (tests)."""
    try:
        args = _build_parser().parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    try:
        day = cli._one_date("--date", args.date)
        out = _out_path(args.out)
        cache_dir = safety.assert_writable_dir(Path(args.cache_dir), "--cache-dir")
        if args.offline and not cache_dir.is_dir():
            raise MissingPathError("--cache-dir", f"{cache_dir} does not exist")
        cache = cli._cache(cache_dir, offline=args.offline, opener=opener)
        cache.verify()
        collected = collect(cache, day)
        data, provenance = encode(collected)
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            safety.write_files_atomically({out: data, provenance_path(out): provenance})
        except OSError as exc:
            raise OutputError(str(out), f"output files could not be written: {exc}") from exc
    except RebuildError as exc:
        print(f"pit_rebuild collect: {exc.field_name}: {exc.reason}", file=sys.stderr)
        return exc.exit_code
    print(f"wrote {out}: {len(collected.rows)} tickers, {len(collected.excluded)} left out")
    return 0
