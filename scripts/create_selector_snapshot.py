#!/usr/bin/env python3
"""Create the selector CSVs of a snapshot: stocks from CVM and B3, FIIs from Status Invest (issue #80, spec #75).

Run from the repository root, then build the manifest from the files it writes:

    python3 scripts/create_selector_snapshot.py --date 2026-11-16 --out-dir snapshots/data/selectors-2026-11-17
    python3 py/snapshot_registry.py build ... --stock-universe <out>/stocks.csv --fii-universe <out>/fiis.csv \\
        --stock-provenance <out>/stocks.provenance.json --fii-provenance <out>/fiis.provenance.json

Stocks: the CVM/B3 collector (py/pit_rebuild/collector.py) at d = --date, over the raw cache. Its universe is
already the B3 recent-trade filter of #63 (a session in the 30 days up to d). Status Invest is never called for
stocks, and a stock collection failure never falls back to it. A stock whose B3 classification is missing because
its GetDetail download failed or did not parse (collector "unclassified" reasons detail_download_failed and
detail_unparseable) fails the creation with exit 7: the selector z-scores by SETOR and would drop that stock silently.
A GetDetail record that has no classification (reason no_classification) is a fact of B3, not a failure; the stock is
written with empty SETOR and SUBSETOR and listed under "unclassified" in stocks.provenance.json.

FIIs: the Status Invest export (py/fetch_status_invest_fii.py), downloaded live, then filtered by the B3 recent-trade
rule of #63 at d (app.services.b3_trading.filter_recent_trades, CODBDI 12), reading the COTAHIST of the raw cache.
Premium runs no longer filter (#17), so the CSV is filtered here, before it is registered.

Writes <out-dir>/stocks.csv, stocks.provenance.json, fiis.csv and fiis.provenance.json together, or none of them.
fiis.provenance.json holds the provider "statusinvest", the download time, the CSV sha256 and the filter record.

Exit codes:

    0  the four files written
    1  the output files could not be written
    2  invalid arguments, a cache directory inside data/, snapshots/ or baselines/, or an output inside baselines/
    3  the cache directory does not exist in --offline mode
    4  integrity violation in the stock raw files: cached file hash mismatch, PIT leak
    5  a required stock raw file could not be downloaded, or is absent from the cache in --offline mode
    6  the FII CSV failed: Status Invest download or validation, or the B3 recent-trade filter
    7  a stock has no B3 classification because its GetDetail download failed or its body did not parse
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from collections.abc import Callable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT / "py", PROJECT_ROOT / "api"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from app.services.b3_trading import CotahistCache, filter_recent_trades  # noqa: E402
from pit_rebuild import cli, collector, safety  # noqa: E402
from pit_rebuild.cache import Cache  # noqa: E402
from pit_rebuild.errors import MissingPathError, OutputError, RebuildError  # noqa: E402

FII_PROVIDER = "statusinvest"
FII_TIMEOUT_SECONDS = 30
EXIT_FII = 6
EXIT_UNCLASSIFIED = 7


class FiiInputError(RuntimeError):
    """The FII CSV could not be had, validated or filtered."""


class UnclassifiedStockError(RuntimeError):
    """A stock lost its B3 classification to a failed or unreadable GetDetail."""


def _pit_cotahist(cache: Cache, root: Path) -> CotahistCache:
    """The yearly COTAHIST of the raw cache (the file the stock collector read), served to the FII filter."""

    def download(url: str) -> bytes:
        return cache.get(url, url.rsplit("/", 1)[1], required=True).read_bytes()

    return CotahistCache(root, downloader=download)


def _fii_csv(day, cache: Cache, fii_collector: Callable | None, cotahist_cache: CotahistCache | None) -> tuple[bytes, bytes]:
    """The filtered FII CSV bytes and its provenance bytes."""
    from fetch_status_invest_fii import refresh, validate_rows

    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        path = work / "fiis.csv"
        try:
            (fii_collector or refresh)(output_path=path, timeout=FII_TIMEOUT_SECONDS)
            retrieved_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            with path.open(encoding="utf-8-sig", newline="") as handle:
                validate_rows(list(csv.DictReader(handle)))
            record = filter_recent_trades(
                {"fiis": path}, work, collected_on=day,
                cache=cotahist_cache or _pit_cotahist(cache, work / "cotahist"),
            )
        except RebuildError:
            raise
        except Exception as exc:
            raise FiiInputError(f"{type(exc).__name__}: {exc}") from exc
        data = path.read_bytes()
    provenance = {
        "provider": FII_PROVIDER,
        "retrieved_at": retrieved_at,
        "csv_sha256": hashlib.sha256(data).hexdigest(),
        "recent_trade_filter": {key: value for key, value in record.items() if key != "csv_sha256"},
    }
    text = json.dumps(provenance, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    return data, text.encode("utf-8")


def create(out_dir: Path, day, cache: Cache, *, fii_collector=None, cotahist_cache=None) -> tuple[int, int]:
    """Write the four files; return (stock rows, FII rows)."""
    collected = collector.collect(cache, day)
    failed = {ticker: reason for ticker, reason in sorted(collected.unclassified.items())
              if reason in collector.UNCLASSIFIED_FAILURES}
    if failed:
        listed = ", ".join(f"{ticker} ({reason})" for ticker, reason in failed.items())
        raise UnclassifiedStockError(f"{len(failed)} stock(s) without a B3 classification: {listed}")
    stocks, stock_provenance = collector.encode(collected)
    fiis, fii_provenance = _fii_csv(day, cache, fii_collector, cotahist_cache)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        safety.write_files_atomically({
            out_dir / "stocks.csv": stocks,
            collector.provenance_path(out_dir / "stocks.csv"): stock_provenance,
            out_dir / "fiis.csv": fiis,
            out_dir / "fiis.provenance.json": fii_provenance,
        })
    except OSError as exc:
        raise OutputError(str(out_dir), f"output files could not be written: {exc}") from exc
    return len(collected.rows), fiis.count(b"\n") - 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Selector CSVs of a snapshot: stocks from CVM and B3, FIIs from Status Invest.")
    parser.add_argument("--date", required=True, metavar="YYYY-MM-DD", help="cutoff date of the snapshot")
    parser.add_argument("--out-dir", required=True, metavar="DIR", help="directory of the selector CSVs, e.g. snapshots/data/selectors-<name>")
    parser.add_argument("--cache-dir", default=str(cli.DEFAULT_CACHE_DIR), help="raw file cache (default .cache/pit-rebuild)")
    parser.add_argument("--offline", action="store_true", help="read the stock raw files only from the cache; FIIs still come from Status Invest")
    return parser


def main(argv: Sequence[str] | None = None, *, opener=None, fii_collector=None, cotahist_cache=None) -> int:
    """Create the selector CSVs; return the exit code. opener, fii_collector and cotahist_cache replace the network (tests)."""
    try:
        args = _build_parser().parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    try:
        day = cli._one_date("--date", args.date)
        out_dir = collector._out_path(args.out_dir)
        cache_dir = safety.assert_writable_dir(Path(args.cache_dir), "--cache-dir")
        if args.offline and not cache_dir.is_dir():
            raise MissingPathError("--cache-dir", f"{cache_dir} does not exist")
        cache = cli._cache(cache_dir, offline=args.offline, opener=opener)
        cache.verify()
        stocks, fiis = create(out_dir, day, cache, fii_collector=fii_collector, cotahist_cache=cotahist_cache)
    except RebuildError as exc:
        print(f"create_selector_snapshot: {exc.field_name}: {exc.reason}", file=sys.stderr)
        return exc.exit_code
    except FiiInputError as exc:
        print(f"create_selector_snapshot: FII CSV failed: {exc}", file=sys.stderr)
        return EXIT_FII
    except UnclassifiedStockError as exc:
        print(f"create_selector_snapshot: stock classification failed: {exc}", file=sys.stderr)
        return EXIT_UNCLASSIFIED
    print(f"wrote {out_dir}: {stocks} stocks (cvm-b3), {fiis} FIIs (statusinvest)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
