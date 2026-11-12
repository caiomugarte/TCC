"""Recent B3 trading filter of the Status Invest universe (issue #63). No network."""

import codecs
import csv
from datetime import date, timedelta
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

from app.services.b3_trading import (
    COTAHIST_CACHE_ENV,
    B3TradingUnavailableError,
    CotahistCache,
    default_cache_root,
)
from app.services.premium_executor import PremiumExecutor
from app.services.status_invest_inputs import refresh_inputs
from fetch_status_invest import OUTPUT_COLUMNS as STOCK_COLUMNS
from fetch_status_invest_fii import OUTPUT_COLUMNS as FII_COLUMNS
from pit_rebuild.sources import cotahist_url
from tests.cotahist_fixture import COLLECTED_ON, cotahist_zip, trade


def _csv_bytes(columns, tickers, *, bom=False) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    for index, ticker in enumerate(tickers):
        writer.writerow({column: ticker if column == "TICKER" else str(index + 1) for column in columns})
    data = buffer.getvalue().encode("utf-8")
    return codecs.BOM_UTF8 + data if bom else data


def _writer(data: bytes):
    def collect(*, output_path, timeout):
        output_path.write_bytes(data)

    return collect


class Downloader:
    def __init__(self, files=None, error=None):
        self.files = files or {}
        self.error = error
        self.calls = []

    def __call__(self, url):
        self.calls.append(url)
        if self.error is not None:
            raise self.error
        answer = self.files[url]
        if isinstance(answer, BaseException):
            raise answer
        return answer


def _http_error(url, code):
    return urllib.error.HTTPError(url, code, "error", {}, None)


class RecentTradeFilterTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def run_refresh(self, stocks, fiis, lines, *, collected_on=COLLECTED_ON, workspace="run-1",
                    downloader=None, stock_bytes=None, fii_bytes=None):
        downloader = downloader or Downloader({cotahist_url(collected_on.year): cotahist_zip(lines)})
        cache = CotahistCache(self.root / "cache", downloader=downloader)
        result = refresh_inputs(
            self.root / workspace,
            stock_collector=_writer(stock_bytes or _csv_bytes(STOCK_COLUMNS, stocks)),
            fii_collector=_writer(fii_bytes or _csv_bytes(FII_COLUMNS, fiis)),
            cotahist_cache=cache,
            collected_on=collected_on,
        )
        record = json.loads((self.root / workspace / "status-invest" / "universe-filter.json").read_text("utf-8"))
        return result, record, downloader

    @staticmethod
    def tickers(path):
        with Path(path).open(encoding="utf-8-sig", newline="") as handle:
            return [row["TICKER"] for row in csv.DictReader(handle)]

    def test_recent_session_stays_and_no_session_is_removed_as_nenhum(self):
        result, record, _ = self.run_refresh(
            ["LIVE3", "DEAD3"], ["LIVE11"],
            [trade("LIVE3", COLLECTED_ON - timedelta(days=2)), trade("LIVE11", COLLECTED_ON, codbdi="12")],
        )

        self.assertEqual(self.tickers(result.stocks.path), ["LIVE3"])
        self.assertEqual(self.tickers(result.fiis.path), ["LIVE11"])
        self.assertEqual(
            record["removed"],
            {"stocks": [{"ticker": "DEAD3", "reason": "no_recent_trade", "last_trade": "nenhum",
                         "other_codbdi_last_trade": None}], "fiis": []},
        )

    def test_last_session_31_days_before_is_removed_and_30_days_stays(self):
        result, record, _ = self.run_refresh(
            ["EDGE3", "OLD3"], ["LIVE11"],
            [
                trade("EDGE3", COLLECTED_ON - timedelta(days=30)),
                trade("OLD3", COLLECTED_ON - timedelta(days=31)),
                trade("LIVE11", COLLECTED_ON, codbdi="12"),
            ],
        )

        self.assertEqual(self.tickers(result.stocks.path), ["EDGE3"])
        self.assertEqual(
            record["removed"]["stocks"],
            [{"ticker": "OLD3", "reason": "no_recent_trade",
              "last_trade": (COLLECTED_ON - timedelta(days=31)).isoformat(), "other_codbdi_last_trade": None}],
        )

    def test_a_session_after_the_collection_date_does_not_count(self):
        result, record, _ = self.run_refresh(
            ["LATE3"], ["LIVE11"],
            [trade("LATE3", COLLECTED_ON + timedelta(days=1)), trade("LIVE11", COLLECTED_ON, codbdi="12")],
        )

        self.assertEqual(self.tickers(result.stocks.path), [])
        self.assertEqual(record["removed"]["stocks"][0]["last_trade"], "nenhum")

    def test_fiis_trade_under_codbdi_12_and_stock_codes_do_not_keep_a_fii(self):
        result, record, _ = self.run_refresh(
            ["LIVE3", "RECO3"], ["FUND11", "LOTE11"],
            [
                trade("LIVE3", COLLECTED_ON),
                trade("RECO3", COLLECTED_ON, codbdi="08"),
                trade("FUND11", COLLECTED_ON, codbdi="12"),
                trade("LOTE11", COLLECTED_ON, codbdi="02"),
            ],
        )

        self.assertEqual(self.tickers(result.stocks.path), ["LIVE3", "RECO3"])
        self.assertEqual(self.tickers(result.fiis.path), ["FUND11"])
        self.assertEqual([item["ticker"] for item in record["removed"]["fiis"]], ["LOTE11"])
        self.assertEqual(record["removed"]["fiis"][0]["other_codbdi_last_trade"],
                         {"date": COLLECTED_ON.isoformat(), "codbdi": "02"})

    def test_removed_ticker_trading_under_another_codbdi_records_that_session(self):
        result, record, _ = self.run_refresh(
            ["G2DI33", "PINE14", "OLDR3", "LIVE3"], ["PPLA11", "CTEM11", "LIVE11"],
            [
                trade("LIVE3", COLLECTED_ON),
                trade("LIVE11", COLLECTED_ON, codbdi="12"),
                trade("G2DI33", date(2026, 11, 3), codbdi="35"),
                trade("G2DI33", date(2026, 11, 6), codbdi="35"),
                trade("G2DI33", COLLECTED_ON + timedelta(days=1), codbdi="35"),
                trade("PINE14", date(2026, 11, 5), codbdi="22"),
                trade("OLDR3", date(2026, 9, 1)),
                trade("OLDR3", date(2026, 8, 20), codbdi="96"),
                trade("PPLA11", date(2026, 11, 5), codbdi="35"),
                trade("CTEM11", date(2026, 9, 1), codbdi="12"),
                trade("CTEM11", date(2026, 10, 30), codbdi="14"),
                trade("CTEM11", date(2026, 10, 30), codbdi="14", tpmerc="030"),
            ],
        )

        self.assertEqual(self.tickers(result.stocks.path), ["LIVE3"])
        self.assertEqual(self.tickers(result.fiis.path), ["LIVE11"])
        self.assertEqual(record["removed"], {
            "stocks": [
                {"ticker": "G2DI33", "reason": "no_recent_trade", "last_trade": "nenhum",
                 "other_codbdi_last_trade": {"date": "2026-11-06", "codbdi": "35"}},
                {"ticker": "OLDR3", "reason": "no_recent_trade", "last_trade": "2026-09-01",
                 "other_codbdi_last_trade": {"date": "2026-08-20", "codbdi": "96"}},
                {"ticker": "PINE14", "reason": "no_recent_trade", "last_trade": "nenhum",
                 "other_codbdi_last_trade": {"date": "2026-11-05", "codbdi": "22"}},
            ],
            "fiis": [
                {"ticker": "CTEM11", "reason": "no_recent_trade", "last_trade": "2026-09-01",
                 "other_codbdi_last_trade": {"date": "2026-10-30", "codbdi": "14"}},
                {"ticker": "PPLA11", "reason": "no_recent_trade", "last_trade": "nenhum",
                 "other_codbdi_last_trade": {"date": "2026-11-05", "codbdi": "35"}},
            ],
        })

    def test_only_the_cash_market_counts(self):
        result, _, _ = self.run_refresh(
            ["LIVE3", "TERM3"], ["LIVE11"],
            [
                trade("LIVE3", COLLECTED_ON),
                trade("TERM3", COLLECTED_ON, tpmerc="030"),
                trade("LIVE11", COLLECTED_ON, codbdi="12"),
            ],
        )

        self.assertEqual(self.tickers(result.stocks.path), ["LIVE3"])

    def test_january_reads_the_previous_year_too(self):
        collected_on = date(2027, 1, 10)
        downloader = Downloader({
            cotahist_url(2026): cotahist_zip([trade("DEC3", date(2026, 12, 20))], member="COTAHIST_A2026.TXT"),
            cotahist_url(2027): cotahist_zip([trade("LIVE11", date(2027, 1, 8), codbdi="12")],
                                             member="COTAHIST_A2027.TXT"),
        })

        result, record, _ = self.run_refresh(
            ["DEC3"], ["LIVE11"], [], collected_on=collected_on, downloader=downloader,
        )

        self.assertEqual(sorted(downloader.calls), [cotahist_url(2026), cotahist_url(2027)])
        self.assertEqual(self.tickers(result.stocks.path), ["DEC3"])
        self.assertEqual([item["url"] for item in record["cotahist"]], [cotahist_url(2026), cotahist_url(2027)])

    def test_january_without_the_new_year_file_continues_with_the_previous_year(self):
        for collected_on in (date(2027, 1, 1), date(2027, 1, 4)):
            with self.subTest(collected_on=collected_on):
                previous = cotahist_zip([trade("DEC3", date(2026, 12, 30)), trade("FUND11", date(2026, 12, 30), codbdi="12")])
                downloader = Downloader({
                    cotahist_url(2026): previous,
                    cotahist_url(2027): _http_error(cotahist_url(2027), 404),
                })

                result, record, _ = self.run_refresh(
                    ["DEC3", "GONE3"], ["FUND11"], [], collected_on=collected_on, downloader=downloader,
                    workspace=f"run-{collected_on.isoformat()}",
                )

                self.assertEqual(self.tickers(result.stocks.path), ["DEC3"])
                self.assertEqual(self.tickers(result.fiis.path), ["FUND11"])
                self.assertEqual([item["url"] for item in record["cotahist"]], [cotahist_url(2026)])
                self.assertEqual(record["cotahist_absent"], [{"url": cotahist_url(2027), "status": 404}])
                self.assertFalse((self.root / "cache" / collected_on.isoformat() / "COTAHIST_A2027.ZIP").exists())

    def test_the_new_year_404_is_not_cached(self):
        collected_on = date(2027, 1, 4)
        downloader = Downloader({
            cotahist_url(2026): cotahist_zip([trade("DEC3", date(2026, 12, 30)), trade("FUND11", date(2026, 12, 30), codbdi="12")]),
            cotahist_url(2027): _http_error(cotahist_url(2027), 404),
        })
        self.run_refresh(["DEC3"], ["FUND11"], [], collected_on=collected_on, downloader=downloader, workspace="run-a")
        downloader.files[cotahist_url(2027)] = cotahist_zip([trade("DEC3", collected_on), trade("FUND11", collected_on, codbdi="12")],
                                                       member="COTAHIST_A2027.TXT")

        _, record, _ = self.run_refresh(["DEC3"], ["FUND11"], [], collected_on=collected_on, downloader=downloader,
                                        workspace="run-b")

        self.assertEqual(downloader.calls.count(cotahist_url(2027)), 2)
        self.assertEqual(downloader.calls.count(cotahist_url(2026)), 1)
        self.assertEqual(record["cotahist_absent"], [])
        self.assertEqual([item["url"] for item in record["cotahist"]], [cotahist_url(2026), cotahist_url(2027)])

    def test_january_other_failures_on_the_new_year_file_fail_the_refresh(self):
        collected_on = date(2027, 1, 4)
        for failure in (_http_error(cotahist_url(2027), 500), OSError("network down"), TimeoutError("timed out")):
            with self.subTest(failure=repr(failure)):
                downloader = Downloader({
                    cotahist_url(2026): cotahist_zip([trade("DEC3", date(2026, 12, 30)), trade("FUND11", date(2026, 12, 30), codbdi="12")]),
                    cotahist_url(2027): failure,
                })
                with self.assertRaises(B3TradingUnavailableError):
                    self.run_refresh(["DEC3"], ["FUND11"], [], collected_on=collected_on, downloader=downloader)

    def test_january_404_on_the_previous_year_file_fails_the_refresh(self):
        collected_on = date(2027, 1, 4)
        downloader = Downloader({
            cotahist_url(2026): _http_error(cotahist_url(2026), 404),
            cotahist_url(2027): cotahist_zip([trade("DEC3", collected_on), trade("FUND11", collected_on, codbdi="12")],
                                                       member="COTAHIST_A2027.TXT"),
        })

        with self.assertRaises(B3TradingUnavailableError):
            self.run_refresh(["DEC3"], ["FUND11"], [], collected_on=collected_on, downloader=downloader)

    def test_january_404_with_a_stale_previous_year_still_fails_without_a_session(self):
        collected_on = date(2027, 1, 20)
        downloader = Downloader({
            cotahist_url(2026): cotahist_zip([trade("DEC3", date(2026, 12, 1)), trade("FUND11", date(2026, 12, 1), codbdi="12")]),
            cotahist_url(2027): _http_error(cotahist_url(2027), 404),
        })

        with self.assertRaisesRegex(B3TradingUnavailableError, "no session"):
            self.run_refresh(["DEC3"], ["FUND11"], [], collected_on=collected_on, downloader=downloader)

    def test_a_404_on_the_current_year_outside_january_fails_the_refresh(self):
        downloader = Downloader({cotahist_url(COLLECTED_ON.year): _http_error(cotahist_url(COLLECTED_ON.year), 404)})

        with self.assertRaises(B3TradingUnavailableError):
            self.run_refresh(["LIVE3"], ["LIVE11"], [], downloader=downloader)

    def test_outside_january_only_the_current_year_is_read(self):
        _, _, downloader = self.run_refresh(
            ["LIVE3"], ["LIVE11"], [trade("LIVE3", COLLECTED_ON), trade("LIVE11", COLLECTED_ON, codbdi="12")],
        )

        self.assertEqual(downloader.calls, [cotahist_url(COLLECTED_ON.year)])

    def test_download_failure_fails_the_refresh_with_its_own_error_code(self):
        with self.assertRaises(B3TradingUnavailableError) as ctx:
            self.run_refresh(["LIVE3"], ["LIVE11"], [], downloader=Downloader(error=OSError("B3 is down")))

        self.assertEqual(PremiumExecutor._failure_code(ctx.exception), "b3_trading_unavailable")
        self.assertEqual(
            PremiumExecutor._safe_failure_message("b3_trading_unavailable"),
            "Os dados de negociação da B3 não estão disponíveis para esta execução.",
        )
        self.assertFalse((self.root / "run-1" / "status-invest" / "universe-filter.json").exists())

    def test_unreadable_cotahist_fails_the_refresh(self):
        downloader = Downloader({cotahist_url(COLLECTED_ON.year): b"not a zip"})

        with self.assertRaises(B3TradingUnavailableError):
            self.run_refresh(["LIVE3"], ["LIVE11"], [], downloader=downloader)

    def test_cotahist_without_a_session_in_the_window_fails_the_refresh(self):
        with self.assertRaisesRegex(B3TradingUnavailableError, "no session"):
            self.run_refresh(["LIVE3"], ["LIVE11"], [trade("LIVE3", COLLECTED_ON - timedelta(days=40))])

    def test_cotahist_is_downloaded_once_per_day_and_reused_across_runs(self):
        lines = [trade("LIVE3", COLLECTED_ON), trade("LIVE11", COLLECTED_ON, codbdi="12")]
        downloader = Downloader({cotahist_url(COLLECTED_ON.year): cotahist_zip(lines)})

        _, first, _ = self.run_refresh(["LIVE3"], ["LIVE11"], [], workspace="run-a", downloader=downloader)
        _, second, _ = self.run_refresh(["LIVE3"], ["LIVE11"], [], workspace="run-b", downloader=downloader)
        self.assertEqual(len(downloader.calls), 1)
        self.assertEqual(first["cotahist"], second["cotahist"])

        later = COLLECTED_ON + timedelta(days=2)
        downloader.files[cotahist_url(later.year)] = cotahist_zip([trade("LIVE3", later), trade("LIVE11", later, codbdi="12")])
        self.run_refresh(["LIVE3"], ["LIVE11"], [], workspace="run-c", downloader=downloader, collected_on=later)
        self.assertEqual(len(downloader.calls), 2)
        self.assertEqual(sorted(path.name for path in (self.root / "cache").iterdir()), [later.isoformat()])

    def test_record_carries_url_sha256_and_the_filtered_csv_hashes(self):
        lines = [trade("LIVE3", COLLECTED_ON), trade("LIVE11", COLLECTED_ON, codbdi="12")]
        archive = cotahist_zip(lines)

        result, record, _ = self.run_refresh(["LIVE3", "DEAD3"], ["LIVE11", "DEAD11"], lines)

        self.assertEqual(record["collected_on"], COLLECTED_ON.isoformat())
        self.assertEqual(record["window_days"], 30)
        self.assertEqual(
            record["cotahist"],
            [{"url": cotahist_url(2026), "sha256": hashlib.sha256(archive).hexdigest(),
              "downloaded_on": COLLECTED_ON.isoformat()}],
        )
        for role, source in (("stocks", result.stocks), ("fiis", result.fiis)):
            filtered = hashlib.sha256(source.path.read_bytes()).hexdigest()
            self.assertEqual(source.sha256, filtered)
            self.assertEqual(record["csv_sha256"][role], filtered)
        self.assertEqual(record["removed"]["fiis"],
                         [{"ticker": "DEAD11", "reason": "no_recent_trade", "last_trade": "nenhum",
                           "other_codbdi_last_trade": None}])

    def test_filtered_csv_keeps_bom_line_ends_header_and_order(self):
        original = _csv_bytes(STOCK_COLUMNS, ["ZZZ3", "DEAD3", "AAA3"], bom=True)
        lines = original.split(b"\r\n")
        expected = b"\r\n".join(line for line in lines if not line.startswith(b"DEAD3,"))

        result, _, _ = self.run_refresh(
            [], ["LIVE11"],
            [trade("ZZZ3", COLLECTED_ON), trade("AAA3", COLLECTED_ON), trade("LIVE11", COLLECTED_ON, codbdi="12")],
            stock_bytes=original,
        )

        self.assertEqual(result.stocks.path.read_bytes(), expected)
        self.assertTrue(expected.startswith(codecs.BOM_UTF8))

    def test_nothing_removed_leaves_the_csv_bytes_untouched(self):
        original = _csv_bytes(STOCK_COLUMNS, ["LIVE3"])

        result, record, _ = self.run_refresh(
            [], ["LIVE11"], [trade("LIVE3", COLLECTED_ON), trade("LIVE11", COLLECTED_ON, codbdi="12")],
            stock_bytes=original,
        )

        self.assertEqual(result.stocks.path.read_bytes(), original)
        self.assertEqual(record["removed"]["stocks"], [])

    def test_cache_root_comes_from_the_environment(self):
        with patch.dict(os.environ, {COTAHIST_CACHE_ENV: str(self.root / "elsewhere")}):
            self.assertEqual(default_cache_root(), self.root / "elsewhere")
        with patch.dict(os.environ, {COTAHIST_CACHE_ENV: "", "XDG_CACHE_HOME": str(self.root / "xdg")}):
            self.assertEqual(default_cache_root(), self.root / "xdg" / "prumo" / "cotahist")


if __name__ == "__main__":
    unittest.main()
