import csv
import hashlib
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from app.services.status_invest_inputs import (
    STATUS_INVEST_TIMEOUT_SECONDS,
    SelectorPriceError,
    SelectorSnapshotChangedError,
    SourceSnapshot,
    StatusInvestInputError,
    load_selector_prices,
    refresh_inputs as _refresh_inputs,
)
from app.services.b3_trading import CotahistCache
from fetch_status_invest import OUTPUT_COLUMNS as STOCK_COLUMNS
from fetch_status_invest_fii import OUTPUT_COLUMNS as FII_COLUMNS
from tests.cotahist_fixture import COLLECTED_ON, cotahist_zip, trade


def _write_dataset(path: Path, columns: tuple[str, ...], ticker: str = "AAA3") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow(
            {column: ticker if column == "TICKER" else "1" for column in columns}
        )


def refresh_inputs(workspace, **kwargs):
    """The real refresh_inputs with a no-network COTAHIST where AAA3 and AAA11 traded on the collection day."""
    data = cotahist_zip([trade("AAA3", COLLECTED_ON), trade("AAA11", COLLECTED_ON, codbdi="12")])
    kwargs.setdefault(
        "cotahist_cache",
        CotahistCache(Path(workspace).parent / "cotahist-cache", downloader=lambda _url: data),
    )
    kwargs.setdefault("collected_on", COLLECTED_ON)
    return _refresh_inputs(workspace, **kwargs)


class StatusInvestInputsTests(unittest.TestCase):
    def collectors(self, *, stock=None, fii=None):
        calls = []

        def collect(role, columns, action):
            def run(*, output_path, timeout):
                calls.append((role, output_path, timeout))
                if action:
                    return action(output_path)
                _write_dataset(
                    output_path, columns, "AAA3" if role == "stocks" else "AAA11"
                )

            return run

        return (
            calls,
            collect("stocks", STOCK_COLUMNS, stock),
            collect("fiis", FII_COLUMNS, fii),
        )

    def test_success_records_run_paths_hash_provider_and_utc_time(self):
        with tempfile.TemporaryDirectory() as directory:
            calls, stock, fii = self.collectors()
            result = refresh_inputs(
                Path(directory) / "run-1", stock_collector=stock, fii_collector=fii
            )

            self.assertEqual([role for role, _, _ in calls], ["stocks", "fiis"])
            self.assertTrue(
                all(
                    path.is_relative_to(Path(directory) / "run-1")
                    for _, path, _ in calls
                )
            )
            self.assertTrue(
                all(
                    timeout == STATUS_INVEST_TIMEOUT_SECONDS
                    for _, _, timeout in calls
                )
            )
            for source in (result.stocks, result.fiis):
                self.assertEqual(source.provider, "statusinvest")
                self.assertEqual(
                    source.sha256,
                    hashlib.sha256(source.path.read_bytes()).hexdigest(),
                )
                self.assertTrue(source.retrieved_at.endswith("Z"))
                retrieved_at = datetime.fromisoformat(
                    source.retrieved_at.replace("Z", "+00:00")
                )
                self.assertEqual(retrieved_at.utcoffset().total_seconds(), 0)

    def test_stock_refresh_failure_aborts_after_attempting_both_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            calls, stock, fii = self.collectors(
                stock=Mock(side_effect=RuntimeError("failed"))
            )

            with self.assertRaisesRegex(StatusInvestInputError, "stocks input failed"):
                refresh_inputs(
                    Path(directory) / "run-1", stock_collector=stock, fii_collector=fii
                )

            self.assertEqual([role for role, _, _ in calls], ["stocks", "fiis"])

    def test_fii_refresh_failure_aborts_combined_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            calls, stock, fii = self.collectors(
                fii=Mock(side_effect=RuntimeError("failed"))
            )

            with self.assertRaisesRegex(StatusInvestInputError, "fiis input failed"):
                refresh_inputs(
                    Path(directory) / "run-1", stock_collector=stock, fii_collector=fii
                )

            self.assertEqual([role for role, _, _ in calls], ["stocks", "fiis"])

    def test_empty_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, stock, fii = self.collectors(
                stock=lambda path: path.write_text("", encoding="utf-8")
            )

            with self.assertRaisesRegex(StatusInvestInputError, "stocks input failed"):
                refresh_inputs(
                    Path(directory) / "run-1", stock_collector=stock, fii_collector=fii
                )

    def test_malformed_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, stock, fii = self.collectors(
                stock=lambda path: path.write_text("TICKER\n", encoding="utf-8")
            )

            with self.assertRaisesRegex(StatusInvestInputError, "stocks input failed"):
                refresh_inputs(
                    Path(directory) / "run-1", stock_collector=stock, fii_collector=fii
                )

    def test_run_paths_are_isolated_and_canonical_files_are_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical_stock = root / "status_invest.csv"
            canonical_fii = root / "status_invest_fii.csv"
            canonical_stock.write_text("canonical stock", encoding="utf-8")
            canonical_fii.write_text("canonical fii", encoding="utf-8")
            _, first_stock, first_fii = self.collectors()
            _, second_stock, second_fii = self.collectors()

            first = refresh_inputs(
                root / "run-a", stock_collector=first_stock, fii_collector=first_fii
            )
            second = refresh_inputs(
                root / "run-b", stock_collector=second_stock, fii_collector=second_fii
            )

            self.assertNotEqual(first.stocks.path, second.stocks.path)
            self.assertNotEqual(first.fiis.path, second.fiis.path)
            self.assertEqual(first.stocks.path.parent.parent.name, "run-a")
            self.assertEqual(second.stocks.path.parent.parent.name, "run-b")
            self.assertEqual(canonical_stock.read_text(encoding="utf-8"), "canonical stock")
            self.assertEqual(canonical_fii.read_text(encoding="utf-8"), "canonical fii")


# Run 3289af79 FII selector order and PRECO (fiis.csv, retrieved 2026-11-06T12:29:37Z).
LOT_TEST_1_PRICES = (
    ("MXRF11", "8.64"),
    ("KNCR11", "102.16"),
    ("BTLG11", "97.51"),
    ("BPML11", "80.63"),
    ("GGRC11", "8.35"),
    ("ALZR11", "9.54"),
    ("XPML11", "96.00"),
    ("KNRI11", "147.89"),
    ("RECT11", "31.82"),
    ("BRCR11", "38.28"),
    ("KNUQ11", "94.98"),
    ("RBRF11", "6.53"),
    ("CPTS11", "7.00"),
)


def _write_price_csv(path: Path, rows, *, columns=("TICKER", "PRECO", "DY"), bom=False) -> SourceSnapshot:
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(row))
    data = ("\n".join(lines) + "\n").encode("utf-8")
    if bom:
        data = b"\xef\xbb\xbf" + data
    path.write_bytes(data)
    return SourceSnapshot(
        path=path,
        provider="statusinvest",
        retrieved_at="2026-11-06T12:29:37Z",
        sha256=hashlib.sha256(data).hexdigest(),
    )


class SelectorPriceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "fiis.csv"

    def test_prices_come_from_the_hashed_csv_in_cents(self):
        rows = [(ticker, price, "1.0") for ticker, price in LOT_TEST_1_PRICES]
        rows[7] = ("  knri11 ", "147.89", "1.0")
        rows.append(("HGLG11", "160.00", "1.0"))
        snapshot = _write_price_csv(self.path, rows, bom=True)
        asked = [ticker for ticker, _ in LOT_TEST_1_PRICES]

        prices = load_selector_prices(snapshot, asked)

        self.assertEqual(
            prices,
            {
                "MXRF11": 864,
                "KNCR11": 10216,
                "BTLG11": 9751,
                "BPML11": 8063,
                "GGRC11": 835,
                "ALZR11": 954,
                "XPML11": 9600,
                "KNRI11": 14789,
                "RECT11": 3182,
                "BRCR11": 3828,
                "KNUQ11": 9498,
                "RBRF11": 653,
                "CPTS11": 700,
            },
        )

    def test_changed_bytes_raise_before_parsing(self):
        original = [("KNRI11", "147.89", "1.0"), ("KNCR11", "102.16", "1.0")]
        for changed_price in ("148.00", "abc"):
            with self.subTest(changed_price=changed_price):
                snapshot = _write_price_csv(self.path, original)
                _write_price_csv(self.path, [("KNRI11", changed_price, "1.0"), original[1]])

                with self.assertRaises(SelectorSnapshotChangedError):
                    load_selector_prices(snapshot, ["KNRI11", "KNCR11"])

    def test_each_bad_price_raises_naming_the_ticker(self):
        cases = {
            "ticker row missing": ([("KNCR11", "102.16", "1.0")], ("TICKER", "PRECO", "DY")),
            "no PRECO column": ([("KNRI11", "147.89", "1.0")], ("TICKER", "COTACAO", "DY")),
            "empty": ([("KNRI11", "", "1.0")], ("TICKER", "PRECO", "DY")),
            "non-numeric": ([("KNRI11", "abc", "1.0")], ("TICKER", "PRECO", "DY")),
            "NaN": ([("KNRI11", "NaN", "1.0")], ("TICKER", "PRECO", "DY")),
            "Infinity": ([("KNRI11", "Infinity", "1.0")], ("TICKER", "PRECO", "DY")),
            "zero": ([("KNRI11", "0", "1.0")], ("TICKER", "PRECO", "DY")),
            "negative": ([("KNRI11", "-1.00", "1.0")], ("TICKER", "PRECO", "DY")),
            "conflicting duplicates": (
                [("KNRI11", "10.00", "1.0"), ("KNRI11", "11.00", "1.0")],
                ("TICKER", "PRECO", "DY"),
            ),
        }
        for label, (rows, columns) in cases.items():
            with self.subTest(case=label):
                snapshot = _write_price_csv(self.path, rows, columns=columns)

                with self.assertRaises(SelectorPriceError) as raised:
                    load_selector_prices(snapshot, ["KNRI11"])

                self.assertIn("KNRI11", str(raised.exception))

        missing = SourceSnapshot(
            path=Path(self.directory.name) / "absent.csv",
            provider="statusinvest",
            retrieved_at="2026-11-06T12:29:37Z",
            sha256="0" * 64,
        )
        with self.subTest(case="file missing"):
            with self.assertRaises(SelectorPriceError):
                load_selector_prices(missing, ["KNRI11"])

    def test_equal_duplicate_rows_are_accepted(self):
        snapshot = _write_price_csv(
            self.path, [("KNRI11", "10.00", "1.0"), ("KNRI11", "10.00", "2.0")]
        )

        self.assertEqual(load_selector_prices(snapshot, ["KNRI11"]), {"KNRI11": 1000})

    def test_price_is_quantised_half_up(self):
        snapshot = _write_price_csv(
            self.path, [("AAA11", "10.005", "1.0"), ("BBB11", "10.004", "1.0")]
        )

        self.assertEqual(
            load_selector_prices(snapshot, ["AAA11", "BBB11"]),
            {"AAA11": 1001, "BBB11": 1000},
        )

    def test_reading_prices_opens_no_socket(self):
        snapshot = _write_price_csv(self.path, [("KNRI11", "147.89", "1.0")])

        with patch("socket.socket", side_effect=AssertionError("network call")):
            prices = load_selector_prices(snapshot, ["KNRI11"])

        self.assertEqual(prices, {"KNRI11": 14789})
