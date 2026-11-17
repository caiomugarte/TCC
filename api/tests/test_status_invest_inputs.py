import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from snapshot_manifest import SnapshotChangedError, SnapshotManifest

from tests.test_premium_snapshot_only import build_snapshot

from app.services.status_invest_inputs import (
    SelectorPriceError,
    load_snapshot_inputs,
    SourceSnapshot,
    load_selector_prices,
)


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

                with self.assertRaises(SnapshotChangedError):
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


class SnapshotInputHashTests(unittest.TestCase):
    def test_selector_bytes_changed_after_the_manifest_loaded_raise_snapshot_changed(self):
        for name in ("stocks.csv", "fiis.csv"):
            with self.subTest(source=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = SnapshotManifest.load(build_snapshot(root / "registry"))
                path = root / "registry" / name
                # Still a valid selector CSV, so only the hash check can refuse it.
                path.write_bytes(path.read_bytes().replace(b"10.00", b"11.00"))
                workspace = root / "workspace"

                with self.assertRaisesRegex(SnapshotChangedError, "SHA-256 mismatch"):
                    load_snapshot_inputs(manifest, workspace)

                self.assertFalse((workspace / "snapshot" / name).exists())
