import csv
import hashlib
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

from app.services.status_invest_inputs import (
    STATUS_INVEST_TIMEOUT_SECONDS,
    StatusInvestInputError,
    refresh_inputs,
)
from fetch_status_invest import OUTPUT_COLUMNS as STOCK_COLUMNS
from fetch_status_invest_fii import OUTPUT_COLUMNS as FII_COLUMNS


def _write_dataset(path: Path, columns: tuple[str, ...], ticker: str = "AAA3") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow(
            {column: ticker if column == "TICKER" else "1" for column in columns}
        )


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
