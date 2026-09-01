import csv
from io import StringIO
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))

from fetch_status_invest import (  # noqa: E402
    SOURCE_COLUMNS,
    StatusInvestFetchError,
    parse_sector_csv,
    write_dataset,
)


class StatusInvestSyncTests(unittest.TestCase):
    def _payload(self, **values):
        row = {column: "" for column in SOURCE_COLUMNS}
        row.update(values)
        output = StringIO()
        writer = csv.writer(output, delimiter=";")
        writer.writerow(SOURCE_COLUMNS)
        writer.writerow([row[column] for column in SOURCE_COLUMNS])
        return output.getvalue()

    def test_parse_normalizes_brazilian_values_and_adds_sector(self):
        rows = parse_sector_csv(
            self._payload(
                TICKER="aeri3",
                PRECO="1.234,56",
                **{"VALOR DE MERCADO": "9.876.543,21"},
            ),
            "Bens Industriais",
        )

        self.assertEqual(rows[0]["TICKER"], "AERI3")
        self.assertEqual(rows[0]["PRECO"], "1234.56")
        self.assertEqual(rows[0]["VALOR DE MERCADO"], "9876543.21")
        self.assertEqual(rows[0]["SETOR"], "Bens Industriais")

    def test_missing_column_is_rejected(self):
        columns = [column for column in SOURCE_COLUMNS if column != "DY"]
        payload = ";".join(columns) + "\n" + ";".join("" for _ in columns)

        with self.assertRaisesRegex(StatusInvestFetchError, "missing columns: DY"):
            parse_sector_csv(payload, "Bens Industriais")

    def test_empty_payload_is_rejected(self):
        with self.assertRaisesRegex(StatusInvestFetchError, "empty CSV"):
            parse_sector_csv("", "Bens Industriais")

    def test_invalid_dataset_does_not_replace_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "status_invest_fundamentals.csv"
            output.write_text("old dataset\n", encoding="utf-8")

            with self.assertRaises(StatusInvestFetchError):
                write_dataset([{"TICKER": "AERI3"}], output)

            self.assertEqual(output.read_text(encoding="utf-8"), "old dataset\n")


if __name__ == "__main__":
    unittest.main()
