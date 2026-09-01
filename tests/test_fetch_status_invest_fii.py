import csv
from io import StringIO
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))

from fetch_status_invest_fii import (  # noqa: E402
    OUTPUT_COLUMNS,
    SOURCE_COLUMNS,
    StatusInvestFiiFetchError,
    fetch_segment,
    parse_segment_csv,
    validate_rows,
    write_dataset,
)


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.payload


class FiiFetchTests(unittest.TestCase):
    def _payload(self, **values):
        columns = list(SOURCE_COLUMNS)
        alias = "CAGR VALOR COTA 3 ANOS"
        columns[columns.index(alias)] = " CAGR VALOR CORA 3 ANOS"
        row = {column: "" for column in SOURCE_COLUMNS}
        row.update(values)
        output = StringIO()
        writer = csv.writer(output, delimiter=";")
        writer.writerow(columns)
        writer.writerow([
            row["CAGR VALOR COTA 3 ANOS"]
            if "CORA" in column
            else row[column.strip()]
            for column in columns
        ])
        return output.getvalue()

    def test_parse_normalizes_alias_numeric_values_and_segment(self):
        rows = parse_segment_csv(
            self._payload(
                TICKER="bbrc11",
                PRECO="103,45",
                **{
                    "CAGR VALOR COTA 3 ANOS": "9,06",
                    "GESTAO": "Ativa",
                    "PATRIMONIO": "165.920.780,61",
                },
            ),
            "Agências de Bancos",
        )

        self.assertEqual(rows[0]["TICKER"], "BBRC11")
        self.assertEqual(rows[0]["PRECO"], "103.45")
        self.assertEqual(rows[0]["CAGR VALOR COTA 3 ANOS"], "9.06")
        self.assertEqual(rows[0]["GESTAO"], "Ativa")
        self.assertEqual(rows[0]["PATRIMONIO"], "165920780.61")
        self.assertEqual(rows[0]["SETOR"], "Agências de Bancos")

    def test_fetch_uses_fii_segment_filter_and_category_type(self):
        payload = self._payload(TICKER="BBRC11")
        with patch(
            "fetch_status_invest_fii.urlopen",
            return_value=_Response(payload.encode()),
        ) as urlopen:
            rows = fetch_segment(91, "Agências de Bancos")

        url = urlopen.call_args.args[0].full_url
        self.assertIn("CategoryType=2", url)
        self.assertIn("Segment%22%3A%2291", url)
        self.assertEqual(rows[0]["SETOR"], "Agências de Bancos")

    def test_duplicate_tickers_are_rejected(self):
        row = {column: "" for column in OUTPUT_COLUMNS}
        row.update({"TICKER": "BBRC11", "SETOR": "Agências de Bancos"})
        with self.assertRaisesRegex(StatusInvestFiiFetchError, "duplicate FII tickers"):
            validate_rows([row, dict(row)])

    def test_invalid_dataset_does_not_replace_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "status_invest_fii.csv"
            output.write_text("old dataset\n", encoding="utf-8")

            with self.assertRaises(StatusInvestFiiFetchError):
                write_dataset([{"TICKER": "BBRC11"}], output)

            self.assertEqual(output.read_text(encoding="utf-8"), "old dataset\n")


if __name__ == "__main__":
    unittest.main()
