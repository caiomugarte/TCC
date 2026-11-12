"""The field list and tolerance classes of the rebuild (IND-02, CMP-01, A-2)."""

import ast
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))

from pit_rebuild import fields  # noqa: E402

CONFIG = PROJECT_ROOT / "py" / "config.py"
A2_PERCENT_FIELDS = {
    "DY",
    "ROE",
    "ROA",
    "ROIC",
    "MARG. LIQUIDA",
    "MARGEM EBIT",
    "CAGR RECEITAS 5 ANOS",
    "CAGR LUCROS 5 ANOS",
}


def config_groups(path):
    """Read METRIC_GROUPS of the config with its own ast walk, independent of fields.py."""
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "METRIC_GROUPS":
            return ast.literal_eval(node.value)
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "METRIC_GROUPS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("no METRIC_GROUPS assignment")


class FieldListTests(unittest.TestCase):
    def test_score_fields_are_the_16_metric_group_names_in_group_order(self):
        expected = [field for group in config_groups(CONFIG).values() for field in group]

        self.assertEqual(len(expected), 16)
        self.assertEqual(fields.SCORE_FIELDS, tuple(expected))

    def test_all_fields_add_the_three_extra_fields_to_19(self):
        self.assertEqual(fields.EXTRA_FIELDS, ("PRECO", "VALOR DE MERCADO", "LIQUIDEZ MEDIA DIARIA"))
        self.assertEqual(fields.ALL_FIELDS, fields.SCORE_FIELDS + fields.EXTRA_FIELDS)
        self.assertEqual(len(fields.ALL_FIELDS), 19)

    def test_a_config_with_an_extra_metric_yields_17_score_fields(self):
        groups = config_groups(CONFIG)
        groups["div"] = list(groups["div"]) + ["EXTRA METRIC"]
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.py"
            config.write_text("METRIC_GROUPS = " + repr(groups) + "\n", encoding="utf-8")

            flattened = [field for group in fields.metric_groups(config).values() for field in group]

        self.assertEqual(len(flattened), 17)
        self.assertIn("EXTRA METRIC", flattened)

    def test_importing_the_package_does_not_import_config(self):
        code = (
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "import pit_rebuild.fields; print('config' in sys.modules)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code, str(PROJECT_ROOT / "py")],
            capture_output=True,
            text=True,
            check=True,
        )

        self.assertEqual(result.stdout.strip(), "False")


class ToleranceClassTests(unittest.TestCase):
    def test_pp_tolerance_applies_to_exactly_the_eight_a2_fields(self):
        pp = {field for field in fields.ALL_FIELDS if fields.tolerance_kind(field) == "pp"}

        self.assertEqual(pp, A2_PERCENT_FIELDS)
        self.assertEqual(len(fields.ALL_FIELDS) - len(pp), 11)
        self.assertEqual(fields.tolerance_kind("P/L"), "relative")

    def test_unknown_field_is_refused_rather_than_given_a_default_tolerance(self):
        with self.assertRaises(ValueError):
            fields.tolerance_kind("DYY")


class StatementAndAdjustedFieldTests(unittest.TestCase):
    """B-7 and B-8: the 15 fields that read a CVM statement value, and the 14 fields of the adjusted verdict."""

    def test_the_accounting_fields_are_all_fields_less_the_four_market_fields_and_there_are_15(self):
        self.assertEqual(
            set(fields.ALL_FIELDS) - set(fields.ACCOUNTING_FIELDS),
            {"PRECO", "VALOR DE MERCADO", "LIQUIDEZ MEDIA DIARIA", "DY"},
        )
        self.assertEqual(len(fields.ACCOUNTING_FIELDS), 15)

    def test_the_accounting_fields_keep_the_all_fields_order(self):
        positions = [fields.ALL_FIELDS.index(field) for field in fields.ACCOUNTING_FIELDS]
        self.assertEqual(positions, sorted(positions))

    def test_the_adjusted_fields_are_the_score_fields_less_roic_and_peg_ratio_and_there_are_14(self):
        self.assertEqual(set(fields.SCORE_FIELDS) - set(fields.ADJUSTED_FIELDS), {"ROIC", "PEG RATIO"})
        self.assertLessEqual(set(fields.ADJUSTED_FIELDS), set(fields.SCORE_FIELDS))
        self.assertEqual(len(fields.ADJUSTED_FIELDS), 14)

    def test_the_adjusted_fields_keep_the_score_fields_order(self):
        positions = [fields.SCORE_FIELDS.index(field) for field in fields.ADJUSTED_FIELDS]
        self.assertEqual(positions, sorted(positions))


if __name__ == "__main__":
    unittest.main()
