"""The FII reference CSV, its eligible universe and the comparison rules (issue #60). No network."""

from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))

from pit_rebuild import compare, fii_compare, fii_status_invest  # noqa: E402
from pit_rebuild.errors import IntegrityError, MissingPathError  # noqa: E402

FIXTURES = PROJECT_ROOT / "tests" / "fixtures" / "pit_rebuild_fii"


class StatusInvestFiiTest(unittest.TestCase):
    def test_the_universe_is_the_selector_core_rule_with_the_widest_premium_floor(self):
        si = fii_status_invest.load(FIXTURES / "si_2026-08-10.csv")
        # SMAL11 has PATRIMONIO below 200 M; NODY11 has no DY
        self.assertEqual(fii_status_invest.eligible(si), ["ALFA11", "AMBI11", "BETA11", "NOCV11"])

    def test_the_universe_matches_the_production_selector(self):
        try:
            from fii_selection import apply_fii_eligibility, load_fii_data
        except ImportError:  # pandas is not installed in this interpreter
            self.skipTest("pandas is not available")
        frame = apply_fii_eligibility(load_fii_data(FIXTURES / "si_2026-08-10.csv"), filters=fii_status_invest.FLOOR)
        si = fii_status_invest.load(FIXTURES / "si_2026-08-10.csv")
        self.assertEqual(sorted(frame["TICKER"]), fii_status_invest.eligible(si))

    def test_text_cells_are_kept_and_numbers_parsed(self):
        si = fii_status_invest.load(FIXTURES / "si_2026-11-05.csv")
        self.assertEqual(si.text["BETA11"], {"GESTAO": "Passiva", "SETOR": "Logística"})
        self.assertEqual(si.rows["ALFA11"]["N COTISTAS"], 52000.0)
        self.assertIsNone(si.rows["BETA11"]["CAGR DIVIDENDOS 3 ANOS"])

    def test_a_missing_file_a_missing_column_or_a_repeated_ticker_is_refused(self):
        with self.assertRaises(MissingPathError):
            fii_status_invest.load(FIXTURES / "absent.csv")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "si.csv"
            path.write_text("TICKER,PRECO\nALFA11,1\n", encoding="utf-8")
            with self.assertRaises(IntegrityError):
                fii_status_invest.load(path)
            text = (FIXTURES / "si_2026-11-05.csv").read_text(encoding="utf-8")
            path.write_text(text + text.splitlines()[1] + "\n", encoding="utf-8")
            with self.assertRaises(IntegrityError):
                fii_status_invest.load(path)


class ToleranceTest(unittest.TestCase):
    def test_percent_indicators_use_half_a_point_and_the_others_five_percent(self):
        self.assertTrue(fii_compare.within("DY", 9.92, 10.42))
        self.assertFalse(fii_compare.within("DY", 9.92, 10.43))
        self.assertTrue(fii_compare.within("PERCENTUAL EM CAIXA", 5.0, 5.5))
        self.assertTrue(fii_compare.within("P/VP", 1.00, 1.05))
        self.assertFalse(fii_compare.within("P/VP", 1.00, 1.06))
        self.assertTrue(fii_compare.within("N COTISTAS", 50000, 52500))
        self.assertFalse(fii_compare.within("N COTISTAS", 50000, 52501))

    def test_summary_counts_the_denominator_the_misses_and_the_worst_cases(self):
        stats = fii_compare.summarize("P/VP", [
            ("A", 1.0, 1.0, None), ("B", 1.0, 1.2, None), ("C", 1.0, None, "no_cvm_match"), ("D", None, 0.9, None),
        ])
        self.assertEqual((stats.denominator, stats.within, stats.reference_blank), (3, 1, 1))
        self.assertAlmostEqual(stats.pct_within, 100 / 3)
        self.assertEqual(stats.missing, [{"ticker": "C", "reason": "no_cvm_match"}])
        self.assertEqual([item["ticker"] for item in stats.worst], ["B", "A"])
        self.assertAlmostEqual(stats.worst[0]["diff"], 20.0)
        self.assertEqual(compare.verdict(stats.pct_within, stats.denominator), compare.NAO_RECONSTRUIVEL)


class CategoryTest(unittest.TestCase):
    def test_segments_are_compared_after_the_fixed_mapping(self):
        stats = fii_compare.summarize_categories("SETOR", [
            ("A", "Shoppings", "Shoppings", None),
            ("B", "Logística", "Logística", None),
            ("C", "Imóveis Industriais e Logísticos", "Logística", None),
            ("D", "Papéis", "Multicategoria", None),
            ("E", "Papéis", None, "no_cvm_match"),
        ])
        self.assertEqual((stats.denominator, stats.within), (5, 3))
        self.assertAlmostEqual(stats.pct_within, 60.0)
        self.assertEqual(stats.crosstab[("Logística", "Imóveis Industriais e Logísticos")], 1)
        self.assertEqual(stats.mismatches, [{"ticker": "D", "reference": "Papéis", "rebuilt": "Multicategoria"}])
        self.assertEqual(stats.missing, [{"ticker": "E", "reason": "no_cvm_match"}])

    def test_management_maps_definida_to_passiva_and_ignores_a_blank_reference(self):
        stats = fii_compare.summarize_categories("GESTAO", [
            ("A", "Ativa", "Ativa", None), ("B", "Passiva", "Definida", None), ("C", "", "Ativa", None),
        ])
        self.assertEqual((stats.denominator, stats.within, stats.reference_blank), (2, 2, 1))


if __name__ == "__main__":
    unittest.main()
