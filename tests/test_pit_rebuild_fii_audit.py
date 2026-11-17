"""Classification of the worst FII cases from the evidence the sources hold (issue #60). No network."""

from datetime import date, timedelta
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from pit_rebuild import fii_audit, fii_indicators  # noqa: E402
from pit_rebuild.dividends import Dividend  # noqa: E402
from test_pit_rebuild_fii_indicators import A, B3, D, REPORTS, YAHOO, informe, informes, quotes  # noqa: E402

SI_ERROR = "erro do Status Invest"
METHOD = "diferença de método"
OURS = "erro nosso"
OPEN = "sem classificação"


class ClassifyTest(unittest.TestCase):
    def context(self, reports=REPORTS, b3=B3, yahoo=YAHOO):
        q = quotes()
        store = informes(*reports)
        rebuilt = fii_indicators.rebuild("ALFA11", D, q, store, b3, yahoo)
        return fii_audit.Context(rebuilt=rebuilt, informes=store, quotes=q, b3_fund=b3, yahoo=yahoo)

    def test_a_reference_equal_to_an_older_report_known_at_d_is_a_status_invest_lag(self):
        reports = REPORTS + (informe(A, date(2026, 5, 1), date(2026, 6, 15), nav=130.0, equity=1_300_000_000, shares=10_000_000, holders=47000),)
        case = fii_audit.classify("N COTISTAS", 47000, self.context(reports))
        self.assertEqual(case["class"], SI_ERROR)
        self.assertEqual(case["evidence"], "informe 2026-05 v1, entregue em 2026-06-15")

    def test_a_reference_equal_to_a_report_delivered_after_d_is_a_method_difference(self):
        case = fii_audit.classify("PATRIMONIO", 1_400_000_000, self.context())  # the 2026-07 report, delivered 2026-08-14
        self.assertEqual(case["class"], METHOD)
        self.assertEqual(case["evidence"], "informe 2026-07 v1, entregue em 2026-08-14")

    def test_a_price_equal_to_the_close_of_another_session_is_a_method_difference(self):
        case = fii_audit.classify("PRECO", 120.0, self.context())  # the close of every session before D
        self.assertEqual(case["class"], METHOD)
        self.assertIn((D - timedelta(days=1)).isoformat(), case["evidence"])

    def test_a_dividend_value_equal_to_the_other_source_is_a_method_difference(self):
        # DY from Yahoo is 12 / 121; the nine B3 records in the window give 9 / 121 = 7.44 %
        case = fii_audit.classify("DY", 7.44, self.context())
        self.assertEqual(case["class"], METHOD)
        self.assertIn("b3", case["evidence"])

    def test_a_reference_equal_to_the_other_fund_of_the_isin_is_our_error(self):
        decoy = informe("22.222.222/0001-22", date(2026, 5, 1), date(2026, 6, 15), nav=22000.0, equity=5e9, shares=224_952, holders=11)
        case = fii_audit.classify("N COTISTAS", 11, self.context(REPORTS + (decoy,)))
        self.assertEqual(case["class"], OURS)

    def test_no_evidence_leaves_the_case_open(self):
        case = fii_audit.classify("N COTISTAS", 12345, self.context())
        self.assertEqual(case["class"], OPEN)
        self.assertEqual(case["evidence"], "informe 2026-06 v1, entregue em 2026-07-15")


if __name__ == "__main__":
    unittest.main()
