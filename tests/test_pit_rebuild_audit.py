"""scripts/pit_rebuild_audit.py: the reproduction rules on small fixtures (no network, no cache, no Status Invest file).

The audit document states these rules (Reproduzir, Ordem das regras, Proventos (DY)); the fixtures pin them.
"""

import importlib.util
import sys
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True  # AD-013: the in-process loader of this suite writes no bytecode
_SPEC = importlib.util.spec_from_file_location("pit_rebuild_audit", REPO / "scripts" / "pit_rebuild_audit.py")
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


def _line(amount, last=date(2025, 1, 1), ex=None):
    return {"amount": amount, "last": last, "ex": ex, "file": "b3_cash_FIXTURE_p1.json"}


class ReproductionTests(unittest.TestCase):
    def test_a_value_reproduces_only_within_the_report_tolerance(self):  # CMP-01: 5 % relative, 0.5 p.p. for percentages
        self.assertTrue(audit.reproduces("P/L", 10.0, 10.4))  # 4 % relative
        self.assertFalse(audit.reproduces("P/L", 10.0, 10.6))  # 6 % relative
        self.assertTrue(audit.reproduces("ROE", 15.91, 15.98))  # 0.07 p.p.
        self.assertFalse(audit.reproduces("ROE", 15.91, 16.50))  # 0.59 p.p.
        self.assertFalse(audit.reproduces("P/L", 10.0, None))  # a view without a value reproduces nothing

    def test_the_oldest_reproducer_is_by_receipt_then_period_end_then_version(self):  # the citation rule of B-10
        a = {"dt_receb": date(2024, 5, 9), "dt": date(2024, 3, 31), "versao": 1}
        b = {"dt_receb": date(2024, 5, 9), "dt": date(2023, 12, 31), "versao": 2}
        c = {"dt_receb": date(2023, 6, 1), "dt": date(2025, 12, 31), "versao": 1}
        self.assertIs(audit.oldest([a, b, c]), c)  # the earliest receipt wins, whatever the period
        self.assertIs(audit.oldest([a, b]), b)  # same receipt: the earlier period end wins
        d = {"dt_receb": date(2024, 5, 9), "dt": date(2023, 12, 31), "versao": 1}
        self.assertIs(audit.oldest([b, d]), d)  # same receipt and period: the lower version wins


class DySubsetTests(unittest.TestCase):
    LINES = [_line(0.5), _line(0.25), _line(0.1), _line(0.7)]

    def test_a_subset_that_reproduces_the_dy_at_two_decimals_is_found(self):  # 0.5 + 0.25 + 0.1 = 0.85 on a price of 100
        found = audit.dy_search(self.LINES, 0.85, [("d", 100.0), ("d − 1", None)])
        self.assertEqual(found["d"], [0, 1, 2])
        self.assertIsNone(found["d − 1"])  # no close of d − 1: nothing to test

    def test_the_same_subset_reproduces_on_the_other_close_when_the_price_changes(self):  # 0.85 / 50 = 1.70 %
        found = audit.dy_search(self.LINES, 1.7, [("d", 100.0), ("d − 1", 50.0)])
        self.assertIsNone(found["d"])
        self.assertEqual(found["d − 1"], [0, 1, 2])

    def test_no_subset_means_no_witness(self):  # 0.3 is not a sum of these lines (0.1, 0.25, 0.5, 0.7 and their sums)
        self.assertIsNone(audit.dy_search(self.LINES, 0.3, [("d", 100.0)])["d"])

    def test_identical_lines_are_separate_items_as_the_status_invest_sums_them(self):  # two lines of 0.4 make 0.8
        found = audit.dy_search([_line(0.4), _line(0.4)], 0.8, [("d", 100.0)])
        self.assertEqual(found["d"], [0, 1])


def _cached(received):
    """A stand-in for a cached Period: the search reads only the receipt date of its filing."""
    return SimpleNamespace(filing=SimpleNamespace(dt_receb=received))


class SearchScopeTests(unittest.TestCase):
    def test_every_cached_version_received_by_R_is_read_with_no_age_limit(self):  # B-10 r4: no age mask, no STL-01
        versions = {
            "DFP": {"C1": {(date(2015, 12, 31), 1): _cached(date(2016, 3, 31)),
                           (date(2025, 12, 31), 2): _cached(date(2026, 3, 30)),
                           (date(2026, 6, 30), 1): _cached(date(2026, 11, 10))}},  # received after R: not read
            "ITR": {"C1": {(date(2026, 3, 31), 1): _cached(date(2026, 5, 10)),
                           (date(2026, 6, 30), 1): _cached(date(2026, 8, 20))}},  # received after d: read at R
        }
        self.assertEqual(audit.candidate_reads(versions, "C1", date(2026, 8, 7)), [
            ("DFP", date(2015, 12, 31), 1, date(2016, 3, 31), date(2026, 8, 7)),  # a period ended 10 years before d is read whole
            ("DFP", date(2025, 12, 31), 2, date(2026, 3, 30), date(2026, 8, 7)),
            ("ITR", date(2026, 3, 31), 1, date(2026, 5, 10), date(2026, 8, 7)),
            ("ITR", date(2026, 6, 30), 1, date(2026, 8, 20), audit.R_DATE),
        ])

    def test_a_company_without_cached_versions_has_no_reads(self):
        self.assertEqual(audit.candidate_reads({"DFP": {}, "ITR": {}}, "C1", date(2026, 8, 7)), [])


def _facts(**kw):
    facts = dict(lag=False, post=False, erro_nosso=False, b3_count=False, cited_af=False, ll_method=False,
                 pl_method=False, cited_any=False, price_two=False)
    facts.update(kw)
    return facts


class RuleOrderTests(unittest.TestCase):  # `Ordem das regras` (HEADER): the first rule whose facts hold gives the class
    def test_each_rule_beats_every_rule_below_it(self):
        cases = [
            (dict(lag=True, post=True, erro_nosso=True, b3_count=True, cited_af=True, ll_method=True), (audit.ERRO_SI, "defasagem (B-9)")),
            (dict(post=True, erro_nosso=True, cited_af=True), (audit.ERRO_SI, "post-d")),
            (dict(erro_nosso=True, cited_af=True, ll_method=True), (audit.ERRO_NOSSO, "erro nosso")),
            (dict(b3_count=True, cited_af=True, ll_method=True), (audit.ERRO_SI, "reproduzido (sobre a contagem da B3)")),
            (dict(b3_count=True, ll_method=True, pl_method=True), (audit.METODO, "contagem da B3 de R")),
            (dict(ll_method=True, pl_method=True, cited_af=True), (audit.METODO, "lucro total 3.11")),
            (dict(pl_method=True, cited_af=True), (audit.METODO, "patrimônio total 2.03")),
            (dict(cited_af=True, cited_any=True, price_two=True), (audit.ERRO_SI, "reproduzido")),
            (dict(cited_any=True, price_two=True), (audit.ERRO_SI, "reproduzido (visão de método)")),
            (dict(price_two=True), (audit.ERRO_SI, "preço d−1")),
            (dict(), (audit.ERRO_SI, "sem base")),
        ]
        for facts, expected in cases:
            with self.subTest(facts=facts):
                self.assertEqual(audit.decide(_facts(**facts)), expected)


if __name__ == "__main__":
    unittest.main()
