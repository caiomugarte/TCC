"""scripts/pit_rebuild_audit.py: the reproduction rules on small fixtures (no network, no cache, no Status Invest file).

The audit document states these rules (Reproduzir, Ordem das regras, Proventos (DY)); the fixtures pin them.
"""

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
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

def _reproducer(received, period, versao, value, ref=10.0):
    """A reproducer of the search, as the citation reads it; exact means equal to the SI value at two decimals."""
    return {"dt_receb": received, "dt": period, "versao": versao, "value": value, "gap": abs(value - ref),
            "exact": round(value, 2) == round(ref, 2)}


class CitationTests(unittest.TestCase):  # B-10 r5: the exact reproducer received first, else the closest to the SI value
    def test_an_exact_reproducer_is_cited_before_an_older_inexact_one(self):
        inexact = _reproducer(date(2019, 5, 1), date(2018, 12, 31), 1, 10.4)  # within 5 %, received first
        exact = _reproducer(date(2024, 5, 1), date(2024, 3, 31), 1, 10.0)
        self.assertIs(audit.citation([inexact, exact]), exact)

    def test_among_exact_reproducers_the_one_received_first_is_cited(self):  # then DT_REFER, then VERSAO
        a = _reproducer(date(2024, 5, 9), date(2024, 3, 31), 1, 10.0)
        b = _reproducer(date(2024, 5, 9), date(2023, 12, 31), 2, 10.0)
        c = _reproducer(date(2023, 6, 1), date(2025, 12, 31), 1, 10.0)
        self.assertIs(audit.citation([a, b, c]), c)  # the earliest receipt wins, whatever the period
        self.assertIs(audit.citation([a, b]), b)  # same receipt: the earlier period end wins
        d = _reproducer(date(2024, 5, 9), date(2023, 12, 31), 1, 10.0)
        self.assertIs(audit.citation([b, d]), d)  # same receipt and period: the lower version wins

    def test_an_exact_reproducer_received_first_beats_a_closer_one_received_later(self):  # exact is the first key, not a distance
        older_further = _reproducer(date(2019, 5, 1), date(2018, 12, 31), 1, 10.004)  # equal at two decimals, 0.004 away
        newer_closer = _reproducer(date(2024, 5, 1), date(2024, 3, 31), 1, 10.0)  # equal, exact, received later
        self.assertIs(audit.citation([newer_closer, older_further]), older_further)

    def test_without_an_exact_reproducer_the_closest_to_the_si_value_is_cited(self):
        far = _reproducer(date(2018, 1, 1), date(2017, 12, 31), 1, 10.4)  # the oldest, 0.4 away
        near = _reproducer(date(2023, 1, 1), date(2022, 12, 31), 1, 10.2)  # 0.2 away
        self.assertIs(audit.citation([far, near]), near)

    def test_a_tie_on_the_gap_goes_to_the_oldest(self):
        later = _reproducer(date(2021, 1, 1), date(2020, 12, 31), 1, 10.25)
        older = _reproducer(date(2019, 1, 1), date(2018, 12, 31), 1, 9.75)  # the same gap, 0.25
        self.assertIs(audit.citation([later, older]), older)


class MeasureWordingTests(unittest.TestCase):  # B-10 r5 (V9): the motive states a measure, never what the SI did with a filing
    def test_an_exact_match_is_stated_as_equal_at_two_decimals(self):
        best = dict(_reproducer(date(2020, 3, 30), date(2019, 12, 31), 1, 10.0), kind="DFP", method=(), view="FULL")
        motive = audit.reproduced_motive([best], best, date(2026, 8, 7), 10.0)
        self.assertTrue(motive.startswith("valor do SI: igual a duas casas a DFP 2019-12-31 v1, recebido em 2020-03-30, "
                                          "antes de d = 2026-08-07 (10); "), motive)
        self.assertIn("1 demonstrativo(s) dentro da tolerância, 1 a duas casas; cita-se o exato recebido primeiro", motive)

    def test_a_match_within_the_tolerance_names_CMP_01_and_says_it_shows_no_origin(self):
        best = dict(_reproducer(date(2020, 3, 30), date(2019, 12, 31), 1, 10.4), kind="DFP", method=(), view="FULL")
        motive = audit.reproduced_motive([best], best, date(2026, 8, 7), 10.0)
        self.assertTrue(motive.startswith("valor do SI: dentro da tolerância CMP-01 de DFP 2019-12-31 v1, recebido em 2020-03-30, "
                                          "antes de d = 2026-08-07 (10.4 contra SI 10); a coincidência não mostra a origem; "), motive)
        self.assertNotIn("reproduz", motive)


class DyWindowTests(unittest.TestCase):  # B-10 r5: only contiguous windows of the class's records, in data-ex order
    LINES = [_line(0.5), _line(0.25), _line(0.1), _line(0.7)]

    def test_a_contiguous_window_that_reproduces_the_dy_at_two_decimals_is_found(self):  # 0.5 + 0.25 + 0.1 = 0.85 on a price of 100
        found = audit.dy_search(self.LINES, 0.85, [("d", 100.0), ("d − 1", None)])
        self.assertEqual(found["d"], (0, 2))  # lines 0 to 2, both ends included
        self.assertIsNone(found["d − 1"])  # no close of d − 1: nothing to test

    def test_the_same_window_reproduces_on_the_other_close_when_the_price_changes(self):  # 0.85 / 50 = 1.70 %
        found = audit.dy_search(self.LINES, 1.7, [("d", 100.0), ("d − 1", 50.0)])
        self.assertIsNone(found["d"])
        self.assertEqual(found["d − 1"], (0, 2))

    def test_a_subset_that_skips_a_line_is_not_a_window(self):  # 0.5 + 0.1 = 0.6 skips 0.25: no contiguous window gives 0.6
        self.assertIsNone(audit.dy_search(self.LINES, 0.6, [("d", 100.0)])["d"])

    def test_no_window_means_no_result(self):  # 0.3 is the sum of no window of these lines
        self.assertIsNone(audit.dy_search(self.LINES, 0.3, [("d", 100.0)])["d"])

    def test_identical_lines_are_separate_items_as_the_status_invest_sums_them(self):  # two lines of 0.4 make 0.8
        self.assertEqual(audit.dy_search([_line(0.4), _line(0.4)], 0.8, [("d", 100.0)])["d"], (0, 1))

    def test_the_shortest_window_is_cited_and_among_equal_lengths_the_earliest(self):
        self.assertEqual(audit.dy_search([_line(0.25), _line(0.25), _line(0.5)], 0.5, [("d", 100.0)])["d"], (2, 2))  # one line beats two
        self.assertEqual(audit.dy_search([_line(0.25)] * 4, 0.5, [("d", 100.0)])["d"], (0, 1))  # three windows of two: the first


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
    facts = dict(lag=False, post=False, erro_nosso=False, b3_count=False, cited_af=False, exact_af=False,
                 ll_method=False, pl_method=False, cited_any=False, price_two=False)
    facts.update(kw)
    return facts


class RuleOrderTests(unittest.TestCase):  # `Ordem das regras` (HEADER): the first rule whose facts hold gives the class
    def test_each_rule_beats_every_rule_below_it(self):
        cases = [
            (dict(lag=True, post=True, erro_nosso=True, b3_count=True, cited_af=True, ll_method=True), (audit.ERRO_SI, "defasagem (B-9)")),
            (dict(post=True, erro_nosso=True, cited_af=True), (audit.ERRO_SI, "post-d")),
            (dict(erro_nosso=True, cited_af=True, ll_method=True), (audit.ERRO_NOSSO, "erro nosso")),
            (dict(b3_count=True, cited_af=True, exact_af=True, ll_method=True), (audit.ERRO_SI, "reproduzido (sobre a contagem da B3)")),
            (dict(b3_count=True, cited_af=True, ll_method=True), (audit.METODO, "contagem da B3 de R")),
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

    def test_the_b3_total_count_yields_to_a_demonstrativo_only_on_an_exact_match(self):  # B-10 r5, rule 6
        self.assertEqual(audit.decide(_facts(b3_count=True, cited_af=True, exact_af=True)),
                         (audit.ERRO_SI, "reproduzido (sobre a contagem da B3)"))
        self.assertEqual(audit.decide(_facts(b3_count=True, cited_af=True, exact_af=False)),  # within the tolerance only
                         (audit.METODO, "contagem da B3 de R"))
        self.assertEqual(audit.decide(_facts(b3_count=True)), (audit.METODO, "contagem da B3 de R"))


class GeneratorWritesNoBytecodeTests(unittest.TestCase):  # V12: the generator writes only its --out file, no bytecode beside the pit package
    def test_a_run_of_the_generator_writes_no_bytecode_into_the_repository(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(REPO / "py", root / "py", ignore=shutil.ignore_patterns("__pycache__"))
            (root / "scripts").mkdir()
            shutil.copy(REPO / "scripts" / "pit_rebuild_audit.py", root / "scripts" / "pit_rebuild_audit.py")
            env = {key: value for key, value in os.environ.items() if key != "PYTHONDONTWRITEBYTECODE"}  # the run must not rely on it
            subprocess.run([sys.executable, str(root / "scripts" / "pit_rebuild_audit.py"), "--help"], check=True, env=env, capture_output=True)
            self.assertEqual(sorted(root.rglob("*.pyc")), [])


if __name__ == "__main__":
    unittest.main()
