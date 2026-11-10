"""Earliest years, DY coverage, the full-rebuild years and the effort estimate (COV-01, COV-02, COV-03, A-13).

tests/fixtures/pit_rebuild/listing_dfp.html is a DFP directory listing in the nginx autoindex
shape, 2010 to 2026. The ITR and FCA listings of the tests are built the same way from the years
given. Effort numbers are the hand-computed ones of the spec: a load of 10 s for 100 MB, a full
rebuild of 1,600 MB, and 5 s of compute over 2 dates.
"""

import sys
import unittest
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))

from pit_rebuild import coverage  # noqa: E402

FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "pit_rebuild" / "listing_dfp.html"


def listing(source: str, first_year: int) -> bytes:
    rows = "\n".join(f'<a href="{source}_cia_aberta_{year}.zip">{source}_cia_aberta_{year}.zip</a>' for year in range(first_year, 2027))
    return f"<html><body><pre>\n{rows}\n</pre></body></html>".encode("utf-8")


def probe(first_present: int) -> dict[int, bool]:
    return {year: year >= first_present for year in range(2005, 2013)}


SOURCES_EARLIEST = {"DFP": 2010, "ITR": 2011, "FCA": 2010, "COTAHIST": 2005}


class EarliestYearTests(unittest.TestCase):
    def test_each_listing_gives_its_first_year_and_the_fixture_starts_in_2010(self):
        years = coverage.earliest_years(
            {"DFP": FIXTURE.read_bytes(), "ITR": listing("itr", 2011), "FCA": listing("fca", 2010)},
            probe(2006),
        )
        self.assertEqual(years["DFP"], {"year": 2010, "at_probe_floor": False})
        self.assertEqual(years["ITR"], {"year": 2011, "at_probe_floor": False})
        self.assertEqual(years["FCA"], {"year": 2010, "at_probe_floor": False})

    def test_cotahist_is_reported_from_the_first_probe_year_present_and_never_an_earlier_one(self):
        self.assertEqual(coverage.earliest_years({}, probe(2006))["COTAHIST"], {"year": 2006, "at_probe_floor": False})

    def test_cotahist_present_in_2005_is_reported_as_the_probe_floor(self):
        self.assertEqual(coverage.earliest_years({}, probe(2005))["COTAHIST"], {"year": 2005, "at_probe_floor": True})

    def test_roe_is_rebuildable_one_year_after_the_later_of_dfp_and_itr(self):
        self.assertEqual(coverage.field_earliest("ROE", SOURCES_EARLIEST), 2012)

    def test_cagr_needs_five_fiscal_years_peg_two_years_and_balance_or_price_fields_none(self):
        self.assertEqual(coverage.field_earliest("CAGR RECEITAS 5 ANOS", SOURCES_EARLIEST), 2015)
        self.assertEqual(coverage.field_earliest("CAGR LUCROS 5 ANOS", SOURCES_EARLIEST), 2015)
        self.assertEqual(coverage.field_earliest("PEG RATIO", SOURCES_EARLIEST), 2013)
        # P/VP: the later of DFP and ITR (2011) and COTAHIST (2005), no lookback.
        self.assertEqual(coverage.field_earliest("P/VP", SOURCES_EARLIEST), 2011)

    def test_a_field_whose_source_year_is_unknown_has_no_earliest_year(self):
        self.assertIsNone(coverage.field_earliest("P/L", {"COTAHIST": 2005, "DFP": None, "ITR": 2011}))


class DyAndRebuildYearsTests(unittest.TestCase):
    def test_dy_takes_its_earliest_year_from_b3_when_any_ticker_used_b3_and_from_yahoo_otherwise(self):
        with_b3 = coverage.dy_coverage(
            {"b3": {"PETR4": date(2016, 3, 1)}, "yahoo": {"ABCD3": date(2000, 6, 1), "EFGH3": date(2001, 6, 1)}}
        )
        self.assertEqual((with_b3["earliest_year_from"], with_b3["earliest_year"]), ("b3", 2016))
        self.assertEqual(with_b3["by_source"]["b3"]["retention"], coverage.RETENTION["b3"])
        without_b3 = coverage.dy_coverage(
            {"b3": {}, "yahoo": {"ABCD3": date(2003, 6, 1), "EFGH3": date(2005, 6, 1)}}
        )
        self.assertEqual((without_b3["earliest_year_from"], without_b3["earliest_year"]), ("yahoo", 2003))
        self.assertEqual(without_b3["by_source"]["yahoo"]["retention"], coverage.RETENTION["yahoo"])

    def test_b3_tickers_give_their_earliest_raw_date_the_year_of_the_earliest_and_the_median_of_the_years(self):
        # The dates are reported as given: 1996-03-22 is not cut by d or moved to a session.
        result = coverage.dy_coverage(
            {"b3": {"PETR4": date(1996, 3, 22), "VALE3": date(2003, 5, 2), "TAEE11": date(2006, 7, 1)}, "yahoo": {}}
        )
        b3 = result["by_source"]["b3"]
        self.assertEqual(
            (b3["tickers"], b3["earliest_raw_date"], b3["earliest_year"], b3["median_ticker_year"]),
            (3, "1996-03-22", 1996, 2003),
        )
        self.assertEqual((result["earliest_year_from"], result["earliest_year"]), ("b3", 1996))

    def test_two_tickers_give_the_median_of_their_years_between_the_two(self):
        result = coverage.dy_coverage({"b3": {"ALFA3": date(2003, 5, 2), "BETA3": date(2006, 7, 1)}, "yahoo": {}})
        self.assertEqual(result["by_source"]["b3"]["median_ticker_year"], 2004.5)

    def test_a_source_with_no_ticker_has_no_dates_and_no_median(self):
        result = coverage.dy_coverage({"b3": {}, "yahoo": {}})
        self.assertEqual(
            result["by_source"]["b3"],
            {"tickers": 0, "earliest_raw_date": None, "earliest_year": None, "median_ticker_year": None,
             "retention": coverage.RETENTION["b3"]},
        )
        self.assertEqual((result["earliest_year_from"], result["earliest_year"]), (None, None))

    def test_full_rebuild_years_follow_the_src_01_rule_clipped_to_each_source(self):
        self.assertEqual(
            coverage.full_rebuild_years(SOURCES_EARLIEST),
            {"DFP": (2010, 2026), "ITR": (2011, 2026), "FCA": (2010, 2026), "COTAHIST": (2011, 2026)},
        )


class EffortTests(unittest.TestCase):
    COTAHIST_SIZES = {"https://example.test/COTAHIST_A2011.ZIP": 800_000_000, "https://example.test/COTAHIST_A2012.ZIP": 800_000_000}

    def test_the_load_is_projected_by_bytes_and_only_the_compute_is_multiplied_by_180(self):
        result = coverage.effort({"COTAHIST": (10.0, 100_000_000)}, 5.0, 2, self.COTAHIST_SIZES, lambda url: "COTAHIST")
        self.assertEqual(result["decision_dates"], 180)
        self.assertAlmostEqual(result["load_projected_seconds"], 160.0)  # 10 s / 100 MB × 1,600 MB
        self.assertAlmostEqual(result["compute_seconds_per_date"], 2.5)
        self.assertAlmostEqual(result["compute_projected_seconds"], 450.0)  # 2.5 s × 180
        self.assertAlmostEqual(result["total_projected_seconds"], 610.0)

    def test_a_kind_measured_with_no_bytes_gives_no_total(self):
        result = coverage.effort(
            {"DFP": (5.0, 0), "COTAHIST": (10.0, 100_000_000)},
            5.0,
            2,
            self.COTAHIST_SIZES,
            lambda url: "COTAHIST",
        )
        self.assertIsNone(result["total_projected_seconds"])
        self.assertIsNone(result["load"]["DFP"]["projected_seconds"])

    def test_the_decision_dates_are_the_180_months_from_2012_01_to_2026_12(self):
        self.assertEqual(coverage.DECISION_DATES, 180)


if __name__ == "__main__":
    unittest.main()
