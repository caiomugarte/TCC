"""The report object and its Markdown: provenance, parameters, mapping, dividends, coverage and determinism (RPT-02, RPT-03, RPT-04, VER-03, VER-04, MAP-02, COV-02, IND-04).

The sample run has two decision dates. Its non-financial group holds two tickers within tolerance
and two unclassified tickers with no rebuilt value, so the class verdict with them is "não
reconstruível" and without them "reconstruível". The financial group is all within tolerance.
The values are chosen so each verdict is easy to check by hand; they are not market data.
"""

from collections import Counter
from copy import deepcopy
from datetime import date
import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))

from pit_rebuild import compare, coverage, report  # noqa: E402
from pit_rebuild.fields import ALL_FIELDS  # noqa: E402

BASE = [("AAAA3", 10.0, 10.0, None), ("BBBB3", 20.0, 20.0, None)]
UNCLASSIFIED = [("UUUU3", 5.0, None, "no_quote"), ("VVVV3", 5.0, None, "no_quote")]
FINANCIAL = [("EEEE3", 1.0, 1.0, None)]
MAPPING = {
    "mapped": 4,
    "unmapped": [{"ticker": "DDDD3", "reason": "no_cvm_code"}],
    "refiled_after_d": ["RRRR3"],
    "mapped_via_b3": ["USIM6"],
    "vm_price_fallback": {
        "count": 1,
        "companies": [{"cd_cvm": "1603", "class_without_quote": "PREF", "price_from": "ON"}],
    },
    "share_scale": {"x1000": ["1603"], "unverified": ["1604"]},
    "financial": [{"ticker": "EEEE3", "rule": "setor_ativ"}],
    "unclassified": [{"ticker": "UUUU3", "si_setor": "Financeiro e Outros", "reason": "not_in_fca_or_b3"}],
}
RAW = [
    {
        "url": "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/DFP/DADOS/dfp_cia_aberta_2025.zip",
        "name": "dfp_cia_aberta_2025.zip",
        "sha256": "b" * 64,
        "bytes": 1234,
    }
]
STATUS_INVEST = [
    {"date": "2026-08-07", "path": "data/raw/status_invest_fundamentals.csv", "sha256": "a" * 64},
    {"date": "2026-11-06", "path": "snapshots/data/selectors-2026-11-06/stocks.csv", "sha256": "c" * 64},
]
DIVIDENDS = {
    "by_source": {"2026-08-07": {"b3": 2, "yahoo": 1, "none": 0}, "2026-11-06": {"b3": 3, "yahoo": 0, "none": 0}},
    "yahoo_reasons": {
        "2026-08-07": {"b3_failed": 0, "b3_empty": 0, "b3_incomplete": 1, "b3_short_history": 0},
        "2026-11-06": {"b3_failed": 0, "b3_empty": 0, "b3_incomplete": 0, "b3_short_history": 0},
    },
    "b3_empty_yahoo_failed": {"2026-08-07": 1, "2026-11-06": 0},
}


def pairs_for(field: str, group: str, unclassified=UNCLASSIFIED):
    blank = ("CCCC3", None, 0.0, None) if field == "DY" else ("CCCC3", None, 5.0, None)
    if group == "financial":
        return FINANCIAL
    if group == "non_financial":
        return BASE + list(unclassified) + [blank]
    return BASE + [blank]


def per_date_entry(eligible: int, unclassified=UNCLASSIFIED) -> dict:
    fields = {}
    for field in ALL_FIELDS:
        fields[field] = {
            "non_financial": compare.summarize(field, pairs_for(field, "non_financial", unclassified)),
            "non_financial_without_unclassified": compare.summarize(field, pairs_for(field, "without", unclassified)),
            "financial": compare.summarize(field, pairs_for(field, "financial", unclassified)),
        }
    return {"eligible": eligible, "mapping": MAPPING, "fields": fields}


def coverage_block() -> dict:
    return {
        "earliest_year_by_source": {
            "DFP": {"year": 2010, "at_probe_floor": False},
            "COTAHIST": {"year": 2005, "at_probe_floor": True},
        },
        "earliest_year_by_field": {"ROE": 2012, "P/VP": 2011},
        "dy": coverage.dy_coverage(
            {
                "b3": {"tickers": 2, "earliest_ex_date": date(2016, 1, 4)},
                "yahoo": {"tickers": 1, "earliest_ex_date": date(2000, 6, 1)},
            }
        ),
        "effort": coverage.effort(
            {"COTAHIST": (10.0, 100_000_000)},
            5.0,
            2,
            {
                "https://example.test/COTAHIST_A2011.ZIP": 800_000_000,
                "https://example.test/COTAHIST_A2012.ZIP": 800_000_000,
            },
            lambda url: "COTAHIST",
        ),
        "years": coverage.full_rebuild_years({"DFP": 2010, "ITR": 2011, "FCA": 2010, "COTAHIST": 2005}),
    }


def run_kwargs(**overrides) -> dict:
    kwargs = {
        "dates": [date(2026, 8, 7), date(2026, 11, 6)],
        "tool_commit": "abc1234",
        "status_invest": STATUS_INVEST,
        "raw_files": RAW,
        "per_date": {"2026-08-07": per_date_entry(3), "2026-11-06": per_date_entry(5)},
        "dividends": DIVIDENDS,
        "coverage": coverage_block(),
    }
    kwargs.update(overrides)
    return kwargs


class ReportCase(unittest.TestCase):
    def setUp(self):
        self.report = report.build(**run_kwargs())
        self.markdown = report.render_markdown(self.report)


class ProvenanceAndParametersTests(ReportCase):
    def test_the_raw_files_and_both_status_invest_csvs_are_in_the_inputs(self):
        raw = self.report["inputs"]["raw_files"][0]
        self.assertEqual((raw["url"], raw["sha256"], raw["bytes"]), (RAW[0]["url"], "b" * 64, 1234))
        paths = {(item["date"], item["path"], item["sha256"]) for item in self.report["inputs"]["status_invest"]}
        self.assertIn(("2026-11-06", "snapshots/data/selectors-2026-11-06/stocks.csv", "c" * 64), paths)
        self.assertIn(RAW[0]["url"], self.markdown)
        self.assertIn("snapshots/data/selectors-2026-11-06/stocks.csv", self.markdown)

    def test_the_parameters_hold_the_thresholds_tolerances_formulas_and_the_share_scale_rule(self):
        parameters = self.report["parameters"]
        self.assertEqual(parameters["tolerance"], {"relative": 0.05, "pp": 0.5, "rounding_decimals": 2})
        self.assertEqual(parameters["thresholds"], {"reconstruivel": 90, "com_ressalvas": 70})
        self.assertEqual(compare.verdict(parameters["thresholds"]["reconstruivel"], 1), compare.RECONSTRUIVEL)
        self.assertEqual(compare.verdict(parameters["thresholds"]["com_ressalvas"], 1), compare.COM_RESSALVAS)
        self.assertIn("DY", parameters["percent_fields"])
        self.assertEqual(parameters["liquidity_sessions"], 21)  # F5: 21 sessions (T21); T28 keeps 21 in the v2 schema
        self.assertEqual(set(parameters["formulas"]), set(ALL_FIELDS))
        self.assertIn("B3", parameters["share_scale_rule"])
        self.assertIn("31,6", parameters["share_scale_rule"])
        self.assertEqual(parameters["eligibility"]["cap_min"], 200_000_000)

    def test_the_non_pit_inputs_are_the_seven_inputs_of_a19_with_their_roles(self):
        rows = self.report["parameters"]["non_pit_inputs"]
        self.assertEqual(len(rows), 7)
        self.assertEqual(
            Counter(row["role"] for row in rows),
            Counter(
                {
                    "unit_only": 1,
                    "identity": 3,
                    "classification": 1,
                    "source_selection": 1,
                    "dated_value_current_amount": 1,
                }
            ),
        )
        for row in rows:
            self.assertTrue(row["use"] and row["full_rebuild_check"])


class MarkdownTests(ReportCase):
    def test_the_markdown_renders_the_rule_and_one_row_per_non_pit_input(self):
        self.assertIn(self.report["parameters"]["share_scale_rule"], self.markdown)
        for row in self.report["parameters"]["non_pit_inputs"]:
            self.assertIn(row["input"], self.markdown)
        without_one = deepcopy(self.report)
        without_one["parameters"]["non_pit_inputs"].pop()
        self.assertNotEqual(report.render_markdown(without_one), self.markdown)
        without_rule = deepcopy(self.report)
        without_rule["parameters"]["share_scale_rule"] = ""
        self.assertNotEqual(report.render_markdown(without_rule), self.markdown)

    def test_a_yahoo_ticker_makes_the_dividend_source_secondary_and_none_keeps_it_primary(self):
        self.assertEqual(self.report["dividends"]["source"], "secondary")
        self.assertIn("`secondary`", self.markdown)
        primary = report.build(
            **run_kwargs(
                dividends={
                    "by_source": {"2026-08-07": {"b3": 2, "yahoo": 0, "none": 0}, "2026-11-06": {"b3": 3, "yahoo": 0, "none": 0}},
                    "yahoo_reasons": DIVIDENDS["yahoo_reasons"],
                    "b3_empty_yahoo_failed": DIVIDENDS["b3_empty_yahoo_failed"],
                }
            )
        )
        self.assertEqual(primary["dividends"]["source"], "primary")
        self.assertIn(report.CAVEAT, self.markdown)
        self.assertIn("não tem empresas que saíram da bolsa", self.markdown)

    def test_mapping_counts_are_per_date_in_the_json_and_rendered(self):
        mapping = self.report["dates"]["2026-08-07"]["mapping"]
        self.assertEqual(mapping["vm_price_fallback"]["count"], 1)
        self.assertEqual(mapping["vm_price_fallback"]["companies"][0]["cd_cvm"], "1603")
        self.assertEqual(mapping["share_scale"], {"x1000": ["1603"], "unverified": ["1604"]})
        self.assertEqual(mapping["mapped_via_b3"], ["USIM6"])
        self.assertEqual(self.report["dividends"]["b3_empty_yahoo_failed"]["2026-08-07"], 1)
        self.assertIn("`mapped_via_b3`", self.markdown)
        self.assertIn("`vm_price_fallback`", self.markdown)
        self.assertIn("`share_scale.unverified`", self.markdown)
        self.assertIn("Não verificada (`share_scale.unverified`): `1604`.", self.markdown)

    def test_the_verdict_line_shows_both_class_verdicts_when_they_differ(self):
        verdicts = self.report["verdicts"]
        self.assertTrue(verdicts["class_differs"])
        self.assertEqual(verdicts["class"], compare.NAO_RECONSTRUIVEL)
        self.assertEqual(verdicts["class_without_unclassified"], compare.RECONSTRUIVEL)
        self.assertIn("o veredito seria `reconstruível`", self.markdown)
        same = report.build(**run_kwargs(per_date={"2026-08-07": per_date_entry(3, []), "2026-11-06": per_date_entry(5, [])}))
        self.assertFalse(same["verdicts"]["class_differs"])
        self.assertIn("não mudam o veredito", report.render_markdown(same))

    def test_unclassified_tickers_and_mapping_reasons_are_listed_per_date(self):
        self.assertEqual(self.report["dates"]["2026-11-06"]["mapping"]["unclassified"][0]["ticker"], "UUUU3")
        self.assertIn("UUUU3", self.markdown)
        self.assertIn("`not_in_fca_or_b3`", self.markdown)
        self.assertIn("DDDD3", self.markdown)

    def test_the_cotahist_probe_floor_renders_as_le_2005_with_the_probe_note(self):
        self.assertIn("≤ 2005 (limite da sondagem)", self.markdown)

    def test_dy_coverage_shows_each_source_with_its_retention_and_the_year_from_b3(self):
        self.assertIn(coverage.RETENTION["b3"], self.markdown)
        self.assertIn(coverage.RETENTION["yahoo"], self.markdown)
        self.assertIn("O DY começa em 2016, pela fonte `b3`.", self.markdown)

    def test_the_worst_cases_render_in_collapsible_sections(self):
        self.assertIn("<details>", self.markdown)
        self.assertIn("10 piores casos", self.markdown)
        self.assertIn("Sem valor reconstruído: `UUUU3` (no_quote), `VVVV3` (no_quote).", self.markdown)

    def test_render_is_deterministic_and_reads_every_number_from_the_dict(self):
        self.assertEqual(report.render_markdown(self.report), self.markdown)
        changed = deepcopy(self.report)
        changed["dates"]["2026-08-07"]["eligible"] = 999
        self.assertIn("Elegíveis: 999.", report.render_markdown(changed))
        self.assertNotEqual(report.render_markdown(changed), self.markdown)


class ObjectTests(ReportCase):
    def test_encode_json_is_sorted_indented_and_newline_terminated(self):
        encoded = report.encode_json(self.report)
        self.assertTrue(encoded.endswith(b"\n"))
        self.assertIn(b'\n  "schema"', encoded)
        data = json.loads(encoded)
        self.assertEqual(list(data), sorted(data))

    def test_build_derives_the_verdicts_from_the_per_date_statistics(self):
        fields = self.report["verdicts"]["fields"]
        self.assertEqual(fields["P/L"]["non_financial"], compare.NAO_RECONSTRUIVEL)
        self.assertEqual(fields["P/L"]["non_financial_without_unclassified"], compare.RECONSTRUIVEL)
        self.assertEqual(fields["ROE"]["financial"], compare.RECONSTRUIVEL)

    def test_build_refuses_per_date_entries_that_do_not_match_the_dates(self):
        extra = run_kwargs(per_date={"2026-08-07": per_date_entry(3), "2026-11-06": per_date_entry(5), "2026-12-01": per_date_entry(1)})
        with self.assertRaises(ValueError):
            report.build(**extra)


if __name__ == "__main__":
    unittest.main()
