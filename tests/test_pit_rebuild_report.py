"""The report object and its Markdown: provenance, parameters, mapping, as-of checks, dividends, coverage, sensitivities and determinism (RPT-02..RPT-04, VER-03..VER-08, MAP-02, COV-02, PIT-04, IND-04).

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
from pit_rebuild.fields import ADJUSTED_FIELDS, ALL_FIELDS, PERCENT_FIELDS  # noqa: E402

GROUP_NAMES = [group for group, _label in report.GROUPS]
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
    "share_scale": {
        "scaled": [{"cd_cvm": "1603", "factor": 1000, "residual": 1.0}, {"cd_cvm": "1605", "factor": 0.1, "residual": 1.02}],
        "unverified": ["1604"],
    },
    "reference_stale": ["ZZZZ3"],
    "vm_stale_class_price": [{"cd_cvm": "1606", "class": "ON", "ticker": "EMAE3", "session": "2025-08-18"}],
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
    {
        "date": "2026-08-07",
        "collected_on": "2026-08-07",
        "path": "data/raw/status_invest_fundamentals.csv",
        "sha256": "a" * 64,
        "as_of_check": {
            "eligible_with_price": 269,
            "matches": {"2026-08-06": 1, "2026-08-07": 260},
            "best_session": "2026-08-07",
        },
    },
    {
        "date": "2026-11-06",
        "collected_on": "2026-11-06",
        "path": "snapshots/data/selectors-2026-11-06/stocks.csv",
        "sha256": "c" * 64,
        "as_of_check": {
            "eligible_with_price": 274,
            "matches": {"2026-11-05": 270, "2026-11-06": 4},
            "best_session": "2026-11-05",
        },
    },
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


def stats_of(field: str, pct: float) -> compare.FieldStats:
    """Statistics of 10 tickers with the given share within tolerance: the verdict is then hand-checkable."""
    unit = "pp" if field in PERCENT_FIELDS else "relative_pct"
    return compare.FieldStats(field=field, diff_unit=unit, denominator=10, within=int(pct // 10), pct_within=float(pct))


def stats_fields(pct_by_field: dict | None = None, default: float = 100.0) -> dict:
    """{field: {group: FieldStats}} with every indicator at default % except those named in pct_by_field."""
    pct_by_field = pct_by_field or {}
    return {
        field: {group: stats_of(field, pct_by_field.get(field, default)) for group in GROUP_NAMES} for field in ALL_FIELDS
    }


def stats_entry(eligible: int, pct_by_field: dict | None = None, default: float = 100.0) -> dict:
    return {"eligible": eligible, "mapping": MAPPING, "fields": stats_fields(pct_by_field, default)}


def coverage_block() -> dict:
    return {
        "earliest_year_by_source": {
            "DFP": {"year": 2010, "at_probe_floor": False},
            "COTAHIST": {"year": 2005, "at_probe_floor": True},
        },
        "earliest_year_by_field": {"ROE": 2012, "P/VP": 2011},
        "dy": coverage.dy_coverage(
            {
                "b3": {"PETR4": date(2016, 1, 4), "VALE3": date(2018, 3, 1)},
                "yahoo": {"ABCD3": date(2000, 6, 1)},
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
        self.assertNotIn("31,6", parameters["share_scale_rule"])  # r5 (F13): the r4 limit of 31,6 is replaced by √10
        self.assertEqual(parameters["eligibility"]["cap_min"], 200_000_000)

    def test_the_non_pit_inputs_are_the_eight_inputs_of_a19_with_their_roles(self):
        rows = self.report["parameters"]["non_pit_inputs"]
        self.assertEqual(len(rows), 8)
        self.assertEqual(
            Counter(row["role"] for row in rows),
            Counter(
                {
                    "unit_only": 1,
                    "identity": 3,
                    "classification": 1,
                    "source_selection": 1,
                    "dated_value_current_amount": 1,
                    "dated_event_current_record": 1,
                }
            ),
        )
        stock_dividends = next(row for row in rows if row["role"] == "dated_event_current_record")
        self.assertIn("stockDividends", stock_dividends["input"])
        for row in rows:
            self.assertTrue(row["use"] and row["full_rebuild_check"])

    def test_the_schema_is_v2_and_the_parameters_hold_the_quote_codes_and_the_filings_cutoffs(self):
        self.assertEqual(self.report["schema"], "pit-rebuild-stocks/v2")
        self.assertEqual(self.report["parameters"]["quote_codbdi"], ["02", "07", "08"])
        self.assertEqual(self.report["parameters"]["filings_cutoffs"], {})
        cut = report.build(**run_kwargs(filings_cutoffs={date(2026, 11, 6): date(2026, 11, 2)}))
        self.assertEqual(cut["parameters"]["filings_cutoffs"], {"2026-11-06": "2026-11-02"})
        self.assertIn("`2026-11-06` com filings recebidos até `2026-11-02`", report.render_markdown(cut))

    def test_the_share_scale_rule_names_the_powers_of_ten_the_sqrt10_limit_and_the_residual(self):
        rule = self.report["parameters"]["share_scale_rule"]
        for text in ("10^-2", "10^3", "√10", "resíduo"):
            self.assertIn(text, rule)
        self.assertNotIn("31,6", rule)


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
        self.assertEqual(
            mapping["share_scale"],
            {
                "scaled": [{"cd_cvm": "1603", "factor": 1000, "residual": 1.0}, {"cd_cvm": "1605", "factor": 0.1, "residual": 1.02}],
                "unverified": ["1604"],
            },
        )
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

    def test_the_markdown_of_the_object_is_the_markdown_of_its_json(self):
        # the committed JSON is written with sorted keys; the Markdown must not depend on the order a dict was built in
        loaded = json.loads(report.encode_json(self.report))
        self.assertEqual(report.render_markdown(loaded), self.markdown)
        changed = deepcopy(self.report)
        changed["dates"]["2026-08-07"]["eligible"] = 999
        self.assertIn("Elegíveis: 999.", report.render_markdown(changed))
        self.assertNotEqual(report.render_markdown(changed), self.markdown)

    def test_each_scaled_factor_and_residual_is_rendered_as_a_power_of_ten_row(self):
        self.assertIn("| `1603` | 10^3 | 1,00 |", self.markdown)
        self.assertIn("| `1605` | 10^-1 | 1,02 |", self.markdown)
        changed = deepcopy(self.report)
        changed["dates"]["2026-08-07"]["mapping"]["share_scale"]["scaled"][1]["residual"] = 1.5
        self.assertIn("| `1605` | 10^-1 | 1,50 |", report.render_markdown(changed))

    def test_reference_stale_and_vm_stale_class_price_are_rendered_and_deleting_either_changes_the_markdown(self):
        mapping = self.report["dates"]["2026-08-07"]["mapping"]
        self.assertEqual(mapping["reference_stale"], ["ZZZZ3"])
        self.assertIn("`ZZZZ3`", self.markdown)
        self.assertIn("| `1606` | ON | `EMAE3` | 2025-08-18 |", self.markdown)
        for key in ("reference_stale", "vm_stale_class_price"):
            changed = deepcopy(self.report)
            del changed["dates"]["2026-08-07"]["mapping"][key]
            rendered = report.render_markdown(changed)
            self.assertNotEqual(rendered, self.markdown, key)
            self.assertIn("não informado", rendered)

    def test_the_as_of_table_is_rendered_and_the_warning_appears_exactly_when_the_best_session_differs(self):
        inputs = self.report["inputs"]["status_invest"]
        self.assertEqual([item["as_of_check"]["best_session"] for item in inputs], ["2026-08-07", "2026-11-05"])
        self.assertEqual(inputs[1]["collected_on"], "2026-11-06")
        self.assertIn("| 2026-11-06 | 4 |", self.markdown)
        warning = "**Atenção:** as cotações do CSV de 2026-11-06 batem melhor com a sessão 2026-11-05, não com 2026-11-06."
        self.assertIn(warning, self.markdown)
        self.assertEqual(self.markdown.count("**Atenção:**"), 1)
        self.assertNotIn("CSV de 2026-08-07 batem", self.markdown)

    def test_the_dividend_filters_slash_names_and_the_dy_coverage_are_in_the_object_and_the_markdown(self):
        extras = dict(
            DIVIDENDS,
            b3_filters={"raw_records": 10, "duplicates": 2, "capital_returns": 1},
            split_adjust_no_isin={"2026-08-07": 1, "2026-11-06": 0},
            slash_names=[{"trading_name": "AMBEV S/A", "queried_as": "AMBEV SA", "total_records": 0}],
        )
        out = report.build(**run_kwargs(dividends=extras))
        self.assertEqual(out["dividends"]["b3_filters"]["duplicates"], 2)
        self.assertEqual(out["dividends"]["slash_names"][0]["queried_as"], "AMBEV SA")
        self.assertEqual(out["dividends"]["split_adjust_no_isin"]["2026-08-07"], 1)
        markdown = report.render_markdown(out)
        self.assertIn("10 brutos; 2 repetidos, contados uma vez; 1 de `REST CAP DIN`, descartados.", markdown)
        self.assertIn("| AMBEV S/A | `AMBEV SA` | 0 |", markdown)
        self.assertIn("| 2026-08-07 | 1 |", markdown)
        b3 = out["coverage"]["dy"]["by_source"]["b3"]
        self.assertEqual(b3["earliest_raw_date"], "2016-01-04")
        self.assertEqual(b3["median_ticker_year"], 2017)
        self.assertIn("| `b3` | 2 | 2016-01-04 | 2016 | 2017 |", markdown)
        self.assertIn("| `yahoo` | 1 | 2000-06-01 | 2000 | 2000 |", markdown)


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

    def test_build_refuses_a_filings_cutoff_for_a_date_outside_the_run_or_after_it(self):
        with self.assertRaises(ValueError):
            report.build(**run_kwargs(filings_cutoffs={date(2026, 8, 1): date(2026, 7, 30)}))
        with self.assertRaises(ValueError):
            report.build(**run_kwargs(filings_cutoffs={date(2026, 11, 6): date(2026, 11, 7)}))


class SensitivityTests(unittest.TestCase):
    """VER-05..VER-08 and A-22: each sensitivity is hand-computed on 10-ticker statistics."""

    def build_with(self, *, per_date=None, sensitivities=None, cutoffs=None):
        entries = per_date or {"2026-08-07": stats_entry(3), "2026-11-06": stats_entry(5)}
        return report.build(**run_kwargs(per_date=entries, filings_cutoffs=cutoffs or {}, sensitivities=sensitivities))

    def test_official_verdicts_do_not_change_when_every_sensitivity_says_not_reconstruivel(self):
        official = self.build_with()
        every = stats_fields(default=0.0)
        out = self.build_with(
            sensitivities={
                "without_reference_stale": {"excluded": {"2026-08-07": ["ZZZZ3"]}, "fields": {"2026-08-07": every, "2026-11-06": every}},
                "filings_cutoff": {"2026-11-06": {"cutoff": date(2026, 11, 2), "fields": every}},
            },
            cutoffs={date(2026, 11, 6): date(2026, 11, 2)},
        )
        self.assertEqual(out["verdicts"], official["verdicts"])
        self.assertEqual(out["dates"], official["dates"])
        self.assertIs(out["sensitivities"]["official"], False)
        self.assertEqual(out["sensitivities"]["without_reference_stale"]["verdicts"]["class"], compare.NAO_RECONSTRUIVEL)
        self.assertEqual(out["sensitivities"]["filings_cutoff"]["2026-11-06"]["verdicts"]["class"], compare.NAO_RECONSTRUIVEL)

    def test_class_without_roic_peg_drops_roic_and_peg_and_carries_the_note(self):
        out = self.build_with(
            per_date={"2026-08-07": stats_entry(3, {"ROIC": 10.0}), "2026-11-06": stats_entry(5, {"ROIC": 10.0})},
            sensitivities={},
        )
        self.assertEqual(out["verdicts"]["fields"]["ROIC"]["non_financial"], compare.NAO_RECONSTRUIVEL)
        self.assertEqual(out["verdicts"]["class"], compare.NAO_RECONSTRUIVEL)
        block = out["sensitivities"]["class_without_roic_peg"]
        self.assertEqual(block["class"], compare.RECONSTRUIVEL)
        self.assertEqual(block["note"], report.ROIC_PEG_NOTE)
        self.assertEqual(block["excluded_fields"], ["PEG RATIO", "ROIC"])
        both = self.build_with(
            per_date={"2026-08-07": stats_entry(3, {"ROIC": 10.0, "PEG RATIO": 10.0}), "2026-11-06": stats_entry(5, {"PEG RATIO": 10.0})},
            sensitivities={},
        )
        self.assertEqual(both["sensitivities"]["class_without_roic_peg"]["class"], compare.RECONSTRUIVEL)

    def test_filings_cutoff_recomputes_one_date_and_combines_it_with_the_other_official_date(self):
        cutoffs = {date(2026, 11, 6): date(2026, 11, 2)}

        def cutoff_block(per_date):
            sensitivity = {"2026-11-06": {"cutoff": date(2026, 11, 2), "fields": stats_fields()}}
            return self.build_with(per_date=per_date, cutoffs=cutoffs, sensitivities={"filings_cutoff": sensitivity})

        # 2026-08-07 is reconstruível and 2026-11-06 is com ressalvas on P/L: the official class is com ressalvas.
        official_bad_second = {"2026-08-07": stats_entry(3), "2026-11-06": stats_entry(5, {"P/L": 75.0})}
        out = cutoff_block(official_bad_second)
        self.assertEqual(out["verdicts"]["class"], compare.COM_RESSALVAS)
        lifted = out["sensitivities"]["filings_cutoff"]["2026-11-06"]
        # overall(official 2026-08-07 = reconstruível, recomputed 2026-11-06 = reconstruível) = reconstruível for P/L.
        self.assertEqual(lifted["verdicts"]["fields"]["P/L"]["non_financial"], compare.RECONSTRUIVEL)
        self.assertEqual(lifted["verdicts"]["class"], compare.RECONSTRUIVEL)
        self.assertEqual(lifted["date_verdicts"]["P/L"]["non_financial"], compare.RECONSTRUIVEL)
        self.assertEqual(lifted["cutoff"], "2026-11-02")
        # overall(official 2026-08-07 = com ressalvas, recomputed 2026-11-06 = reconstruível) = com ressalvas.
        official_bad_first = {"2026-08-07": stats_entry(3, {"P/L": 75.0}), "2026-11-06": stats_entry(5)}
        kept = cutoff_block(official_bad_first)["sensitivities"]["filings_cutoff"]["2026-11-06"]
        self.assertEqual(kept["verdicts"]["class"], compare.COM_RESSALVAS)
        self.assertEqual(kept["verdicts"]["fields"]["P/L"]["non_financial"], compare.COM_RESSALVAS)

    def test_without_reference_stale_recomputes_verdicts_and_lists_the_excluded_tickers_per_date(self):
        stale = {
            "2026-08-07": stats_fields({"P/L": 75.0}),
            "2026-11-06": stats_fields(),
        }
        out = self.build_with(
            sensitivities={"without_reference_stale": {"excluded": {"2026-08-07": ["ZZZZ3", "YYYY3"]}, "fields": stale}},
        )
        block = out["sensitivities"]["without_reference_stale"]
        self.assertEqual(block["excluded"], {"2026-08-07": ["YYYY3", "ZZZZ3"], "2026-11-06": []})
        self.assertEqual(block["dates"]["2026-08-07"]["verdicts"]["P/L"]["non_financial"], compare.COM_RESSALVAS)
        self.assertEqual(block["verdicts"]["fields"]["P/L"]["non_financial"], compare.COM_RESSALVAS)
        self.assertEqual(block["verdicts"]["class"], compare.COM_RESSALVAS)
        self.assertEqual(out["verdicts"]["class"], compare.RECONSTRUIVEL)

    def test_without_sensitivities_the_object_and_the_markdown_have_no_sensitivity_section(self):
        out = self.build_with()
        self.assertNotIn("sensitivities", out)
        self.assertNotIn("Sensibilidades (não são o veredito oficial)", report.render_markdown(out))

    def test_the_sensitivity_section_follows_the_verdict_and_changing_one_sensitivity_changes_the_markdown(self):
        out = self.build_with(
            sensitivities={"without_reference_stale": {"excluded": {}, "fields": {"2026-08-07": stats_fields({"P/L": 75.0}), "2026-11-06": stats_fields()}}},
        )
        markdown = report.render_markdown(out)
        verdict_at = markdown.index("## Veredito")
        section_at = markdown.index("## Sensibilidades (não são o veredito oficial)")
        parameters_at = markdown.index("## Parâmetros")
        self.assertTrue(verdict_at < section_at < parameters_at)
        self.assertIn("| Sem referências obsoletas (D-A) | `com ressalvas` |", markdown)
        self.assertIn("Nota (D-C): " + report.ROIC_PEG_NOTE, markdown)
        changed = deepcopy(out)
        changed["sensitivities"]["without_reference_stale"]["verdicts"]["class"] = compare.NAO_RECONSTRUIVEL
        rendered = report.render_markdown(changed)
        self.assertNotEqual(rendered, markdown)
        self.assertIn("| Sem referências obsoletas (D-A) | `não reconstruível` |", rendered)

    def test_the_verdict_line_holds_only_official_verdicts(self):
        out = self.build_with(
            sensitivities={"without_reference_stale": {"excluded": {}, "fields": {"2026-08-07": stats_fields({"P/L": 0.0}), "2026-11-06": stats_fields({"P/L": 0.0})}}},
        )
        markdown = report.render_markdown(out)
        verdict_line = next(line for line in markdown.splitlines() if line.startswith("**Veredito da classe:**"))
        self.assertEqual(verdict_line, "**Veredito da classe:** `reconstruível`. Os tickers não classificados não mudam o veredito.")
        self.assertEqual(out["sensitivities"]["without_reference_stale"]["verdicts"]["class"], compare.NAO_RECONSTRUIVEL)


def adjusted_stats(pct_by_field: dict | None = None, default: float = 100.0) -> dict:
    """{field: FieldStats} of the non-financial group, the statistics the adjusted scope reads for every indicator."""
    return {field: groups["non_financial"] for field, groups in stats_fields(pct_by_field, default).items()}


def adjusted_entry(fields: dict | None = None, *, excluded=(), si_lag=None, worst_details=None) -> dict:
    """The stale-free statistics of one date for the adjusted block: all at 100 % unless given."""
    return {
        "excluded": list(excluded),
        "fields": adjusted_stats() if fields is None else fields,
        "si_lag": si_lag or {},
        "worst_details": worst_details or {},
    }


def adjusted_run(first: dict | None = None, second: dict | None = None) -> dict:
    """The report object of the run with the adjusted statistics of its two dates."""
    adjusted = {"2026-08-07": first or adjusted_entry(), "2026-11-06": second or adjusted_entry()}
    return report.build(**run_kwargs(adjusted=adjusted))


class AdjustedVerdictTests(unittest.TestCase):
    """ADJ-02..ADJ-07, AUD-05, LAG-02 and LAG-04: the adjusted block is built after the official verdicts, from the given statistics."""

    def test_the_block_lists_the_14_adjusted_fields_and_no_roic_or_peg_ratio(self):
        block = adjusted_run()["adjusted_verdict"]
        self.assertEqual(block["fields"], list(ADJUSTED_FIELDS))
        self.assertEqual(len(block["fields"]), 14)
        self.assertNotIn("ROIC", block["fields"])
        self.assertNotIn("PEG RATIO", block["fields"])
        self.assertEqual(set(block["dates"]["2026-11-06"]["fields"]), set(ADJUSTED_FIELDS))
        self.assertEqual(block["class"], compare.RECONSTRUIVEL)  # every field at 100 % on both dates

    def test_each_field_takes_the_worse_of_its_two_dates_and_the_class_the_worst_field(self):
        block = adjusted_run(adjusted_entry(adjusted_stats({"P/L": 75.0})), adjusted_entry(adjusted_stats({"ROE": 50.0})))["adjusted_verdict"]
        self.assertEqual(block["dates"]["2026-08-07"]["fields"]["P/L"]["verdict"], compare.COM_RESSALVAS)
        self.assertEqual(block["verdicts"]["P/L"], compare.COM_RESSALVAS)  # com ressalvas and reconstruível: the worse
        self.assertEqual(block["verdicts"]["ROE"], compare.NAO_RECONSTRUIVEL)
        self.assertEqual(block["verdicts"]["DIVIDA LIQUIDA / EBIT"], compare.RECONSTRUIVEL)
        self.assertEqual(block["class"], compare.NAO_RECONSTRUIVEL)  # the worst of the 14 fields

    def test_worst_rows_carry_si_lag_the_filing_and_the_rounded_previous_value_and_dy_rows_have_no_filing(self):
        filing = {"kind": "ITR", "dt_refer": date(2026, 3, 31), "versao": 1, "dt_receb": date(2026, 5, 14)}
        pl = adjusted_stats()["P/L"]
        pl.worst = [
            {"ticker": "AAAA3", "reference": 10.0, "rebuilt": 15.0, "diff": 50.0},
            {"ticker": "BBBB3", "reference": 20.0, "rebuilt": 30.0, "diff": 50.0},
        ]
        dy = adjusted_stats()["DY"]
        dy.worst = [{"ticker": "CCCC3", "reference": 3.0, "rebuilt": 0.0, "diff": 3.0}]
        fields = {**adjusted_stats(), "P/L": pl, "DY": dy}
        details = {
            "P/L": {"AAAA3": {"filing": filing, "previous_value": 10.4567, "si_lag": True}},  # BBBB3 has no details
            "DY": {"CCCC3": {"filing": filing, "previous_value": 0.3, "si_lag": False}},
        }
        block = adjusted_run(adjusted_entry(fields, worst_details=details))["adjusted_verdict"]
        rows = {row["ticker"]: row for row in block["dates"]["2026-08-07"]["fields"]["P/L"]["worst"]}
        self.assertTrue(rows["AAAA3"]["si_lag"])
        self.assertEqual(rows["AAAA3"]["filing"], {"kind": "ITR", "dt_refer": "2026-03-31", "versao": 1, "dt_receb": "2026-05-14"})
        self.assertEqual(rows["AAAA3"]["previous_value"], 10.46)
        self.assertEqual((rows["BBBB3"]["si_lag"], rows["BBBB3"]["filing"], rows["BBBB3"]["previous_value"]), (False, None, None))
        dy_row = block["dates"]["2026-08-07"]["fields"]["DY"]["worst"][0]
        self.assertEqual((dy_row["filing"], dy_row["previous_value"]), (None, None))

    def test_the_criterion_is_met_at_70_percent_on_every_field_and_date(self):
        criterion = adjusted_run(adjusted_entry(adjusted_stats(default=70.0)), adjusted_entry(adjusted_stats(default=70.0)))["adjusted_verdict"]["criterion"]
        self.assertEqual(
            criterion,
            {"part": 1, "threshold_pct": 70, "met": True, "failing": [], "part2": "docs/validation/pit-rebuild-stocks-audit.md"},
        )

    def test_the_criterion_fails_at_69_99_percent_and_names_that_field_and_date(self):
        second = adjusted_entry(adjusted_stats({"P/L": 69.99}, default=70.0))
        criterion = adjusted_run(adjusted_entry(adjusted_stats(default=70.0)), second)["adjusted_verdict"]["criterion"]
        self.assertFalse(criterion["met"])
        self.assertEqual(criterion["failing"], [{"field": "P/L", "date": "2026-11-06", "pct_within": 69.99}])

    def test_a_field_with_no_denominator_on_a_date_fails_the_criterion_with_a_null_pct_within(self):
        empty = adjusted_stats()
        empty["ROE"] = compare.FieldStats(field="ROE", diff_unit="pp")  # denominator 0 and no pct_within on 2026-11-06
        block = adjusted_run(adjusted_entry(), adjusted_entry(empty))["adjusted_verdict"]
        self.assertFalse(block["criterion"]["met"])
        self.assertEqual(block["criterion"]["failing"], [{"field": "ROE", "date": "2026-11-06", "pct_within": None}])
        self.assertEqual(block["dates"]["2026-11-06"]["fields"]["ROE"]["verdict"], compare.SEM_REFERENCIA)

    def test_each_field_and_date_keeps_the_missing_list_of_its_statistics(self):
        stats = adjusted_stats()
        stats["P/L"].missing = [{"ticker": "UUUU3", "reason": "stale_filing"}]
        block = adjusted_run(adjusted_entry(stats))["adjusted_verdict"]
        self.assertEqual(block["dates"]["2026-08-07"]["fields"]["P/L"]["missing"], [{"ticker": "UUUU3", "reason": "stale_filing"}])
        self.assertEqual(block["dates"]["2026-11-06"]["fields"]["P/L"]["missing"], [])

    def test_the_si_lag_count_is_the_number_of_listed_tickers_and_the_within_counts_do_not_change(self):
        entry = adjusted_entry(adjusted_stats({"P/L": 60.0}), si_lag={"P/L": ["BBBB3", "AAAA3"]})
        block = adjusted_run(entry)["adjusted_verdict"]
        pl = block["dates"]["2026-08-07"]["fields"]["P/L"]
        self.assertEqual(pl["si_lag"], {"count": 2, "tickers": ["AAAA3", "BBBB3"]})
        self.assertEqual((pl["denominator"], pl["within"], pl["pct_within"]), (10, 6, 60.0))
        self.assertEqual(block["dates"]["2026-08-07"]["fields"]["ROE"]["si_lag"], {"count": 0, "tickers": []})

    def test_the_excluded_tickers_are_listed_per_date_in_order(self):
        block = adjusted_run(adjusted_entry(excluded=["ZZZZ3", "YYYY3"]))["adjusted_verdict"]
        self.assertEqual(block["excluded"], {"2026-08-07": ["YYYY3", "ZZZZ3"], "2026-11-06": []})

    def test_the_official_verdicts_are_byte_identical_with_and_without_the_adjusted_block(self):
        official = report.build(**run_kwargs())
        with_block = adjusted_run(adjusted_entry(), adjusted_entry())  # every adjusted field at 100 %: a class that differs from the official one
        self.assertEqual(official["verdicts"]["class"], compare.NAO_RECONSTRUIVEL)
        self.assertEqual(with_block["adjusted_verdict"]["class"], compare.RECONSTRUIVEL)
        self.assertNotIn("adjusted_verdict", official)
        self.assertEqual(report.encode_json(with_block["verdicts"]), report.encode_json(official["verdicts"]))
        self.assertEqual(report.encode_json(with_block["dates"]), report.encode_json(official["dates"]))

    def test_build_refuses_adjusted_statistics_that_miss_one_of_the_dates(self):
        with self.assertRaises(ValueError):
            report.build(**run_kwargs(adjusted={"2026-08-07": adjusted_entry()}))


if __name__ == "__main__":
    unittest.main()
