"""The committed PIT rebuild report is the output of the real run (RPT-04, RPT-06, VER-03, VER-05..VER-08, PIT-04, SRC-02, CLI-01).

The tests read only committed files: docs/validation/pit-rebuild-stocks.json and .md, the README of the validation
folder, cli.py's docstring, and the tracked Status Invest CSV. They make no request and read no cache. The README is
pt-BR and the docstring is English, so the exit-code check compares the codes, not the wording.
"""

import ast
import hashlib
import json
import re
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))

from pit_rebuild import report  # noqa: E402

VALIDATION = PROJECT_ROOT / "docs" / "validation"
JSON_PATH = VALIDATION / "pit-rebuild-stocks.json"
MD_PATH = VALIDATION / "pit-rebuild-stocks.md"
README = VALIDATION / "README.md"
CLI = PROJECT_ROOT / "py" / "pit_rebuild" / "cli.py"
SI_TRACKED = PROJECT_ROOT / "data" / "raw" / "status_invest_fundamentals.csv"
VERDICTS = {"reconstruível", "com ressalvas", "não reconstruível"}
SOURCE_KINDS = ("DFP", "ITR", "FCA", "cadastro", "COTAHIST", "B3 company", "B3 dividends")
DATES = ["2026-08-07", "2026-11-05"]


def source_kind(url: str) -> str | None:
    """The kind of a raw file from its URL: the yearly CVM zips, the cadastro, COTAHIST and the two B3 APIs."""
    if "/DOC/DFP/" in url:
        return "DFP"
    if "/DOC/ITR/" in url:
        return "ITR"
    if "/DOC/FCA/" in url:
        return "FCA"
    if "/CAD/" in url:
        return "cadastro"
    if "COTAHIST" in url:
        return "COTAHIST"
    if "GetListedSupplementCompany" in url:
        return "B3 company"
    if "GetListedCashDividends" in url:
        return "B3 dividends"
    return None


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads(JSON_PATH.read_text(encoding="utf-8"))
        cls.markdown = MD_PATH.read_text(encoding="utf-8")

    def test_the_committed_markdown_is_the_render_of_the_committed_json_byte_for_byte(self):
        # RPT-04: every number in the Markdown comes from the JSON written in the same run
        self.assertEqual(MD_PATH.read_bytes(), report.render_markdown(self.report).encode("utf-8"))

    def test_the_dates_are_the_two_runs_with_their_collection_dates_and_their_as_of_sessions(self):
        self.assertEqual(self.report["parameters"]["dates"], DATES)
        self.assertEqual(sorted(self.report["dates"]), DATES)
        items = {item["date"]: item for item in self.report["inputs"]["status_invest"]}
        self.assertEqual(sorted(items), DATES)
        self.assertEqual(items["2026-08-07"]["collected_on"], "2026-08-07")  # PIT-04: collected on its own date
        self.assertEqual(items["2026-11-05"]["collected_on"], "2026-11-06")  # the snapshot of 2026-11-06, as of 11-05
        for day, item in items.items():
            # A-11: the CSV's PRECO values belong to the session it is evaluated at
            self.assertEqual(item["as_of_check"]["best_session"], day)

    def test_the_class_and_the_financial_group_verdicts_are_in_the_verdict_set(self):
        verdicts = self.report["verdicts"]
        self.assertIn(verdicts["class"], VERDICTS)
        self.assertIn(verdicts["class_without_unclassified"], VERDICTS)
        self.assertIn(verdicts["financial_group"], VERDICTS)
        # VER-03: the verdict line of the Markdown is the official class verdict
        self.assertIn(f"**Veredito da classe:** `{verdicts['class']}`", self.markdown)

    def test_the_sensitivities_are_labelled_not_official_and_the_run_arguments_are_recorded(self):
        # VER-08: the sensitivities never replace the verdict
        self.assertIs(self.report["sensitivities"]["official"], False)
        self.assertIn("## Sensibilidades (não são o veredito oficial)", self.markdown)
        # VER-06 and A-22: the filings cutoff of the run, the liquidity window and the quote codes
        self.assertEqual(self.report["parameters"]["filings_cutoffs"], {"2026-11-05": "2026-11-02"})
        self.assertEqual(self.report["parameters"]["liquidity_sessions"], 21)
        self.assertEqual(self.report["parameters"]["quote_codbdi"], ["02", "07", "08"])
        self.assertIn("√10", self.report["parameters"]["share_scale_rule"])
        self.assertIn("(DT_REFER, d]", self.report["parameters"]["share_scale_rule"])  # RPT-06: the AD-033 share count

    def test_every_source_kind_has_a_raw_file_identified_by_a_sha256_and_its_bytes(self):
        found = {kind: 0 for kind in SOURCE_KINDS}
        for item in self.report["inputs"]["raw_files"]:
            kind = source_kind(item["url"])
            if kind in found and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) and item["bytes"] > 0:
                found[kind] += 1
        self.assertEqual([kind for kind, count in found.items() if count == 0], [])

    def test_the_status_invest_hash_of_the_tracked_csv_is_the_hash_of_the_file_today(self):
        entry = next(item for item in self.report["inputs"]["status_invest"] if item["date"] == "2026-08-07")
        self.assertEqual(entry["path"], "data/raw/status_invest_fundamentals.csv")
        self.assertEqual(entry["sha256"], hashlib.sha256(SI_TRACKED.read_bytes()).hexdigest())

    def test_no_as_of_warning_appears_because_each_csv_matches_its_own_date(self):
        self.assertNotIn("**Atenção:**", self.markdown)

    def test_the_exit_code_table_of_the_readme_lists_the_codes_of_the_cli_docstring(self):
        docstring = ast.get_docstring(ast.parse(CLI.read_text(encoding="utf-8"))) or ""
        cli_codes = re.findall(r"^    (\d)  ", docstring, re.MULTILINE)
        readme = README.read_text(encoding="utf-8")
        section = readme.split("## Códigos de saída", 1)[1].split("\n## ", 1)[0]
        readme_codes = re.findall(r"^\| (\d) \|", section, re.MULTILINE)
        self.assertEqual(cli_codes, ["0", "1", "2", "3", "4", "5"])
        self.assertEqual(readme_codes, cli_codes)


class V3ArtifactTests(unittest.TestCase):
    """The #64 report (RPT-01, RPT-06, ADJ-05, MBR-02): schema v3, the adjusted verdict and its criterion, the rule texts, the lists, the investigation."""

    @classmethod
    def setUpClass(cls):
        cls.report = json.loads(JSON_PATH.read_text(encoding="utf-8"))
        cls.markdown = MD_PATH.read_text(encoding="utf-8")

    def test_the_report_is_schema_v3_amended_by_64_and_names_the_v2_report_it_replaces(self):
        self.assertEqual(self.report["schema"], "pit-rebuild-stocks/v3")
        self.assertEqual(self.report["amended_by"], [64])
        self.assertEqual(self.report["previous_report"], {"schema": "pit-rebuild-stocks/v2", "commit": "ebaf1fd"})
        self.assertIn("Regra de veredito da #59, aplicada aos valores com as correções da #64; os números da v2 estão no commit `ebaf1fd`.", self.markdown)

    def test_the_adjusted_verdict_has_the_14_fields_on_both_dates_and_its_criterion_follows_its_own_percentages(self):
        adjusted = self.report["adjusted_verdict"]
        self.assertEqual(len(adjusted["fields"]), 14)
        self.assertNotIn("ROIC", adjusted["fields"])
        self.assertNotIn("PEG RATIO", adjusted["fields"])
        self.assertEqual(sorted(adjusted["dates"]), DATES)
        for iso in DATES:
            self.assertEqual(sorted(adjusted["dates"][iso]["fields"]), sorted(adjusted["fields"]))
        criterion = adjusted["criterion"]
        self.assertEqual((criterion["part"], criterion["threshold_pct"]), (1, 70))
        # ADJ-05 and ADJ-06: a field with no pct_within counts as failing; the order is the report's (field, then date)
        failing = []
        for field in adjusted["fields"]:
            for iso in DATES:
                pct = adjusted["dates"][iso]["fields"][field]["pct_within"]
                if pct is None or pct < 70:
                    failing.append({"field": field, "date": iso, "pct_within": pct})
        self.assertEqual(criterion["failing"], failing)
        self.assertEqual(criterion["met"], not failing)
        self.assertEqual(criterion["part2"], "docs/validation/pit-rebuild-stocks-audit.md")

    def test_the_markdown_prints_part_one_of_the_criterion_and_never_the_whole_criterion_as_met(self):
        met = self.report["adjusted_verdict"]["criterion"]["met"]
        self.assertIn(f"Critério de liberação (#64), parte 1 (≥ 70 % nas duas datas): {'cumprida' if met else 'não cumprida'}.", self.markdown)
        self.assertIn("Parte 2 (10 piores conferidos e explicados): ver `pit-rebuild-stocks-audit.md`.", self.markdown)
        self.assertNotIn("Critério de liberação (#64): cumprido", self.markdown)  # ADJ-05: the whole criterion is in the audit (AUD-06)

    def test_the_rule_texts_state_the_ad033_share_count_and_drop_the_59_sentences(self):
        parameters = self.report["parameters"]
        dividends = next(item for item in parameters["non_pit_inputs"] if item["input"] == "B3 `stockDividends`")
        formula = parameters["formulas"]["VALOR DE MERCADO"]
        self.assertIn("(DT_REFER, d]", parameters["share_scale_rule"])
        self.assertIn("AD-033", dividends["use"])
        self.assertIn("(DT_REFER, d]", dividends["use"])
        self.assertIn("(DT_REFER, d]", formula)
        for text in (parameters["share_scale_rule"], dividends["use"], formula):
            self.assertNotIn("só eventos com data-ex ≤ d", text)  # RPT-06: the #59 share-count sentence is gone
            self.assertNotIn("um desdobramento de 1:5 dá ×10", text)  # and so is its unconditional form

    def test_each_date_has_the_five_lists_of_the_share_stale_and_ttm_rules(self):
        for iso in DATES:
            mapping = self.report["dates"][iso]["mapping"]
            self.assertIsInstance(mapping["share_events"]["adjusted"], list)
            self.assertIsInstance(mapping["share_events"]["no_isin"], list)
            self.assertIsInstance(mapping["share_count_suspect"], list)
            self.assertIsInstance(mapping["ttm_prior_from_itr"], list)
            stale = mapping["stale_filing"]
            self.assertEqual([row["ticker"] for row in stale], sorted(row["ticker"] for row in stale))  # STL-03: sorted by ticker

    def test_the_mbrf3_investigation_is_in_the_json_and_its_section_renders_it(self):
        notes = self.report["investigations"]
        self.assertEqual([(note["ticker"], note["date"], note["field"]) for note in notes], [("MBRF3", "2026-11-05", "DY")])
        self.assertIn("## Investigações", self.markdown)
        self.assertIn(notes[0]["finding"], self.markdown)  # MBR-02: the note is in the Markdown too


if __name__ == "__main__":
    unittest.main()
