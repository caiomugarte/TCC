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


if __name__ == "__main__":
    unittest.main()
