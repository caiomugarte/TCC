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


AUDIT = VALIDATION / "pit-rebuild-stocks-audit.md"
AUDIT_CLASSES = ("erro nosso", "erro do Status Invest", "diferença de método", "não classificado")  # AUD-02
AUDIT_PREFIXES = ("CVM ", "B3 evento ", "COTAHIST ", "B3 proventos ", "Yahoo ")  # B-10: the fixed evidence prefixes
AUDIT_CVM_ITEM = re.compile(r"^CVM (DFP|ITR) \d{4}-\d{2}-\d{2} v\d+ rec \d{4}-\d{2}-\d{2}: .+$")
AUDIT_WITHIN = "dentro da tolerância"  # a row of the worst list inside the tolerance: no error to explain


def parse_audit(text):
    """The audit document as data, per "## <date>" section: rows [(field, cells)], counts {field: cells}, total cells, missing rows."""
    dates, current, section, field = {}, None, None, None
    for line in text.splitlines():
        match = re.match(r"^## (\d{4}-\d{2}-\d{2})$", line)
        if match:
            current = match.group(1)
            dates[current] = {"rows": [], "counts": {}, "total": None, "missing": []}
            section = None
            continue
        if line.startswith("### "):
            section = line[4:].split(" (")[0]
            continue
        if line.startswith("#### "):
            field = line[5:].strip()
            continue
        if current is None or not line.startswith("|") or line.startswith(("| ---", "| #", "| Indicador")):
            continue
        cells = [cell.strip() for cell in line.strip()[1:-1].split("|")]
        block = dates[current]
        if section == "Contagem por indicador":
            if cells[0].startswith("**Total**"):
                block["total"] = cells
            else:
                block["counts"][cells[0]] = cells
        elif section == "Linhas":
            block["rows"].append((field, cells))
        elif section.startswith("Sem valor"):
            block["missing"].append(cells)
    return dates


def audit_items(cell):
    return [item.strip() for item in cell.split(";") if item.strip()]


class AuditDocumentTests(unittest.TestCase):
    """docs/validation/pit-rebuild-stocks-audit.md classifies the worst rows of the adjusted verdict (AUD-01..AUD-05, B-10).

    The document is checked against the committed JSON only: no cache, no request. Every row repeats the JSON's
    reference, rebuilt value and difference, and the class counts and the missing summary are recounted from the rows.
    """

    @classmethod
    def setUpClass(cls):
        cls.adjusted = json.loads(JSON_PATH.read_text(encoding="utf-8"))["adjusted_verdict"]
        cls.text = AUDIT.read_text(encoding="utf-8")
        cls.doc = parse_audit(cls.text)

    def test_each_audited_date_has_one_row_per_worst_case_of_the_json_and_nothing_else(self):  # AUD-01, AUD-03
        self.assertIn("2026-08-07", self.doc)
        for iso, block in self.doc.items():
            expected = sorted((field, w["ticker"]) for field, st in self.adjusted["dates"][iso]["fields"].items() for w in st["worst"])
            got = sorted((field, cells[1]) for field, cells in block["rows"])
            self.assertEqual(got, expected, iso)
            self.assertEqual(len(got), len(set(got)), iso)

    def test_each_row_repeats_the_json_values_rank_and_si_lag(self):  # AUD-03
        for iso, block in self.doc.items():
            for field, st in self.adjusted["dates"][iso]["fields"].items():
                worst = {w["ticker"]: (rank, w) for rank, w in enumerate(st["worst"], 1)}
                for row_field, cells in block["rows"]:
                    if row_field != field:
                        continue
                    rank, w = worst[cells[1]]
                    self.assertEqual(int(cells[0]), rank, (iso, field, cells[1]))
                    self.assertAlmostEqual(float(cells[2]), w["reference"], places=9, msg=(iso, field, cells[1]))
                    self.assertAlmostEqual(float(cells[3]), w["rebuilt"], places=9, msg=(iso, field, cells[1]))
                    self.assertAlmostEqual(float(cells[4]), round(w["diff"], 2), places=9, msg=(iso, field, cells[1]))
                    self.assertEqual(cells[5], "sim" if w["si_lag"] else "não", (iso, field, cells[1]))

    def test_each_row_has_one_class_of_the_set_and_a_tolerance_row_is_named_as_such(self):  # AUD-02
        for iso, block in self.doc.items():
            for field, cells in block["rows"]:
                self.assertIn(cells[6], AUDIT_CLASSES, (iso, field, cells[1]))
                if cells[7].startswith(AUDIT_WITHIN):
                    self.assertEqual(cells[6], "não classificado", (iso, field, cells[1]))

    def test_accounting_rows_carry_a_cvm_item_and_dy_rows_a_dividend_and_a_close_never_only_cvm(self):  # AUD-02, S2
        for iso, block in self.doc.items():
            for field, cells in block["rows"]:
                items = audit_items(cells[8])
                if field == "DY":
                    self.assertTrue(any(item.startswith(("B3 proventos ", "Yahoo ")) for item in items), (iso, cells[1]))
                    self.assertTrue(any(item.startswith("COTAHIST ") for item in items), (iso, cells[1]))
                    self.assertFalse(all(item.startswith("CVM ") for item in items), (iso, cells[1]))
                else:
                    self.assertTrue(any(item.startswith("CVM ") for item in items), (iso, field, cells[1]))

    def test_every_evidence_item_has_a_b10_prefix_and_every_cvm_item_its_fields(self):  # B-10
        for iso, block in self.doc.items():
            for field, cells in block["rows"]:
                for item in audit_items(cells[8]):
                    self.assertTrue(item.startswith(AUDIT_PREFIXES), (iso, field, cells[1], item))
                    if item.startswith("CVM "):
                        self.assertRegex(item, AUDIT_CVM_ITEM, (iso, field, cells[1]))

    def test_the_class_counts_per_field_and_their_total_equal_the_rows(self):  # AUD-04
        for iso, block in self.doc.items():
            per = {}
            for field, cells in block["rows"]:
                per.setdefault(field, {name: 0 for name in AUDIT_CLASSES})[cells[6]] += 1
            for field in self.adjusted["fields"]:
                counts = block["counts"][field]
                self.assertEqual([int(value) for value in counts[1:5]], [per[field][name] for name in AUDIT_CLASSES], (iso, field))
                self.assertEqual(int(counts[5]), sum(per[field].values()), (iso, field))
            total = [int(value.strip("*")) for value in block["total"][1:6]]
            self.assertEqual(total[:4], [sum(per[field][name] for field in per) for name in AUDIT_CLASSES], iso)
            self.assertEqual(total[4], sum(sum(per[field].values()) for field in per), iso)

    def test_the_missing_summary_equals_the_json_counts_per_date_field_and_reason(self):  # AUD-05
        for iso, block in self.doc.items():
            expected = {}
            for field in self.adjusted["fields"]:
                for missing in self.adjusted["dates"][iso]["fields"][field]["missing"]:
                    key = (field, missing["reason"])
                    expected[key] = expected.get(key, 0) + 1
            got = {(cells[0], cells[1]): int(cells[2]) for cells in block["missing"] if cells[1] != "—"}
            self.assertEqual(got, expected, iso)
            for cells in block["missing"]:
                if cells[1] == "—":
                    self.assertEqual(cells[2], "0", (iso, cells[0]))
                    self.assertFalse(self.adjusted["dates"][iso]["fields"][cells[0]]["missing"], (iso, cells[0]))

    def test_the_header_names_the_limits_of_the_worst_list_and_the_four_classes(self):  # AUD-05, S6
        self.assertIn("tickers sem valor reconstruído", self.text)
        self.assertIn("referência zero", self.text)
        self.assertIn(AUDIT_WITHIN, self.text)
        for name in AUDIT_CLASSES:
            self.assertIn(name, self.text)


if __name__ == "__main__":
    unittest.main()
