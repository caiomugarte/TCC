"""The report command end to end on the universe fixture: exit codes, edge cases, safety (CLI-01..CLI-05, SAFE-01, SAFE-02, RPT-01, RPT-05, PIT-01, COV-03, IND-04).

The universe (tests/fixtures/pit_rebuild/universe) holds one company per edge case, all with round
numbers, so every expectation below is worked out by hand from the fixture values:

- ALFA (ALFA3 ON, ALFA4 PN): 50 M shares each, LL 100 M, so EPS 1.00; VM 1 100 M; P/L 10.00 and 12.00.
- EUCA (EUCA3 ON, not eligible at 5.00; EUCA4 PN, eligible at 8.00): VM of EUCA4 = 20 M x 5.00 + 30 M x 8.00 = 340 M.
- EUCX (thousands, B3 count 92.619.256): share scale x1000, VM = 31,258,000 x 3.00 + 61,361,000 x 2.00 = 216 496 000.
- COBR (composition 0;0;0): no share count, so VM and the per-share fields have no value (no_share_count).
- NOBX (no B3 record): share scale unverified; P/L 6.00.
- SEMP (PN quoted only in May 2026): VM of SEMP3 = 10 M x 20.00 + 10 M x 20.00 (PN priced at the ON quote).
- BANC (Bancos): the financial group.
- REFL (DFP 2025 delivered 2026-09-15): refiled after the first date; P/L 10.00 on the first, 5.00 on the second.
- NOUS (DFP delivered 2026-12-01): no usable filing. NOQT (no quote): no_quote. ZZZZ (not in FCA nor B3): unclassified.
- SOLO (B3 only, pages 1 and 2): DY 5.00 on the first date (the 2026-10-29 record is not known yet), 15.00 on the second.
- SHRT (B3 history from 2026-02-10): Yahoo, DY 5.00 and 8.00; the 2026-10-15 Yahoo event is after the first date.
- AZUL (B3 history complete and empty, Yahoo answers 404): DY 0 with reason b3_empty_yahoo_failed.

Each test copies a cache populated once for the class, so tests that damage files do not interfere.
"""

import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from pit_rebuild import cli, cvm, safety  # noqa: E402
from pit_rebuild_support import UNIVERSE, populate_cache  # noqa: E402

D1 = date(2026, 8, 7)
D2 = date(2026, 11, 6)
ISO1, ISO2 = D1.isoformat(), D2.isoformat()
SI1 = UNIVERSE / "si_2026-08-07.csv"
SI2 = UNIVERSE / "si_2026-11-06.csv"
REPORT = "pit-rebuild-stocks.json"
MARKDOWN = "pit-rebuild-stocks.md"


def no_network(request, timeout=None):
    raise AssertionError(f"a request was made while --offline is set: {request}")


class StepClock:
    """Each call advances the clock by a fixed step, so every timed block measures exactly one step."""

    def __init__(self, step: float):
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        self.now += self.step
        return self.now


def report_args(cache: Path, out: Path, *, offline: bool = True, dates=(D1, D2), si=(SI1, SI2)) -> list[str]:
    args = ["report"]
    for day, path in zip(dates, si):
        args += ["--date", day.isoformat(), "--si", str(path)]
    args += ["--cache-dir", str(cache), "--out-dir", str(out)]
    if offline:
        args.append("--offline")
    return args


def run(args, opener=no_network, clock=None) -> tuple[int, str]:
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        code = cli.main(args, opener=opener, clock=clock)
    return code, err.getvalue()


def _flip_byte(data: bytes) -> bytes:
    """The same bytes with one bit flipped near the end of a zip (its central directory)."""
    flipped = bytearray(data)
    flipped[-30] ^= 0x01
    return bytes(flipped)


def tree_hashes(paths) -> dict[str, str]:
    hashes = {}
    for base in paths:
        if base.exists():
            for path in sorted(base.rglob("*")):
                if path.is_file():
                    hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


PROTECTED = [safety.REPO_ROOT / name for name in ("data", "snapshots", "baselines")]


class ReportCase(unittest.TestCase):
    """Shared fixture: one populated cache per class, copied into a fresh directory per test."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.master = Path(cls._tmp.name) / "master"
        populate_cache(cls.master, (D1, D2))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.work = Path(self._dir.name)
        self.cache = self.work / "cache"
        shutil.copytree(self.master, self.cache)
        self.out = self.work / "out"

    def tearDown(self):
        self._dir.cleanup()

    def report_json(self) -> dict:
        return json.loads((self.out / REPORT).read_text(encoding="utf-8"))

    def success(self) -> dict:
        code, err = run(report_args(self.cache, self.out))
        self.assertEqual(code, 0, err)
        return self.report_json()


class ReportSuccessTests(ReportCase):
    def test_exit_0_writes_both_files_and_the_hand_computed_verdicts(self):
        code, err = run(report_args(self.cache, self.out), opener=no_network)
        self.assertEqual(code, 0, err)
        self.assertEqual(err, "")
        self.assertTrue((self.out / MARKDOWN).is_file())
        report = self.report_json()
        self.assertEqual(report["parameters"]["dates"], [ISO1, ISO2])
        # P/L, non-financial (the 15 eligible non-financial tickers with a Status Invest P/L): 11 within
        # 5 %; COBR3 (no share count), NOUS3 (no usable filing), NOQT3 (no quote) and ZZZZ3 (unmapped) are out.
        for iso in (ISO1, ISO2):
            pl = report["dates"][iso]["fields"]["P/L"]["non_financial"]
            self.assertEqual((pl["denominator"], pl["within"]), (15, 11), iso)
            self.assertEqual(report["dates"][iso]["verdicts"]["P/L"]["non_financial"], "com ressalvas")
        self.assertEqual(report["dates"][ISO1]["fields"]["P/L"]["non_financial"]["pct_within"], 100 * 11 / 15)
        # PRECO, LIQUIDEZ and DY: 15 in the denominator, 12 within (NOUS3, NOQT3 and ZZZZ3 out).
        for field in ("PRECO", "LIQUIDEZ MEDIA DIARIA", "DY"):
            stats = report["dates"][ISO1]["fields"][field]["non_financial"]
            self.assertEqual((stats["denominator"], stats["within"]), (15, 12), field)
        # P/VP, PSR, EV/EBIT and PEG RATIO: COBR3 is the only reference and has no share count, so 0 of 1.
        self.assertEqual(report["dates"][ISO1]["verdicts"]["P/VP"]["non_financial"], "não reconstruível")
        self.assertEqual(report["verdicts"]["class"], "não reconstruível")
        self.assertEqual(report["verdicts"]["financial_group"], "reconstruível")  # BANC3 alone, 100 % within
        self.assertFalse(report["verdicts"]["class_differs"])

    def test_the_report_is_written_from_a_run_that_made_no_request(self):
        # no_network raises if any request is made, and the offline run finished, so none was.
        self.success()
        self.assertTrue((self.out / REPORT).is_file())

    def test_preferred_price_of_a_company_is_its_own_quote_not_the_preferred_one(self):
        # EUCA4's VM = 20 M x 5.00 (EUCA3, not eligible, quoted) + 30 M x 8.00 = 340 M. If the ON share were
        # priced at the PN quote, VM would be 400 M, outside 5 %, and the within count would be 10.
        report = self.success()
        vm = report["dates"][ISO1]["fields"]["VALOR DE MERCADO"]["non_financial"]
        self.assertEqual((vm["denominator"], vm["within"]), (15, 11))
        # SEMP's PN has no quote in the 30 days before the date: it is counted once, priced at the ON quote.
        fallback = report["dates"][ISO1]["mapping"]["vm_price_fallback"]
        self.assertEqual(fallback["count"], 1)
        self.assertEqual(fallback["companies"], [{"cd_cvm": "1006", "class_without_quote": "PREF", "price_from": "SEMP3"}])

    def test_share_scale_lists_the_thousands_company_and_the_one_without_b3_record(self):
        report = self.success()
        for iso in (ISO1, ISO2):
            scale = report["dates"][iso]["mapping"]["share_scale"]
            self.assertEqual(scale["x1000"], ["1003"], iso)  # EUCX: CVM counts in thousands, B3 count 92.619.256
            self.assertEqual(scale["unverified"], ["1005"], iso)  # NOBX: no B3 record, counts used as filed
        # the thousands company is priced in full units: 31 258 000 x 3.00 + 61 361 000 x 2.00 = 216 496 000
        self.assertEqual(report["dates"][ISO1]["fields"]["VALOR DE MERCADO"]["non_financial"]["within"], 11)

    def test_company_without_share_count_has_no_value_with_reason_and_is_in_no_scale_list(self):
        report = self.success()
        for iso in (ISO1, ISO2):
            day = report["dates"][iso]
            for field in ("VALOR DE MERCADO", "P/L", "P/VP", "PSR", "EV/EBIT", "PEG RATIO"):
                missing = day["fields"][field]["non_financial"]["missing"]
                self.assertIn({"ticker": "COBR3", "reason": "no_share_count"}, missing, (iso, field))
            self.assertNotIn("1004", day["mapping"]["share_scale"]["x1000"])
            self.assertNotIn("1004", day["mapping"]["share_scale"]["unverified"])

    def test_mapping_reasons_and_classification_per_date(self):
        report = self.success()
        unmapped = {row["ticker"]: row["reason"] for row in report["dates"][ISO1]["mapping"]["unmapped"]}
        self.assertEqual(unmapped["NOUS3"], "no_usable_filing")
        self.assertEqual(unmapped["NOQT3"], "no_quote")
        self.assertEqual(unmapped["ZZZZ3"], "not_in_fca_or_b3")
        self.assertNotIn("SOLO3", unmapped)
        self.assertEqual(report["dates"][ISO1]["mapping"]["mapped_via_b3"], ["SOLO3"])
        self.assertEqual(report["dates"][ISO1]["mapping"]["unclassified"], [
            {"ticker": "ZZZZ3", "si_setor": "Financeiro e Outros", "reason": "not_in_fca_or_b3"},
        ])
        self.assertEqual(report["dates"][ISO1]["mapping"]["financial"], [{"ticker": "BANC3", "rule": "setor_ativ"}])
        # REFL's 2025 filing reached the CVM after the first date: that date uses its 2024 filing.
        self.assertEqual(report["dates"][ISO1]["mapping"]["refiled_after_d"], ["REFL3"])
        self.assertEqual(report["dates"][ISO2]["mapping"]["refiled_after_d"], [])
        self.assertEqual(report["dates"][ISO2]["fields"]["P/L"]["non_financial"]["within"], 11)

    def test_dividends_are_cut_at_the_decision_date_and_counted_per_source(self):
        report = self.success()
        div = report["dividends"]
        # SOLO's 2026-10-29 record is ex on 2026-10-30: unknown on the first date, counted on the second.
        self.assertEqual(report["dates"][ISO1]["fields"]["DY"]["non_financial"]["within"], 12)
        self.assertEqual(report["dates"][ISO2]["fields"]["DY"]["non_financial"]["within"], 12)
        self.assertEqual(div["by_source"][ISO1], {"b3": 12, "yahoo": 1, "none": 0})
        self.assertEqual(div["yahoo_reasons"][ISO1], {"b3_failed": 0, "b3_empty": 0, "b3_incomplete": 0, "b3_short_history": 1})
        # the eleven companies whose complete B3 history is empty and whose Yahoo lookup failed: DY 0, source b3
        self.assertEqual(div["b3_empty_yahoo_failed"][ISO1], 11)
        self.assertEqual(div["source"], "secondary")

    def test_coverage_years_and_the_full_rebuild_effort(self):
        report = self.success()
        coverage_out = report["coverage"]
        self.assertEqual(coverage_out["earliest_year_by_source"]["DFP"], {"year": 2010, "at_probe_floor": False})
        self.assertEqual(coverage_out["earliest_year_by_source"]["COTAHIST"], {"year": 2011, "at_probe_floor": False})
        self.assertEqual(coverage_out["earliest_year_by_field"]["P/L"], 2012)  # latest source year 2011 + 1 year of TTM
        self.assertEqual(coverage_out["years"]["COTAHIST"], [2011, 2026])
        self.assertEqual(coverage_out["effort"]["decision_dates"], 180)
        self.assertEqual(coverage_out["effort"]["dates_measured"], 2)

    def test_every_raw_file_and_the_status_invest_csv_are_identified_by_sha256(self):
        report = self.success()
        raw = report["inputs"]["raw_files"]
        self.assertTrue(raw)
        for item in raw:
            self.assertRegex(item["sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(item["bytes"], 0)
            self.assertTrue(item["url"].startswith("https://"))
        names = {item["name"] for item in raw}
        self.assertIn("dfp_cia_aberta_2025.zip", names)
        self.assertIn("COTAHIST_A2026.ZIP", names)
        self.assertIn("cad_cia_aberta.csv", names)
        expected = {ISO1: hashlib.sha256(SI1.read_bytes()).hexdigest(), ISO2: hashlib.sha256(SI2.read_bytes()).hexdigest()}
        self.assertEqual({item["date"]: item["sha256"] for item in report["inputs"]["status_invest"]}, expected)

    def test_effort_load_and_compute_follow_the_injected_clock(self):
        # Each clock call advances 2 s: every timed block measures one step, and each date's compute too.
        code, err = run(report_args(self.cache, self.out), clock=StepClock(2.0))
        self.assertEqual(code, 0, err)
        effort = self.report_json()["coverage"]["effort"]
        for kind in ("DFP", "ITR", "FCA", "CADASTRO", "COTAHIST"):
            self.assertEqual(effort["load"][kind]["seconds"], 2.0, kind)
            self.assertGreater(effort["load"][kind]["bytes"], 0, kind)
        self.assertEqual(effort["compute_seconds_per_date"], 2.0)
        self.assertEqual(effort["compute_projected_seconds"], 360.0)  # 2 s per date x 180 decision dates
        self.assertIsNotNone(effort["load_projected_seconds"])


class ReportExitCodeTests(ReportCase):
    def test_exit_1_when_the_output_directory_cannot_be_created(self):
        blocker = self.work / "a-file"
        blocker.write_text("not a directory", encoding="utf-8")
        code, err = run(report_args(self.cache, blocker / "out"))
        self.assertEqual(code, 1)
        self.assertEqual(len(err.strip().splitlines()), 1, err)
        self.assertTrue(err.startswith("pit_rebuild: "), err)
        self.assertFalse((blocker / "out").exists())

    def test_exit_2_for_bad_dates_repeated_dates_count_mismatch_and_forbidden_roots(self):
        cases = {
            "not ISO": ["--date", "07/08/2026", "--si", str(SI1), "--cache-dir", str(self.cache), "--out-dir", str(self.out)],
            "not a calendar date": ["--date", "2026-13-01", "--si", str(SI1), "--cache-dir", str(self.cache), "--out-dir", str(self.out)],
            "repeated date": ["--date", ISO1, "--si", str(SI1), "--date", ISO1, "--si", str(SI1), "--cache-dir", str(self.cache), "--out-dir", str(self.out)],
            "count mismatch": ["--date", ISO1, "--date", ISO2, "--si", str(SI1), "--cache-dir", str(self.cache), "--out-dir", str(self.out)],
            "out-dir in data": ["--date", ISO1, "--si", str(SI1), "--cache-dir", str(self.cache), "--out-dir", str(safety.REPO_ROOT / "data" / "pit-test-out")],
            "out-dir in snapshots": ["--date", ISO1, "--si", str(SI1), "--cache-dir", str(self.cache), "--out-dir", str(safety.REPO_ROOT / "snapshots" / "pit-test-out")],
            "out-dir in baselines": ["--date", ISO1, "--si", str(SI1), "--cache-dir", str(self.cache), "--out-dir", str(safety.REPO_ROOT / "baselines" / "pit-test-out")],
            "cache in data": ["--date", ISO1, "--si", str(SI1), "--cache-dir", str(safety.REPO_ROOT / "data" / "pit-test-cache"), "--out-dir", str(self.out), "--offline"],
            "cache in snapshots": ["--date", ISO1, "--si", str(SI1), "--cache-dir", str(safety.REPO_ROOT / "snapshots" / "pit-test-cache"), "--out-dir", str(self.out), "--offline"],
            "cache in baselines": ["--date", ISO1, "--si", str(SI1), "--cache-dir", str(safety.REPO_ROOT / "baselines" / "pit-test-cache"), "--out-dir", str(self.out), "--offline"],
            "cache in a nested child of data": ["--date", ISO1, "--si", str(SI1), "--cache-dir", str(safety.REPO_ROOT / "data" / "raw" / "child"), "--out-dir", str(self.out), "--offline"],
        }
        for name, extra in cases.items():
            with self.subTest(case=name):
                code, err = run(["report", *extra])
                self.assertEqual(code, 2, err)
                self.assertEqual(len(err.strip().splitlines()), 1, err)
                self.assertFalse(self.out.exists(), name)
        for path in PROTECTED[:1] + PROTECTED[2:]:
            self.assertFalse((path / "pit-test-out").exists())
        self.assertFalse((safety.REPO_ROOT / "data" / "raw" / "child").exists())

    def test_exit_3_for_a_missing_si_csv_and_a_missing_cache_directory(self):
        code, err = run(report_args(self.cache, self.out, si=(self.work / "missing.csv", SI2)))
        self.assertEqual(code, 3)
        self.assertIn("missing.csv", err)
        self.assertFalse(self.out.exists())
        code, err = run(report_args(self.work / "no-cache", self.out))
        self.assertEqual(code, 3)
        self.assertIn("no-cache", err)
        self.assertFalse(self.out.exists())

    def test_exit_4_for_a_tampered_cached_file_naming_it(self):
        target = self.cache / "dfp_cia_aberta_2025.zip"
        data = bytearray(target.read_bytes())
        data[-30] ^= 0x01
        target.write_bytes(bytes(data))
        code, err = run(report_args(self.cache, self.out))
        self.assertEqual(code, 4)
        self.assertIn("dfp_cia_aberta_2025.zip", err)
        self.assertFalse((self.out / REPORT).exists())

    def test_exit_4_for_a_status_invest_csv_missing_a_required_column(self):
        text = SI1.read_text(encoding="utf-8").splitlines()
        header = text[0].split(",")
        keep = [index for index, column in enumerate(header) if column != "DY"]
        broken = self.work / "si-without-dy.csv"
        broken.write_text("\n".join(",".join(line.split(",")[i] for i in keep) for line in text) + "\n", encoding="utf-8")
        code, err = run(report_args(self.cache, self.out, si=(broken, SI2)))
        self.assertEqual(code, 4)
        self.assertIn("DY", err)
        self.assertFalse(self.out.exists())

    def test_exit_4_when_a_dated_input_after_the_decision_date_reaches_the_computation(self):
        # The loader never returns a filing after d, so the leak is forced: the balance sheet of every
        # company arrives dated 2026-12-01, after the first decision date.
        future = cvm.Dated(1.0, date(2026, 12, 1))
        with mock.patch.object(cvm, "balance", return_value=future):
            code, err = run(report_args(self.cache, self.out))
        self.assertEqual(code, 4)
        self.assertIn("2026-12-01", err)
        self.assertFalse((self.out / REPORT).exists())

    def test_exit_5_for_a_required_file_absent_offline_naming_its_url(self):
        (self.cache / "dfp_cia_aberta_2025.zip").unlink()
        code, err = run(report_args(self.cache, self.out))
        self.assertEqual(code, 5)
        self.assertIn("dfp_cia_aberta_2025.zip", err)
        self.assertIn("--offline", err)
        self.assertFalse(self.out.exists())


class ReportSafetyTests(ReportCase):
    def test_a_failed_run_leaves_the_previous_report_byte_identical_with_one_stderr_line(self):
        self.success()
        before = {name: (self.out / name).read_bytes() for name in (REPORT, MARKDOWN)}
        target = self.cache / "dfp_cia_aberta_2025.zip"
        pristine = target.read_bytes()
        damages = {
            4: lambda: target.write_bytes(_flip_byte(pristine)),  # hash no longer matches the manifest
            5: target.unlink,  # absent from the cache in --offline mode
        }
        for code_expected, damage in damages.items():
            with self.subTest(exit_code=code_expected):
                damage()
                code, err = run(report_args(self.cache, self.out))
                self.assertEqual(code, code_expected, err)
                self.assertEqual(len(err.strip().splitlines()), 1, err)
                self.assertEqual({name: (self.out / name).read_bytes() for name in (REPORT, MARKDOWN)}, before)
                target.write_bytes(pristine)

    def test_the_status_invest_csvs_and_the_protected_trees_are_unchanged_by_the_run(self):
        si_before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (SI1, SI2)}
        trees_before = tree_hashes(PROTECTED)
        self.success()
        self.assertEqual({path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (SI1, SI2)}, si_before)
        self.assertEqual(tree_hashes(PROTECTED), trees_before)

    def test_help_exits_0_in_a_subprocess_and_the_package_does_not_import_config(self):
        done = subprocess.run(
            [sys.executable, "py/pit_rebuild", "report", "--help"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120, check=False,
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("--offline", done.stdout)
        probe = subprocess.run(
            [sys.executable, "-c", "import sys; sys.path.insert(0, 'py'); import pit_rebuild.cli; print('config' in sys.modules)"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120, check=False,
        )
        self.assertEqual(probe.stdout.strip(), "False", probe.stderr)


if __name__ == "__main__":
    unittest.main()
