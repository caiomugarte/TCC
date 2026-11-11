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
- REFL (DFP 2025: version 1 delivered 2026-03-31, version 2 on 2026-09-15): re-filed across the first date; P/L 10.00 on the first, 5.00 on the second.
- NOUS (DFP delivered 2026-12-01): no usable filing. NOQT (no quote): no_quote. ZZZZ (not in FCA nor B3): unclassified.
- SOLO (B3 only, pages 1 and 2): DY 5.00 on the first date (the 2026-10-29 record is not known yet), 15.00 on the second.
- SHRT (B3 history from 2026-02-10): Yahoo, DY 5.00 and 8.00; the 2026-10-15 Yahoo event is after the first date.
- AZUL (B3 history complete and empty, Yahoo answers 404): DY 0 with reason b3_empty_yahoo_failed.

Each test copies a cache populated once for the class, so tests that damage files do not interfere.
"""

import base64
import contextlib
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from urllib.parse import unquote

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# This file names the frozen bundle (the protected trees), so no bytecode is written before the imports (AD-013).
sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT_ROOT / "py"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from pit_rebuild import cli, cotahist, cvm, safety, sources  # noqa: E402
from pit_rebuild_support import UNIVERSE, FakeOpener, FakeResponse, make_zip, populate_cache, universe_routes  # noqa: E402

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
        # EUCA4's VM = 20 M x 5.00 (EUCA3, not eligible, quoted) + 30 M x 8.00 = 340 M, within 5 %.
        # SEMP (r5, F14): SEMP4 last traded 9.00 on 2026-05-29, 70 days before 2026-08-07. Its class is priced at
        # that close, not at the ON quote, and listed in vm_stale_class_price: VM = 10 M x 20.00 + 10 M x 9.00
        # = 290 M against the reference 400 M (-27.5 %), so SEMP3 is out and the within count is 10, not 11.
        report = self.success()
        vm = report["dates"][ISO1]["fields"]["VALOR DE MERCADO"]["non_financial"]
        self.assertEqual((vm["denominator"], vm["within"]), (15, 10))
        mapping = report["dates"][ISO1]["mapping"]
        self.assertEqual(mapping["vm_price_fallback"]["count"], 0)
        self.assertEqual(mapping["vm_stale_class_price"], [{"cd_cvm": "1006", "class": "PREF", "ticker": "SEMP4", "session": "2026-05-29"}])

    def test_share_scale_lists_the_thousands_company_and_the_one_without_b3_record(self):
        report = self.success()
        for iso in (ISO1, ISO2):
            scale = report["dates"][iso]["mapping"]["share_scale"]
            # EUCX: CVM counts in thousands (31 258 and 61 361), B3 count 92 619 256: factor 10^3, residual
            # 92 619 000 / 92 619 256 (the counts are scaled and rounded, A-18)
            self.assertEqual([row["cd_cvm"] for row in scale["scaled"]], ["1003"], iso)
            self.assertEqual(scale["scaled"][0]["factor"], 1000, iso)
            self.assertAlmostEqual(scale["scaled"][0]["residual"], 92_619_000 / 92_619_256, places=9, msg=iso)
            self.assertEqual(scale["unverified"], ["1005"], iso)  # NOBX: no B3 record, counts used as filed
        # EUCX's VM is 31 258 000 x 3.00 + 61 361 000 x 2.00 = 216 496 000, the reference, so it is within. The count
        # of 10 (11 in r4) is SEMP alone moving out, on its stale PN close (see the SEMP test above).
        self.assertEqual(report["dates"][ISO1]["fields"]["VALOR DE MERCADO"]["non_financial"]["within"], 10)

    def test_company_without_share_count_has_no_value_with_reason_and_is_in_no_scale_list(self):
        report = self.success()
        for iso in (ISO1, ISO2):
            day = report["dates"][iso]
            for field in ("VALOR DE MERCADO", "P/L", "P/VP", "PSR", "EV/EBIT", "PEG RATIO"):
                missing = day["fields"][field]["non_financial"]["missing"]
                self.assertIn({"ticker": "COBR3", "reason": "no_share_count"}, missing, (iso, field))
            self.assertNotIn("1004", [row["cd_cvm"] for row in day["mapping"]["share_scale"]["scaled"]])
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
        # REFL's 2025 filing was re-filed across the first date: version 1 had arrived by then, version 2 after it,
        # so that date uses its 2024 filing and REFL3 is flagged.
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
        self.assertEqual(div["yahoo_reasons"][ISO1], {"b3_failed": 0, "b3_empty": 11, "b3_incomplete": 0, "b3_short_history": 1})
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


def fetch_args(cache: Path, dates=(D1, D2), si=(SI1, SI2)) -> list[str]:
    args = ["fetch"]
    for day, path in zip(dates, si):
        args += ["--date", day.isoformat(), "--si", str(path)]
    return args + ["--cache-dir", str(cache)]


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://example.invalid/", code, "error", {}, io.BytesIO(b""))


def json_response(body: dict) -> FakeResponse:
    data = (json.dumps(body, separators=(",", ":")) + "\n").encode("utf-8")
    return FakeResponse(data, {"Content-Length": str(len(data))})


class FetchTests(unittest.TestCase):
    """fetch downloads what report needs, through the same acquisition (SRC-01..SRC-05, COV-01, COV-03, CLI-05)."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.work = Path(self._dir.name)
        self.cache = self.work / "cache"
        self.out = self.work / "out"
        self.routes = universe_routes((D1, D2))
        self.planned = [url for url, _name in sources.plan_files((D1, D2))]

    def tearDown(self):
        self._dir.cleanup()

    def fetch(self, routes=None, cache=None):
        fake = FakeOpener(self.routes if routes is None else routes)
        code, err = run(fetch_args(cache or self.cache), opener=fake)
        return code, err, fake

    @staticmethod
    def gets(fake) -> list[str]:
        return [url for method, url in fake.requests if method == "GET"]

    def manifest(self) -> dict:
        return json.loads((self.cache / "manifest.json").read_text(encoding="utf-8"))

    def test_fetch_requests_every_planned_url_once_and_writes_the_manifest(self):
        code, err, fake = self.fetch()
        self.assertEqual(code, 0, err)
        gets = self.gets(fake)
        for url in self.planned:
            self.assertEqual(gets.count(url), 1, url)
        names = {entry["name"] for entry in self.manifest()["files"].values()}
        for _url, name in sources.plan_files((D1, D2)):
            self.assertIn(name, names)

    def test_fetch_records_the_listings_probes_and_content_lengths_for_the_full_rebuild(self):
        code, err, _fake = self.fetch()
        self.assertEqual(code, 0, err)
        manifest = self.manifest()
        self.assertIn("listing_dfp.html", {entry["name"] for entry in manifest["files"].values()})
        self.assertEqual(manifest["probes"][sources.cotahist_url(2011)], True)
        self.assertEqual(manifest["probes"][sources.cotahist_url(2005)], False)
        self.assertEqual(sorted(sources.cotahist_url(y) for y in range(2005, 2013)), sorted(manifest["probes"]))
        self.assertIn(sources.dfp_url(2010), manifest["sizes"])
        self.assertIn(sources.cotahist_url(2026), manifest["sizes"])
        self.assertIn(sources.CAD_URL, manifest["sizes"])

    def test_a_second_fetch_makes_no_request_for_cached_files(self):
        self.fetch()
        code, err, fake = self.fetch()
        self.assertEqual(code, 0, err)
        repeated = [url for url in self.gets(fake) if url in set(self.planned)]
        self.assertEqual(repeated, [])

    def test_a_b3_cash_history_over_two_pages_fetches_both_pages(self):
        _code, _err, fake = self.fetch()
        gets = self.gets(fake)
        self.assertEqual(gets.count(sources.b3_cash_url("SOLO S.A.", 1)), 1)
        self.assertEqual(gets.count(sources.b3_cash_url("SOLO S.A.", 2)), 1)

    def test_a_b3_company_record_that_fails_sends_each_eligible_ticker_of_that_company_to_yahoo(self):
        routes = dict(self.routes)
        routes[sources.b3_company_url("ALFA")] = http_error(500)
        code, err, fake = self.fetch(routes)
        self.assertEqual(code, 0, err)
        alfa = [url for url in self.gets(fake) if "/ALFA3.SA?" in url or "/ALFA4.SA?" in url]
        self.assertEqual(len(alfa), 2)  # one Yahoo request per eligible ticker of ALFA
        self.assertEqual(run(report_args(self.cache, self.out))[0], 0)
        self.assertEqual(self.report_dividends()["yahoo_reasons"][ISO1]["b3_failed"], 2)

    def test_a_cash_history_with_no_records_goes_to_yahoo_and_a_failed_lookup_is_a_confirmed_non_payer(self):
        routes = dict(self.routes)
        routes[sources.b3_cash_url("SOLO S.A.", 1)] = json_response(
            {"page": {"pageNumber": 1, "pageSize": 120, "totalRecords": 0, "totalPages": 1}, "results": []}
        )
        code, err, fake = self.fetch(routes)
        self.assertEqual(code, 0, err)
        self.assertEqual(len([url for url in self.gets(fake) if "/SOLO3.SA?" in url]), 1)
        self.assertEqual(run(report_args(self.cache, self.out))[0], 0)
        div = self.report_dividends()
        self.assertEqual(div["yahoo_reasons"][ISO1], {"b3_failed": 0, "b3_empty": 12, "b3_incomplete": 0, "b3_short_history": 1})
        self.assertEqual(div["b3_empty_yahoo_failed"][ISO1], 12)  # the eleven of the fixture plus SOLO3

    def test_a_missing_second_page_is_incomplete_and_goes_to_yahoo(self):
        routes = dict(self.routes)
        routes[sources.b3_cash_url("SOLO S.A.", 2)] = http_error(500)
        code, err, fake = self.fetch(routes)
        self.assertEqual(code, 0, err)
        self.assertEqual(len([url for url in self.gets(fake) if "/SOLO3.SA?" in url]), 1)
        self.assertEqual(run(report_args(self.cache, self.out))[0], 0)
        div = self.report_dividends()
        self.assertEqual(div["yahoo_reasons"][ISO1]["b3_incomplete"], 1)
        self.assertEqual(div["by_source"][ISO1]["none"], 1)  # Yahoo failed too: no dividend source

    def test_a_complete_history_starting_after_d_minus_365_asks_yahoo(self):
        _code, _err, fake = self.fetch()
        self.assertEqual(len([url for url in self.gets(fake) if "/SHRT3.SA?" in url]), 1)

    def test_a_complete_history_reaching_d_minus_365_makes_no_yahoo_request(self):
        _code, _err, fake = self.fetch()
        self.assertEqual([url for url in self.gets(fake) if "/SOLO3.SA?" in url], [])

    def test_a_failed_cotahist_download_exits_5_naming_it_and_keeps_the_cached_files(self):
        code, err, _fake = self.fetch()
        self.assertEqual(code, 0, err)
        cached = {entry["name"] for entry in self.manifest()["files"].values()}
        (self.cache / "COTAHIST_A2026.ZIP").unlink()
        manifest = self.manifest()
        del manifest["files"]["COTAHIST_A2026.ZIP"]
        (self.cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        routes = dict(self.routes)
        routes[sources.cotahist_url(2026)] = http_error(500)
        code, err, _fake = self.fetch(routes)
        self.assertEqual(code, 5)
        self.assertIn(sources.cotahist_url(2026), err)
        still = {entry["name"] for entry in self.manifest()["files"].values()}
        self.assertEqual(still, cached - {"COTAHIST_A2026.ZIP"})
        self.assertTrue((self.cache / "dfp_cia_aberta_2025.zip").is_file())

    def test_fetch_then_report_offline_exits_0_end_to_end(self):
        code, err, _fake = self.fetch()
        self.assertEqual(code, 0, err)
        code, err = run(report_args(self.cache, self.out))
        self.assertEqual(code, 0, err)
        self.assertTrue((self.out / REPORT).is_file())

    def test_a_ticker_absent_from_every_fca_year_is_requested_and_mapped_through_its_b3_record(self):
        _code, _err, fake = self.fetch()
        self.assertEqual(self.gets(fake).count(sources.b3_company_url("SOLO")), 1)
        self.assertEqual(run(report_args(self.cache, self.out))[0], 0)
        mapping_out = json.loads((self.out / REPORT).read_text(encoding="utf-8"))["dates"][ISO1]["mapping"]
        self.assertEqual(mapping_out["mapped_via_b3"], ["SOLO3"])
        self.assertNotIn("SOLO3", [row["ticker"] for row in mapping_out["unmapped"]])

    def test_a_confirmed_non_payer_is_asked_of_yahoo_once_and_counted_under_b3(self):
        _code, _err, fake = self.fetch()
        self.assertEqual(len([url for url in self.gets(fake) if "/AZUL4.SA?" in url]), 1)  # Yahoo answers 404
        self.assertEqual(run(report_args(self.cache, self.out))[0], 0)
        div = self.report_dividends()
        self.assertEqual(div["by_source"][ISO1]["none"], 0)
        self.assertEqual(div["by_source"][ISO1]["b3"], 12)  # AZUL4 among them, with DY 0 from b3

    def test_fetch_exit_codes_for_missing_si_and_forbidden_cache(self):
        code, err, _fake = self.fetch_with(si=(self.work / "absent.csv", SI2))
        self.assertEqual(code, 3)
        self.assertIn("absent.csv", err)
        forbidden = safety.REPO_ROOT / "data" / "pit-test-fetch-cache"
        code, err = run(fetch_args(forbidden), opener=FakeOpener(self.routes))
        self.assertEqual(code, 2, err)
        self.assertFalse(forbidden.exists())

    def fetch_with(self, si):
        fake = FakeOpener(self.routes)
        code, err = run(fetch_args(self.cache, si=si), opener=fake)
        return code, err, fake

    def report_dividends(self) -> dict:
        return json.loads((self.out / REPORT).read_text(encoding="utf-8"))["dividends"]


# --- shape universe (T29): one small company per r5 shape, hand-computed; no network -----------------------

SHAPE_SESSIONS = [day for day in (date(2026, 5, 1) + timedelta(days=n) for n in range(100)) if day.weekday() < 5 and day <= D1]
SHAPE_COMPANIES = {  # root: (CNPJ, CD_CVM, name, ON shares, PN shares)
    "CODB": ("21.000.000/0001-21", "2001", "CODB S.A.", 1_000_000, 0),
    "UNIT": ("22.000.000/0001-22", "2002", "UNIT S.A.", 50_000_000, 50_000_000),
    "MINO": ("23.000.000/0001-23", "2003", "MINO S.A.", 50_000_000, 50_000_000),
    "ZERO": ("24.000.000/0001-24", "2004", "ZERO S.A.", 50_000_000, 50_000_000),
    "TIMX": ("25.000.000/0001-25", "2005", "TIMX S.A.", 50_000_000, 50_000_000),
    "ESPA": ("26.000.000/0001-26", "2006", "ESPA S.A.", 50_000_000, 50_000_000),
    "EMAE": ("27.000.000/0001-27", "2007", "EMAE S.A.", 50_000_000, 50_000_000),
    "NOPR": ("28.000.000/0001-28", "2008", "NOPR S.A.", 50_000_000, 50_000_000),
    "SBSP": ("29.000.000/0001-29", "2009", "SBSP S.A.", 100_000_000, 0),
    "DUPL": ("30.000.000/0001-30", "2010", "DUPL S.A.", 100_000_000, 0),
    "FOOX": ("31.000.000/0001-31", "2011", "FOO S.A.", 100_000_000, 0),
    "OKPX": ("32.000.000/0001-32", "2012", "OKPX S.A.", 1_000_000, 0),
    "STAL": ("33.000.000/0001-33", "2013", "STAL S.A.", 1_000_000, 0),
}
# the stale ticker of the as-of and reference-stale tests: its last session is d - 31 (2026-07-07)
SHAPE_STALE = ("STAL3", "STAL", "ON", 10.0, "BRSTAL3CNOR1", date(2026, 7, 7))
# ticker, company root, ESPECI, close on every session, ISIN (12 columns)
SHAPE_QUOTES = [
    ("UNIT11", "UNIT", "UNT", 33.0, "BRUNIT11CNOR"),
    ("MINO3", "MINO", "ON", 10.0, "BRMINO3CNOR1"),
    ("ZERO3", "ZERO", "ON", 10.0, "BRZERO3CNOR1"),
    ("TIMX3", "TIMX", "ON", 10.0, "BRTIMX3CNOR1"),
    ("ESPA3", "ESPA", "ON", 10.0, "BRESPA3CNOR1"),
    ("ESPA4", "ESPA", "PN", 12.0, "BRESPA4CNOR1"),
    ("EMAE4", "EMAE", "PN", 10.0, "BREMAE4CNOR1"),
    ("NOP3", "NOPR", "ON", 10.0, "BRNOP3CNOR1"),
    ("SBSP3", "SBSP", "ON", 10.0, "BRSBSP3CNOR5"),
    ("DUPL3", "DUPL", "ON", 10.0, "BRDUPL3CNOR1"),
    ("FOOX3", "FOOX", "ON", 10.0, "BRFOOX3CNOR1"),
    ("OKP3", "OKPX", "ON", 10.0, "BROKP3CNOR1"),
]
# FCA rows: ticker, root, valor mobiliario, composition (units only)
SHAPE_FCA = [
    ("CODB3", "CODB", "Ações Ordinárias", ""),
    ("UNIT11", "UNIT", "Units", "1 ON e 2 PN"),
    ("MINO3", "MINO", "Ações Ordinárias", ""),
    ("ZERO3", "ZERO", "Ações Ordinárias", ""),
    ("TIMX3", "TIMX", "Ações Ordinárias", ""),
    ("ESPA3", "ESPA", "Ações Ordinárias", ""),
    ("ESPA4", "ESPA", "Ações Preferenciais", ""),
    ("EMAE3", "EMAE", "Ações Ordinárias", ""),
    ("EMAE4", "EMAE", "Ações Preferenciais", ""),
    ("NOP3", "NOPR", "Ações Ordinárias", ""),
    ("NOP4", "NOPR", "Ações Preferenciais", ""),
    ("SBSP3", "SBSP", "Ações Ordinárias", ""),
    ("DUPL3", "DUPL", "Ações Ordinárias", ""),
    ("FOOX3", "FOOX", "Ações Ordinárias", ""),
    ("OKP3", "OKPX", "Ações Ordinárias", ""),
    (SHAPE_STALE[0], SHAPE_STALE[1], "Ações Ordinárias", ""),
]
# Status Invest rows of the eligible tickers (VM and liquidity make them eligible; the reference values are off on purpose)
SHAPE_SI = [
    {"TICKER": "CODB3", "VALOR DE MERCADO": 500_000_000, "LIQUIDEZ MEDIA DIARIA": 1_000_000, "PRECO": 1.0},
    {"TICKER": "UNIT11", "VALOR DE MERCADO": 1_000_000_000, "LIQUIDEZ MEDIA DIARIA": 1_000_000, "P/L": 20.0},
    {"TICKER": "MINO3", "VALOR DE MERCADO": 1_000_000_000, "LIQUIDEZ MEDIA DIARIA": 1_000_000, "ROE": 1.0, "P/VP": 1.0},
    {"TICKER": "ZERO3", "VALOR DE MERCADO": 1_000_000_000, "LIQUIDEZ MEDIA DIARIA": 1_000_000, "P/L": 1.0},
    {"TICKER": "TIMX3", "VALOR DE MERCADO": 1_000_000_000, "LIQUIDEZ MEDIA DIARIA": 1_000_000, "P/L": 1.0},
    {"TICKER": "ESPA3", "VALOR DE MERCADO": 200_000_000, "LIQUIDEZ MEDIA DIARIA": 1_000_000},
    {"TICKER": "ESPA4", "VALOR DE MERCADO": 200_000_000, "LIQUIDEZ MEDIA DIARIA": 1_000_000},
    {"TICKER": "EMAE4", "VALOR DE MERCADO": 500_000_000, "LIQUIDEZ MEDIA DIARIA": 1_000_000},
    {"TICKER": "NOP3", "VALOR DE MERCADO": 1_000_000_000, "LIQUIDEZ MEDIA DIARIA": 1_000_000},
    {"TICKER": "SBSP3", "VALOR DE MERCADO": 1_000_000_000, "LIQUIDEZ MEDIA DIARIA": 1_000_000, "DY": 1.0},
    {"TICKER": "DUPL3", "VALOR DE MERCADO": 1_000_000_000, "LIQUIDEZ MEDIA DIARIA": 1_000_000, "DY": 1.0},
    {"TICKER": "FOOX3", "VALOR DE MERCADO": 1_000_000_000, "LIQUIDEZ MEDIA DIARIA": 1_000_000},
    {"TICKER": "OKP3", "VALOR DE MERCADO": 500_000_000, "LIQUIDEZ MEDIA DIARIA": 1_000_000, "PRECO": 10.0},
    {"TICKER": SHAPE_STALE[0], "VALOR DE MERCADO": 500_000_000, "LIQUIDEZ MEDIA DIARIA": 1_000_000, "PRECO": 9.0},
]
SHAPE_DFP_STATEMENTS = {  # root: (BPP rows, consolidated DRE rows, individual DRE rows)
    "MINO": ([("2.03", 200_000_000.0), ("2.03.09", 50_000_000.0)], [("3.11", 100_000_000.0), ("3.11.01", 100_000_000.0)], []),
    "UNIT": ([], [("3.11", 100_000_000.0), ("3.11.01", 100_000_000.0)], []),
    "ZERO": ([], [("3.11", -100_000_000.0), ("3.11.01", 0.0)], []),
    "TIMX": ([], [("3.01", 0.0), ("3.11", 0.0)], [("3.01", 1_000_000_000.0), ("3.11", 100_000_000.0)]),
}


def _cotahist_line(day: date, ticker: str, codbdi: str, especi: str, close: float, voltot: float, isin: str) -> str:
    """One 245-column COTAHIST record in the layout cotahist._parse reads (TIPREG 01, TPMERC 010, FATCOT 1)."""
    line = [" "] * 245

    def put(column: int, text: str) -> None:  # 1-based columns, as the layout is documented
        line[column - 1 : column - 1 + len(text)] = list(text)

    put(1, "01")
    put(3, day.strftime("%Y%m%d"))
    put(11, codbdi)
    put(13, ticker.ljust(12))
    put(25, "010")
    put(40, especi.ljust(10))
    put(109, f"{round(close * 100):013d}")
    put(171, f"{round(voltot * 100):018d}")
    put(211, "0000001")
    put(231, isin.ljust(12))
    return "".join(line)


def _shape_cotahist_2026() -> str:
    lines = []
    last = len(SHAPE_SESSIONS)
    for index, day in enumerate(SHAPE_SESSIONS, start=1):
        # CODB3 is CODBDI 08 only: its VOLTOT on session j is 1 000 000 + 100 000 j and its last close is 30.00
        lines.append(_cotahist_line(day, "CODB3", "08", "ON", 30.0 if index == last else 20.0, 1_000_000.0 + 100_000.0 * index, "BRCODB3CNOR1"))
        for ticker, _root, especi, close, isin in SHAPE_QUOTES:
            lines.append(_cotahist_line(day, ticker, "02", especi, close, 1_000_000.0, isin))
    stale_ticker, _root, stale_especi, stale_close, stale_isin, stale_last = SHAPE_STALE
    for day in SHAPE_SESSIONS:
        if day <= stale_last:
            lines.append(_cotahist_line(day, stale_ticker, "02", stale_especi, stale_close, 1_000_000.0, stale_isin))
    lines.append(_cotahist_line(date(2026, 8, 10), "NOP4", "02", "PN", 10.0, 1_000_000.0, "BRNOP4CNOR1"))  # after d only
    return "\n".join(lines) + "\n"


def _shape_cotahist_2025() -> str:
    return _cotahist_line(date(2025, 8, 18), "EMAE3", "02", "ON", 150.0, 1_000_000.0, "BREMAE3CNOR1") + "\n"


def _latin1(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("latin-1"))


def _shape_b3_pages(root: Path) -> None:
    def page(results: list, total: int) -> str:
        return json.dumps({"page": {"pageNumber": 1, "pageSize": 120, "totalRecords": total, "totalPages": 1}, "results": results})

    dividend = {"typeStock": "ON", "valueCash": "0,50", "lastDatePriorEx": "10/05/2026", "corporateAction": "DIVIDENDO"}
    capital = {"typeStock": "ON", "valueCash": "9,99", "lastDatePriorEx": "10/06/2026", "corporateAction": "REST CAP DIN"}
    old = {"typeStock": "ON", "valueCash": "0,10", "lastDatePriorEx": "01/05/2025", "corporateAction": "DIVIDENDO"}
    sbsp = {"typeStock": "ON", "valueCash": "5,00", "lastDatePriorEx": "01/06/2026", "corporateAction": "DIVIDENDO"}
    _latin1(root / "b3" / "cash" / "SBSP_p1.json", page([sbsp, old], 2))
    _latin1(root / "b3" / "cash" / "DUPL_p1.json", page([dividend, dict(dividend), capital, old], 4))
    _latin1(root / "b3" / "cash" / "FOOX_p1.json", page([], 0))
    companies = {
        "SBSP": {"tradingName": "SBSP S.A.", "code": "SBSP", "codeCVM": "2009", "totalNumberShares": "100.000.000",
                 "stockDividends": [{"label": "DESDOBRAMENTO", "factor": "400", "isinCode": "BRSBSP3CNOR5", "lastDatePrior": "10/07/2026"}]},
        "DUPL": {"tradingName": "DUPL S.A.", "code": "DUPL", "codeCVM": "2010", "totalNumberShares": "100.000.000"},
        "FOOX": {"tradingName": "FOO S/A", "code": "FOOX", "codeCVM": "2011", "totalNumberShares": "100.000.000"},
        "ESPA": {"tradingName": "ESPA S.A.", "code": "ESPA", "codeCVM": "2006", "totalNumberShares": "10.000.000"},
    }
    for root_name, body in companies.items():
        _latin1(root / "b3" / f"company_{root_name}.json", json.dumps([body]))


def write_shape_universe(root: Path) -> Path:
    """A fixture universe with one company per r5 shape (tasks T29). The shared universe is not touched."""
    si_header = (UNIVERSE / "si_2026-08-07.csv").read_text(encoding="utf-8").splitlines()[0]
    bpp_header = (UNIVERSE / "dfp" / "2025" / "dfp_cia_aberta_BPP_con_2025.csv").read_text(encoding="latin-1").splitlines()[0]
    dre_header = (UNIVERSE / "dfp" / "2025" / "dfp_cia_aberta_DRE_con_2025.csv").read_text(encoding="latin-1").splitlines()[0]
    cnpj = {key: value[0] for key, value in SHAPE_COMPANIES.items()}
    name = {key: value[2] for key, value in SHAPE_COMPANIES.items()}
    cd = {key: value[1] for key, value in SHAPE_COMPANIES.items()}
    root.mkdir(parents=True, exist_ok=True)
    _latin1(
        root / "cad_cia_aberta.csv",
        "CNPJ_CIA;DENOM_SOCIAL;SIT;CD_CVM;SETOR_ATIV\n"
        + "".join(f"{cnpj[k]};{name[k]};ATIVO;{cd[k]};Comercio\n" for k in SHAPE_COMPANIES),
    )
    si_columns = si_header.split(",")
    si_rows = [{**{column: "" for column in si_columns}, "SETOR": "Comercio", **row} for row in SHAPE_SI]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=si_columns)
    writer.writeheader()
    writer.writerows(si_rows)
    _latin1(root / "si_2026-08-07.csv", buffer.getvalue())
    dfp = root / "dfp" / "2025"
    _latin1(
        dfp / "dfp_cia_aberta_2025.csv",
        "CNPJ_CIA;DT_REFER;VERSAO;DENOM_CIA;CD_CVM;CATEG_DOC;ID_DOC;DT_RECEB;LINK_DOC\n"
        + "".join(f"{cnpj[k]};2025-12-31;1;{name[k]};{cd[k]};DFP;1;2026-03-01 18:00:00;\n" for k in SHAPE_COMPANIES),
    )
    _latin1(
        dfp / "dfp_cia_aberta_composicao_capital_2025.csv",
        "CNPJ_CIA;DT_REFER;VERSAO;DENOM_CIA;QT_ACAO_ORDIN_CAP_INTEGR;QT_ACAO_PREF_CAP_INTEGR;QT_ACAO_TOTAL_CAP_INTEGR;"
        "QT_ACAO_ORDIN_TESOURO;QT_ACAO_PREF_TESOURO;QT_ACAO_TOTAL_TESOURO\n"
        + "".join(
            f"{cnpj[k]};2025-12-31;1;{name[k]};{value[3]};{value[4]};{value[3] + value[4]};0;0;0\n"
            for k, value in SHAPE_COMPANIES.items()
        ),
    )

    def statement_rows(index: int) -> str:
        rows = []
        for key, (bpp, consolidated, individual) in SHAPE_DFP_STATEMENTS.items():
            for account, value in (bpp if index == 0 else []) + (consolidated if index == 1 else []) + (individual if index == 2 else []):
                rows.append((key, account, value))
        return rows

    bpp_rows = "".join(
        f"{cnpj[key]};2025-12-31;1;{name[key]};{cd[key]};UNIDADE;ÚLTIMO;{account};Conta {account};{value:.2f};S\n"
        for key, account, value in statement_rows(0)
    )
    _latin1(dfp / "dfp_cia_aberta_BPP_con_2025.csv", bpp_header + "\n" + bpp_rows)
    dre_con = "".join(
        f"{cnpj[key]};2025-12-31;1;{name[key]};{cd[key]};UNIDADE;ÚLTIMO;2025-01-01;2025-12-31;{account};Conta {account};{value:.2f};S\n"
        for key, account, value in statement_rows(1)
    )
    _latin1(dfp / "dfp_cia_aberta_DRE_con_2025.csv", dre_header + "\n" + dre_con)
    dre_ind = "".join(
        f"{cnpj[key]};2025-12-31;1;{name[key]};{cd[key]};UNIDADE;ÚLTIMO;2025-01-01;2025-12-31;{account};Conta {account};{value:.2f};S\n"
        for key, account, value in statement_rows(2)
    )
    _latin1(dfp / "dfp_cia_aberta_DRE_ind_2025.csv", dre_header + "\n" + dre_ind)
    fca_rows = []
    for ticker, company_root, valor, composition in SHAPE_FCA:
        fca_rows.append(f"{cnpj[company_root]};2025-12-31;1;{valor};{ticker};{composition}\n")
    _latin1(
        root / "fca" / "2025" / "fca_cia_aberta_valor_mobiliario_2025.csv",
        "CNPJ_Companhia;Data_Referencia;Versao;Valor_Mobiliario;Codigo_Negociacao;Composicao_BDR_Unit\n" + "".join(fca_rows),
    )
    _latin1(root / "cotahist" / "2025.txt", _shape_cotahist_2025())
    _latin1(root / "cotahist" / "2026.txt", _shape_cotahist_2026())
    _shape_b3_pages(root)  # the B3 company records and the cash pages of SBSP, DUPL, FOOX and ESPA
    return root


def worst_value(report_object: dict, field: str, ticker: str, group: str = "non_financial") -> float:
    """The rebuilt value of one ticker among the worst cases of a field (the ten largest differences)."""
    rows = report_object["dates"][ISO1]["fields"][field][group]["worst"]
    return {row["ticker"]: row["rebuilt"] for row in rows}[ticker]


class ShapeReportTests(unittest.TestCase):
    """One report over the shape universe, shared by the class: each test checks one r5 shape."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        base = Path(cls._tmp.name)
        cls.universe = write_shape_universe(base / "universe")
        cls.cache = base / "cache"
        cls.out = base / "out"
        populate_cache(cls.cache, (D1,), cls.universe)
        code, err = run(
            ["report", "--date", ISO1, "--si", str(cls.universe / "si_2026-08-07.csv"), "--cache-dir", str(cls.cache),
             "--out-dir", str(cls.out), "--offline"]
        )
        if code != 0:
            raise RuntimeError(f"the shape report exited {code}: {err}")
        cls.report = json.loads((cls.out / REPORT).read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def unmapped(self) -> set[str]:
        return {row["ticker"] for row in self.report["dates"][ISO1]["mapping"]["unmapped"]}

    def test_a_quote_only_on_codbdi_08_is_mapped_priced_and_averaged_over_21_sessions(self):
        self.assertNotIn("CODB3", self.unmapped())
        self.assertEqual(worst_value(self.report, "PRECO", "CODB3"), 30.0)  # the close of 2026-08-07
        n = len(SHAPE_SESSIONS)
        # VOLTOT of session j is 1 000 000 + 100 000 j; the last 21 sessions are j = N - 20 .. N, whose mean is
        # 1 000 000 + 100 000 (N - 10)
        self.assertEqual(worst_value(self.report, "LIQUIDEZ MEDIA DIARIA", "CODB3"), 1_000_000 + 100_000 * (n - 10))

    def test_a_unit_with_composition_1_on_e_2_pn_is_mapped_and_its_pl_divides_the_price_by_k_3(self):
        self.assertNotIn("UNIT11", self.unmapped())
        # 33.00 / k = 3 is 11.00 per unit share; LL 100 M over 100 M shares is 1.00 per share: P/L = 11.00
        self.assertEqual(worst_value(self.report, "P/L", "UNIT11"), 11.0)

    def test_the_controllers_equity_takes_out_the_minority_interests_in_roe_and_pvp(self):
        # MINO: PL 200 M less 2.03.09 of 50 M is 150 M. ROE = 100 x 100 M / 150 M = 66.67; P/VP = 10.00 / (150 M / 100 M) = 6.67
        self.assertEqual(worst_value(self.report, "ROE", "MINO3"), 66.67)
        self.assertEqual(worst_value(self.report, "P/VP", "MINO3"), 6.67)

    def test_a_zero_3_11_01_row_takes_the_3_11_figure_and_its_sign_for_pl(self):
        # ZERO: 3.11.01 is zero, so LL is 3.11 = -100 M over 100 M shares: -1.00 per share, P/L = 10.00 / -1.00
        self.assertEqual(worst_value(self.report, "P/L", "ZERO3"), -10.0)

    def test_a_company_with_a_zero_consolidated_dre_takes_its_values_from_the_individual_statements(self):
        # TIMX: the consolidated DRE has 3.01 and 3.11 at zero, so the individual DRE is used (A-21, F18): LL 100 M, P/L 10.00
        self.assertEqual(worst_value(self.report, "P/L", "TIMX3"), 10.0)

    def test_an_espa_shaped_company_is_scaled_by_a_tenth_and_its_value_uses_the_rounded_counts(self):
        mapping = self.report["dates"][ISO1]["mapping"]
        scaled = [row for row in mapping["share_scale"]["scaled"] if row["cd_cvm"] == "2006"]
        self.assertEqual(len(scaled), 1)
        self.assertEqual(scaled[0]["factor"], 0.1)
        self.assertAlmostEqual(scaled[0]["residual"], 1.0, places=9)  # 100 M x 0.1 / 10 M
        # counts 5 M ON and 5 M PN after the factor: 5 M x 10.00 + 5 M x 12.00 = 110 M
        self.assertEqual(worst_value(self.report, "VALOR DE MERCADO", "ESPA3"), 110_000_000.0)

    def test_a_class_last_traded_before_the_window_is_priced_at_that_close_and_listed_as_stale_not_as_a_fallback(self):
        mapping = self.report["dates"][ISO1]["mapping"]
        self.assertIn({"cd_cvm": "2007", "class": "ON", "ticker": "EMAE3", "session": "2025-08-18"}, mapping["vm_stale_class_price"])
        self.assertNotIn("2007", [row["cd_cvm"] for row in mapping["vm_price_fallback"]["companies"]])
        # EMAE: 50 M ON at EMAE3's 150.00 from 2025-08-18, plus 50 M PN at EMAE4's 10.00: 7 500 M + 500 M = 8 000 M
        self.assertEqual(worst_value(self.report, "VALOR DE MERCADO", "EMAE4"), 8_000_000_000.0)

    def test_a_class_with_no_session_on_or_before_the_date_is_priced_at_the_other_class_as_a_fallback(self):
        # NOPR's PN has no session on or before d (its only line is after d), so its class takes NOP3's close.
        # MINO, ZERO and TIMX have PN shares and no PN ticker at all: the same fallback applies to them.
        fallback = self.report["dates"][ISO1]["mapping"]["vm_price_fallback"]
        self.assertIn({"cd_cvm": "2008", "class_without_quote": "PREF", "price_from": "NOP3"}, fallback["companies"])
        self.assertEqual(fallback["count"], len(fallback["companies"]))

    def test_a_b3_dividend_before_a_split_is_divided_by_the_split_multiplier(self):
        # SBSP: 5.00 with ex-date 2026-06-02, before the DESDOBRAMENTO 400 (multiplier 5) whose ex-date 2026-07-13 is
        # on or before d: 5.00 / 5 = 1.00 and DY = 100 x 1.00 / 10.00 = 10.00. Its ISIN is in the quotes: no ISIN missing.
        self.assertEqual(worst_value(self.report, "DY", "SBSP3"), 10.0)
        self.assertEqual(self.report["dividends"]["split_adjust_no_isin"], {ISO1: 0})

    def test_repeated_b3_records_count_once_and_a_capital_return_is_not_a_dividend(self):
        # DUPL: 0.50 twice in the response, counted once; REST CAP DIN 9.99 dropped; 0.10 of 2025-05 is outside the window.
        # DY = 100 x 0.50 / 10.00 = 5.00
        self.assertEqual(worst_value(self.report, "DY", "DUPL3"), 5.0)
        filters = self.report["dividends"]["b3_filters"]
        self.assertEqual(filters["duplicates"], 1)
        self.assertEqual(filters["capital_returns"], 1)

    def test_a_slash_name_with_an_empty_complete_history_is_b3_failed_and_is_listed(self):
        dividends_out = self.report["dividends"]
        self.assertEqual(dividends_out["slash_names"], [{"trading_name": "FOO S/A", "queried_as": "FOO SA", "total_records": 0}])
        self.assertGreaterEqual(dividends_out["yahoo_reasons"][ISO1]["b3_failed"], 1)
        missing = self.report["dates"][ISO1]["fields"]["DY"]["non_financial"]["missing"]
        self.assertIn({"ticker": "FOOX3", "reason": "no_dividend_source"}, missing)
        self.assertEqual(dividends_out["b3_empty_yahoo_failed"][ISO1], 0)


class ShapeAsOfTests(unittest.TestCase):
    """The as-of check (A-11) on the shape universe, whose CODB3 close moves from 20.00 to 30.00 on the last session."""

    WARN_PRECO = {  # the 2026-08-06 closes: 13 of the 14 tickers have one (STAL3 has no session on or after 07-08)
        "CODB3": 20.0, "UNIT11": 33.0, "MINO3": 10.0, "ZERO3": 10.0, "TIMX3": 10.0, "ESPA3": 10.0, "ESPA4": 12.0,
        "EMAE4": 10.0, "NOP3": 10.0, "SBSP3": 10.0, "DUPL3": 10.0, "FOOX3": 10.0, "OKP3": 10.0, "STAL3": 10.0,
    }

    def test_the_as_of_check_names_the_best_session_and_warns_exactly_when_it_differs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            universe = write_shape_universe(base / "universe")
            cache = base / "cache"
            populate_cache(cache, (D1,), universe)
            # as built: PRECO of CODB3 (1.00) and STAL3 (9.00) match no close; OKP3 (10.00) matches 08-06 and 08-07: a tie, so 08-07
            out = base / "out"
            code, err = run(["report", "--date", ISO1, "--si", str(universe / "si_2026-08-07.csv"), "--cache-dir", str(cache),
                             "--out-dir", str(out), "--offline"])
            self.assertEqual(code, 0, err)
            as_built = json.loads((out / REPORT).read_text(encoding="utf-8"))
            self.assertEqual(
                as_built["inputs"]["status_invest"][0]["as_of_check"],
                {"eligible_with_price": 3, "matches": {"2026-08-06": 1, "2026-08-07": 1}, "best_session": "2026-08-07"},
            )
            self.assertNotIn("**Atenção:**", (out / MARKDOWN).read_text(encoding="utf-8"))
            # a CSV whose PRECO values are the 2026-08-06 closes: 13 match on 08-06; on 08-07 CODB3's close is 30.00,
            # so 12 match. The best session is 08-06 and the Markdown warns.
            with (universe / "si_2026-08-07.csv").open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                columns = reader.fieldnames
                rows = list(reader)
            for row in rows:
                if row["TICKER"] in self.WARN_PRECO:
                    row["PRECO"] = str(self.WARN_PRECO[row["TICKER"]])
            warn = base / "si_warn.csv"
            with warn.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
            out2 = base / "out2"
            code, err = run(["report", "--date", ISO1, "--si", str(warn), "--cache-dir", str(cache), "--out-dir", str(out2), "--offline"])
            self.assertEqual(code, 0, err)
            warned = json.loads((out2 / REPORT).read_text(encoding="utf-8"))
            self.assertEqual(
                warned["inputs"]["status_invest"][0]["as_of_check"],
                {"eligible_with_price": 14, "matches": {"2026-08-06": 13, "2026-08-07": 12}, "best_session": "2026-08-06"},
            )
            self.assertIn(
                "**Atenção:** as cotações do CSV de 2026-08-07 batem melhor com a sessão 2026-08-06, não com 2026-08-07.",
                (out2 / MARKDOWN).read_text(encoding="utf-8"),
            )

    def test_a_ticker_whose_last_session_is_d_minus_31_is_reference_stale_and_leaves_only_the_sensitivity(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            universe = write_shape_universe(base / "universe")
            cache = base / "cache"
            populate_cache(cache, (D1,), universe)
            out = base / "out"
            code, err = run(["report", "--date", ISO1, "--si", str(universe / "si_2026-08-07.csv"), "--cache-dir", str(cache),
                             "--out-dir", str(out), "--offline"])
            self.assertEqual(code, 0, err)
            report_object = json.loads((out / REPORT).read_text(encoding="utf-8"))
        self.assertIn("STAL3", report_object["dates"][ISO1]["mapping"]["reference_stale"])  # its last session is 2026-07-07
        preco = report_object["dates"][ISO1]["fields"]["PRECO"]["non_financial"]
        # PRECO: CODB3 (1.00 against 30.00) and STAL3 (no quote) are out; OKP3 (10.00 against 10.00) is within: 1 of 3
        self.assertEqual((preco["denominator"], preco["within"]), (3, 1))
        self.assertAlmostEqual(preco["pct_within"], 100 / 3, places=9)
        self.assertIn({"ticker": "STAL3", "reason": "no_quote"}, preco["missing"])
        # without STAL3 the denominator is 2 and the within count stays 1: 50.0 %
        stale_free = report_object["sensitivities"]["without_reference_stale"]["dates"][ISO1]["fields"]["PRECO"]["non_financial"]
        self.assertEqual((stale_free["denominator"], stale_free["within"]), (2, 1))
        self.assertEqual(stale_free["pct_within"], 50.0)


class ShapeFetchTests(unittest.TestCase):
    def test_fetch_asks_b3_for_the_query_name_and_never_with_a_slash(self):
        with tempfile.TemporaryDirectory() as tmp:
            universe = write_shape_universe(Path(tmp) / "universe")
            fake = FakeOpener(universe_routes((D1,), universe))
            code, err = run(
                ["fetch", "--date", ISO1, "--si", str(universe / "si_2026-08-07.csv"), "--cache-dir", str(Path(tmp) / "cache")],
                opener=fake,
            )
            self.assertEqual(code, 0, err)
            names = [
                json.loads(base64.b64decode(unquote(url.rsplit("/", 1)[1])))["tradingName"]
                for _method, url in fake.requests
                if "GetListedCashDividends" in url
            ]
            self.assertIn("FOO SA", names)
            self.assertTrue(names and all("/" not in name for name in names))


class CollectedOnAndCutoffTests(ReportCase):
    """--collected-on, --filings-cutoff and the untimed sensitivities on the shared universe (CLI-03, A-11, A-22)."""

    def invalid(self, extra: list[str]) -> tuple[int, str]:
        return run(report_args(self.cache, self.out) + extra)

    def assert_exit_2_with_one_line(self, extra: list[str], argument: str) -> None:
        code, err = self.invalid(extra)
        self.assertEqual(code, 2, err)
        self.assertEqual(len(err.strip().splitlines()), 1, err)
        self.assertIn(argument, err)
        self.assertFalse(self.out.exists())

    def test_bad_collected_on_arguments_exit_2_naming_the_argument_and_write_nothing(self):
        cases = {
            "malformed": ["--collected-on", ISO1, "--collected-on", "06/11/2026"],
            "given once for two dates": ["--collected-on", ISO2],
            "before its --date": ["--collected-on", "2026-08-06", "--collected-on", ISO2],
        }
        for name, extra in cases.items():
            with self.subTest(name):
                self.assert_exit_2_with_one_line(extra, "--collected-on")

    def test_bad_filings_cutoff_arguments_exit_2_naming_the_argument_and_write_nothing(self):
        cases = {
            "without a colon": ["--filings-cutoff", ISO2],
            "malformed side": ["--filings-cutoff", "2026-11-06:2026-13-40"],
            "date not given with --date": ["--filings-cutoff", "2026-10-01:2026-09-30"],
            "date repeated": ["--filings-cutoff", "2026-11-06:2026-09-14", "--filings-cutoff", "2026-11-06:2026-09-30"],
            "cutoff after its date": ["--filings-cutoff", "2026-11-06:2026-11-07"],
        }
        for name, extra in cases.items():
            with self.subTest(name):
                self.assert_exit_2_with_one_line(extra, "--filings-cutoff")

    def test_without_collected_on_every_csv_is_collected_on_its_own_date(self):
        for item in self.success()["inputs"]["status_invest"]:
            self.assertEqual(item["collected_on"], item["date"])

    def test_a_filings_cutoff_is_a_sensitivity_on_the_older_filing_and_leaves_verdicts_and_dates_unchanged(self):
        plain = self.success()
        out2 = self.work / "out2"
        code, err = run(report_args(self.cache, out2) + ["--filings-cutoff", f"{ISO2}:2026-09-14"])
        self.assertEqual(code, 0, err)
        cut = json.loads((out2 / REPORT).read_text(encoding="utf-8"))
        # VER-08: the official objects of the two runs are equal
        self.assertEqual(cut["verdicts"], plain["verdicts"])
        self.assertEqual(cut["dates"], plain["dates"])
        self.assertEqual(cut["parameters"]["filings_cutoffs"], {ISO2: "2026-09-14"})
        self.assertIs(cut["sensitivities"]["official"], False)
        # REFL3 (2025 filing received 2026-09-15): P/L 5.00 in the official run, within; with the 2024 filing (10.00) it is out
        official = cut["dates"][ISO2]["fields"]["P/L"]["non_financial"]
        self.assertEqual((official["denominator"], official["within"]), (15, 11))
        recomputed = cut["sensitivities"]["filings_cutoff"][ISO2]
        self.assertEqual(recomputed["cutoff"], "2026-09-14")
        self.assertEqual((recomputed["fields"]["P/L"]["non_financial"]["denominator"],
                          recomputed["fields"]["P/L"]["non_financial"]["within"]), (15, 10))

    def test_class_without_roic_peg_is_on_every_exit_0_run(self):
        sensitivities = self.success()["sensitivities"]
        block = sensitivities["class_without_roic_peg"]
        self.assertEqual(block["excluded_fields"], ["PEG RATIO", "ROIC"])
        self.assertIn(block["class"], {"reconstruível", "com ressalvas", "não reconstruível"})
        self.assertIn("não são públicas", block["note"])
        self.assertIs(sensitivities["official"], False)

    def test_compute_seconds_per_date_is_the_same_with_and_without_filings_cutoff(self):
        code, err = run(report_args(self.cache, self.out), clock=StepClock(2.0))
        self.assertEqual(code, 0, err)
        plain_per_date = self.report_json()["coverage"]["effort"]["compute_seconds_per_date"]
        out2 = self.work / "out2"
        code, err = run(report_args(self.cache, out2) + ["--filings-cutoff", f"{ISO2}:2026-09-14"], clock=StepClock(2.0))
        self.assertEqual(code, 0, err)
        cut_per_date = json.loads((out2 / REPORT).read_text(encoding="utf-8"))["coverage"]["effort"]["compute_seconds_per_date"]
        self.assertEqual(cut_per_date, plain_per_date)
        self.assertEqual(plain_per_date, 2.0)  # 2 s per date: the untimed sensitivity evaluations add no clock step


class CollectedFetchTests(unittest.TestCase):
    def test_fetch_with_a_collection_date_one_day_after_asks_yahoo_for_the_day_after_it_and_report_finds_the_files(self):
        collected = D1 + timedelta(days=1)
        period2 = collected + timedelta(days=1)  # Yahoo runs up to the day after the last collection date (A-11)
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cache = base / "cache"
            fake = FakeOpener(universe_routes((collected,), UNIVERSE))  # the universe serves Yahoo up to max(dates) + 1
            args = ["--date", ISO1, "--si", str(SI1), "--collected-on", collected.isoformat(), "--cache-dir", str(cache)]
            code, err = run(["fetch", *args], opener=fake)
            self.assertEqual(code, 0, err)
            epoch = int(datetime(period2.year, period2.month, period2.day, tzinfo=timezone.utc).timestamp())
            yahoo = [url for _method, url in fake.requests if "query1.finance.yahoo.com" in url]
            self.assertTrue(yahoo)
            self.assertTrue(all(f"period2={epoch}&" in url for url in yahoo), yahoo)
            code, err = run(["report", *args, "--out-dir", str(base / "out"), "--offline"])
            self.assertEqual(code, 0, err)


class RawDateAndCounterTests(unittest.TestCase):
    """The shared universe: DY's raw B3 date, the B3 filter totals, and the empty split and slash counters (COV-02, F16, A-20)."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.cache = Path(cls._tmp.name) / "cache"
        populate_cache(cls.cache, (D1, D2))
        cls.out = Path(cls._tmp.name) / "out"
        code, err = run(report_args(cls.cache, cls.out))
        if code != 0:
            raise RuntimeError(err)
        cls.report = json.loads((cls.out / REPORT).read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_the_b3_earliest_raw_date_is_the_first_last_date_prior_ex_plus_one_day_before_the_first_session(self):
        b3 = self.report["coverage"]["dy"]["by_source"]["b3"]
        # SOLO's earliest B3 record has lastDatePriorEx 2024-03-15; the raw date is 2024-03-16, before the first COTAHIST session
        self.assertEqual(b3["earliest_raw_date"], "2024-03-16")
        self.assertEqual(b3["earliest_year"], 2024)

    def test_the_shared_run_reports_no_slash_name_no_split_without_isin_and_the_b3_filter_totals(self):
        dividends_out = self.report["dividends"]
        self.assertEqual(dividends_out["slash_names"], [])
        self.assertEqual(dividends_out["split_adjust_no_isin"], {ISO1: 0, ISO2: 0})
        self.assertEqual(dividends_out["b3_filters"], {"raw_records": 4, "duplicates": 0, "capital_returns": 0})


class ScaledShareCountTests(unittest.TestCase):
    """A-18: a share count scaled by its factor is rounded to whole shares before it is valued (T35, N1).

    cli._company is the smallest unit that reads the rounded counts. Its inputs are inline: one DFP with a
    composicao_capital row, an empty ITR, one B3 record of 4 000 001 shares and no quotes, so the factor is
    0.1 and the counts are the only thing that decides n_on and n_pref.
    """

    CNPJ = "10.101.101/0001-01"
    D = date(2026, 8, 7)

    def test_the_scaled_counts_are_rounded_to_whole_shares(self):
        # ON 30 000 007 and PREF 10 000 003 at 0.1 are 3 000 000.7 and 1 000 000.3: the counts are 3 000 001 and 1 000 000.
        members = {
            "dfp_cia_aberta_2025.csv": (
                "CNPJ_CIA;DT_REFER;VERSAO;DENOM_CIA;CD_CVM;CATEG_DOC;ID_DOC;DT_RECEB;LINK_DOC\n"
                f"{self.CNPJ};2025-12-31;1;CASE S.A.;1;DFP;1;2026-03-20 18:00:00;\n"
            ),
            "dfp_cia_aberta_composicao_capital_2025.csv": (
                "CNPJ_CIA;DT_REFER;VERSAO;QT_ACAO_ORDIN_CAP_INTEGR;QT_ACAO_PREF_CAP_INTEGR;QT_ACAO_TOTAL_CAP_INTEGR\n"
                f"{self.CNPJ};2025-12-31;1;30000007;10000003;40000010\n"
            ),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dfp_cia_aberta_2025.zip"
            path.write_bytes(make_zip(members))
            dfp = cvm.load_filings([path], cvm.DFP)
        acq = SimpleNamespace(
            dfp=dfp,
            itr=cvm.load_filings([], cvm.ITR),
            b3={"ROUN": SimpleNamespace(total_shares=4_000_001)},
            fca=SimpleNamespace(tickers_of=lambda cnpj: []),
            si={self.D: SimpleNamespace(rows={})},
            quotes=cotahist.Quotes(),
        )
        company = cli._company(acq, self.D, self.CNPJ, "9999", ["ROUN3"], self.D)
        self.assertEqual(company.scaled["factor"], 0.1)
        self.assertEqual((company.n_on, company.n_pref), (3_000_001, 1_000_000))


if __name__ == "__main__":
    unittest.main()
