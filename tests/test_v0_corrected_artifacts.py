"""v0-corrected (P0-02) artifact verification (spec.md VC-09 to VC-11, VC-24, VC-26; tasks T10-T11).

The tests derive from the spec's acceptance criteria, not from the implementation:

- VC-11 (T10): the recomputation uses the frozen fitness semantics - `sel["SCORE"].sum()`
  with pandas skipna behavior minus `lambda * hhi * n` - over data roots resolved from
  the bundle path (never the live `data/`), and never substitutes `n * score_mean`;
  hhi and the score fields recompute from the row's own tickers; a row whose chromosome
  does not reconstruct inside the frozen snapshot is `not_verifiable`, never a silent
  pass.

Interpreter split (spec.md): these tests need numpy/pandas and run under
`/tmp/opencode/tcc-venv/bin/python`; on an interpreter without that stack every test
SKIPs with an explicit reason. Bundle imports set `sys.dont_write_bytecode` first and
run with cwd inside a scratch root (AD-013); nothing here writes under either bundle.
"""

import ast
import csv
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy  # noqa: F401
    import pandas as pd

    HAVE_NUMERIC_STACK = True
    MISSING_STACK = ""
except ImportError as exc:  # pragma: no cover - exercised by the system interpreter
    HAVE_NUMERIC_STACK = False
    MISSING_STACK = f"numpy/pandas unavailable in this interpreter: {exc}"

REPO = Path(__file__).resolve().parents[1]
LEGACY = REPO / "baselines" / "v0-legacy"
CORRECTED = REPO / "baselines" / "v0-corrected"
CORRECTED_REFERENCE = CORRECTED / "reference"
LIVE_DATA = REPO / "data" / "processed"
PROFILES = ("conservador", "moderado", "arrojado", "caio")
TOL = 1e-9  # VC-01/VC-11/VC-29 recompute tolerance


def load_bundle_module(path, name):
    """Import one bundle source file under a unique module name (AD-013 hygiene).

    Callers must have set sys.dont_write_bytecode first and must run with cwd inside
    a scratch root.
    """
    sys.dont_write_bytecode = True  # before the bundle import (AD-013)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def reference_rows(profile):
    """Committed reference rows of one profile keyed by run_id - the test's own read."""
    path = CORRECTED_REFERENCE / f"metrics_stability_{profile}.csv"
    rows = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, strict=True):
            rows[int(row["run_id"])] = {
                "run_id": int(row["run_id"]),
                "seed": int(row["seed"]),
                "tickers": list(ast.literal_eval(row["tickers"])),
                "fitness": float(row["fitness"]),
                "hhi": float(row["hhi"]),
                "score_median": float(row["score_median"]),
                "score_mean": float(row["score_mean"]),
                "score_std": float(row["score_std"]),
            }
    return rows


def live_only_tickers(frozen_universe):
    """Tickers present at HEAD data/processed but absent from the frozen bundle data.

    D-f's "live-only ticker" fixture material: if the recomputation ever resolved its
    data root from the live tree, these would silently resolve; against the bundle's
    frozen data they must be `not_verifiable`.
    """
    found = set()
    for path in sorted(LIVE_DATA.glob("*.csv")):
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, strict=True)
            if not reader.fieldnames or "TICKER" not in reader.fieldnames:
                continue
            for row in reader:
                ticker = str(row["TICKER"]).strip()
                if ticker and ticker not in frozen_universe:
                    found.add(ticker)
    return found


@unittest.skipUnless(HAVE_NUMERIC_STACK, MISSING_STACK)
class FrozenFitnessTests(unittest.TestCase):
    """VC-11 (T10): _frozen_fitness.py is the frozen recomputation owner."""

    @classmethod
    def setUpClass(cls):
        sys.dont_write_bytecode = True  # before any bundle import (AD-013)
        cls._scratch = tempfile.TemporaryDirectory()
        cls._saved_cwd = os.getcwd()
        os.chdir(cls._scratch.name)  # bundle imports run with cwd in a scratch root
        cls.ff = load_bundle_module(
            CORRECTED / "tools" / "_frozen_fitness.py", "_test_v0_frozen_fitness"
        )
        cls.ga = load_bundle_module(
            CORRECTED / "code" / "ga.py", "_test_v0_frozen_ga"
        )
        cls.fb = cls.ff.FrozenBundle.load(CORRECTED, Path(cls._scratch.name) / "scratch")

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._saved_cwd)
        cls._scratch.cleanup()

    @staticmethod
    def expected_formula(frame, tickers, lam, n):
        """VC-11's formula over the row's own rows: skipna SCORE sum minus lambda*hhi*n."""
        sel = frame[frame["TICKER"].isin(set(tickers))]
        weights = sel["SETOR"].value_counts(normalize=True)
        hhi = (weights ** 2).sum()
        return sel["SCORE"].sum() - lam * hhi * n

    def test_frozen_fitness_equals_the_skipna_score_sum_minus_lambda_hhi_n(self):
        for profile in PROFILES:
            frame = self.fb.ranked(profile)
            n, lam = self.fb.n_assets(profile), self.fb.lam(profile)
            for run_id in (0, 7):
                with self.subTest(profile=profile, run_id=run_id):
                    row = reference_rows(profile)[run_id]
                    tickers = row["tickers"]
                    self.assertEqual(len(tickers), n)
                    expected = self.expected_formula(frame, tickers, lam, n)
                    self.assertEqual(self.fb.fitness_of(profile, tickers), expected)
                    # the frozen function itself is the semantics source - the module
                    # must reproduce ga.fitness over the row's own mask bit-for-bit
                    mask = frame["TICKER"].isin(set(tickers)).to_numpy()
                    self.assertEqual(
                        self.fb.fitness_of(profile, tickers),
                        float(self.ga.fitness(frame, mask, lam, n)),
                    )

    def test_n_times_score_mean_is_never_the_recomputation(self):
        frame = self.fb.ranked("caio")
        n, lam = self.fb.n_assets("caio"), self.fb.lam("caio")
        nan_rows = frame[frame["SCORE"].isna()]
        full_rows = frame[frame["SCORE"].notna()]
        self.assertGreater(len(nan_rows), 0)  # the frozen caio frame carries NaN SCORE rows
        tickers = list(nan_rows["TICKER"][:3]) + list(full_rows["TICKER"][: n - 3])
        sel = frame[frame["TICKER"].isin(set(tickers))]
        self.assertGreater(int(sel["SCORE"].isna().sum()), 0)
        recomputed = self.fb.fitness_of("caio", tickers)
        expected = self.expected_formula(frame, tickers, lam, n)
        self.assertEqual(recomputed, expected)  # skipna sum, NaN rows excluded from the sum
        weights = sel["SETOR"].value_counts(normalize=True)
        hhi = (weights ** 2).sum()
        mean_shortcut = n * sel["SCORE"].mean() - lam * hhi * n
        self.assertGreater(abs(recomputed - mean_shortcut), TOL)  # distinguishable...
        self.assertNotEqual(recomputed, mean_shortcut)  # ...and never the recomputation

    def test_hhi_and_score_fields_recompute_from_the_rows_own_tickers(self):
        frozen_any = set()
        for profile in PROFILES:
            frozen_any |= self.fb.universe(profile)
        live_only = live_only_tickers(frozen_any)
        self.assertTrue(live_only, "the live-only fixture material must exist at HEAD")
        rows_using_frozen_only = 0
        for profile in PROFILES:
            frame = self.fb.ranked(profile)
            live_universe = set()
            with open(LIVE_DATA / f"fundamentals_clean_{profile}.csv", newline="",
                      encoding="utf-8") as handle:
                for live_row in csv.DictReader(handle, strict=True):
                    live_universe.add(str(live_row["TICKER"]))
            computed = self.fb.score_fields_of
            for run_id, row in sorted(reference_rows(profile).items()):
                with self.subTest(profile=profile, run_id=run_id):
                    tickers = row["tickers"]
                    self.assertAlmostEqual(
                        self.fb.hhi_of(profile, tickers), row["hhi"], delta=TOL
                    )
                    fields = computed(profile, tickers)
                    self.assertAlmostEqual(fields["score_mean"], row["score_mean"], delta=TOL)
                    self.assertAlmostEqual(
                        fields["score_median"], row["score_median"], delta=TOL
                    )
                    self.assertAlmostEqual(fields["score_std"], row["score_std"], delta=TOL)
                if set(tickers) - live_universe:
                    # the row can only reconstruct from the bundle's frozen data -
                    # proof the recomputation never resolved the live data/ root
                    rows_using_frozen_only += 1
        self.assertGreater(rows_using_frozen_only, 0)

    def test_evidence_for_row_reports_reconstruction_snapshot_and_field_pairs(self):
        profile = "conservador"
        row = reference_rows(profile)[0]
        tickers = row["tickers"]
        evidence = self.fb.evidence_for_row(profile, tickers, row)
        self.assertEqual(evidence.row_reconstruction, "exact")
        self.assertIs(evidence.data_snapshot_match, True)
        self.assertEqual(evidence.reported, row["fitness"])
        self.assertEqual(evidence.recomputed, self.fb.fitness_of(profile, tickers))
        self.assertEqual(evidence.hhi_reported, row["hhi"])
        self.assertEqual(evidence.hhi_recomputed, self.fb.hhi_of(profile, tickers))
        computed = self.fb.score_fields_of(profile, tickers)
        self.assertEqual(
            set(evidence.score_fields), {"score_mean", "score_median", "score_std"}
        )
        for name, (reported, recomputed) in evidence.score_fields.items():
            with self.subTest(field=name):
                self.assertEqual(reported, row[name])
                self.assertEqual(recomputed, computed[name])
        # a chromosome of the wrong size cannot reconstruct: inexact, nothing recomputed
        short = self.fb.evidence_for_row(profile, tickers[:-1], row)
        self.assertEqual(short.row_reconstruction, "inexact")
        self.assertIsNone(short.recomputed)
        self.assertIsNone(short.hhi_recomputed)
        # a live-only ticker (present at HEAD data/processed, absent from the frozen
        # bundle data) yields not_verifiable - never a silent pass (VC-11)
        frozen_any = set()
        for name in PROFILES:
            frozen_any |= self.fb.universe(name)
        live_only = sorted(live_only_tickers(frozen_any))
        self.assertTrue(live_only)
        forged = list(tickers)
        forged[-1] = live_only[0]
        evidence = self.fb.evidence_for_row(profile, forged, row)
        self.assertEqual(evidence.row_reconstruction, "inexact")
        self.assertIs(evidence.data_snapshot_match, False)
        self.assertIsNone(evidence.recomputed)
        self.assertIsNone(evidence.hhi_recomputed)
        self.assertTrue(all(recomputed is None for _, recomputed in
                            evidence.score_fields.values()))


PINNED_BASE = "181453f986cce7941addf9092e38c6e005e18844"


def git_show(rev_path):
    """Bytes of a git object at the pinned base - local object database only."""
    completed = subprocess.run(
        ["git", "show", rev_path], cwd=str(REPO), capture_output=True, check=True
    )
    return completed.stdout


def resolve_pointer(root, pointer):
    """RFC-6901 JSON Pointer resolution - the test's own read of a pinned leaf."""
    node = root
    for part in pointer.split("/")[1:]:
        part = part.replace("~1", "/").replace("~0", "~")
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def find_run_rows(node, pointer="", list_key=None, out=None):
    """The test's own all_runs[i] finder - independent of the verifier's walker."""
    if out is None:
        out = []
    if isinstance(node, dict):
        for key, value in node.items():
            find_run_rows(value, f"{pointer}/{key}", key, out)
    elif isinstance(node, list):
        if list_key == "all_runs":
            for index, item in enumerate(node):
                out.append((f"{pointer}/{index}", item))
        else:
            for index, item in enumerate(node):
                find_run_rows(item, f"{pointer}/{index}", None, out)
    return out


@unittest.skipUnless(HAVE_NUMERIC_STACK, MISSING_STACK)
class ArtifactVerifierTests(unittest.TestCase):
    """VC-01/VC-09/VC-10/VC-11/VC-24/VC-26 (T11): verify_artifacts.py and status derivation.

    Every negative fixture is a runtime mutation inside `tempfile.TemporaryDirectory()`
    (D-f): mutated ticker set, duplicated TICKER, live-only ticker, NaN-SCORE cells -
    zero committed fixture files.
    """

    @classmethod
    def setUpClass(cls):
        sys.dont_write_bytecode = True  # before any bundle import (AD-013)
        cls._scratch = tempfile.TemporaryDirectory()
        cls._saved_cwd = os.getcwd()
        os.chdir(cls._scratch.name)  # bundle imports run with cwd in a scratch root
        cls.va = load_bundle_module(
            CORRECTED / "tools" / "verify_artifacts.py", "_test_v0_verify_artifacts"
        )
        cls.vd = cls.va.vd
        cls.fb = cls.va.ff.FrozenBundle.load(
            CORRECTED, Path(cls._scratch.name) / "scratch"
        )
        cls.report = cls.va.verify_reference(
            CORRECTED_REFERENCE, CORRECTED, Path(cls._scratch.name) / "scratch"
        )
        cls.derived = cls.va.derive_status_entries(
            REPO, CORRECTED, PINNED_BASE, Path(cls._scratch.name) / "scratch"
        )

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._saved_cwd)
        cls._scratch.cleanup()

    # ---- runtime-mutation fixture machinery (D-f) ---- #

    @staticmethod
    def copy_reference(tmp):
        dest = Path(tmp) / "reference"
        shutil.copytree(CORRECTED_REFERENCE, dest)
        return dest

    @staticmethod
    def rewrite_csv_row(dest, profile, run_id, mutate):
        path = dest / f"metrics_stability_{profile}.csv"
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, strict=True)
            columns = list(reader.fieldnames)
            rows = [dict(row) for row in reader]
        for row in rows:
            if int(row["run_id"]) == run_id:
                mutate(row)
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def artifact(self, report, name):
        matches = [a for a in report["artifacts"] if a["path"] == name]
        self.assertEqual(len(matches), 1, f"expected exactly one artifact entry for {name}")
        return matches[0]

    def spec_formula_cells(self, frame, tickers, profile):
        """The recorded cells a correct writer would emit for one selection (VC-11)."""
        sel = frame[frame["TICKER"].isin(set(tickers))]
        weights = sel["SETOR"].value_counts(normalize=True)
        hhi = (weights ** 2).sum()
        n, lam = self.fb.n_assets(profile), self.fb.lam(profile)
        return {
            "fitness": repr(float(sel["SCORE"].sum() - lam * hhi * n)),
            "hhi": repr(float(hhi)),
            "score_mean": repr(float(sel["SCORE"].mean())),
            "score_median": repr(float(sel["SCORE"].median())),
            "score_std": repr(float(sel["SCORE"].std())),
        }

    # ---- the six T11 tests ---- #

    def test_every_reference_row_reports_fitness_equal_to_recomputed(self):
        self.assertEqual(self.report["status"], "PASS")
        metrics = [a for a in self.report["artifacts"]
                   if a["path"].startswith("metrics_stability_")]
        self.assertEqual(len(metrics), 4)
        self.assertEqual(sum(a["rows_total"] for a in metrics), 120)
        self.assertEqual(sum(a["rows_passed"] for a in metrics), 120)
        self.assertEqual(sum(a["rows_failed"] for a in metrics), 0)
        self.assertEqual(sum(a["rows_unverifiable"] for a in metrics), 0)
        for artifact in metrics:
            with self.subTest(artifact=artifact["path"]):
                self.assertEqual(artifact["rows_total"], 30)
                self.assertEqual(artifact["rows_passed"], 30)
                self.assertEqual(artifact["findings"], [])
        # NaN-SCORE semantics: a fixture row whose cells are the skipna-sum formula
        # passes; the same row carrying n*score_mean cells FAILS (VC-11) - the mean
        # shortcut is never the recomputation, at module or verifier level
        frame = self.fb.ranked("caio")
        nan_rows = frame[frame["SCORE"].isna()]
        full_rows = frame[frame["SCORE"].notna()]
        self.assertGreater(len(nan_rows), 0)
        n = self.fb.n_assets("caio")
        tickers = list(nan_rows["TICKER"][:3]) + list(full_rows["TICKER"][: n - 3])
        sel = frame[frame["TICKER"].isin(set(tickers))]
        self.assertGreater(int(sel["SCORE"].isna().sum()), 0)
        with tempfile.TemporaryDirectory() as tmp:
            dest = self.copy_reference(tmp)
            cells = self.spec_formula_cells(frame, tickers, "caio")

            def write(row):
                row["tickers"] = repr(tickers)
                row.update(cells)

            self.rewrite_csv_row(dest, "caio", 0, write)
            report = self.va.verify_reference(dest, CORRECTED, Path(tmp) / "scratch")
            artifact = self.artifact(report, "metrics_stability_caio.csv")
            self.assertEqual(report["status"], "PASS", artifact["findings"])
            self.assertEqual(artifact["rows_passed"], 30)
        with tempfile.TemporaryDirectory() as tmp:
            dest = self.copy_reference(tmp)
            cells = self.spec_formula_cells(frame, tickers, "caio")
            weights = sel["SETOR"].value_counts(normalize=True)
            hhi = (weights ** 2).sum()
            cells["fitness"] = repr(
                float(n * sel["SCORE"].mean() - self.fb.lam("caio") * hhi * n)
            )  # the forbidden n*score_mean shortcut

            def write(row):
                row["tickers"] = repr(tickers)
                row.update(cells)

            self.rewrite_csv_row(dest, "caio", 0, write)
            report = self.va.verify_reference(dest, CORRECTED, Path(tmp) / "scratch")
            artifact = self.artifact(report, "metrics_stability_caio.csv")
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(artifact["rows_failed"], 1)
            self.assertEqual(artifact["rows_passed"], 29)

    def test_summary_n_runs_equals_the_csv_row_count(self):
        summary = json.loads(
            (CORRECTED_REFERENCE / "multiple_runs_summary.json").read_text(encoding="utf-8")
        )
        for profile in PROFILES:
            with self.subTest(profile=profile):
                path = CORRECTED_REFERENCE / f"metrics_stability_{profile}.csv"
                with open(path, newline="", encoding="utf-8") as handle:
                    data_rows = sum(1 for _ in csv.DictReader(handle, strict=True))
                self.assertEqual(summary[profile]["n_runs"], data_rows)
                self.assertEqual(data_rows, 30)
        artifact = self.artifact(self.report, "multiple_runs_summary.json")
        self.assertEqual(artifact["rows_total"], len(PROFILES))
        self.assertEqual(artifact["rows_passed"], len(PROFILES))
        self.assertEqual(artifact["rows_failed"], 0)
        # the FAIL branch: a runtime-mutated summary with a wrong n_runs fails loudly
        with tempfile.TemporaryDirectory() as tmp:
            dest = self.copy_reference(tmp)
            summary_path = dest / "multiple_runs_summary.json"
            document = json.loads(summary_path.read_text(encoding="utf-8"))
            document["caio"]["n_runs"] = 29
            summary_path.write_text(json.dumps(document), encoding="utf-8")
            report = self.va.verify_reference(dest, CORRECTED, Path(tmp) / "scratch")
            artifact = self.artifact(report, "multiple_runs_summary.json")
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(artifact["rows_failed"], 1)
            self.assertEqual(artifact["rows_passed"], 3)
            problems = [p for f in artifact["findings"] for p in f["problems"]]
            self.assertTrue(any("n_runs" in p for p in problems), problems)

    def test_mutated_ticker_set_fails_the_artifact_and_keeps_the_row_in_the_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = self.copy_reference(tmp)
            row_cells = {}

            def drop_one(row):
                tickers = list(ast.literal_eval(row["tickers"]))
                row_cells.update(row)
                row["tickers"] = repr(tickers[:-1])  # size != n

            self.rewrite_csv_row(dest, "conservador", 5, drop_one)
            report = self.va.verify_reference(dest, CORRECTED, Path(tmp) / "scratch")
            self.assertEqual(report["status"], "FAIL")
            artifact = self.artifact(report, "metrics_stability_conservador.csv")
            self.assertEqual(artifact["rows_total"], 30)  # the row stays in the denominator
            self.assertEqual(artifact["rows_passed"], 29)  # never counted as passing
            self.assertEqual(artifact["rows_failed"] + artifact["rows_unverifiable"], 1)
            self.assertEqual(
                artifact["rows_passed"] + artifact["rows_failed"] + artifact["rows_unverifiable"],
                30,
            )
            findings = {f["row_key"]["run_id"]: f["problems"] for f in artifact["findings"]}
            self.assertIn(5, findings)
            self.assertTrue(
                any("exactly 10" in p for p in findings[5]),
                findings[5],
            )
            for name in ("metrics_stability_moderado.csv", "metrics_stability_arrojado.csv",
                         "metrics_stability_caio.csv"):
                with self.subTest(artifact=name):
                    self.assertEqual(self.artifact(report, name)["status"], "PASS")

    def test_duplicated_ticker_fails_loudly_instead_of_mis_reconstructing(self):
        # variant 1: the artifact row's ticker set carries a duplicate (10 values,
        # 9 distinct) - loud FAIL, never a silent de-duplicated reconstruction
        with tempfile.TemporaryDirectory() as tmp:
            dest = self.copy_reference(tmp)

            def duplicate(row):
                tickers = list(ast.literal_eval(row["tickers"]))
                tickers[-1] = tickers[0]
                row["tickers"] = repr(tickers)

            self.rewrite_csv_row(dest, "conservador", 7, duplicate)
            report = self.va.verify_reference(dest, CORRECTED, Path(tmp) / "scratch")
            self.assertEqual(report["status"], "FAIL")
            artifact = self.artifact(report, "metrics_stability_conservador.csv")
            self.assertEqual(artifact["rows_total"], 30)
            self.assertEqual(artifact["rows_passed"], 29)
            findings = {f["row_key"]["run_id"]: f["problems"] for f in artifact["findings"]}
            self.assertIn(7, findings)
            self.assertTrue(
                any("duplicated ticker" in p for p in findings[7]),
                findings[7],
            )
        # variant 2: duplicate TICKER rows inside the frozen ranked frame itself -
        # the frame is rejected loudly, never used to mis-reconstruct a chromosome
        with tempfile.TemporaryDirectory() as tmp:
            fake_bundle = Path(tmp) / "bundle"
            shutil.copytree(CORRECTED / "code", fake_bundle / "code")
            shutil.copytree(CORRECTED / "data", fake_bundle / "data")
            data_path = fake_bundle / "data" / "processed" / "fundamentals_clean_conservador.csv"
            with open(data_path, newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle, strict=True)
                columns = list(reader.fieldnames)
                rows = [dict(row) for row in reader]
            rows.append(dict(rows[0]))  # a duplicated TICKER row
            with open(data_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
            report = self.va.verify_reference(
                CORRECTED_REFERENCE, fake_bundle, Path(tmp) / "scratch"
            )
            self.assertEqual(report["status"], "FAIL")
            artifact = self.artifact(report, "metrics_stability_conservador.csv")
            self.assertEqual(artifact["rows_total"], 30)
            self.assertEqual(artifact["rows_unverifiable"], 30)  # loud, nothing passed
            problems = [p for f in artifact["findings"] for p in f["problems"]]
            self.assertTrue(any("duplicate TICKER rows" in p for p in problems), problems)

    def test_live_only_ticker_yields_not_verifiable_not_a_silent_pass(self):
        frozen_any = set()
        for profile in PROFILES:
            frozen_any |= self.fb.universe(profile)
        live_only = sorted(live_only_tickers(frozen_any))
        self.assertTrue(live_only, "the live-only fixture material must exist at HEAD")
        with tempfile.TemporaryDirectory() as tmp:
            dest = self.copy_reference(tmp)

            def forge(row):
                tickers = list(ast.literal_eval(row["tickers"]))
                tickers[-1] = live_only[0]
                row["tickers"] = repr(tickers)

            self.rewrite_csv_row(dest, "conservador", 3, forge)
            report = self.va.verify_reference(dest, CORRECTED, Path(tmp) / "scratch")
            self.assertEqual(report["status"], "FAIL")
            artifact = self.artifact(report, "metrics_stability_conservador.csv")
            self.assertEqual(artifact["rows_total"], 30)  # kept in the denominator
            self.assertEqual(artifact["rows_passed"], 29)  # not a silent pass
            self.assertEqual(artifact["rows_unverifiable"], 1)
            findings = {f["row_key"]["run_id"]: f["problems"] for f in artifact["findings"]}
            self.assertIn(3, findings)
            self.assertTrue(
                any(live_only[0] in p for p in findings[3]),
                findings[3],
            )
            # the evidence the verifier derives says not_verifiable, not a value
            evidence, notes = self.va._evidence_for(
                self.fb, "conservador", [live_only[0]] * 10, {"fitness": "1.0"}
            )
            self.assertEqual(evidence.row_reconstruction, "inexact")
            self.assertIs(evidence.data_snapshot_match, False)
            self.assertIsNone(evidence.recomputed)

    def test_derived_status_entries_rederive_evidence_from_the_pinned_blobs(self):
        derived = self.derived
        self.assertTrue(derived)
        blob_digests = {}
        blob_rows = {}
        for entry in derived:
            path = entry["path"]
            if path not in blob_digests:
                data = git_show(f"{PINNED_BASE}:{path}")
                blob_digests[path] = hashlib.sha256(data).hexdigest()
                blob_rows[path] = data
            with self.subTest(entry=path, locator=entry["locator"]):
                # (a) every digest is the pinned blob's own bytes (VC-14)
                self.assertEqual(entry["sha256"], blob_digests[path])
                if entry["kind"] == "derived":
                    self.assertIsNone(entry["row_key"])
                elif path.endswith(".csv"):
                    self.assertIsInstance(entry["row_key"], dict)
                    self.assertIn("run_id", entry["row_key"])
                else:
                    self.assertIsNone(entry["row_key"])  # JSON pointers identify the row
        # (b+c+d) run-row entries: reported == the pinned cell, producer == the
        # mechanical key/columns test, evidence == fresh _frozen_fitness re-derivation
        run_row_entries = {}
        for entry in derived:
            if entry["artifact_class"] in ("csv_run_rows", "json_run_rows"):
                run_row_entries.setdefault(entry["path"], []).append(entry)
        checked_rows = 0
        for path, entries in sorted(run_row_entries.items()):
            data = blob_rows[path]
            if path.endswith(".csv"):
                reader = csv.DictReader(io.StringIO(data.decode("utf-8")), strict=True)
                rows = []
                for row in reader:
                    key = json.dumps(
                        {"run_id": int(row["run_id"])}, sort_keys=True, separators=(",", ":")
                    )
                    rows.append((key, dict(row)))
                profile = Path(path).stem.split("metrics_stability_")[1]
            else:
                document = json.loads(data.decode("utf-8"))
                rows = find_run_rows(document)
                profile = None
            by_row = {}
            for entry in entries:
                key = (
                    json.dumps(entry["row_key"], sort_keys=True, separators=(",", ":"))
                    if path.endswith(".csv")
                    else entry["locator"].rsplit("/", 1)[0]
                )
                by_row.setdefault(key, []).append(entry)
            self.assertEqual(set(by_row), {key for key, _ in rows})
            for row_key, row in rows:
                row_entries = by_row[row_key]
                checked_rows += 1
                tickers = (
                    ast.literal_eval(row["tickers"])
                    if isinstance(row["tickers"], str)
                    else row["tickers"]
                )
                row_profile = profile or row_key.lstrip("/").split("/", 1)[0]
                keys = list(row.keys())
                expected_producer = (
                    "corrected_capture"
                    if "generations_run" in keys and "converged_early" in keys
                    else "legacy_capture"
                )
                try:
                    evidence = self.fb.evidence_for_row(row_profile, tickers, row)
                    expected = {
                        "row_reconstruction": evidence.row_reconstruction,
                        "data_snapshot_match": evidence.data_snapshot_match,
                        "recomputed": evidence.recomputed,
                    }
                    score_pairs = evidence.score_fields
                except self.va.ff.FrozenError:
                    expected = {
                        "row_reconstruction": "inexact",
                        "data_snapshot_match": False,
                        "recomputed": None,
                    }
                    score_pairs = {
                        name: (
                            float(row[name]) if row.get(name) not in (None, "") else None,
                            None,
                        )
                        for name in row
                        if name.startswith("score_")
                    }
                expected["score_recompute"] = self.vd.score_recompute(score_pairs)
                for entry in row_entries:
                    name = entry["locator"].split("/")[-1]
                    if entry["kind"] == "score":
                        _, expected_recomputed = score_pairs.get(name, (None, None))
                    else:
                        expected_recomputed = expected["recomputed"]
                    with self.subTest(path=path, locator=entry["locator"]):
                        self.assertEqual(entry["producer"], expected_producer)
                        self.assertEqual(
                            entry["row_reconstruction"], expected["row_reconstruction"]
                        )
                        self.assertEqual(
                            entry["data_snapshot_match"], expected["data_snapshot_match"]
                        )
                        self.assertEqual(entry["score_recompute"], expected["score_recompute"])
                        if expected_recomputed is None:
                            self.assertIsNone(entry["recomputed"])
                        else:
                            self.assertLessEqual(
                                abs(entry["recomputed"] - expected_recomputed), TOL
                            )
                        cell = (
                            resolve_pointer(document, entry["locator"])
                            if path.endswith(".json")
                            else row[entry["locator"]]
                        )
                        cell = None if cell in (None, "") else float(cell)
                        self.assertEqual(entry["reported"], cell)
        self.assertGreaterEqual(checked_rows, 1000)  # 790 CSV + 240 JSON run rows
        # (e) every verdict recomputes from its own recorded evidence through the
        # shared rule (VC-24) and the annotation discipline holds
        verdicts = self.independent_verdicts(derived)
        by_id = {
            self.vd.entry_id(e["path"], e["locator"], e["row_key"]): e for e in derived
        }
        for entry_id, entry in by_id.items():
            with self.subTest(entry=entry_id):
                self.assertEqual(verdicts[entry_id], entry["verdict"])
                self.assertEqual(
                    self.vd.annotation_errors(entry, entry["verdict"]), []
                )
        self.assertTrue(any(e["verdict"] == "invalid_fitness" for e in derived))
        # (f) the map round-trips through check_status clean, and a tampered map fails
        with tempfile.TemporaryDirectory() as tmp:
            status_path = Path(tmp) / "status.json"
            status_path.write_text(
                json.dumps({"schema_version": 1, "pinned_base_commit": PINNED_BASE,
                            "entries": derived}),
                encoding="utf-8",
            )
            self.assertEqual(self.va.check_status(status_path, derived), [])
            tampered = json.loads(status_path.read_text(encoding="utf-8"))
            victim = next(e for e in tampered["entries"] if e["kind"] == "fitness")
            victim["reported"] = float(victim["reported"]) + 1.0
            status_path.write_text(json.dumps(tampered), encoding="utf-8")
            problems = self.va.check_status(status_path, derived)
            self.assertTrue(any("reported" in p for p in problems), problems)
            tampered = json.loads(json.dumps({"schema_version": 1, "entries": derived}))
            victim = next(e for e in tampered["entries"] if e["kind"] == "fitness")
            victim["verdict"] = "valid"
            status_path.write_text(json.dumps(tampered), encoding="utf-8")
            problems = self.va.check_status(status_path, derived)
            self.assertTrue(any("verdict" in p for p in problems), problems)

    def independent_verdicts(self, entries):
        """Bottom-up verdict recomputation via the shared `_verdict` rule only."""
        by_id = {self.vd.entry_id(e["path"], e["locator"], e["row_key"]): e for e in entries}
        verdicts = {}

        def verdict_of(entry_id):
            if entry_id in verdicts:
                return verdicts[entry_id]
            entry = by_id[entry_id]
            if entry["kind"] == "derived":
                value = self.vd.verdict_for(
                    {**entry, "contributor_verdicts": [verdict_of(c) for c in entry["contributors"]]}
                )
            else:
                value = self.vd.verdict_for(entry)
            verdicts[entry_id] = value
            return value

        for entry_id in by_id:
            verdict_of(entry_id)
        return verdicts


if __name__ == "__main__":
    unittest.main()
