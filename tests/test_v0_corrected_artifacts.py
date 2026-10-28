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
import importlib.util
import os
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


if __name__ == "__main__":
    unittest.main()
