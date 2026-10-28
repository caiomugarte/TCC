"""v0-corrected (P0-02) replay checker (spec.md VC-20..VC-22, VC-27; tasks T17-T18).

The tests derive from the spec's acceptance criteria, not from the implementation:

- VC-20: a run is one (profile, seed) pair, scope 20 seeds x 4 frozen profiles = 80
  runs (run_ids 0-19 = seeds 42-61), and the proof's top-level summary carries the
  computed run denominator so the gate cannot silently change scope.
- VC-21: the E1-E4 expectation rows are recomputed from `runs[]`; any violated row is
  reported as `unexpected divergence`, naming the row, and exits non-zero - never
  folded into "expected".
- VC-22: nothing executes unless `TCC_V0_CORRECTED_REPLAY=1`; without the flag every
  mode exits 2 before any driver spawns.
- VC-27: a fixture that injects one expectation-row violation proves the checker's
  failure branch executes (non-zero exit, `unexpected divergence` named).

Standard library only, synthetic in-memory/tempdir proofs - no driver ever executes
here. Nothing is written under either bundle.
"""

import contextlib
import importlib.util
import io
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CORRECTED = REPO / "baselines" / "v0-corrected"
PROOF_PATH = (
    REPO / ".specs" / "features" / "p0-02-v0-corrected" / "proofs" / "replay_20seeds.json"
)
PROFILES = ("conservador", "moderado", "arrojado", "caio")
N_RUN_IDS = 20
SEED_BASE = 42
TOL = 1e-9


def load_replay_tool():
    """Import compare_replay.py from the bundle's tools/ under a unique module name.

    Callers must have set sys.dont_write_bytecode first and must run with cwd inside a
    scratch root (AD-013).
    """
    sys.dont_write_bytecode = True  # before the bundle import (AD-013)
    path = CORRECTED / "tools" / "compare_replay.py"
    spec = importlib.util.spec_from_file_location("_test_v0_corrected_compare_replay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_runs(e1_flips=(), e2_flips=(), e3_equal=(), e4_agree=()):
    """One synthetic 80-run grid with per-expectation misses injected at known indices.

    Index i is (profile, run_id) in canonical order. Defaults make all four
    expectations hold on 80/80; each parameter moves exactly its own numerator:
    `e1_flips` runs whose two reported fitness values differ bit-wise, `e2_flips`
    runs whose corrected recomputed fitness differs from its reported, `e3_equal`
    runs whose ticker sets are equal (so E3 counts 80-len), `e4_agree` runs whose
    legacy recomputed fitness equals its reported (so E4 counts 80-len).
    """
    e1_flips, e2_flips = set(e1_flips), set(e2_flips)
    e3_equal, e4_agree = set(e3_equal), set(e4_agree)
    runs = []
    index = 0
    for profile in PROFILES:
        for run_id in range(N_RUN_IDS):
            base = 4.0 + index / 100.0
            corrected_reported = base + 0.125 if index in e1_flips else base
            corrected_recomputed = (
                corrected_reported + 0.5 if index in e2_flips else corrected_reported
            )
            legacy_reported = base
            legacy_recomputed = base if index in e4_agree else base + 0.5
            legacy_tickers = [f"LT{index:03d}{i:02d}3" for i in range(10)]
            corrected_tickers = (
                list(legacy_tickers) if index in e3_equal else [f"CT{index:03d}{i:02d}3" for i in range(10)]
            )
            runs.append(
                {
                    "profile": profile,
                    "run_id": run_id,
                    "seed": SEED_BASE + run_id,
                    "legacy": {
                        "reported_fitness": legacy_reported,
                        "recomputed_fitness": legacy_recomputed,
                        "tickers": legacy_tickers,
                        "hhi": 0.1,
                    },
                    "corrected": {
                        "reported_fitness": corrected_reported,
                        "recomputed_fitness": corrected_recomputed,
                        "tickers": corrected_tickers,
                        "hhi": 0.2,
                    },
                    "ticker_sets_equal": index in e3_equal,
                }
            )
            index += 1
    return runs


def independent_fold(runs):
    """E1-E4 numerators recomputed from the run rows by the spec's own semantics.

    E1: packed-float64 bit-identity of the two reported fitness values;
    E2: corrected recomputed == corrected reported within 1e-9;
    E3: the two ticker sets differ (as sets);
    E4: legacy recomputed != legacy reported beyond 1e-9.
    """

    def equal(a, b):
        return abs(float(a) - float(b)) <= TOL

    return {
        "E1": sum(
            1
            for r in runs
            if struct.pack(">d", float(r["legacy"]["reported_fitness"]))
            == struct.pack(">d", float(r["corrected"]["reported_fitness"]))
        ),
        "E2": sum(
            1
            for r in runs
            if equal(r["corrected"]["recomputed_fitness"], r["corrected"]["reported_fitness"])
        ),
        "E3": sum(
            1 for r in runs if set(r["legacy"]["tickers"]) != set(r["corrected"]["tickers"])
        ),
        "E4": sum(
            1
            for r in runs
            if not equal(r["legacy"]["recomputed_fitness"], r["legacy"]["reported_fitness"])
        ),
    }


class ReplayCheckerTests(unittest.TestCase):
    """VC-20/VC-21/VC-27: the expectation fold, the scope pin, the failure branch."""

    @classmethod
    def setUpClass(cls):
        sys.dont_write_bytecode = True  # before any bundle import (AD-013)
        cls._scratch = tempfile.TemporaryDirectory()
        cls._saved_cwd = os.getcwd()
        os.chdir(cls._scratch.name)  # bundle imports run with cwd in a scratch root
        cls.cr = load_replay_tool()

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._saved_cwd)
        cls._scratch.cleanup()

    @staticmethod
    def run_main(module, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = module.main(argv)
        return code, out.getvalue(), err.getvalue()

    @staticmethod
    def with_replay_flag():
        """Save/restore wrapper around TCC_V0_CORRECTED_REPLAY=1 (the opt-in gate)."""
        return _EnvFlag("1")

    def test_check_expectations_recomputes_e1_e4_from_the_run_rows(self):
        cr = self.cr
        # all four hold over non-trivial numerators: E1 80/80, E2 80/80, E3 41/80, E4 1/80
        runs = synthetic_runs(e3_equal=range(39), e4_agree=range(79))
        fold = independent_fold(runs)
        self.assertEqual(fold, {"E1": 80, "E2": 80, "E3": 41, "E4": 1})

        proof = cr.build_proof(runs, {"python_VV": "Python test", "numpy": "2.5.4", "pandas": "3.1.0"})
        self.assertEqual(proof["denominator"], 80)
        recorded = {row["name"]: row for row in proof["expectations"]}
        for name, numerator in fold.items():
            self.assertEqual(recorded[name]["numerator"], numerator, name)
            self.assertEqual(recorded[name]["denominator"], 80, name)
            self.assertIs(recorded[name]["held"], True, name)
        self.assertEqual(cr.check_expectations(proof), [])

        # the fold comes from runs[], not from the summary: a tampered summary is flagged
        tampered = json.loads(json.dumps(proof))
        tampered["expectations"][2]["numerator"] = 99
        problems = cr.check_expectations(tampered)
        self.assertTrue(
            any("summary disagrees" in p and "E3" in p for p in problems), problems
        )

        # threshold semantics over the computed denominator, one flip at a time
        cases = (
            (synthetic_runs(e1_flips=[0]), "E1", 79),          # needs 80/80
            (synthetic_runs(e2_flips=[0]), "E2", 79),          # needs 80/80 rows
            (synthetic_runs(e3_equal=range(40)), "E3", 40),    # needs >= 41/80
            (synthetic_runs(e4_agree=range(80)), "E4", 0),     # needs >= 1 run
        )
        for runs, name, numerator in cases:
            proof = cr.build_proof(runs, {"python_VV": "Python t", "numpy": None, "pandas": None})
            self.assertEqual(independent_fold(runs)[name], numerator, name)
            problems = cr.check_expectations(proof)
            violations = [p for p in problems if p.startswith("unexpected divergence")]
            self.assertEqual(len(violations), 1, (name, problems))
            self.assertIn(name, violations[0])
            self.assertIn(f"{numerator}/80", violations[0])

    def test_env_gate_refuses_execution_without_the_replay_flag(self):
        cr = self.cr
        sentinel = []

        def refuse(*args, **kwargs):
            sentinel.append(args)
            raise AssertionError("run_pair must not be called without the opt-in flag")

        original = cr.run_pair
        cr.run_pair = refuse
        try:
            with _EnvFlag(None):
                with tempfile.TemporaryDirectory() as tmp:
                    proof_path = Path(tmp) / "proof.json"
                    code, out, err = self.run_main(cr, ["--proof", str(proof_path)])
                    self.assertEqual(code, 2)
                    self.assertIn("TCC_V0_CORRECTED_REPLAY", err)
                    self.assertFalse(proof_path.exists())  # zero execution, nothing recorded
                    code, out, err = self.run_main(
                        cr, ["--check-proof", str(PROOF_PATH)]
                    )
                    self.assertEqual(code, 2)
                    self.assertIn("TCC_V0_CORRECTED_REPLAY", err)
        finally:
            cr.run_pair = original
        self.assertEqual(sentinel, [])  # no driver was ever spawned

    def test_injected_expectation_violation_reports_unexpected_divergence_and_exits_nonzero(self):
        cr = self.cr
        env = {"python_VV": "Python test", "numpy": "2.5.4", "pandas": "3.1.0"}
        with tempfile.TemporaryDirectory() as tmp:
            clean_path = Path(tmp) / "clean.json"
            clean_path.write_text(
                json.dumps(cr.build_proof(synthetic_runs(), env)) + "\n", encoding="utf-8"
            )
            with self.with_replay_flag():
                code, out, err = self.run_main(cr, ["--check-proof", str(clean_path)])
            self.assertEqual(code, 0, err)  # the positive control passes

            # inject exactly one expectation-row violation: run 0's two reported fitness
            # values stop being bit-identical, so E1 folds to 79/80
            runs = synthetic_runs(e1_flips=[0])
            proof = cr.build_proof(runs, env)
            violated_path = Path(tmp) / "violated.json"
            violated_path.write_text(json.dumps(proof) + "\n", encoding="utf-8")

            with self.with_replay_flag():
                code, out, err = self.run_main(cr, ["--check-proof", str(violated_path)])
            self.assertNotEqual(code, 0)
            self.assertIn("unexpected divergence", err)
            self.assertIn("E1", err)

            # a summary that claims held=True can never launder the violation
            laundered = json.loads(json.dumps(proof))
            laundered["expectations"][0]["held"] = True
            laundered_path = Path(tmp) / "laundered.json"
            laundered_path.write_text(json.dumps(laundered) + "\n", encoding="utf-8")
            with self.with_replay_flag():
                code, out, err = self.run_main(cr, ["--check-proof", str(laundered_path)])
            self.assertNotEqual(code, 0)
            self.assertIn("unexpected divergence", err)
            self.assertIn("E1", err)


class CommittedProofTests(unittest.TestCase):
    """VC-20/VC-29 over the committed proof: schema + denominator, fold, reference tie."""

    @classmethod
    def setUpClass(cls):
        sys.dont_write_bytecode = True  # before any bundle import (AD-013)
        cls._scratch = tempfile.TemporaryDirectory()
        cls._saved_cwd = os.getcwd()
        os.chdir(cls._scratch.name)  # bundle imports run with cwd in a scratch root
        cls.cr = load_replay_tool()
        cls.proof = json.loads(PROOF_PATH.read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._saved_cwd)
        cls._scratch.cleanup()

    def test_committed_proof_schema_carries_the_80_denominator_runs_and_env_stamp(self):
        cr, proof = self.cr, self.proof
        self.assertEqual(cr.check_schema(proof), [])
        self.assertEqual(proof["schema_version"], 1)
        self.assertEqual(proof["denominator"], 80)
        self.assertEqual(proof["denominator"], len(proof["runs"]))  # computed, not asserted
        self.assertEqual(proof["run_id_range"], [0, 19])
        self.assertEqual(proof["seeds"], list(range(42, 62)))
        self.assertEqual(proof["profiles"], list(PROFILES))
        self.assertEqual(
            [(r["profile"], r["run_id"]) for r in proof["runs"]],
            [(p, run_id) for p in PROFILES for run_id in range(20)],
        )
        # per run BOTH drivers record reported and recomputed fitness, tickers and hhi
        for run in proof["runs"]:
            for side in ("legacy", "corrected"):
                driver = run[side]
                for key in ("reported_fitness", "recomputed_fitness", "hhi"):
                    self.assertIsInstance(driver[key], (int, float), (run["profile"], run["run_id"], side, key))
                self.assertTrue(driver["tickers"], (run["profile"], run["run_id"], side))
        # the env stamp names the declared environment both drivers ran in (VC-19 pins)
        env = proof["env"]
        self.assertEqual(set(env), {"python_VV", "numpy", "pandas"})
        self.assertTrue(env["python_VV"].startswith("Python "))
        pins = {}
        for line in (CORRECTED / "requirements.txt").read_text(encoding="utf-8").splitlines():
            name, _, version = line.partition("==")
            pins[name.strip()] = version.strip()
        self.assertEqual(env["numpy"], pins["numpy"])
        self.assertEqual(env["pandas"], pins["pandas"])

    def test_proof_fold_matches_its_summary_numerators_over_the_denominator(self):
        cr, proof = self.cr, self.proof
        fold = independent_fold(proof["runs"])
        recorded = {row["name"]: row for row in proof["expectations"]}
        self.assertEqual(sorted(recorded), ["E1", "E2", "E3", "E4"])
        for name, numerator in fold.items():
            self.assertEqual(recorded[name]["numerator"], numerator, name)
            self.assertEqual(recorded[name]["denominator"], 80, name)
        # E1-E4 hold over the computed denominator (the spec's thresholds as data)
        self.assertEqual(fold["E1"], 80)   # bit-identical on 80/80 runs
        self.assertEqual(fold["E2"], 80)   # corrected recomputed == reported on 80/80 rows
        self.assertGreaterEqual(fold["E3"], 41)  # ticker sets differ on >= 41/80 runs
        self.assertGreaterEqual(fold["E4"], 1)   # legacy recomputed != reported on >= 1 run
        for row in proof["expectations"]:
            self.assertIs(row["held"], True, row["name"])
        self.assertEqual(cr.check_expectations(proof), [])

    def test_proof_corrected_rows_tie_to_the_committed_reference_for_run_id_0_19(self):
        cr, proof = self.cr, self.proof
        self.assertEqual(len(proof["runs"]), 80)
        problems = cr.check_proof_vs_reference(proof, CORRECTED / "reference")
        self.assertEqual(problems, [])


class _EnvFlag:
    """Context manager: set TCC_V0_CORRECTED_REPLAY to a value (None removes it)."""

    def __init__(self, value):
        self.value = value

    def __enter__(self):
        self.saved = os.environ.get("TCC_V0_CORRECTED_REPLAY")
        if self.value is None:
            os.environ.pop("TCC_V0_CORRECTED_REPLAY", None)
        else:
            os.environ["TCC_V0_CORRECTED_REPLAY"] = self.value
        return self

    def __exit__(self, *exc):
        if self.saved is None:
            os.environ.pop("TCC_V0_CORRECTED_REPLAY", None)
        else:
            os.environ["TCC_V0_CORRECTED_REPLAY"] = self.saved
        return False


if __name__ == "__main__":
    unittest.main()
