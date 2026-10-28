"""tests/test_v0_corrected_property.py - the deterministic property enumeration
(spec.md VC-06, VC-07, VC-08; tasks T12; AD-012).

The tests derive from the spec's acceptance criteria, not from the implementation:

- VC-06: when the corrected GA records a generation best, the recorded pair
  (fitness, chromosome) SHALL be a member of that generation's evaluated
  (fitness, chromosome) pairs - the winning chromosome belongs to the generation
  of the winning fitness - for every enumerated generation and seed. Asserted by
  instrumenting the REAL `run_ga` (`ga.fitness` monkeypatched as a recorder,
  generations recovered by chunking the call log by `pop_size`) with both `ga`
  modules loaded via `importlib.util.spec_from_file_location` under unique module
  names. The loop is never re-implemented.
- VC-07: the enumeration is deterministic (no hypothesis) and covers synthetic
  frames containing NaN SCORE rows, duplicate SCORE ties, duplicate SETORs, and
  the even pop_size parity of the four frozen configs (200/250/300/276).
- VC-08: the same property evaluated against the frozen `baselines/v0-legacy`
  `ga.py` observes at least one violated generation, recorded as the expected
  defect signature - and every chromosome fed through `run_ga` stays feasible
  (`mask.sum() == n`), because any `-inf` score makes the frozen probs
  computation all-NaN and `np.random.choice` raises instead of exposing a
  violation.

The generation dimension is enumerated as a prefix sweep: `run_ga` is called with
`generations = G` for G = 1..N under one seed, so each generation's record state
is observed as an endpoint (deterministic seeding makes the first G chunks of a
G-run identical to the first G chunks of any longer run).

Interpreter: numpy/pandas required - runs under `/tmp/opencode/tcc-venv/bin/python`
and SKIPs with an explicit reason otherwise. Both `ga` modules are imported with
`sys.dont_write_bytecode` set first and cwd inside a scratch root (AD-013);
nothing here writes under either bundle.
"""

import contextlib
import importlib.util
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import pandas as pd

    HAVE_NUMERIC_STACK = True
    MISSING_STACK = ""
except ImportError as exc:  # pragma: no cover - exercised by the system interpreter
    HAVE_NUMERIC_STACK = False
    MISSING_STACK = f"numpy/pandas unavailable in this interpreter: {exc}"

REPO = Path(__file__).resolve().parents[1]
LEGACY = REPO / "baselines" / "v0-legacy"
CORRECTED = REPO / "baselines" / "v0-corrected"

FRAMES = ("nan_score", "tied_score", "duplicate_setor")
SEEDS = (3, 11)
FRAME_ROWS = 24
# the injected small even-parity config for the per-generation prefix sweep
SWEEP_PROFILE = "sweep"
SWEEP_N = 5
SWEEP_LAMBDA = 0.5
SWEEP_POP = 24
SWEEP_GENERATIONS = 6
# the four frozen configs keep their real n_assets/lambda/pop_size (VC-07 parity);
# only `generations` is shortened so the suite stays fast
FROZEN_POPS = {"conservador": 200, "moderado": 250, "arrojado": 300, "caio": 276}
PARITY_GENERATIONS = 3


def load_ga(path, name):
    """One frozen ga.py under a unique module name (AD-013 hygiene).

    The module is deliberately NOT registered in `sys.modules`: the P0-01 gate
    `test_v0_legacy_protocol` asserts no module from the frozen legacy `code/`
    ever lingers there, and `ga.py` needs no registration to execute.
    """
    sys.dont_write_bytecode = True  # before the bundle import (AD-013)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_frame(kind):
    """One deterministic synthetic ranked frame (unique tickers, duplicate SETORs)."""
    sectors = 3 if kind == "duplicate_setor" else 4
    tickers = [f"SYN{i:02d}3" for i in range(FRAME_ROWS)]
    setores = [f"S{i % sectors}" for i in range(FRAME_ROWS)]
    if kind == "tied_score":
        scores = [float(i % 4) for i in range(FRAME_ROWS)]  # duplicate SCORE ties
    elif kind == "nan_score":
        scores = [((i * 5) % 7) * 0.25 for i in range(FRAME_ROWS)]
        for index in (3, 10, 17):
            scores[index] = float("nan")  # NaN SCORE rows
    else:
        scores = [i * 0.1 for i in range(FRAME_ROWS)]  # all distinct
    return pd.DataFrame({"TICKER": tickers, "SETOR": setores, "SCORE": scores})


@contextlib.contextmanager
def frozen_config(ga_module, profile, override):
    """Run with one PERFIL_CONFIG entry overridden - the loop code stays untouched."""
    original = ga_module.PERFIL_CONFIG
    ga_module.PERFIL_CONFIG = {
        **original,
        profile: {**original.get(profile, {}), **override},
    }
    try:
        yield
    finally:
        ga_module.PERFIL_CONFIG = original


class FitnessRecorder:
    """`ga.fitness` monkeypatched as a recorder (VC-06) - observation, not re-implementation.

    The real frozen fitness still computes every value; the recorder logs each
    (fitness, chromosome) pair as it is evaluated and flags any infeasible mask
    instead of letting it reach the selection path unnoticed.
    """

    def __init__(self, real_fitness):
        self._real = real_fitness
        self.log = []  # (fitness, chromosome 0/1 tuple, feasible)
        self.infeasible_masks = 0

    def __call__(self, df, mask, lam, n):
        feasible = int(mask.sum()) == n
        if not feasible:
            self.infeasible_masks += 1
        value = float(self._real(df, mask, lam, n))
        self.log.append((value, tuple(int(v) for v in mask), feasible))
        return value


def instrumented_run(ga_module, frame, profile, seed_value):
    """One REAL run_ga call with the recorder installed; returns the run record."""
    config = ga_module.PERFIL_CONFIG[profile]
    pop_size = config["pop_size"]
    recorder = FitnessRecorder(ga_module.fitness)
    ga_module.fitness = recorder
    try:
        random.seed(seed_value)
        np.random.seed(seed_value)
        carteira = ga_module.run_ga(frame, profile)
    finally:
        ga_module.fitness = recorder._real
    tickers = set(carteira["TICKER"])
    best_chrom = tuple(1 if t in tickers else 0 for t in frame["TICKER"])
    chunks = [
        recorder.log[i:i + pop_size] for i in range(0, len(recorder.log), pop_size)
    ]
    return {
        "profile": profile,
        "seed": seed_value,
        "pop_size": pop_size,
        "generations": config["generations"],
        "n_assets": config["n_assets"],
        "chunks": chunks,
        "log_length": len(recorder.log),
        "best": (float(carteira.attrs["fitness"]), best_chrom),
        "infeasible_masks": recorder.infeasible_masks,
    }


def analyze_record(record):
    """VC-06 for one run: the generation of the winning fitness and its pair.

    `winner_generation` is the first generation whose evaluated maximum equals the
    recorded fitness (strict `>` keeps the first record); the property is that the
    recorded (fitness, chromosome) is a member of that generation's evaluated
    pairs - and is exactly the evaluated individual at the argmax index (first
    maximum) of that generation.
    """
    chunks = record["chunks"]
    best = record["best"]
    winner = None
    for generation, chunk in enumerate(chunks):
        if max(value for value, _, _ in chunk) == best[0]:
            winner = generation
            break
    if winner is None:
        raise AssertionError("the recorded fitness is not any generation's evaluated maximum")
    pairs = [(value, chrom) for value, chrom, _ in chunks[winner]]
    argmax_index = max(range(len(pairs)), key=lambda i: pairs[i][0])  # first maximum wins
    return {
        "winner_generation": winner,
        "pair_is_member": best in pairs,
        "winner_is_argmax": pairs[argmax_index] == best,
        "argmax_pair": pairs[argmax_index],
    }


@unittest.skipUnless(HAVE_NUMERIC_STACK, MISSING_STACK)
class PropertyEnumerationTests(unittest.TestCase):
    """VC-06/VC-07/VC-08 (T12): the deterministic enumeration over real run_ga."""

    maxDiff = None

    @classmethod
    def setUpClass(cls):
        sys.dont_write_bytecode = True  # before any bundle import (AD-013)
        cls._scratch = tempfile.TemporaryDirectory()
        cls._saved_cwd = os.getcwd()
        os.chdir(cls._scratch.name)  # bundle imports run with cwd in a scratch root
        cls.ga_corrected = load_ga(
            CORRECTED / "code" / "ga.py", "_test_v0_property_ga_corrected"
        )
        cls.ga_legacy = load_ga(
            LEGACY / "code" / "ga.py", "_test_v0_property_ga_legacy"
        )
        cls.frames = {kind: make_frame(kind) for kind in FRAMES}
        cls.sweep_runs = {"corrected": [], "legacy": []}
        cls.parity_runs = {"corrected": [], "legacy": []}
        for label, ga in (("corrected", cls.ga_corrected), ("legacy", cls.ga_legacy)):
            for frame_kind in FRAMES:
                for seed_value in SEEDS:
                    for generations in range(1, SWEEP_GENERATIONS + 1):
                        with frozen_config(
                            ga,
                            SWEEP_PROFILE,
                            {
                                "n_assets": SWEEP_N,
                                "lambda": SWEEP_LAMBDA,
                                "pop_size": SWEEP_POP,
                                "generations": generations,
                            },
                        ):
                            run = instrumented_run(
                                ga, cls.frames[frame_kind], SWEEP_PROFILE, seed_value
                            )
                        run["frame"] = frame_kind
                        cls.sweep_runs[label].append(run)
            for profile in FROZEN_POPS:
                for seed_value in SEEDS:
                    with frozen_config(ga, profile, {"generations": PARITY_GENERATIONS}):
                        run = instrumented_run(
                            ga, cls.frames["nan_score"], profile, seed_value
                        )
                    run["frame"] = "nan_score"
                    cls.parity_runs[label].append(run)

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._saved_cwd)
        cls._scratch.cleanup()
        # nothing this module loaded may linger in sys.modules (P0-01 gate)
        for name in ("_test_v0_property_ga_corrected", "_test_v0_property_ga_legacy"):
            sys.modules.pop(name, None)

    def corrected_runs(self):
        return self.sweep_runs["corrected"] + self.parity_runs["corrected"]

    def legacy_runs(self):
        return self.sweep_runs["legacy"] + self.parity_runs["legacy"]

    def test_recorded_pair_belongs_to_the_evaluated_pairs_of_its_generation(self):
        for run in self.corrected_runs():
            analysis = analyze_record(run)
            with self.subTest(
                frame=run["frame"], profile=run["profile"], seed=run["seed"],
                generations=run["generations"],
            ):
                # the recorded pair is a member of the winning generation's evaluated pairs
                self.assertTrue(analysis["pair_is_member"])
                # and the winner is the evaluated individual at the argmax index of
                # its own generation (first-max resolution is part of the protocol)
                self.assertTrue(analysis["winner_is_argmax"])
                self.assertEqual(analysis["argmax_pair"], run["best"])

    def test_enumeration_covers_nan_score_ties_duplicate_setor_and_pop_parity(self):
        nan_frame = self.frames["nan_score"]
        self.assertGreater(int(nan_frame["SCORE"].isna().sum()), 0)  # NaN SCORE rows
        tied = self.frames["tied_score"]
        self.assertGreater(len(tied["SCORE"]), len(set(tied["SCORE"])))  # duplicate ties
        for kind, frame in self.frames.items():
            with self.subTest(frame=kind):
                self.assertGreater(int(frame["SETOR"].duplicated().sum()), 0)  # duplicate SETORs
                self.assertTrue(frame["TICKER"].is_unique)
        # the even pop_size parity of the four frozen configs is exercised as-is
        for profile, expected_pop in FROZEN_POPS.items():
            with self.subTest(profile=profile):
                self.assertEqual(self.ga_corrected.PERFIL_CONFIG[profile]["pop_size"],
                                 expected_pop)
                self.assertEqual(expected_pop % 2, 0)
                parity = [
                    r for r in self.parity_runs["corrected"] if r["profile"] == profile
                ]
                self.assertEqual(len(parity), len(SEEDS))
                for run in parity:
                    self.assertEqual(run["pop_size"], expected_pop)
        self.assertEqual(set(FROZEN_POPS), set(self.ga_corrected.PERFIL_CONFIG))
        self.assertEqual(self.ga_legacy.PERFIL_CONFIG, self.ga_corrected.PERFIL_CONFIG)
        # the enumeration is deterministic: identical frames and identical re-runs
        for kind in FRAMES:
            self.assertTrue(self.frames[kind].equals(make_frame(kind)))
        original = next(
            r for r in self.sweep_runs["corrected"]
            if r["frame"] == FRAMES[0] and r["seed"] == SEEDS[0]
            and r["generations"] == SWEEP_GENERATIONS
        )
        with frozen_config(
            self.ga_corrected,
            SWEEP_PROFILE,
            {
                "n_assets": SWEEP_N,
                "lambda": SWEEP_LAMBDA,
                "pop_size": SWEEP_POP,
                "generations": SWEEP_GENERATIONS,
            },
        ):
            rerun = instrumented_run(
                self.ga_corrected, self.frames[FRAMES[0]], SWEEP_PROFILE, SEEDS[0]
            )
        self.assertEqual(rerun["best"], original["best"])
        self.assertEqual(rerun["chunks"], original["chunks"])

    def test_population_length_stays_pop_size_every_generation(self):
        for label, runs in (
            ("corrected", self.corrected_runs()),
            ("legacy", self.legacy_runs()),
        ):
            for run in runs:
                with self.subTest(
                    module=label, frame=run["frame"], profile=run["profile"],
                    seed=run["seed"], generations=run["generations"],
                ):
                    self.assertEqual(len(run["chunks"]), run["generations"])
                    for chunk in run["chunks"]:
                        # every generation evaluated exactly pop_size individuals,
                        # so the recorded index always addresses an evaluated member
                        self.assertEqual(len(chunk), run["pop_size"])
                    self.assertEqual(
                        run["log_length"], run["generations"] * run["pop_size"]
                    )
                    self.assertEqual(
                        sum(len(chunk) for chunk in run["chunks"]), run["log_length"]
                    )

    def test_legacy_module_shows_a_violated_generation_as_defect_signature(self):
        signatures = []
        for run in self.legacy_runs():
            analysis = analyze_record(run)
            if analysis["pair_is_member"]:
                continue
            # the defect signature: the recorded fitness is the generation's own
            # argmax value, but the recorded chromosome does not belong to that
            # generation's evaluated pairs
            self.assertEqual(analysis["argmax_pair"][0], run["best"][0])
            self.assertNotEqual(analysis["argmax_pair"][1], run["best"][1])
            signatures.append(
                {
                    "frame": run["frame"],
                    "profile": run["profile"],
                    "seed": run["seed"],
                    "generations": run["generations"],
                    "winner_generation": analysis["winner_generation"],
                    "recorded_fitness": run["best"][0],
                    "generation_fitness": analysis["argmax_pair"][0],
                }
            )
        self.assertGreaterEqual(len(signatures), 1)  # >= 1 violated generation observed
        # the discrimination: the corrected module has zero violations over the
        # very same enumeration (proven in the positive test) while the frozen
        # defect shows them - the test distinguishes the two
        corrected_violations = [
            run for run in self.corrected_runs()
            if not analyze_record(run)["pair_is_member"]
        ]
        self.assertEqual(corrected_violations, [])

    def test_infeasible_masks_are_never_fed_through_the_selection_path(self):
        for runs in (self.sweep_runs, self.parity_runs):
            for label in ("corrected", "legacy"):
                for run in runs[label]:
                    with self.subTest(module=label, profile=run["profile"],
                                      seed=run["seed"]):
                        self.assertEqual(run["infeasible_masks"], 0)
                        for chunk in run["chunks"]:
                            for _, _, feasible in chunk:
                                self.assertTrue(feasible)
        # infeasible masks are exercised only against the pure fitness() function
        frame = self.frames["nan_score"]
        mask = np.zeros(len(frame), dtype=int)
        mask[: SWEEP_N - 1] = 1  # n-1 selected: infeasible
        self.assertEqual(
            float(self.ga_corrected.fitness(frame, mask, SWEEP_LAMBDA, SWEEP_N)),
            float("-inf"),
        )
        # and never through the frozen selection path: an -inf score makes the
        # frozen probs computation all-NaN and np.random.choice raises instead of
        # exposing a violation (VC-08's measured mechanism)
        scores = np.array([2.0, 1.0, float("-inf")])
        min_fit = scores.min()
        with np.errstate(invalid="ignore"):  # -inf - -inf is the NaN the spec measures
            probs = scores - min_fit + 1e-9
            probs /= probs.sum()
        self.assertTrue(np.isnan(probs).all())
        with self.assertRaises(ValueError):
            np.random.choice(len(scores), p=probs, size=2, replace=False)


if __name__ == "__main__":
    unittest.main()
