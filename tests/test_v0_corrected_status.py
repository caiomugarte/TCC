"""v0-corrected (P0-02) verdict semantics (spec.md VC-12, VC-25, VC-29; tasks T6).

The tests derive from the spec's acceptance criteria, not from the implementation:

- VC-12 "Verdict Attribution Rule": every decision-table row resolves in the spec's
  order for the `fitness`, `score` and `derived` kinds; `data_snapshot_match == false`
  resolves before any producer rule; the corrected-capture mispairing row raises
  instead of becoming a verdict; `defect_ref` attaches only to
  `invalid_fitness`/`invalid_beyond_fitness` and every other non-`valid` verdict
  requires a `reason`.
- VC-25 "bit-identical": bit identity is equality of the packed float64 cells
  (`struct.pack('>d')`), not of nearby values.
- VC-29 row tie: tickers as a set + `hhi` + `fitness` within 1e-9; a forged row fails.

One spec-precision note (not a gap in these assertions): the table writes
`reported == recomputed` without a tolerance while VC-01/VC-11 recompute "within
1e-9"; `verdict_for` reads that comparison at the same 1e-9 as the recomputation
checks. The assertions here use crisp values that resolve under any reading.

Standard library only. The module under test is the bundle's own `tools/_verdict.py`,
loaded under a unique name with `sys.dont_write_bytecode` set first and cwd inside a
scratch root (AD-013) - the single verdict path shared by generator and checker.
"""

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CORRECTED = REPO / "baselines" / "v0-corrected"


def load_bundle_tool(name):
    """Import a bundle tool from baselines/v0-corrected/tools/ under a unique module name.

    Callers must have set sys.dont_write_bytecode first and must run with cwd inside a
    scratch root (AD-013).
    """
    sys.dont_write_bytecode = True  # before the bundle import (AD-013)
    path = CORRECTED / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_test_v0_corrected_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fitness_entry(**overrides):
    """One evidence-complete fitness entry: exact reconstruction, snapshot match."""
    entry = {
        "kind": "fitness",
        "producer": "legacy_capture",
        "row_reconstruction": "exact",
        "data_snapshot_match": True,
        "score_recompute": "match",
        "reported": 4.604745955851804,
        "recomputed": 4.604745955851804,
    }
    entry.update(overrides)
    return entry


class VerdictRuleOrderTests(unittest.TestCase):
    """VC-12: the decision-table rows resolve in the spec's order for all three kinds."""

    @classmethod
    def setUpClass(cls):
        sys.dont_write_bytecode = True  # before any bundle import (AD-013)
        cls._scratch = tempfile.TemporaryDirectory()
        cls._saved_cwd = os.getcwd()
        os.chdir(cls._scratch.name)  # bundle imports run with cwd in a scratch root
        cls.vd = load_bundle_tool("_verdict")

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._saved_cwd)
        cls._scratch.cleanup()

    def test_decision_table_rows_resolve_in_order_for_fitness_score_and_derived(self):
        cases = [
            # fitness rows, in the spec's order
            ("fitness inexact", fitness_entry(row_reconstruction="inexact",
                                              producer="legacy_capture", reported=1.0, recomputed=2.0),
             "not_verifiable_at_head"),
            ("fitness snapshot false", fitness_entry(data_snapshot_match=False,
                                                     producer="legacy_capture", reported=1.0, recomputed=2.0),
             "not_verifiable_at_head"),
            ("fitness score mismatch", fitness_entry(score_recompute="mismatch",
                                                     producer="legacy_capture", reported=1.0, recomputed=1.0),
             "score_provenance_inconsistent"),
            ("fitness producer unrecorded", fitness_entry(producer="unrecorded"),
             "not_verifiable_at_head"),
            ("fitness legacy reported != recomputed", fitness_entry(reported=4.604745955851804,
                                                                    recomputed=0.8156341406041691),
             "invalid_fitness"),
            ("fitness legacy reported == recomputed", fitness_entry(), "valid"),
            ("fitness corrected reported == recomputed", fitness_entry(producer="corrected_capture"),
             "valid"),
            # score rows: inexact / snapshot / mismatch / match
            ("score inexact", {"kind": "score", "producer": "legacy_capture",
                               "row_reconstruction": "inexact", "data_snapshot_match": True,
                               "score_recompute": "match", "reported": 1.0, "recomputed": 1.0},
             "not_verifiable_at_head"),
            ("score snapshot false", {"kind": "score", "producer": "legacy_capture",
                                      "row_reconstruction": "exact", "data_snapshot_match": False,
                                      "score_recompute": "match", "reported": 1.0, "recomputed": 1.0},
             "not_verifiable_at_head"),
            ("score recompute mismatch", {"kind": "score", "producer": "legacy_capture",
                                          "row_reconstruction": "exact", "data_snapshot_match": True,
                                          "score_recompute": "mismatch", "reported": 1.0, "recomputed": 2.0},
             "score_provenance_inconsistent"),
            ("score recompute match", {"kind": "score", "producer": "legacy_capture",
                                       "row_reconstruction": "exact", "data_snapshot_match": True,
                                       "score_recompute": "match", "reported": 1.0, "recomputed": 1.0},
             "valid"),
            # derived rows: contributor precedence in the spec's order
            ("derived any invalid_fitness", {"kind": "derived",
                                             "contributor_verdicts": ["valid", "invalid_fitness"]},
             "invalid_beyond_fitness"),
            ("derived else any not_verifiable", {"kind": "derived", "contributor_verdicts":
                                                 ["valid", "not_verifiable_at_head",
                                                  "score_provenance_inconsistent"]},
             "not_verifiable_at_head"),
            ("derived else any score_provenance", {"kind": "derived", "contributor_verdicts":
                                                   ["valid", "score_provenance_inconsistent"]},
             "score_provenance_inconsistent"),
            ("derived all valid", {"kind": "derived", "contributor_verdicts": ["valid", "valid"]},
             "valid"),
            ("derived no contributors", {"kind": "derived", "contributor_verdicts": []},
             "not_verifiable_at_head"),
        ]
        for label, entry, expected in cases:
            with self.subTest(row=label):
                self.assertEqual(self.vd.verdict_for(entry), expected)

        # "in order": an earlier row wins over a later one the entry also satisfies
        self.assertEqual(
            self.vd.verdict_for(fitness_entry(row_reconstruction="inexact",
                                              producer="legacy_capture",
                                              reported=1.0, recomputed=2.0)),
            "not_verifiable_at_head",
            "row_reconstruction must resolve before the producer rows",
        )
        self.assertEqual(
            self.vd.verdict_for(fitness_entry(score_recompute="mismatch",
                                              producer="legacy_capture",
                                              reported=1.0, recomputed=2.0)),
            "score_provenance_inconsistent",
            "score_recompute must resolve before the producer rows",
        )

    def test_data_snapshot_mismatch_resolves_before_every_producer_rule(self):
        for producer in self.vd.PRODUCERS:
            entry = fitness_entry(producer=producer, data_snapshot_match=False,
                                  reported=1.0, recomputed=2.0)
            with self.subTest(producer=producer):
                # legacy would otherwise be invalid_fitness and corrected would raise
                # MispairingError; the snapshot discriminator resolves first, always
                self.assertEqual(self.vd.verdict_for(entry), "not_verifiable_at_head")

    def test_corrected_capture_mispairing_raises_instead_of_becoming_a_verdict(self):
        mispaired = fitness_entry(producer="corrected_capture", score_recompute="match",
                                  reported=6.2445628, recomputed=4.604745955851804)
        with self.assertRaises(self.vd.MispairingError):
            self.vd.verdict_for(mispaired)
        self.assertTrue(issubclass(self.vd.MispairingError, self.vd.VerdictError))
        self.assertNotIn("mispairing", self.vd.VERDICTS)
        for verdict in self.vd.VERDICTS:
            self.assertNotEqual(verdict, "mispairing")
        # the corrected row that does reproduce is valid - the fix's own row
        self.assertEqual(
            self.vd.verdict_for(fitness_entry(producer="corrected_capture")),
            "valid",
        )

    def test_defect_ref_only_attaches_to_the_two_defect_verdicts(self):
        defect_verdicts = ("invalid_fitness", "invalid_beyond_fitness")
        for verdict in self.vd.VERDICTS:
            with self.subTest(verdict=verdict, annotation="defect_ref"):
                problems = self.vd.annotation_errors({"defect_ref": "P0-02 capture defect"}, verdict)
                if verdict in defect_verdicts:
                    self.assertEqual(problems, [])
                else:
                    self.assertTrue(any("attaches only" in p for p in problems))
            with self.subTest(verdict=verdict, annotation="reason"):
                problems = self.vd.annotation_errors(
                    {"reason": "recorded score fields do not reproduce"}, verdict
                )
                if verdict in defect_verdicts:
                    self.assertTrue(any("defect_ref" in p for p in problems))
                else:
                    self.assertEqual(problems, [])
            with self.subTest(verdict=verdict, annotation="bare"):
                problems = self.vd.annotation_errors({}, verdict)
                if verdict in defect_verdicts:
                    self.assertTrue(any("defect_ref" in p for p in problems))
                elif verdict == "valid":
                    self.assertEqual(problems, [])
                else:
                    self.assertTrue(any("reason" in p for p in problems))


class BitIdentityAndTieTests(unittest.TestCase):
    """VC-25 bit-identity and VC-29 row-tie semantics."""

    @classmethod
    def setUpClass(cls):
        sys.dont_write_bytecode = True  # before any bundle import (AD-013)
        cls._scratch = tempfile.TemporaryDirectory()
        cls._saved_cwd = os.getcwd()
        os.chdir(cls._scratch.name)  # bundle imports run with cwd in a scratch root
        cls.vd = load_bundle_tool("_verdict")

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._saved_cwd)
        cls._scratch.cleanup()

    def test_bitwise_float_equality_compares_packed_bits(self):
        vd = self.vd
        # the same parsed cell compares equal
        self.assertTrue(vd.float_bitwise_equal(4.480605754022773, 4.480605754022773))
        # two floats that are mathematically "close" but differ in their packed bits
        self.assertFalse(vd.float_bitwise_equal(0.1 + 0.2, 0.3))
        self.assertAlmostEqual(0.1 + 0.2, 0.3, places=15)  # closeness is not bit-identity
        # zero and negative zero are different cells
        self.assertFalse(vd.float_bitwise_equal(0.0, -0.0))

    def test_tie_row_rejects_forged_fitness_and_mismatched_hhi(self):
        vd = self.vd
        recorded = {"tickers": ["ALUP11", "AZZA3", "CAML3"], "fitness": 6.2445628, "hhi": 0.16}
        driver = {"tickers": ["ALUP11", "AZZA3", "CAML3"], "fitness": 6.2445628, "hhi": 0.16}
        self.assertEqual(vd.tie_row(recorded, driver), [])
        # ticker order is not part of the tie - tickers compare as a set
        self.assertEqual(
            vd.tie_row(recorded, {"tickers": ["CAML3", "ALUP11", "AZZA3"],
                                  "fitness": 6.2445628, "hhi": 0.16}),
            [],
        )
        # a forged fitness (beyond 1e-9) fails
        forged = dict(driver, fitness=4.604745955851804)
        problems = vd.tie_row(recorded, forged)
        self.assertTrue(any("fitness" in p for p in problems))
        # a mismatched hhi fails
        problems = vd.tie_row(recorded, dict(driver, hhi=0.26))
        self.assertTrue(any("hhi" in p for p in problems))
        # a legacy chromosome under corrected fitness fails on the ticker set
        legacy = {"tickers": ["BRFS3", "BRKM5", "USIM5"], "fitness": 6.2445628, "hhi": 0.16}
        problems = vd.tie_row(recorded, legacy)
        self.assertTrue(any("tickers" in p for p in problems))
        # the tie rejects a row missing a compared field instead of passing it
        problems = vd.tie_row(recorded, {"tickers": ["ALUP11"], "fitness": 6.2445628})
        self.assertTrue(any("missing hhi" in p for p in problems))


if __name__ == "__main__":
    unittest.main()
