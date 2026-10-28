"""v0-corrected (P0-02) verdict semantics and the marking layer (spec.md VC-12 to
VC-16, VC-24, VC-25, VC-29; tasks T6, T13).

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
- VC-13/VC-14/VC-24/VC-15 (the committed `status.json`): every strict `fitness`
  locator of the two subject trees appears exactly once per recorded value,
  enumerated by JSON/CSV structure; every digest equals the git-blob computation at
  the pinned base (the map never pins itself); every verdict recomputes from its own
  evidence fields through the same `_verdict.verdict_for` path; the v0-legacy gates
  stay `VERIFIED 17` and the 52 P0-01 tests pass unmodified.

One spec-precision note (not a gap in these assertions): the table writes
`reported == recomputed` without a tolerance while VC-01/VC-11 recompute "within
1e-9"; `verdict_for` reads that comparison at the same 1e-9 as the recomputation
checks. The assertions here use crisp values that resolve under any reading. A second
note: "strict `fitness` locator" is read as every recorded fitness value - run-row
CSV `fitness` cells, scalar JSON leaves keyed exactly `fitness`, and scalar leaves
under a `fitness` object (VC-13's "nested and aggregate `fitness` values ... by JSON
structure"). Only that reading reproduces the spec's own subject counts (5 of 9
legacy-reference files, 15 of 160 tracked `outputs/` files).

Standard library only. The module under test is the bundle's own `tools/_verdict.py`,
loaded under a unique name with `sys.dont_write_bytecode` set first and cwd inside a
scratch root (AD-013) - the single verdict path shared by generator and checker.
"""

import csv
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CORRECTED = REPO / "baselines" / "v0-corrected"
PINNED_BASE = "181453f986cce7941addf9092e38c6e005e18844"  # spec.md "Pinned Base Commit"
LEGACY_REFERENCE_PREFIX = "baselines/v0-legacy/reference/"
OUTPUTS_PREFIX = "outputs/"
STATUS_PATH = "baselines/v0-corrected/status.json"


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


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True)


def git_show(rev, path):
    completed = git("show", f"{rev}:{path}")
    if completed.returncode != 0:
        raise AssertionError(
            f"git cannot read {rev}:{path}: {completed.stderr.decode().strip()}"
        )
    return completed.stdout


def git_ls_tree(rev, prefix):
    completed = git("ls-tree", "-r", "--name-only", rev, "--", prefix)
    if completed.returncode != 0:
        raise AssertionError(
            f"git cannot list {prefix} at {rev}: {completed.stderr.decode().strip()}"
        )
    return [line for line in completed.stdout.decode("utf-8").splitlines() if line]


def is_fitness_entry(entry):
    """True for every entry whose locator resolves to a recorded fitness value."""
    locator = entry["locator"]
    return locator == "fitness" or locator.endswith("/fitness") or "/fitness/" in locator


def fitness_values_in(path, data):
    """(locator, row_key) for every recorded fitness value of one pinned artifact.

    VC-13's enumeration is structural: the CSV `fitness` column per run row
    (row_key = run_id, CSV rows only), every scalar leaf whose JSON key is exactly
    `fitness`, and every scalar leaf under a `fitness` object - never a text search.
    """
    if path.endswith(".csv"):
        reader = csv.DictReader(io.StringIO(data.decode("utf-8")), strict=True)
        columns = list(reader.fieldnames or [])
        if "fitness" not in columns:
            return []
        if "run_id" not in columns:
            raise AssertionError(f"{path} records fitness cells without run_id row keys")
        return [("fitness", {"run_id": int(row["run_id"])}) for row in reader]
    if path.endswith(".json"):
        found = []
        _json_fitness(json.loads(data.decode("utf-8")), "", False, found)
        return found
    return []


def _json_fitness(node, pointer, inside_fitness, out):
    if isinstance(node, dict):
        for key, value in node.items():
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            child = f"{pointer}/{escaped}"
            if key == "fitness" and _is_leaf(value):
                out.append((child, None))
                continue
            _json_fitness(value, child, key == "fitness" or inside_fitness, out)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _json_fitness(value, f"{pointer}/{index}", inside_fitness, out)
    elif inside_fitness:
        out.append((pointer, None))


def _is_leaf(value):
    return not isinstance(value, (dict, list))


class StatusMapMarkingTests(unittest.TestCase):
    """The committed verdict map and its checks (VC-13, VC-14, VC-24, VC-15).

    Subjects: `baselines/v0-legacy/reference/**` and every `outputs/` file tracked
    at the pinned base commit. Every oracle here is re-derived from the pinned git
    blobs, from `v0-legacy/baseline.json`, or from the spec's own counts -
    `status.json` is never trusted to confirm itself.
    """

    @classmethod
    def setUpClass(cls):
        sys.dont_write_bytecode = True  # before any bundle import (AD-013)
        cls._scratch = tempfile.TemporaryDirectory()
        cls._saved_cwd = os.getcwd()
        os.chdir(cls._scratch.name)  # bundle imports run with cwd in a scratch root
        cls.vd = load_bundle_tool("_verdict")
        document = json.loads((CORRECTED / "status.json").read_text(encoding="utf-8"))
        cls.pinned = document["pinned_base_commit"]
        cls.entries = document["entries"]
        cls.tracked = git_ls_tree(cls.pinned, "outputs/") + git_ls_tree(
            cls.pinned, "baselines/v0-legacy/reference/"
        )
        cls._pinned_bytes = {}

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._saved_cwd)
        cls._scratch.cleanup()

    def pinned_file(self, path):
        if path not in self._pinned_bytes:
            self._pinned_bytes[path] = git_show(self.pinned, path)
        return self._pinned_bytes[path]

    def test_every_strict_fitness_locator_appears_exactly_once(self):
        # VC-13: the subject is baselines/v0-legacy/reference/** (9 covered files,
        # 5 fitness-bearing) and every outputs/ artifact tracked at the pinned base
        # (160 tracked files, 15 fitness-bearing); every recorded fitness value
        # appears exactly once per recorded value in status.json
        legacy_files = [p for p in self.tracked if p.startswith(LEGACY_REFERENCE_PREFIX)]
        outputs_files = [p for p in self.tracked if p.startswith(OUTPUTS_PREFIX)]
        self.assertEqual(len(legacy_files), 9)
        self.assertEqual(len(outputs_files), 160)

        expected = set()
        for path in sorted(legacy_files + outputs_files):
            for locator, row_key in fitness_values_in(path, self.pinned_file(path)):
                key = (path, locator, json.dumps(row_key, sort_keys=True))
                self.assertNotIn(key, expected, f"{path}: value enumerated twice: {locator}")
                expected.add(key)
        self.assertTrue(expected, "no recorded fitness value found in the subject trees")

        recorded = set()
        for entry in self.entries:
            if not is_fitness_entry(entry):
                continue
            key = (entry["path"], entry["locator"], json.dumps(entry["row_key"], sort_keys=True))
            self.assertNotIn(key, recorded, f"{key}: the same value is recorded twice")
            recorded.add(key)
        self.assertEqual(recorded, expected)

        # the spec's own subject counts: 5 of 9 and 15 of 160 carry fitness values
        legacy_bearing = {path for path, _, _ in expected if path.startswith("baselines/")}
        outputs_bearing = {path for path, _, _ in expected if path.startswith(OUTPUTS_PREFIX)}
        self.assertEqual(len(legacy_bearing), 5)
        self.assertEqual(len(outputs_bearing), 15)

        # nested and aggregate fitness values are enumerated by JSON structure
        self.assertIn(
            ("baselines/v0-legacy/reference/multiple_runs_summary.json",
             "/conservador/stability_metrics/fitness/mean", "null"),
            expected,
        )
        self.assertIn(
            ("outputs/comparison_conservador.json",
             "/metrics_comparison/fitness/consensus", "null"),
            expected,
        )

    def test_every_entry_digest_matches_the_git_blob_at_the_pinned_base(self):
        # VC-14: each digest is the SHA-256 of the file's bytes at the pinned base
        # commit, recomputed here from git - status.json never pins itself
        self.assertEqual(self.pinned, PINNED_BASE)
        digests = {}
        for entry in self.entries:
            path = entry["path"]
            self.assertIn(path, self.tracked, f"{path} is not a pinned-base file")
            self.assertNotEqual(path, STATUS_PATH, "status.json must never enumerate itself")
            if path not in digests:
                digests[path] = hashlib.sha256(self.pinned_file(path)).hexdigest()
            self.assertEqual(entry["sha256"], digests[path], path)

        # legacy-reference digests cross-check against v0-legacy/baseline.json
        manifest = json.loads(
            (REPO / "baselines" / "v0-legacy" / "baseline.json").read_text(encoding="utf-8")
        )
        by_path = {entry["path"]: entry for entry in manifest["files"]}
        for path in sorted(digests):
            if path.startswith(LEGACY_REFERENCE_PREFIX):
                rel = path[len("baselines/v0-legacy/"):]
                self.assertIn(rel, by_path, path)
                self.assertEqual(digests[path], by_path[rel]["sha256"], path)

    def test_working_tree_outputs_bytes_still_equal_the_pinned_digests(self):
        # AD-007: outputs/ is never regenerated or modified - every tracked file's
        # working-tree bytes still hash to its pinned-base bytes
        checked = 0
        for path in sorted(self.tracked):
            if not path.startswith(OUTPUTS_PREFIX):
                continue
            checked += 1
            worktree = hashlib.sha256((REPO / path).read_bytes()).hexdigest()
            pinned = hashlib.sha256(self.pinned_file(path)).hexdigest()
            self.assertEqual(worktree, pinned, path)
        self.assertEqual(checked, 160)

    def test_every_verdict_recomputes_from_its_recorded_evidence(self):
        # VC-24/VC-12: the verdict is a pure function of the entry's own recorded
        # evidence fields, resolved bottom-up for derived entries through the same
        # `_verdict.verdict_for` the generator used - writer and checker cannot drift
        by_id = {}
        for entry in self.entries:
            entry_id = self.vd.entry_id(entry["path"], entry["locator"], entry["row_key"])
            self.assertNotIn(entry_id, by_id, f"{entry_id}: one value recorded twice")
            by_id[entry_id] = entry

        resolved = {}

        def verdict_of(entry):
            entry_id = self.vd.entry_id(entry["path"], entry["locator"], entry["row_key"])
            if entry_id not in resolved:
                if entry["kind"] == "derived":
                    contributor_verdicts = [
                        verdict_of(by_id[cid]) for cid in entry["contributors"]
                    ]
                    resolved[entry_id] = self.vd.verdict_for(
                        {**entry, "contributor_verdicts": contributor_verdicts}
                    )
                else:
                    resolved[entry_id] = self.vd.verdict_for(entry)
            return resolved[entry_id]

        for entry in self.entries:
            entry_id = self.vd.entry_id(entry["path"], entry["locator"], entry["row_key"])
            verdict = verdict_of(entry)
            self.assertEqual(verdict, entry["verdict"], entry_id)
            # VC-12 annotation discipline: the defect reference attaches only where
            # the rule attributes the defect; every other non-valid verdict carries
            # its own reason and never borrows the defect's name
            self.assertEqual(self.vd.annotation_errors(entry, verdict), [], entry_id)
            if entry["defect_ref"]:
                self.assertIn("P0-02", entry["defect_ref"], entry_id)
                self.assertIn("capture defect", entry["defect_ref"], entry_id)
            if verdict not in self.vd.DEFECT_VERDICTS and verdict != "valid":
                reason = str(entry["reason"] or "")
                self.assertTrue(reason.strip(), entry_id)
                self.assertNotIn("P0-02", reason, entry_id)
                self.assertNotIn("capture defect", reason, entry_id)

    def test_legacy_manifest_gate_is_verified_with_seventeen_files(self):
        # VC-15: the baselines/v0-legacy gates stay green - verify_manifest.py exits
        # 0 with VERIFIED over 17 files, and the 52 P0-01 tests pass unmodified
        gate = subprocess.run(
            [sys.executable, str(REPO / "baselines" / "v0-legacy" / "tools" / "verify_manifest.py")],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        self.assertEqual(gate.returncode, 0, gate.stdout + gate.stderr)
        report = json.loads(gate.stdout)
        self.assertEqual(report["status"], "VERIFIED")
        self.assertEqual(report["files"], 17)

        p01_tests = [
            "tests/test_v0_legacy_isolation.py",
            "tests/test_v0_legacy_manifest.py",
            "tests/test_v0_legacy_protocol.py",
            "tests/test_v0_legacy_replay.py",
        ]
        diff = git("diff", "--exit-code", PINNED_BASE, "--", *p01_tests)
        self.assertEqual(diff.returncode, 0, "a P0-01 test file was modified")
        suite = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests",
             "-p", "test_v0_legacy_*.py"],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        self.assertEqual(suite.returncode, 0, suite.stdout + suite.stderr)
        self.assertIn("Ran 52 tests", suite.stderr)


if __name__ == "__main__":
    unittest.main()
