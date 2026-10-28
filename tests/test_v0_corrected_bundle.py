"""v0-corrected (P0-02) bundle proofs (spec.md VC-02..VC-05, VC-01/VC-09/VC-29, tasks T1-T8).

The tests derive from the spec's acceptance criteria, not from the implementation:

- VC-03: the live `py/ga.py` equals the canonical (old_bytes, new_bytes) transform applied
  to `git show <pinned base>:py/ga.py`, byte-for-byte, so strict `>`, RNG call order,
  roulette selection, crossover, mutation and seeding stay untouched.
- VC-02: `baselines/v0-corrected/code/ga.py` is the same transform applied to the frozen
  `baselines/v0-legacy/code/ga.py` bytes - the one deliberate difference of the bundle.
- VC-04: every other copy is byte-identical to its v0-legacy counterpart and its sha256
  matches `baselines/v0-legacy/baseline.json`.
- VC-05: `protocol.json` is deeply equal to the v0-legacy protocol except `baseline_id` and
  `derived_from`; the AST-extracted constants are equal with the four frozen configs only.
- VC-01/VC-09/VC-29 (generation machinery): 30 run_ids x 4 profiles completeness, the
  double-pass byte-digest compare, the generation proof schema, and the per-row driver tie
  that rejects a row the driver did not produce.

Standard library only: bytes are hashed and compared, constants are AST-extracted, never
imported. Nothing here writes under either bundle.
"""

import ast
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PINNED_BASE = "181453f986cce7941addf9092e38c6e005e18844"
LIVE_GA = REPO / "py" / "ga.py"
LEGACY = REPO / "baselines" / "v0-legacy"
CORRECTED = REPO / "baselines" / "v0-corrected"

# The canonical transform from spec.md "Canonical transform" (exact byte pair, LF-terminated).
OLD_BYTES = (
    "        population = new_pop\n"
    "\n"
    "        # regista melhor\n"
    "        gen_best_idx = int(scores.argmax())\n"
    "        if scores[gen_best_idx] > best_fitness:\n"
    "            best_fitness = scores[gen_best_idx]\n"
    "            best_chrom = population[gen_best_idx].copy()\n"
).encode("utf-8")

NEW_BYTES = (
    "        # regista melhor\n"
    "        gen_best_idx = int(scores.argmax())\n"
    "        if scores[gen_best_idx] > best_fitness:\n"
    "            best_fitness = scores[gen_best_idx]\n"
    "            best_chrom = population[gen_best_idx].copy()\n"
    "\n"
    "        population = new_pop\n"
).encode("utf-8")


def canonical_transform(data):
    """The spec's canonical (old_bytes, new_bytes) pair applied exactly once."""
    if data.count(OLD_BYTES) != 1:
        raise AssertionError("old_bytes must occur exactly once in the input")
    return data.replace(OLD_BYTES, NEW_BYTES)


def git_show(rev_path):
    """Bytes of a git object at the pinned base - local object database only."""
    completed = subprocess.run(
        ["git", "show", rev_path],
        cwd=str(REPO),
        capture_output=True,
        check=True,
    )
    return completed.stdout


class LiveGaByteEqualityTests(unittest.TestCase):
    """VC-03: the live fix is exactly the canonical transform of the pinned blob."""

    def test_canonical_old_bytes_occur_exactly_once_in_the_pinned_live_file(self):
        pinned = git_show(f"{PINNED_BASE}:py/ga.py")
        self.assertEqual(pinned.count(OLD_BYTES), 1)
        self.assertEqual(pinned.count(NEW_BYTES), 0)

    def test_transformed_pinned_blob_equals_the_worktree_ga_py_byte_for_byte(self):
        pinned = git_show(f"{PINNED_BASE}:py/ga.py")
        expected = canonical_transform(pinned)
        actual = LIVE_GA.read_bytes()
        self.assertEqual(len(actual) - len(expected), 0)  # 0-byte delta
        self.assertEqual(actual, expected)

    def test_one_byte_mutation_fails_the_byte_equality(self):
        pinned = git_show(f"{PINNED_BASE}:py/ga.py")
        expected = canonical_transform(pinned)
        mutated = bytearray(LIVE_GA.read_bytes())
        # one byte inside the relocated statement is flipped
        anchor = mutated.find(b"population = new_pop")
        self.assertGreaterEqual(anchor, 0)
        mutated[anchor] ^= 0x01
        self.assertNotEqual(bytes(mutated), expected)


class CorrectedCodeCopyTests(unittest.TestCase):
    """VC-02: the frozen code copy is the canonical transform of the legacy bundle file."""

    def test_corrected_code_ga_equals_the_transform_of_the_legacy_bundle_file(self):
        legacy = (LEGACY / "code" / "ga.py").read_bytes()
        expected = canonical_transform(legacy)
        actual = (CORRECTED / "code" / "ga.py").read_bytes()
        self.assertEqual(len(actual) - len(expected), 0)  # 0-byte delta
        self.assertEqual(actual, expected)

    def test_old_bytes_occur_exactly_once_in_the_legacy_bundle_file(self):
        legacy = (LEGACY / "code" / "ga.py").read_bytes()
        self.assertEqual(legacy.count(OLD_BYTES), 1)
        self.assertEqual(legacy.count(NEW_BYTES), 0)

    def test_transform_replacement_never_touches_the_rng_or_tie_semantics(self):
        legacy = (LEGACY / "code" / "ga.py").read_bytes()
        corrected = (CORRECTED / "code" / "ga.py").read_bytes()
        old_at = legacy.find(OLD_BYTES)
        new_at = corrected.find(NEW_BYTES)
        self.assertGreaterEqual(old_at, 0)
        self.assertGreaterEqual(new_at, 0)
        # every byte outside the replaced region is untouched: strict ">", RNG call order,
        # roulette selection, crossover, mutation and seeding all live there
        self.assertEqual(legacy[:old_at], corrected[:new_at])
        self.assertEqual(
            legacy[old_at + len(OLD_BYTES):],
            corrected[new_at + len(NEW_BYTES):],
        )


class FrozenCopyTests(unittest.TestCase):
    """VC-04: every other copy is byte-identical to v0-legacy, sha256-verified."""

    @staticmethod
    def _manifest_entries():
        manifest = json.loads((LEGACY / "baseline.json").read_text(encoding="utf-8"))
        return {entry["path"]: entry for entry in manifest["files"]}

    def test_profiles_and_ga_multiple_runs_are_byte_identical_to_the_legacy_copies(self):
        for rel in ("code/profiles.py", "code/ga_multiple_runs.py"):
            with self.subTest(path=rel):
                legacy = (LEGACY / rel).read_bytes()
                corrected = (CORRECTED / rel).read_bytes()
                self.assertEqual(corrected, legacy)

    def test_every_data_processed_file_matches_the_v0_legacy_manifest_sha256(self):
        entries = self._manifest_entries()
        data_paths = sorted(p for p in entries if p.startswith("data/processed/"))
        self.assertEqual(len(data_paths), 5)
        for rel in data_paths:
            with self.subTest(path=rel):
                digest = hashlib.sha256((CORRECTED / rel).read_bytes()).hexdigest()
                self.assertEqual(digest, entries[rel]["sha256"])
                legacy_digest = hashlib.sha256((LEGACY / rel).read_bytes()).hexdigest()
                self.assertEqual(legacy_digest, entries[rel]["sha256"])

    def test_every_corrected_copy_has_zero_byte_delta_against_its_counterpart(self):
        entries = self._manifest_entries()
        copies = ["code/profiles.py", "code/ga_multiple_runs.py"] + sorted(
            p for p in entries if p.startswith("data/processed/")
        )
        self.assertEqual(len(copies), 7)
        for rel in copies:
            with self.subTest(path=rel):
                legacy = (LEGACY / rel).read_bytes()
                corrected = (CORRECTED / rel).read_bytes()
                self.assertEqual(len(corrected) - len(legacy), 0)  # 0-byte delta
                self.assertEqual(corrected, legacy)


def deep_equal_except(left, right, ignored):
    """Dict deep-equality after dropping `ignored` keys from both sides."""
    drop = set(ignored)
    a = {k: v for k, v in left.items() if k not in drop}
    b = {k: v for k, v in right.items() if k not in drop}
    return a == b


def last_assignment(tree, name):
    found = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                found = node.value
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == name:
                found = node.value
    return found


def ast_constant(code_dir, module, name):
    """A module-level constant read out of the source - extracted, never imported."""
    path = Path(code_dir) / f"{module}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    value = last_assignment(tree, name)
    if value is None:
        raise AssertionError(f"{name} is not assigned at module level in {path}")
    return ast.literal_eval(value)


class ProtocolTests(unittest.TestCase):
    """VC-05: derived protocol deep-equal except baseline_id/derived_from, AST-equal constants."""

    FROZEN_CONFIGS = {"conservador", "moderado", "arrojado", "caio"}
    CONSTANTS = (
        ("ga", "PERFIL_CONFIG"),
        ("ga", "NCROSS_RATE"),
        ("ga", "MUT_RATE"),
        ("profiles", "GROUPS"),
        ("profiles", "PROFILE_WEIGHTS"),
    )

    def test_protocol_is_deep_equal_except_baseline_id_and_derived_from(self):
        legacy = json.loads((LEGACY / "protocol.json").read_text(encoding="utf-8"))
        corrected = json.loads((CORRECTED / "protocol.json").read_text(encoding="utf-8"))
        self.assertTrue(
            deep_equal_except(corrected, legacy, ("baseline_id", "derived_from")),
            "protocol.json must be deeply equal to the legacy protocol except "
            "baseline_id and derived_from",
        )
        self.assertEqual(corrected["baseline_id"], "v0-corrected")
        derived = corrected["derived_from"]
        self.assertEqual(derived["baseline_id"], "v0-legacy")
        self.assertEqual(derived["pinned_base_commit"], PINNED_BASE)

    def test_ast_extracted_constants_equal_the_legacy_values_without_caio_last(self):
        for module, name in self.CONSTANTS:
            with self.subTest(constant=name):
                legacy_value = ast_constant(LEGACY / "code", module, name)
                corrected_value = ast_constant(CORRECTED / "code", module, name)
                self.assertEqual(corrected_value, legacy_value)
        configs = ast_constant(CORRECTED / "code", "ga", "PERFIL_CONFIG")
        self.assertEqual(set(configs), self.FROZEN_CONFIGS)
        self.assertNotIn("caio_last", configs)
        weights = ast_constant(CORRECTED / "code", "profiles", "PROFILE_WEIGHTS")
        self.assertEqual(set(weights), self.FROZEN_CONFIGS)

    def test_an_extra_protocol_key_is_rejected_by_the_deep_equal_check(self):
        legacy = json.loads((LEGACY / "protocol.json").read_text(encoding="utf-8"))
        corrected = json.loads((CORRECTED / "protocol.json").read_text(encoding="utf-8"))
        self.assertTrue(
            deep_equal_except(corrected, legacy, ("baseline_id", "derived_from"))
        )
        tampered = dict(corrected)
        tampered["extra_key"] = 1
        self.assertFalse(
            deep_equal_except(tampered, legacy, ("baseline_id", "derived_from")),
            "an extra protocol key must break the deep-equal check",
        )


def load_bundle_tool(name):
    """Import a bundle tool from baselines/v0-corrected/tools/ under a unique module name.

    Callers must have set sys.dont_write_bytecode first and must run with cwd inside a
    scratch root (AD-013): importing bundle code writes __pycache__ and `ga_multiple_runs`
    creates outputs/ at import time.
    """
    sys.dont_write_bytecode = True  # before the bundle import (AD-013)
    path = CORRECTED / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_test_v0_corrected_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ManifestToolingTests(unittest.TestCase):
    """VC-04/VC-28 machinery: build/verify contract on synthetic TemporaryDirectory bundles."""

    EXCLUDED_PATHS = {
        "baseline.json",
        "protocol.json",
        "README.md",
        "tools/",
        "status.json",
        "environment.txt",
        "requirements.txt",
    }
    REQUIRED_FIELDS = (
        "path",
        "source_path",
        "sha256",
        "git_blob_sha1",
        "bytes",
        "mode",
        "role",
    )

    @classmethod
    def setUpClass(cls):
        sys.dont_write_bytecode = True  # before any bundle import (AD-013)
        cls._scratch = tempfile.TemporaryDirectory()
        cls._saved_cwd = os.getcwd()
        os.chdir(cls._scratch.name)  # bundle imports run with cwd in a scratch root
        cls.vm = load_bundle_tool("verify_manifest")
        cls.bm = load_bundle_tool("build_manifest")

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._saved_cwd)
        cls._scratch.cleanup()

    @staticmethod
    def make_bundle(root):
        """A synthetic bundle: files across all covered subtrees - never the real bundles."""
        (root / "code").mkdir(parents=True)
        (root / "data" / "processed").mkdir(parents=True)
        (root / "reference").mkdir(parents=True)
        (root / "code" / "ga.py").write_bytes(b"print('ga')\n")
        (root / "code" / "profiles.py").write_bytes(b"print('profiles')\n")
        (root / "code" / "ga_multiple_runs.py").write_bytes(b"print('runs')\n")
        (root / "data" / "processed" / "fundamentals_clean_caio.csv").write_bytes(
            b"TICKER,SCORE\nX,1.0\n"
        )
        (root / "data" / "processed" / "eligibility_filters.parquet").write_bytes(b"PARQUET")
        (root / "reference" / "metrics_stability_caio.csv").write_bytes(b"run_id,fitness\n0,1.0\n")
        (root / "reference" / "multiple_runs_summary.json").write_bytes(b"{}")
        return root

    @staticmethod
    def run_main(module, argv):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return module.main(argv)

    def test_manifest_covers_three_subtrees_and_names_the_seven_exclusions_with_reasons(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self.make_bundle(Path(tmp) / "bundle")
            manifest = self.bm.build(bundle)
            self.assertEqual(
                manifest["coverage"]["included_subtrees"], ["code/", "data/", "reference/"]
            )
            excluded = {
                entry["path"]: entry["reason"] for entry in manifest["coverage"]["excluded"]
            }
            self.assertEqual(set(excluded), self.EXCLUDED_PATHS)
            for path, reason in excluded.items():
                with self.subTest(excluded=path):
                    self.assertTrue(reason.strip(), "every exclusion needs a reason")
            for entry in manifest["files"]:
                with self.subTest(path=entry["path"]):
                    for field in self.REQUIRED_FIELDS:
                        self.assertIn(field, entry)
            by_path = {entry["path"]: entry for entry in manifest["files"]}
            data = (bundle / "code" / "ga.py").read_bytes()
            self.assertEqual(by_path["code/ga.py"]["sha256"], hashlib.sha256(data).hexdigest())
            self.assertEqual(by_path["code/ga.py"]["bytes"], len(data))
            self.assertEqual(
                by_path["code/ga.py"]["source_path"],
                "baselines/v0-legacy/code/ga.py#canonical-transform",
            )
            self.assertEqual(
                by_path["reference/metrics_stability_caio.csv"]["source_path"],
                "generated:tools/generate_reference.py#pass-a",
            )
            # the manifest refuses overwrite
            self.assertEqual(self.run_main(self.bm, ["--bundle", str(bundle)]), 0)
            self.assertEqual(self.run_main(self.bm, ["--bundle", str(bundle)]), 3)

    def test_missing_modified_and_unexpected_files_are_reported_by_path_and_kind(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self.make_bundle(Path(tmp) / "bundle")
            self.assertEqual(self.run_main(self.bm, ["--bundle", str(bundle)]), 0)
            clean = self.vm.verify_bundle(bundle)
            self.assertEqual(clean["status"], "VERIFIED")
            self.assertEqual(clean["findings"], [])
            (bundle / "data" / "processed" / "eligibility_filters.parquet").unlink()
            with open(bundle / "code" / "profiles.py", "ab") as handle:
                handle.write(b"# drift\n")
            pycache = bundle / "code" / "__pycache__"
            pycache.mkdir()
            (pycache / "ga.cpython-314.pyc").write_bytes(b"\x00")
            (bundle / "reference" / "extra.json").write_bytes(b"{}")
            report = self.vm.verify_bundle(bundle)
            self.assertEqual(report["status"], "DRIFT")
            kinds = {(finding["path"], finding["kind"]) for finding in report["findings"]}
            self.assertEqual(
                kinds,
                {
                    ("data/processed/eligibility_filters.parquet", "missing"),
                    ("code/profiles.py", "modified"),
                    ("code/__pycache__/ga.cpython-314.pyc", "unexpected"),
                    ("reference/extra.json", "unexpected"),
                },
            )

    def test_verified_bundle_exits_zero_and_broken_manifest_exits_three(self):
        self.assertEqual(self.vm.DEFAULT_BUNDLE, "baselines/v0-corrected")
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self.make_bundle(Path(tmp) / "bundle")
            self.assertEqual(self.run_main(self.bm, ["--bundle", str(bundle)]), 0)
            self.assertEqual(self.run_main(self.vm, ["--bundle", str(bundle)]), 0)
            (bundle / "code" / "profiles.py").write_bytes(b"# drifted\n")
            self.assertEqual(self.run_main(self.vm, ["--bundle", str(bundle)]), 1)
            (bundle / "baseline.json").write_text('{"schema_version": 1,', encoding="utf-8")
            self.assertEqual(self.run_main(self.vm, ["--bundle", str(bundle)]), 3)

    def test_excluded_paths_appearing_later_are_not_unexpected(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self.make_bundle(Path(tmp) / "bundle")
            self.assertEqual(self.run_main(self.bm, ["--bundle", str(bundle)]), 0)
            for rel in ("README.md", "protocol.json", "status.json", "environment.txt",
                        "requirements.txt"):
                (bundle / rel).write_text("authored\n", encoding="utf-8")
            tools = bundle / "tools"
            tools.mkdir()
            (tools / "compare_replay.py").write_text("# authored\n", encoding="utf-8")
            report = self.vm.verify_bundle(bundle)
            self.assertEqual(report["status"], "VERIFIED")
            self.assertEqual(report["findings"], [])
            unexpected = [f for f in report["findings"] if f["kind"] == "unexpected"]
            self.assertEqual(unexpected, [])


class GenerationPipelineTests(unittest.TestCase):
    """VC-01/VC-09/VC-29 generation machinery: completeness, determinism, proof, driver tie.

    Pure functions over synthetic `TemporaryDirectory()` outputs - the real generation
    is executed at T8 under tcc-venv and pinned by its proof.
    """

    @classmethod
    def setUpClass(cls):
        sys.dont_write_bytecode = True  # before any bundle import (AD-013)
        cls._scratch = tempfile.TemporaryDirectory()
        cls._saved_cwd = os.getcwd()
        os.chdir(cls._scratch.name)  # bundle imports run with cwd in a scratch root
        cls.gr = load_bundle_tool("generate_reference")

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._saved_cwd)
        cls._scratch.cleanup()

    @staticmethod
    def synthetic_row(profile, run_id):
        """One recorded row exactly as the frozen writer serializes it."""
        return {
            "run_id": run_id,
            "seed": 42 + run_id,
            "tickers": [f"{profile[:2].upper()}{i:02d}3" for i in range(run_id % 5, run_id % 5 + 10)],
            "fitness": 4.0 + run_id / 100.0,
            "hhi": 0.10 + (run_id % 7) / 100.0,
        }

    def make_outputs(self, root):
        """A synthetic outputs/ set: 9 files, 30 run_ids x 4 profiles, VC-09-consistent."""
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        summary = {}
        for profile in self.gr.PROFILES:
            rows = [self.synthetic_row(profile, run_id) for run_id in range(30)]
            with open(root / f"metrics_stability_{profile}.csv", "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["run_id", "seed", "tickers", "fitness", "hhi"])
                for row in rows:
                    writer.writerow([row["run_id"], row["seed"], repr(row["tickers"]),
                                     row["fitness"], row["hhi"]])
            (root / f"carteira_{profile}_consensus.json").write_text("[]\n", encoding="utf-8")
            summary[profile] = {"n_runs": len(rows), "all_runs": rows}
        (root / "multiple_runs_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        return root

    def rewrite_csv(self, root, profile, rows):
        with open(Path(root) / f"metrics_stability_{profile}.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["run_id", "seed", "tickers", "fitness", "hhi"])
            for row in rows:
                writer.writerow([row["run_id"], row["seed"], repr(row["tickers"]),
                                 row["fitness"], row["hhi"]])

    def test_assert_complete_rejects_a_missing_run_id_or_profile(self):
        gr = self.gr
        with tempfile.TemporaryDirectory() as tmp:
            outputs = self.make_outputs(Path(tmp) / "outputs")
            self.assertEqual(gr.assert_complete(outputs), [])

            rows = [r for r in gr.read_metrics_rows(outputs, "caio") if r["run_id"] != 7]
            self.rewrite_csv(outputs, "caio", rows)
            problems = gr.assert_complete(outputs)
            self.assertTrue(any("caio" in p and "missing run_id 7" in p for p in problems))

        with tempfile.TemporaryDirectory() as tmp:
            outputs = self.make_outputs(Path(tmp) / "outputs")
            (outputs / "metrics_stability_moderado.csv").unlink()
            problems = gr.assert_complete(outputs)
            self.assertTrue(any("missing output file: metrics_stability_moderado.csv" in p
                                for p in problems))

        with tempfile.TemporaryDirectory() as tmp:
            outputs = self.make_outputs(Path(tmp) / "outputs")
            rows = gr.read_metrics_rows(outputs, "conservador")
            rows[3]["seed"] = 99  # the seed rule is seed_base + run_id
            self.rewrite_csv(outputs, "conservador", rows)
            problems = gr.assert_complete(outputs)
            self.assertTrue(any("seed" in p and "run_id 3" in p for p in problems))

    def test_compare_passes_catches_a_single_byte_digest_difference(self):
        gr = self.gr
        pass_a = {"metrics_stability_caio.csv": "aa", "multiple_runs_summary.json": "bb"}
        pass_b = {"metrics_stability_caio.csv": "aa", "multiple_runs_summary.json": "bb"}
        self.assertEqual(gr.compare_passes(pass_a, pass_b), [])
        pass_b["metrics_stability_caio.csv"] = "cc"  # one file, one byte
        problems = gr.compare_passes(pass_a, pass_b)
        self.assertEqual(len(problems), 1)
        self.assertIn("metrics_stability_caio.csv", problems[0])
        self.assertTrue(any(p.startswith("digest differs") for p in problems))
        self.assertTrue(gr.compare_passes(pass_a, {"multiple_runs_summary.json": "bb"}))

    def test_emit_proof_records_the_schema_and_120_driver_rows(self):
        gr = self.gr
        driver_rows = [
            {
                "profile": profile,
                "run_id": run_id,
                "seed": 42 + run_id,
                "tickers": ["AAA3", "BBB3"],
                "fitness": 6.2445628,
                "hhi": 0.16,
            }
            for profile in gr.PROFILES
            for run_id in range(30)
        ]
        pass_a = {"metrics_stability_caio.csv": "aa"}
        pass_b = {"metrics_stability_caio.csv": "aa"}
        proof = gr.emit_proof(pass_a, pass_b, driver_rows)
        self.assertEqual(
            set(proof),
            {"schema_version", "env", "passes", "deterministic", "driver_rows"},
        )
        self.assertEqual(proof["schema_version"], 1)
        self.assertEqual(set(proof["env"]), {"python_VV", "numpy", "pandas"})
        self.assertEqual(set(proof["passes"]), {"a", "b"})
        self.assertIs(proof["deterministic"], True)
        self.assertEqual(len(proof["driver_rows"]), 120)
        for row in proof["driver_rows"]:
            self.assertEqual(set(row), {"profile", "run_id", "seed", "tickers", "fitness", "hhi"})
        self.assertEqual(
            {(r["profile"], r["run_id"]) for r in proof["driver_rows"]},
            {(p, run_id) for p in gr.PROFILES for run_id in range(30)},
        )
        diverging = gr.emit_proof(pass_a, {"metrics_stability_caio.csv": "zz"}, driver_rows)
        self.assertIs(diverging["deterministic"], False)

    def test_tie_to_driver_flags_a_row_the_stub_driver_did_not_produce(self):
        gr = self.gr
        with tempfile.TemporaryDirectory() as tmp:
            outputs = self.make_outputs(Path(tmp) / "outputs")

            def faithful(profile, run_id):
                return self.synthetic_row(profile, run_id)

            problems, driver_rows = gr.tie_to_driver(
                outputs, CORRECTED, Path(tmp) / "scratch", runner=faithful, max_workers=1
            )
            self.assertEqual(problems, [])
            self.assertEqual(len(driver_rows), 120)

            def dropping(profile, run_id):
                if (profile, run_id) == ("caio", 7):
                    return None  # the row the stub driver did not produce
                return self.synthetic_row(profile, run_id)

            problems, driver_rows = gr.tie_to_driver(
                outputs, CORRECTED, Path(tmp) / "scratch", runner=dropping, max_workers=1
            )
            self.assertTrue(
                any("caio run_id 7" in p and "driver produced no row" in p for p in problems)
            )
            self.assertEqual(len(problems), 1)
            self.assertEqual(len(driver_rows), 119)


PROOFS = REPO / ".specs" / "features" / "p0-02-v0-corrected" / "proofs"
GENERATION_PROOF = PROOFS / "generation_proof.json"
CORRECTED_REFERENCE = CORRECTED / "reference"


def reference_rows(profile):
    """Committed reference rows of one profile, keyed cells parsed - the test's own read."""
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
            }
    return rows


class GenerationEvidenceTests(unittest.TestCase):
    """VC-01/VC-29: the committed reference is provably the corrected driver's output.

    Fast legs over the committed artifacts: reference <-> proofs/generation_proof.json
    coherence for every driver row, the double-pass digests vs the committed bytes, and
    the manifest pin. The execution that produced them ran at build time (T8).
    """

    @classmethod
    def setUpClass(cls):
        sys.dont_write_bytecode = True  # before any bundle import (AD-013)
        cls._scratch = tempfile.TemporaryDirectory()
        cls._saved_cwd = os.getcwd()
        os.chdir(cls._scratch.name)  # bundle imports run with cwd in a scratch root
        cls.vd = load_bundle_tool("_verdict")
        cls.vm = load_bundle_tool("verify_manifest")
        cls.proof = json.loads(GENERATION_PROOF.read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._saved_cwd)
        cls._scratch.cleanup()

    def test_generation_proof_ties_every_reference_row_to_the_driver_for_120_rows(self):
        recorded = {}
        for profile in ("conservador", "moderado", "arrojado", "caio"):
            for run_id, row in reference_rows(profile).items():
                recorded[(profile, run_id)] = row
        driver = {
            (row["profile"], row["run_id"]): row for row in self.proof["driver_rows"]
        }
        self.assertEqual(len(self.proof["driver_rows"]), 120)
        self.assertEqual(set(driver), set(recorded))
        for key in sorted(recorded):
            with self.subTest(row=key):
                self.assertEqual(self.vd.tie_row(recorded[key], driver[key]), [])

    def test_generation_proof_passes_digests_match_the_committed_reference_bytes(self):
        proof = self.proof
        self.assertIs(proof["deterministic"], True)
        self.assertEqual(set(proof["passes"]), {"a", "b"})
        pass_a, pass_b = proof["passes"]["a"], proof["passes"]["b"]
        self.assertEqual(pass_a, pass_b, "the two passes must be byte-identical")
        committed = sorted(p.name for p in CORRECTED_REFERENCE.iterdir() if p.is_file())
        self.assertEqual(sorted(pass_a), committed)
        for name, digest in sorted(pass_a.items()):
            with self.subTest(file=name):
                data = (CORRECTED_REFERENCE / name).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), digest)

    def test_baseline_manifest_pins_the_generated_reference(self):
        manifest = json.loads((CORRECTED / "baseline.json").read_text(encoding="utf-8"))
        by_path = {entry["path"]: entry for entry in manifest["files"]}
        reference_files = sorted(
            f"reference/{p.name}" for p in CORRECTED_REFERENCE.iterdir() if p.is_file()
        )
        self.assertEqual(len(reference_files), 9)
        for rel in reference_files:
            with self.subTest(path=rel):
                entry = by_path[rel]
                self.assertEqual(
                    entry["source_path"], "generated:tools/generate_reference.py#pass-a"
                )
                data = (CORRECTED / rel).read_bytes()
                self.assertEqual(entry["sha256"], hashlib.sha256(data).hexdigest())
                self.assertEqual(entry["bytes"], len(data))
        self.assertEqual(
            by_path["code/ga.py"]["source_path"],
            "baselines/v0-legacy/code/ga.py#canonical-transform",
        )
        report = self.vm.verify_bundle(CORRECTED)
        self.assertEqual(report["status"], "VERIFIED")
        self.assertEqual(report["findings"], [])


ARCHIVE_REFERENCE = LEGACY / "reference"
CONTROL_PROFILES = ("conservador", "moderado", "arrojado")
ULP_BOUND = 4  # AD-016: a control row is bit-identical or within 4 ULP; beyond = FAIL
CONTROL_ULP_DRIFT = PROOFS / "control_ulp_drift.json"
REFERENCE_DIVERGENCES = PROOFS / "reference_divergences.json"


def ordered_bits(value):
    """Packed float64 bits mapped to sign-flipped ordered integers (ULP order)."""
    bits = struct.unpack(">Q", struct.pack(">d", float(value)))[0]
    return bits ^ 0xFFFFFFFFFFFFFFFF if bits & (1 << 63) else bits | (1 << 63)


def from_ordered_bits(ordered):
    """The float64 whose ordered-bit representation is `ordered`."""
    bits = (
        ordered ^ 0xFFFFFFFFFFFFFFFF if not ordered & (1 << 63) else ordered ^ (1 << 63)
    )
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


def ulp_distance(a, b):
    """Absolute ULP distance between two float64 values (0 = bit-identical)."""
    return abs(ordered_bits(a) - ordered_bits(b))


def shift_ulp(value, distance):
    """The float64 `distance` ULP above `value` (ordered-bit arithmetic)."""
    return from_ordered_bits(ordered_bits(value) + distance)


def archive_rows(profile):
    """Archived run rows of one profile keyed by run_id - the archive CSV's own cells."""
    path = ARCHIVE_REFERENCE / f"metrics_stability_{profile}.csv"
    rows = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, strict=True):
            rows[int(row["run_id"])] = {
                "run_id": int(row["run_id"]),
                "seed": int(row["seed"]),
                "tickers": list(ast.literal_eval(row["tickers"])),
                "fitness": float(row["fitness"]),
                "hhi": float(row["hhi"]),
            }
    return rows


def measure_archive_rows():
    """(profile, run_id) -> both trees' archived and corrected cells, re-read from disk.

    The VC-25 comparison is re-measured here from the two committed CSVs - never read
    back from the proof tables - so a forged table cannot confirm itself (T9: the old
    blocking-evidence JSON is not the oracle, the artifacts are).
    """
    rows = {}
    for profile in ("conservador", "moderado", "arrojado", "caio"):
        archived = archive_rows(profile)
        corrected = reference_rows(profile)
        if set(archived) != set(corrected):
            raise AssertionError(f"{profile}: run_id sets differ between the two trees")
        for run_id in sorted(archived):
            if archived[run_id]["seed"] != corrected[run_id]["seed"]:
                raise AssertionError(
                    f"{profile} run_id {run_id}: seed mismatch, (run_id, seed) key is broken"
                )
            rows[(profile, run_id)] = {
                "seed": archived[run_id]["seed"],
                "archived": archived[run_id]["fitness"],
                "corrected": corrected[run_id]["fitness"],
                "archived_row": archived[run_id],
                "corrected_row": corrected[run_id],
            }
    return rows


class ArchiveComparisonTests(unittest.TestCase):
    """VC-25/VC-29: the committed reference against the archive, per (run_id, seed).

    Control profiles: bit-identical (`struct.pack('>d')`) or within 4 ULP with every
    non-identical row recorded in `proofs/control_ulp_drift.json`; beyond 4 ULP the
    gate fails outright and the row never becomes a table entry. caio: exactly the
    divergent rows in `proofs/reference_divergences.json`, no control row. Both proof
    tables are checked against a fresh measurement of the two CSV trees.
    """

    @classmethod
    def setUpClass(cls):
        sys.dont_write_bytecode = True  # before any bundle import (AD-013)
        cls._scratch = tempfile.TemporaryDirectory()
        cls._saved_cwd = os.getcwd()
        os.chdir(cls._scratch.name)  # bundle imports run with cwd in a scratch root
        cls.vd = load_bundle_tool("_verdict")
        cls.rows = measure_archive_rows()
        cls.ulp_table = json.loads(CONTROL_ULP_DRIFT.read_text(encoding="utf-8"))
        cls.divergence_table = json.loads(REFERENCE_DIVERGENCES.read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._saved_cwd)
        cls._scratch.cleanup()

    def check_archive_tables(self, rows, ulp_table, divergence_table, max_ulp=ULP_BOUND):
        """Problems proving both VC-25 proof tables exactly match the measured rows ([] = clean).

        Control rows: bit-identical, or recorded in the ULP table with (archived,
        corrected, ULP distance) while within `max_ulp`; a control row beyond the bound
        is a blocking finding for a human - it fails outright and never becomes a table
        entry. caio rows: every divergence recorded with (archived, corrected) values.
        Unrecorded divergences, stale entries and recorded-value drift all fail.
        """
        problems = []
        measured_ulp = {}
        measured_div = {}
        for key in sorted(rows):
            row = rows[key]
            profile, run_id = key
            if row["archived"] is None or row["corrected"] is None:
                problems.append(f"row {profile} run_id {run_id} is missing from one tree")
                continue
            distance = ulp_distance(row["archived"], row["corrected"])
            if profile in CONTROL_PROFILES:
                if distance == 0:
                    continue
                if distance <= max_ulp:
                    measured_ulp[key] = (row, distance)
                else:
                    problems.append(
                        f"blocking finding for a human: control row {profile} run_id {run_id} "
                        f"is {distance} ULP from the archive (bound {max_ulp}) - fails the "
                        f"gate outright, never a table entry"
                    )
                continue
            if distance != 0:
                measured_div[key] = (row, distance)
        problems += self._ulp_table_problems(measured_ulp, ulp_table)
        problems += self._divergence_table_problems(measured_div, divergence_table)
        return problems

    @staticmethod
    def _indexed(entries):
        indexed = {}
        for entry in entries:
            key = (entry.get("profile"), entry.get("run_id"))
            if key in indexed:
                indexed[key] = "duplicate"
            else:
                indexed[key] = entry
        return indexed

    def _ulp_table_problems(self, measured, ulp_table):
        problems = []
        recorded = self._indexed(ulp_table.get("entries", []))
        for key, entry in sorted(recorded.items(), key=lambda kv: str(kv[0])):
            if entry == "duplicate":
                problems.append(f"control_ulp_drift.json records {key[0]} run_id {key[1]} twice")
        recorded = {k: v for k, v in recorded.items() if v != "duplicate"}
        for key in sorted(set(measured) - set(recorded)):
            row, distance = measured[key]
            problems.append(
                f"unrecorded control drift: {key[0]} run_id {key[1]} is not bit-identical "
                f"to the archive ({distance} ULP) but has no control_ulp_drift.json entry"
            )
        for key in sorted(set(recorded) - set(measured)):
            problems.append(
                f"stale control_ulp_drift.json entry: {key[0]} run_id {key[1]} no longer "
                f"drifts from the archive"
            )
        for key in sorted(set(measured) & set(recorded)):
            row, distance = measured[key]
            entry = recorded[key]
            expected = {
                "seed": row["seed"],
                "archived_fitness": row["archived"],
                "corrected_fitness": row["corrected"],
                "ulp_distance": distance,
            }
            for field, value in expected.items():
                got = entry.get(field)
                if field in ("archived_fitness", "corrected_fitness"):
                    same = got is not None and self.vd.float_bitwise_equal(got, value)
                else:
                    same = got == value
                if not same:
                    problems.append(
                        f"control_ulp_drift.json entry {key[0]} run_id {key[1]} records "
                        f"{field}={got!r}, measured {value!r}"
                    )
        return problems

    def _divergence_table_problems(self, measured, divergence_table):
        problems = []
        recorded = self._indexed(divergence_table.get("entries", []))
        for key, entry in sorted(recorded.items(), key=lambda kv: str(kv[0])):
            if entry == "duplicate":
                problems.append(f"reference_divergences.json records {key[0]} run_id {key[1]} twice")
        recorded = {k: v for k, v in recorded.items() if v != "duplicate"}
        for key in sorted(recorded):
            if key[0] in CONTROL_PROFILES:
                problems.append(
                    f"control row {key[0]} run_id {key[1]} appears in the divergence table: "
                    f"the table is caio-only and a control row never becomes an entry"
                )
        for key in sorted(set(measured) - set(recorded)):
            problems.append(
                f"unrecorded divergence: caio run_id {key[1]} diverges from the archive but "
                f"has no reference_divergences.json entry"
            )
        for key in sorted(set(recorded) - set(measured)):
            problems.append(
                f"stale divergence entry: {key[0]} run_id {key[1]} no longer diverges from "
                f"the archive"
            )
        for key in sorted(set(measured) & set(recorded)):
            row, distance = measured[key]
            entry = recorded[key]
            expected = {
                "seed": row["seed"],
                "archived_fitness": row["archived"],
                "corrected_fitness": row["corrected"],
            }
            for field, value in expected.items():
                got = entry.get(field)
                if field.endswith("_fitness"):
                    same = got is not None and self.vd.float_bitwise_equal(got, value)
                else:
                    same = got == value
                if not same:
                    problems.append(
                        f"reference_divergences.json entry {key[0]} run_id {key[1]} records "
                        f"{field}={got!r}, measured {value!r}"
                    )
        return problems

    def test_control_profile_fitness_is_bit_identical_or_within_4_ulp_of_the_archive_per_run_id_seed(self):
        control = [key for key in self.rows if key[0] in CONTROL_PROFILES]
        self.assertEqual(len(control), 90)  # 3 control profiles x 30 runs
        identical = 0
        drifted = []
        for key in sorted(control):
            row = self.rows[key]
            if self.vd.float_bitwise_equal(row["archived"], row["corrected"]):
                identical += 1
            else:
                drifted.append((key, ulp_distance(row["archived"], row["corrected"])))
        self.assertEqual(identical, 78)  # 78 of 90 control rows bit-identical (T9)
        self.assertEqual(len(drifted), 12)  # the rest drift within the bound
        for key, distance in drifted:
            with self.subTest(row=key):
                self.assertLessEqual(distance, ULP_BOUND)
        self.assertEqual(self.check_archive_tables(self.rows, self.ulp_table, self.divergence_table), [])

    def test_control_ulp_drift_table_records_exactly_the_non_identical_control_rows(self):
        measured = {}
        for key, row in sorted(self.rows.items()):
            if key[0] in CONTROL_PROFILES and not self.vd.float_bitwise_equal(
                row["archived"], row["corrected"]
            ):
                measured[key] = (row, ulp_distance(row["archived"], row["corrected"]))
        self.assertEqual(len(measured), 12)
        entries = self.ulp_table["entries"]
        self.assertEqual(len(entries), 12)
        recorded = {(entry["profile"], entry["run_id"]): entry for entry in entries}
        self.assertEqual(set(recorded), set(measured))
        for key, (row, distance) in measured.items():
            with self.subTest(row=key):
                entry = recorded[key]
                self.assertEqual(
                    set(entry),
                    {"profile", "run_id", "seed", "archived_fitness", "corrected_fitness",
                     "ulp_distance"},
                )
                self.assertEqual(entry["seed"], row["seed"])
                self.assertTrue(
                    self.vd.float_bitwise_equal(entry["archived_fitness"], row["archived"])
                )
                self.assertTrue(
                    self.vd.float_bitwise_equal(entry["corrected_fitness"], row["corrected"])
                )
                self.assertEqual(entry["ulp_distance"], distance)
                self.assertGreaterEqual(entry["ulp_distance"], 1)
                self.assertLessEqual(entry["ulp_distance"], ULP_BOUND)
        summary = self.ulp_table["summary"]
        self.assertEqual(summary["control_rows"], 90)
        self.assertEqual(summary["bit_identical"], 78)
        self.assertEqual(summary["recorded_drift_rows"], 12)
        self.assertEqual(summary["max_ulp_distance"], 2)

    def test_divergence_table_holds_caio_rows_only_and_exactly_the_divergent_ones(self):
        measured = {}
        for key, row in sorted(self.rows.items()):
            if key[0] == "caio" and not self.vd.float_bitwise_equal(
                row["archived"], row["corrected"]
            ):
                measured[key] = row
        self.assertEqual(len(measured), 30)  # exactly the divergent caio rows
        entries = self.divergence_table["entries"]
        recorded = {(entry["profile"], entry["run_id"]): entry for entry in entries}
        self.assertEqual(len(recorded), len(entries))  # no duplicates
        self.assertEqual(set(recorded), set(measured))
        self.assertTrue(all(entry["profile"] == "caio" for entry in entries))  # no control row
        for key, row in measured.items():
            with self.subTest(row=key):
                entry = recorded[key]
                self.assertEqual(
                    set(entry),
                    {"profile", "run_id", "seed", "archived_fitness", "corrected_fitness",
                     "reason"},
                )
                self.assertEqual(entry["seed"], row["seed"])
                self.assertTrue(
                    self.vd.float_bitwise_equal(entry["archived_fitness"], row["archived"])
                )
                self.assertTrue(
                    self.vd.float_bitwise_equal(entry["corrected_fitness"], row["corrected"])
                )
                self.assertTrue(str(entry["reason"]).strip())
        summary = self.divergence_table["summary"]
        self.assertEqual(summary["caio_rows"], 30)
        self.assertEqual(summary["divergent_rows"], 30)
        self.assertEqual(summary["control_rows_in_table"], 0)

    def test_stale_divergence_entry_fails_the_gate(self):
        # the real tables are clean - the gate only reddens on mutation
        self.assertEqual(self.check_archive_tables(self.rows, self.ulp_table, self.divergence_table), [])
        # a recorded entry whose row no longer diverges is stale
        rows = dict(self.rows)
        key = ("caio", 0)
        rows[key] = dict(rows[key], corrected=rows[key]["archived"])
        problems = self.check_archive_tables(rows, self.ulp_table, self.divergence_table)
        self.assertTrue(any("stale divergence entry" in p for p in problems), problems)
        # an entry that no measurement supports is stale too
        ghost = {"profile": "caio", "run_id": 99, "seed": 141, "archived_fitness": 1.0,
                 "corrected_fitness": 2.0, "reason": "ghost"}
        entries = [dict(e) for e in self.divergence_table["entries"]]
        problems = self.check_archive_tables(self.rows, self.ulp_table, {"entries": entries + [ghost]})
        self.assertTrue(any("stale divergence entry" in p for p in problems), problems)
        # an unrecorded divergence fails just as loudly
        problems = self.check_archive_tables(self.rows, self.ulp_table, {"entries": entries[1:]})
        self.assertTrue(any("unrecorded divergence" in p for p in problems), problems)
        # a stale ULP-drift entry is caught by the same gate
        rows = dict(self.rows)
        key = ("moderado", 2)
        rows[key] = dict(rows[key], corrected=rows[key]["archived"])
        problems = self.check_archive_tables(rows, self.ulp_table, self.divergence_table)
        self.assertTrue(any("stale control_ulp_drift.json entry" in p for p in problems), problems)

    def test_control_profile_divergence_beyond_4_ulp_fails_outright_instead_of_becoming_an_entry(self):
        rows = dict(self.rows)
        key = ("conservador", 0)  # bit-identical in the real measurement - clean to mutate
        self.assertTrue(
            self.vd.float_bitwise_equal(rows[key]["archived"], rows[key]["corrected"])
        )
        rows[key] = dict(rows[key], corrected=shift_ulp(rows[key]["archived"], ULP_BOUND + 1))
        problems = self.check_archive_tables(rows, self.ulp_table, self.divergence_table)
        blocking = [p for p in problems if "blocking finding" in p and "conservador run_id 0" in p]
        self.assertEqual(len(blocking), 1, problems)  # fails the gate outright
        # it never becomes a table entry: no unrecorded/stale problem is raised for it
        self.assertFalse(any("conservador run_id 0" in p and "unrecorded" in p for p in problems))
        self.assertFalse(any("stale" in p for p in problems), problems)
        # and recording it in either table is refused
        ghost = {"profile": "conservador", "run_id": 0, "seed": 42,
                 "archived_fitness": rows[key]["archived"],
                 "corrected_fitness": rows[key]["corrected"], "ulp_distance": ULP_BOUND + 1}
        ulp_entries = [dict(e) for e in self.ulp_table["entries"]] + [ghost]
        problems = self.check_archive_tables(rows, {"entries": ulp_entries}, self.divergence_table)
        self.assertTrue(any("stale control_ulp_drift.json entry" in p for p in problems), problems)
        div_ghost = {"profile": "conservador", "run_id": 0, "seed": 42,
                     "archived_fitness": rows[key]["archived"],
                     "corrected_fitness": rows[key]["corrected"], "reason": "refused"}
        div_entries = [dict(e) for e in self.divergence_table["entries"]] + [div_ghost]
        problems = self.check_archive_tables(rows, self.ulp_table, {"entries": div_entries})
        self.assertTrue(any("caio-only" in p for p in problems), problems)

    def test_forged_reference_row_with_legacy_chromosomes_fails_the_row_tie(self):
        proof = json.loads(GENERATION_PROOF.read_text(encoding="utf-8"))
        driver = {(row["profile"], row["run_id"]): row for row in proof["driver_rows"]}
        flagged = 0
        for key in sorted(self.rows):
            row = self.rows[key]
            # forged row: the legacy chromosome kept, the corrected fitness re-written on
            # it (D-f VC-29 fixture) - what a relabelled legacy reference would look like
            forged = {
                "tickers": row["archived_row"]["tickers"],
                "fitness": row["corrected_row"]["fitness"],
                "hhi": row["corrected_row"]["hhi"],
            }
            problems = self.vd.tie_row(forged, driver[key])
            if problems:
                flagged += 1
                with self.subTest(row=key):
                    self.assertTrue(any("tickers" in p for p in problems), problems)
        self.assertEqual(flagged, 120)  # every archived chromosome differs from its row


if __name__ == "__main__":
    unittest.main()
