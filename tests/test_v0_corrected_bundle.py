"""v0-corrected (P0-02) bundle proofs (spec.md VC-02..VC-05, tasks T1-T5).

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

Standard library only: bytes are hashed and compared, constants are AST-extracted, never
imported. Nothing here writes under either bundle.
"""

import ast
import hashlib
import json
import subprocess
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


if __name__ == "__main__":
    unittest.main()
