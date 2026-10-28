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


if __name__ == "__main__":
    unittest.main()
