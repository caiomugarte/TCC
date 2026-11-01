# IBOV and IFIX as the stock and FII series of the Premium allocation snapshot - verification

**Verdict**: FAIL
**Mutations**: 3 - probed C4, C2, C11 (round 1, all killed at 669e841); round 2 added 8 more over the touched production code, 1 not killed (V-3, now killed); round 3 probed the door-1 identity rule with 10 more, 3 not killed (V-4, V-5)
**Proofs**: `proofs.md` written by `check_proofs.py --mutations` at the round-3 verification commit (28 pass, 0 fail; mutations 4 killed, 3 not - V-4, V-5). At 801b752, before the round-3 mutations were appended: 28 pass, 4 killed (M8 included)
**Verifier**: independent sub-agent (author != verifier) - round 1 at 52ee519, round 2 (re-verification after fixes) at 669e841, round 3 (re-verification of V-3) at 801b752

## Findings

| Finding | Evidence | Fix | Closed |
| --- | --- | --- | --- |
| V-1 (blocking, manifest validation): the door-1 rule "a derived series must be declared by exactly `ibovespa.csv`/`ifix.csv`, character for character" has no test. Removing `declared == expected` from `_validate_index_series` keeps every test green | mutation `C4: drop declared == expected` (`py/snapshot_manifest.py:491`) not killed by the C4 proof, and not killed by the full suites either (`tests` 282 OK, `api/tests` 110 OK) | add the case `metadata.files.fiis == "./ifix.csv"` to `test_derived_index_series_must_live_in_allocation_dir`, asserting `SnapshotManifestError` with the message containing `./ifix.csv`; append it as an example to C4 / AC 3 | e6c8f11 - C4. Round 2: `check_proofs.py --mutations` at 669e841 kills it; re-applied by hand on a copy of HEAD, `tests.test_snapshot_manifest` fails at `test_derived_index_series_must_live_in_allocation_dir` |
| V-2 (nonblocking): the C6 proofs do not tell the new rule from the old code. They also pass on the base `20c3bbe` production code, because deleting the file changes the allocation directory hash, and they assert only exit 4 / `SnapshotError`, not the cause | HEAD tests run against the `20c3bbe` versions of `py/snapshot_manifest.py`, `py/snapshot_registry.py` and `py/allocation_data.py` passed | recompute the allocation sha256 after the deletion in the C6 tests and assert `source fiis not found` / `source brazilian_stocks not found` | 26b63a4 - C6. Round 2: the 669e841 tests run against the 20c3bbe versions of those 3 files fail (`test_validate_missing_index_file_exits_4`: 2 failures, `0 != 4`; `test_missing_premium_index_input_raises_without_substitution`: 2 failures, `SnapshotError not raised`) |
| V-3 (blocking, manifest validation, round 2): the door-1 identity rule for an explicit per-class source (`checked.source_for(key).resolved_path().name` must *equal* `ifix.csv`/`ibovespa.csv`) is proven on a single name, `caio_fiis.csv`, which shares no suffix with `ifix.csv`. Loosening the comparison to `resolved.name.endswith(expected)` keeps every test green. A `sources.fiis` path naming `caio_ifix.csv` or `old_ifix.csv` would then be accepted as the FII series | mutation `C2: explicit per-class source accepts any name ending in the index file name (== -> endswith)` (`py/snapshot_manifest.py:499`) not killed by `tests.test_snapshot_manifest` (12 OK). Own example at HEAD (throwaway): an explicit `sources.fiis` at `allocation/caio_ifix.csv`, `allocation/old_ifix.csv` or `allocation/IFIX.csv`, each with a matching hash, is rejected with `source fiis must be ifix.csv; manifest declares allocation/<name>`. The code is correct and only the proof is looser than the rule | add to `test_series_identity_rule_rejects_non_index_files` an explicit `sources.fiis` at `allocation/caio_ifix.csv` (a copy of `ifix.csv`, hash matching) asserting `SnapshotManifestError` with `source fiis must be ifix.csv` and `caio_ifix.csv` in the message; append the example to C2 / AC 2; re-run `check_proofs.py --mutations` (the mutation is now in `mutations.json`) | e919d54 - C2. M8 re-applied by hand on HEAD: `test_series_identity_rule_rejects_non_index_files` fails at `explicit='caio_ifix.csv'` (`SnapshotManifestError not raised`); file restored byte-identical |
| V-4 (blocking, manifest validation, round 3): two conjuncts of the derived-series rule (door 1, AC 2 / AC 3) are not proven. (a) `resolved.name == expected` in the derived branch can be dropped. The only shapes tested either fail `declared == expected` first or resolve outside the directory. So `allocation/ifix.csv` as a symlink to `allocation/caio_fiis.csv` would be accepted: the basket served as IFIX, with the directory hash matching. That is the failure this feature exists to stop. (b) The parent check is proven only against a differently named outside directory. Comparing directory names instead of directories keeps every test green, and then a symlink `ifix.csv` -> `<other>/allocation/ifix.csv` would be accepted | mutations `C4: drop resolved.name == expected` and `C4: parent.name == allocation_dir.name` (`py/snapshot_manifest.py:492`, `:493`) not killed by `tests.test_snapshot_manifest`, `test_snapshot_registry`, `test_allocation_data` or `test_allocation` (115 OK under each). Own examples at HEAD (throwaway, allocation hash recomputed): the in-directory symlink is rejected with `source fiis must be ifix.csv; manifest declares ifix.csv (resolves to <allocation>/caio_fiis.csv)`, and the symlink into `<root>/older/allocation/ifix.csv` is rejected too. Under the respective mutant, each is ACCEPTED. The code is correct and only the proof is looser than the rule | in `test_derived_index_series_must_live_in_allocation_dir`, add the in-directory symlink `ifix.csv` -> `caio_fiis.csv` (allocation hash recomputed) asserting `SnapshotManifestError` with `source fiis must be ifix.csv` and `caio_fiis.csv`. Move the existing symlink case's outside target to `<root>/older/allocation/ifix.csv`, or add that case. Append both examples to C4 / AC 3. Re-run `check_proofs.py --mutations`: both mutations are now in `mutations.json` | |
| V-5 (blocking, manifest validation, round 3): the explicit per-class source identity (`resolved.name == expected`, door 1 "must equal") is not proven case-sensitive. Comparing `resolved.name.lower()` keeps every test green, and then an explicit `sources.fiis` at `allocation/IFIX.csv` (a different file on Linux, with its own matching hash and any content) would be accepted as the FII series | mutation `C2: case-insensitive explicit name` (`py/snapshot_manifest.py:499`) not killed (115 OK). Own example at HEAD (throwaway): `sources.fiis` -> `allocation/IFIX.csv` holding the `caio_fiis.csv` bytes is rejected with `source fiis must be ifix.csv; manifest declares allocation/IFIX.csv`. Under the mutant it is ACCEPTED | add `IFIX.csv` to the `explicit=` subTest loop in `test_series_identity_rule_rejects_non_index_files`, asserting `source fiis must be ifix.csv; manifest declares allocation/IFIX.csv`. Append the example to C2 / AC 2. Re-run `check_proofs.py --mutations` | |

## Mutations injected

The first 3 are the owed `Mutations: 3` and are in `mutations.json` from round 1. The round-2 M8 is appended there, so the builder's re-run must show it killed. The rest ran on throwaway `git archive HEAD` copies, with the narrowest suite:

| Mutation | Location | Killed | Blocking |
| --- | --- | --- | --- |
| C4: drop `declared == expected` (M1, re-applied) | `py/snapshot_manifest.py` `_validate_index_series` | yes - `test_derived_index_series_must_live_in_allocation_dir` | - (V-1 closed) |
| C2: explicit source `valid = True` | `py/snapshot_manifest.py` `_validate_index_series` | yes | - |
| C11: build `fiis` reads `brazilian_stocks` | `py/snapshot_registry.py` `premium_level_series` | yes | - |
| C4: drop `resolved.parent == allocation_dir` | `py/snapshot_manifest.py` `_validate_index_series` | yes - symlink case | - |
| C2: `INDEX_SERIES_CLASSES = ("fiis",)` | `py/snapshot_manifest.py:43` | yes | - |
| C14: loader `fiis` reads `brazilian_stocks` | `py/allocation_data.py` `load_premium_snapshot_bundle` | yes | - |
| C16: `_single_series` takes the first of several columns | `py/allocation_data.py:108` | yes | - |
| C2: explicit-source branch `valid = True` (full `tests`) | `py/snapshot_manifest.py` | yes | - |
| C21: fetch `sources.ifix.role` back to `benchmark-only` | `py/fetch_allocation_snapshot.py` | yes | - |
| C2: explicit source `==` -> `endswith` (M8) | `py/snapshot_manifest.py:499` | **no** at 669e841; yes at 801b752 - `test_series_identity_rule_rejects_non_index_files` (`explicit='caio_ifix.csv'`) | - (V-3 closed) |

Round 3, at 801b752, on a throwaway `git archive HEAD` copy. Each mutant was run against `tests.test_snapshot_manifest`, `test_snapshot_registry`, `test_allocation_data` and `test_allocation` (baseline 115 OK), comparing the failing tests with the baseline. The full `tests` suite cannot run in an archive copy: its git- and data-bound tests fail there at baseline.

| Mutation | Location | Killed | Blocking |
| --- | --- | --- | --- |
| C2: explicit `==` -> `endswith` (M8, re-applied) | `py/snapshot_manifest.py:499` | yes - `test_series_identity_rule_rejects_non_index_files` | - (V-3 closed) |
| C2: explicit name compared case-insensitively (`.lower()`) | `py/snapshot_manifest.py:499` | **no** | yes - V-5 |
| C4: derived `resolved.name == expected` -> `endswith` | `py/snapshot_manifest.py:492` | **no** - subsumed by the next row | yes - V-4 |
| C4: derived, drop `resolved.name == expected` | `py/snapshot_manifest.py:492` | **no** | yes - V-4 |
| C4: derived parent compared by directory name | `py/snapshot_manifest.py:493` | **no** | yes - V-4 |
| C4: derived `declared == expected` -> `endswith` | `py/snapshot_manifest.py:491` | yes - `test_derived_index_series_must_live_in_allocation_dir` | - |
| C2/C4: derived/explicit branch condition inverted | `py/snapshot_manifest.py:487` | yes - 3 tests | - |
| C2: explicit-branch message declares `expected`, not `source.path` | `py/snapshot_manifest.py:498` | yes | - |
| C2/C15: `_validate_index_series` not called | `py/snapshot_manifest.py:531` | yes - 3 tests | - |
| C2: `INDEX_SERIES_CLASSES = ("fiis",)` / `("brazilian_stocks",)` | `py/snapshot_manifest.py:43` | yes / yes | - |

## Own examples (HEAD, throwaway, not committed)

- AC 2 / door 1, explicit source: `allocation/caio_ifix.csv`, `allocation/old_ifix.csv` and `allocation/IFIX.csv` with matching hashes are each rejected (correct; untested, V-3).
- Round 1, still valid: `./ifix.csv`, `sub/../ifix.csv` and the absolute path are rejected (now pinned by C4). A missing index file with the hash recomputed is rejected with no substitution (now pinned by C6).

## Notes

- Round 3: `git diff 52ee519..HEAD -- py api` is empty, so production code is unchanged since round 1. e919d54 touches only `tests/test_snapshot_manifest.py`, `checks.md` and `plan.md`. The plan edit appends one example (`allocation/caio_ifix.csv`) to criterion 2, which follows from its existing text ("a file not named `ifix.csv`"). Full suites at 801b752: `tests` 282 OK (1 skipped); `api/tests` with `PYTHONPATH=../py` 110 OK. Every mutation was reverted: the copy's `py/snapshot_manifest.py` is byte-identical to HEAD (`cmp`), and the worktree was never mutated.
- Round 3 own examples (HEAD, throwaway, not committed): an in-directory symlink `ifix.csv` -> `caio_fiis.csv`, a symlink into `<root>/older/allocation/ifix.csv`, and an explicit `sources.fiis` at `allocation/IFIX.csv` are each rejected at HEAD (correct; untested, V-4 and V-5).

- Production code is unchanged between 52ee519 and 669e841 (`git diff 52ee519..669e841 -- py api` is empty). The fixes touch only tests and spec. The plan edit appends one example (`./ifix.csv`) to criterion 3. That example follows from the criterion's existing text ("not exactly `ibovespa.csv`/`ifix.csv`"), so the approved obligation does not change.
- Full suites at 669e841: `tests` 282 OK (1 skipped); `api/tests` with `PYTHONPATH=../py` 110 OK.
- `validate_verification.py` needs `proofs.md`'s HEAD to equal the current HEAD. Committing `proofs.md` moves HEAD, so after this docs-only commit the gate also reports the HEAD mismatch. The builder's post-fix `check_proofs.py` run clears it. That run is the last step before the gate, and its commit touches only `proofs.md`.
