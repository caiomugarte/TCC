# IBOV and IFIX as the stock and FII series of the Premium allocation snapshot - verification

**Verdict**: FAIL
**Mutations**: 3 - probed C4, C2, C11 (round 1, all killed at 669e841); round 2 added 8 more over the touched production code, 1 not killed
**Proofs**: `proofs.md` at 73a045f (round 2, run by the Verifier after adding M8: 28 pass, 0 fail; mutations 3 killed, 1 not - V-3). The same run at 669e841 without M8 gave 28 pass, 3 killed
**Verifier**: independent sub-agent (author != verifier) - round 1 at 52ee519, round 2 (re-verification after fixes) at 669e841

## Findings

| Finding | Evidence | Fix | Closed |
| --- | --- | --- | --- |
| V-1 (blocking, manifest validation): the door-1 rule "a derived series must be declared by exactly `ibovespa.csv`/`ifix.csv`, character for character" has no test. Removing `declared == expected` from `_validate_index_series` keeps every test green | mutation `C4: drop declared == expected` (`py/snapshot_manifest.py:491`) not killed by the C4 proof, and not killed by the full suites either (`tests` 282 OK, `api/tests` 110 OK) | add the case `metadata.files.fiis == "./ifix.csv"` to `test_derived_index_series_must_live_in_allocation_dir`, asserting `SnapshotManifestError` with the message containing `./ifix.csv`; append it as an example to C4 / AC 3 | e6c8f11 - C4. Round 2: `check_proofs.py --mutations` at 669e841 kills it; re-applied by hand on a copy of HEAD, `tests.test_snapshot_manifest` fails at `test_derived_index_series_must_live_in_allocation_dir` |
| V-2 (nonblocking): the C6 proofs do not tell the new rule from the old code. They also pass on the base `20c3bbe` production code, because deleting the file changes the allocation directory hash, and they assert only exit 4 / `SnapshotError`, not the cause | HEAD tests run against the `20c3bbe` versions of `py/snapshot_manifest.py`, `py/snapshot_registry.py` and `py/allocation_data.py` passed | recompute the allocation sha256 after the deletion in the C6 tests and assert `source fiis not found` / `source brazilian_stocks not found` | 26b63a4 - C6. Round 2: the 669e841 tests run against the 20c3bbe versions of those 3 files fail (`test_validate_missing_index_file_exits_4`: 2 failures, `0 != 4`; `test_missing_premium_index_input_raises_without_substitution`: 2 failures, `SnapshotError not raised`) |
| V-3 (blocking, manifest validation, round 2): the door-1 identity rule for an explicit per-class source (`checked.source_for(key).resolved_path().name` must *equal* `ifix.csv`/`ibovespa.csv`) is proven on a single name, `caio_fiis.csv`, which shares no suffix with `ifix.csv`. Loosening the comparison to `resolved.name.endswith(expected)` keeps every test green. A `sources.fiis` path naming `caio_ifix.csv` or `old_ifix.csv` would then be accepted as the FII series | mutation `C2: explicit per-class source accepts any name ending in the index file name (== -> endswith)` (`py/snapshot_manifest.py:499`) not killed by `tests.test_snapshot_manifest` (12 OK). Own example at HEAD (throwaway): an explicit `sources.fiis` at `allocation/caio_ifix.csv`, `allocation/old_ifix.csv` or `allocation/IFIX.csv`, each with a matching hash, is rejected with `source fiis must be ifix.csv; manifest declares allocation/<name>`. The code is correct and only the proof is looser than the rule | add to `test_series_identity_rule_rejects_non_index_files` an explicit `sources.fiis` at `allocation/caio_ifix.csv` (a copy of `ifix.csv`, hash matching) asserting `SnapshotManifestError` with `source fiis must be ifix.csv` and `caio_ifix.csv` in the message; append the example to C2 / AC 2; re-run `check_proofs.py --mutations` (the mutation is now in `mutations.json`) | |

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
| C2: explicit source `==` -> `endswith` (M8) | `py/snapshot_manifest.py:499` | **no** | yes - V-3 |

## Own examples (HEAD, throwaway, not committed)

- AC 2 / door 1, explicit source: `allocation/caio_ifix.csv`, `allocation/old_ifix.csv` and `allocation/IFIX.csv` with matching hashes are each rejected (correct; untested, V-3).
- Round 1, still valid: `./ifix.csv`, `sub/../ifix.csv` and the absolute path are rejected (now pinned by C4). A missing index file with the hash recomputed is rejected with no substitution (now pinned by C6).

## Notes

- Production code is unchanged between 52ee519 and 669e841 (`git diff 52ee519..669e841 -- py api` is empty). The fixes touch only tests and spec. The plan edit appends one example (`./ifix.csv`) to criterion 3. That example follows from the criterion's existing text ("not exactly `ibovespa.csv`/`ifix.csv`"), so the approved obligation does not change.
- Full suites at 669e841: `tests` 282 OK (1 skipped); `api/tests` with `PYTHONPATH=../py` 110 OK.
- `validate_verification.py` needs `proofs.md`'s HEAD to equal the current HEAD. Committing `proofs.md` moves HEAD, so after this docs-only commit the gate also reports the HEAD mismatch. The builder's post-fix `check_proofs.py` run clears it. That run is the last step before the gate, and its commit touches only `proofs.md`.
