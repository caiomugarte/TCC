# IBOV and IFIX as the stock and FII series of the Premium allocation snapshot - verification

**Verdict**: PASS
**Mutations**: 3 - probed C4, C2, C11 (round 1, all killed at 669e841); round 2 added 8 more over the touched production code, 1 not killed (V-3, now killed); round 3 probed the door-1 identity rule with 10 more, 3 not killed (V-4, V-5, now killed); round 4 (closure) re-ran all 19 in `mutations.json`: 19 killed, 0 not
**Proofs**: `proofs.md` written by `check_proofs.py --mutations` at the round-4 closure commit (28 pass, 0 fail; mutations 19 killed, 0 not). At 8d557b2, before the closure commit: the same result
**Verifier**: independent sub-agent (author != verifier) - round 1 at 52ee519, round 2 (re-verification after fixes) at 669e841, round 3 (re-verification of V-3) at 801b752, round 4 (closure of V-4 and V-5) at 8d557b2

## Findings

| Finding | Evidence | Fix | Closed |
| --- | --- | --- | --- |
| V-1 (blocking, manifest validation): the door-1 rule "a derived series must be declared by exactly `ibovespa.csv`/`ifix.csv`, character for character" has no test. Removing `declared == expected` from `_validate_index_series` keeps every test green | mutation `C4: drop declared == expected` (`py/snapshot_manifest.py:491`) not killed by the C4 proof, and not killed by the full suites either (`tests` 282 OK, `api/tests` 110 OK) | add the case `metadata.files.fiis == "./ifix.csv"` to `test_derived_index_series_must_live_in_allocation_dir`, asserting `SnapshotManifestError` with the message containing `./ifix.csv`; append it as an example to C4 / AC 3 | e6c8f11 - C4. Round 2: `check_proofs.py --mutations` at 669e841 kills it; re-applied by hand on a copy of HEAD, `tests.test_snapshot_manifest` fails at `test_derived_index_series_must_live_in_allocation_dir` |
| V-2 (nonblocking): the C6 proofs do not tell the new rule from the old code. They also pass on the base `20c3bbe` production code, because deleting the file changes the allocation directory hash, and they assert only exit 4 / `SnapshotError`, not the cause | HEAD tests run against the `20c3bbe` versions of `py/snapshot_manifest.py`, `py/snapshot_registry.py` and `py/allocation_data.py` passed | recompute the allocation sha256 after the deletion in the C6 tests and assert `source fiis not found` / `source brazilian_stocks not found` | 26b63a4 - C6. Round 2: the 669e841 tests run against the 20c3bbe versions of those 3 files fail (`test_validate_missing_index_file_exits_4`: 2 failures, `0 != 4`; `test_missing_premium_index_input_raises_without_substitution`: 2 failures, `SnapshotError not raised`) |
| V-3 (blocking, manifest validation, round 2): the door-1 identity rule for an explicit per-class source (`checked.source_for(key).resolved_path().name` must *equal* `ifix.csv`/`ibovespa.csv`) is proven on a single name, `caio_fiis.csv`, which shares no suffix with `ifix.csv`. Loosening the comparison to `resolved.name.endswith(expected)` keeps every test green. A `sources.fiis` path naming `caio_ifix.csv` or `old_ifix.csv` would then be accepted as the FII series | mutation `C2: explicit per-class source accepts any name ending in the index file name (== -> endswith)` (`py/snapshot_manifest.py:499`) not killed by `tests.test_snapshot_manifest` (12 OK). Own example at HEAD (throwaway): an explicit `sources.fiis` at `allocation/caio_ifix.csv`, `allocation/old_ifix.csv` or `allocation/IFIX.csv`, each with a matching hash, is rejected with `source fiis must be ifix.csv; manifest declares allocation/<name>`. The code is correct and only the proof is looser than the rule | add to `test_series_identity_rule_rejects_non_index_files` an explicit `sources.fiis` at `allocation/caio_ifix.csv` (a copy of `ifix.csv`, hash matching) asserting `SnapshotManifestError` with `source fiis must be ifix.csv` and `caio_ifix.csv` in the message; append the example to C2 / AC 2; re-run `check_proofs.py --mutations` (the mutation is now in `mutations.json`) | e919d54 - C2. M8 re-applied by hand on HEAD: `test_series_identity_rule_rejects_non_index_files` fails at `explicit='caio_ifix.csv'` (`SnapshotManifestError not raised`); file restored byte-identical |
| V-4 (blocking, manifest validation, round 3): two conjuncts of the derived-series rule (door 1, AC 2 / AC 3) are not proven. (a) `resolved.name == expected` in the derived branch can be dropped. The only shapes tested either fail `declared == expected` first or resolve outside the directory. So `allocation/ifix.csv` as a symlink to `allocation/caio_fiis.csv` would be accepted: the basket served as IFIX, with the directory hash matching. That is the failure this feature exists to stop. (b) The parent check is proven only against a differently named outside directory. Comparing directory names instead of directories keeps every test green, and then a symlink `ifix.csv` -> `<other>/allocation/ifix.csv` would be accepted | mutations `C4: drop resolved.name == expected` and `C4: parent.name == allocation_dir.name` (`py/snapshot_manifest.py:492`, `:493`) not killed by `tests.test_snapshot_manifest`, `test_snapshot_registry`, `test_allocation_data` or `test_allocation` (115 OK under each). Own examples at HEAD (throwaway, allocation hash recomputed): the in-directory symlink is rejected with `source fiis must be ifix.csv; manifest declares ifix.csv (resolves to <allocation>/caio_fiis.csv)`, and the symlink into `<root>/older/allocation/ifix.csv` is rejected too. Under the respective mutant, each is ACCEPTED. The code is correct and only the proof is looser than the rule | in `test_derived_index_series_must_live_in_allocation_dir`, add the in-directory symlink `ifix.csv` -> `caio_fiis.csv` (allocation hash recomputed) asserting `SnapshotManifestError` with `source fiis must be ifix.csv` and `caio_fiis.csv`. Move the existing symlink case's outside target to `<root>/older/allocation/ifix.csv`, or add that case. Append both examples to C4 / AC 3. Re-run `check_proofs.py --mutations`: both mutations are now in `mutations.json` | a439ee5 - C4. The builder's round-4 sweep on a throwaway `git archive` copy with the a439ee5 test file: `drop resolved.name == expected` and `resolved.name.endswith` are killed by `test_derived_index_series_must_live_in_allocation_dir` (`symlink='allocation/caio_fiis.csv'`, `'allocation/caio_ifix.csv'`), and `parent.name == allocation_dir.name` is killed by the same test (`symlink='older/allocation/ifix.csv'`). Both `mutations.json` entries are re-checked by `check_proofs.py --mutations` |
| V-5 (blocking, manifest validation, round 3): the explicit per-class source identity (`resolved.name == expected`, door 1 "must equal") is not proven case-sensitive. Comparing `resolved.name.lower()` keeps every test green, and then an explicit `sources.fiis` at `allocation/IFIX.csv` (a different file on Linux, with its own matching hash and any content) would be accepted as the FII series | mutation `C2: case-insensitive explicit name` (`py/snapshot_manifest.py:499`) not killed (115 OK). Own example at HEAD (throwaway): `sources.fiis` -> `allocation/IFIX.csv` holding the `caio_fiis.csv` bytes is rejected with `source fiis must be ifix.csv; manifest declares allocation/IFIX.csv`. Under the mutant it is ACCEPTED | add `IFIX.csv` to the `explicit=` subTest loop in `test_series_identity_rule_rejects_non_index_files`, asserting `source fiis must be ifix.csv; manifest declares allocation/IFIX.csv`. Append the example to C2 / AC 2. Re-run `check_proofs.py --mutations` | a439ee5 - C2. In the round-4 sweep, `resolved.name.lower()` on the explicit branch is killed by `test_series_identity_rule_rejects_non_index_files` (`explicit='IFIX.csv'`, and `'IBOVESPA.csv'` for `brazilian_stocks`). It is re-checked by `check_proofs.py --mutations` |

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

## Builder sweep of the identity rule (round 4, not a verification round)

The builder swept every single-point mutant of `_validate_index_series`, its call site and its constants. It used a throwaway `git archive 7c38f25` copy under `/tmp` and ran `tests.test_snapshot_manifest`, `test_snapshot_registry`, `test_allocation_data` and `test_allocation` (baseline 115 OK, 0 failing), comparing each mutant's failing tests with the baseline. The copy's `py/snapshot_manifest.py` was restored and checked with `assert` after every mutant. The "HEAD tests" column is the sweep with the 7c38f25 tests. The "a439ee5 tests" column is the sweep with the a439ee5 test file overlaid. The 15 mutants that survived the HEAD tests and are now killed are marked *. The 12 of them not already in `mutations.json` were appended to it.

| # | Mutant | HEAD tests | a439ee5 tests | Killed by / why equivalent |
| --- | --- | --- | --- | --- |
| M01 | `INDEX_SERIES_CLASSES = ("fiis",)` | killed | killed | C2/C4 tests, C15 loader test |
| M02 | `INDEX_SERIES_CLASSES = ("brazilian_stocks",)` | killed | killed | C2/C4 tests, C5, C15 |
| M03 | `INDEX_SERIES_CLASSES = ()` | killed | killed | C2/C4/C8/C15 |
| M04 | `ALLOCATION_FILES["fiis"] = "caio_fiis.csv"` | killed | killed | C1 `test_allocation_files_name_ibov_and_ifix`, build tests |
| M05 | `ALLOCATION_FILES["brazilian_stocks"] = "caio_stocks.csv"` | killed | killed | C1, build tests |
| M06 | `allocation = None` (every series takes the explicit branch) | killed | killed | C2, C4, C5, C15 |
| M07* | `resolved = Path(source.path)` (no resolve) | **survived** | killed | C2 `explicit='explicit/ifix.csv'` (symlink to `caio_fiis.csv`) |
| M08 | branch `derived_from != "allocation"` (branches swapped) | killed | killed | C2, C4, C5, C15 |
| M09 | branch drops `and allocation is not None` | survived | survived | equivalent: `derived_from` is set only by `source_for` when `allocation` exists; with no allocation, an explicit source claiming `derived_from` would raise `AttributeError` instead of `SnapshotManifestError` - rejected either way, never accepted |
| M10 | branch `and` -> `or allocation is not None` | killed | killed | C2 |
| M11 | `metadata.get("filename", "")` default | survived | survived | equivalent: `source_for` always sets `filename` on a derived source; the default is unreachable (and the mutant only rejects more) |
| M12 | `allocation_dir` without `.resolve()` | survived | survived | not a loosening: `resolved_path()` already resolves relative paths; for an absolute allocation path through a symlinked parent the mutant *rejects* a valid manifest (probe: accepted at HEAD, rejected under the mutant) - tighter, never looser |
| M13 | derived: drop `declared == expected` | killed | killed | C4 (`./ifix.csv`) |
| M14 | derived: `declared.endswith(expected)` | killed | killed | C4 (`./ifix.csv`) |
| M15* | derived: `declared.startswith(expected)` | **survived** | killed | C4 `declared='ifix.csv/../ifix.csv'` |
| M16 | derived: `expected in declared` | killed | killed | C4 (`./ifix.csv`) |
| M17* | derived: `declared.lower() == expected` | **survived** | killed | C4 `declared='IFIX.csv'` (symlink to `ifix.csv`) |
| M18 | derived: `declared != expected` | killed | killed | every valid-manifest test |
| M19* | derived: drop `resolved.name == expected` (V-4) | **survived** | killed | C4 `symlink='allocation/caio_fiis.csv'` |
| M20* | derived: `resolved.name.endswith(expected)` (V-4) | **survived** | killed | C4 `symlink='allocation/caio_ifix.csv'` |
| M21* | derived: `resolved.name.startswith(expected)` | **survived** | killed | C4 `symlink='allocation/ifix.csv.old'` |
| M22* | derived: `resolved.name.lower() == expected` | **survived** | killed | C4 `symlink='allocation/IFIX.csv'` |
| M23 | derived: `resolved.name != expected` | killed | killed | every valid-manifest test |
| M24 | derived: drop `resolved.parent == allocation_dir` | killed | killed | C4 (symlink outside) |
| M25* | derived: `resolved.parent.name == allocation_dir.name` (V-4) | **survived** | killed | C4 `symlink='older/allocation/ifix.csv'` |
| M26* | derived: `allocation_dir in resolved.parents` | **survived** | killed | C4 `symlink='allocation/sub/ifix.csv'` |
| M27* | derived: `str(resolved).startswith(str(allocation_dir))` | **survived** | killed | C4 `symlink='allocation/sub/ifix.csv'` |
| M28 | derived: `resolved.parent != allocation_dir` | killed | killed | every valid-manifest test |
| M29 | derived: `declared == expected or resolved.name ...` | killed | killed | C4 |
| M30 | derived: `resolved.name ... or resolved.parent ...` | killed | killed | C2, C4, C8, C15 |
| M31 | message: always append `(resolves to ...)` | survived | survived | equivalent for the rule: changes only the message suffix when the declared name is already wrong; the message still names the class, the expected and the declared file (C9), and nothing is accepted |
| M32* | message: never append `(resolves to ...)` | **survived** | killed | C4 symlink cases assert `(resolves to <target>)` |
| M33 | explicit: `declared = expected` in the message | killed | killed | C2 |
| M34 | explicit: `valid = True` | killed | killed | C2 |
| M35 | explicit: `resolved.name.endswith(expected)` (M8) | killed | killed | C2 `explicit='caio_ifix.csv'` |
| M36* | explicit: `resolved.name.lower() == expected` (V-5) | **survived** | killed | C2 `explicit='IFIX.csv'`, `'IBOVESPA.csv'` |
| M37* | explicit: `resolved.name.startswith(expected)` | **survived** | killed | C2 `explicit='ifix.csv.old'` |
| M38 | explicit: `expected in resolved.name` | killed | killed | C2 `explicit='caio_ifix.csv'` |
| M39 | explicit: `resolved.name != expected` | killed | killed | C2 |
| M40* | explicit: `Path(source.path).name == expected` | **survived** | killed | C2 `explicit='explicit/ifix.csv'` |
| M41 | `if not valid:` -> `if False:` | killed | killed | C2, C4, C8, C15 |
| M42 | message `{key}` and `{expected}` swapped | killed | killed | C2, C4, C5, C15 |
| M43 | message drops `{declared}` | killed | killed | C2, C4, C5, C15 |
| M44 | call site removed | killed | killed | C2, C4, C8, C15 |
| M45* | call site gated on `verify_hashes` | **survived** | killed | C2 `caio_*` manifests re-validated with `verify_hashes=False` |
| M46 | call moved after the hash/existence loop | survived | survived | equivalent: changes only which error a manifest that fails both checks reports first; the loop raises or passes independently of the series name, so every manifest rejected at HEAD is still rejected |

M19, M25 and M36 were already in `mutations.json` from round 3. M12 is the only survivor with an observable difference, and it is a tightening (a false reject of an absolute allocation path through a symlink, a shape no `build`/`fetch` writes), so no test pins it.

## Own examples (HEAD, throwaway, not committed)

- AC 2 / door 1, explicit source: `allocation/caio_ifix.csv`, `allocation/old_ifix.csv` and `allocation/IFIX.csv` with matching hashes are each rejected (correct; untested, V-3).
- Round 1, still valid: `./ifix.csv`, `sub/../ifix.csv` and the absolute path are rejected (now pinned by C4). A missing index file with the hash recomputed is rejected with no substitution (now pinned by C6).

## Notes

- Round 3: `git diff 52ee519..HEAD -- py api` is empty, so production code is unchanged since round 1. e919d54 touches only `tests/test_snapshot_manifest.py`, `checks.md` and `plan.md`. The plan edit appends one example (`allocation/caio_ifix.csv`) to criterion 2, which follows from its existing text ("a file not named `ifix.csv`"). Full suites at 801b752: `tests` 282 OK (1 skipped); `api/tests` with `PYTHONPATH=../py` 110 OK. Every mutation was reverted: the copy's `py/snapshot_manifest.py` is byte-identical to HEAD (`cmp`), and the worktree was never mutated.
- Round 3 own examples (HEAD, throwaway, not committed): an in-directory symlink `ifix.csv` -> `caio_fiis.csv`, a symlink into `<root>/older/allocation/ifix.csv`, and an explicit `sources.fiis` at `allocation/IFIX.csv` are each rejected at HEAD (correct; untested, V-4 and V-5).

- Production code is unchanged between 52ee519 and 669e841 (`git diff 52ee519..669e841 -- py api` is empty). The fixes touch only tests and spec. The plan edit appends one example (`./ifix.csv`) to criterion 3. That example follows from the criterion's existing text ("not exactly `ibovespa.csv`/`ifix.csv`"), so the approved obligation does not change.
- Full suites at 669e841: `tests` 282 OK (1 skipped); `api/tests` with `PYTHONPATH=../py` 110 OK.
- `validate_verification.py` needs `proofs.md`'s HEAD to equal the current HEAD. Committing `proofs.md` moves HEAD, so after this docs-only commit the gate also reports the HEAD mismatch. The builder's post-fix `check_proofs.py` run clears it. That run is the last step before the gate, and its commit touches only `proofs.md`.

## Round 4 closure (independent Verifier, at 8d557b2)

Scope set by the human: closure only. No new mutants were hunted beyond the ones listed.

1. `git diff 52ee519..HEAD -- py api` is empty. Production code is unchanged since round 1.
2. `check_proofs.py --mutations mutations.json` at 8d557b2: 28 pass, 0 fail; 19 mutations killed, 0 not. Spot-checked by hand on a throwaway `git archive HEAD` copy (`/tmp/iw38-v4-copy`, removed afterwards, `py/snapshot_manifest.py` restored and checked with `cmp` against the worktree) with `tests.test_snapshot_manifest` (baseline 12 OK):
   - V-4 (a), line `and resolved.name == expected` deleted: killed. `test_derived_index_series_must_live_in_allocation_dir` fails 4 subtests (`symlink='allocation/caio_fiis.csv'`, `'allocation/caio_ifix.csv'`, `'allocation/ifix.csv.old'`, `'allocation/IFIX.csv'`).
   - V-4 (b), `resolved.parent.name == allocation_dir.name`: killed. The same test fails at `symlink='older/allocation/ifix.csv'`.
   - V-5, explicit `resolved.name.lower() == expected`: killed. `test_series_identity_rule_rejects_non_index_files` fails at `explicit='IFIX.csv'` and `'IBOVESPA.csv'`.
3. The survivors the builder classified as equivalent or tighter:
   - M09 confirmed. Without `and allocation is not None`, the only changed shape is an explicit source with `derived_from: allocation` and no allocation source. That shape now crashes on `None.resolved_path()` with an `AttributeError` instead of being checked by name. It is rejected either way and never accepted.
   - M11 confirmed. The `""` default is reached only by an explicit source that claims `derived_from: allocation` with no `filename`. `declared` then fails `== expected`, so the mutant only rejects more.
   - M12 confirmed. `resolved` is a fully resolved path from `source_for`. An unresolved `allocation_dir` can equal its parent only when it is already canonical, so it accepts nothing HEAD rejects. It is tighter, never looser.
   - M31 confirmed. It changes only the message suffix, and only when the declared name is already wrong. The class, the expected name and the declared name stay in the message, and the accept/reject decision is identical.
   - M46 confirmed. `validate_manifest` has no early return between the call and `return checked`. Every check either raises or passes on its own, so moving the call changes only which error is reported first. The set of accepted manifests is the same.
   No classification is rejected, so this item has no blocking gap.
4. Full suites at 8d557b2: `tests` 282 OK (1 skipped); `api/tests` with `PYTHONPATH=../py` 110 OK.
5. V-4 and V-5 are closed by a439ee5 and re-checked by the mutations above. Every finding has a closing commit.

