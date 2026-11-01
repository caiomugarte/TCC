# IBOV and IFIX as the stock and FII series of the Premium allocation snapshot - verification

**Verdict**: FAIL
**Mutations**: 3 - probed C4, C2, C11
**Proofs**: `proofs.md` at d305235 (re-run by the Verifier: 28 pass, 0 fail; mutations 2 killed, 1 not)
**Verifier**: independent sub-agent (author != verifier)

## Findings

| Finding | Evidence | Fix | Closed |
| --- | --- | --- | --- |
| V-1 (blocking, manifest validation): the door-1 rule "a derived series must be declared by exactly `ibovespa.csv`/`ifix.csv`, character for character" has no test. Removing `declared == expected` from `_validate_index_series` keeps every test green | mutation `C4: drop declared == expected` (`py/snapshot_manifest.py:491`) not killed by the C4 proof, and not killed by the full suites either (`tests` 282 OK, `api/tests` 110 OK). At HEAD the code is correct: `metadata.files.fiis` set to `./ifix.csv`, `sub/../ifix.csv` or the absolute `<dir>/allocation/ifix.csv` is rejected with `source fiis must be ifix.csv; manifest declares ./ifix.csv`. Each C4 example that is rejected also fails the name check or the parent check, so the literal-name condition is never the deciding one | add the case `metadata.files.fiis == "./ifix.csv"` (it resolves to `<allocation>/ifix.csv` and the hash matches) to `test_derived_index_series_must_live_in_allocation_dir`, asserting `SnapshotManifestError` with the message containing `./ifix.csv`; append it as an example to C4 / AC 3 | |
| V-2 (nonblocking): the C6 proofs do not tell the new rule from the old code. They also pass on the base `20c3bbe` production code, because deleting the file changes the allocation directory hash, and they assert only exit 4 / `SnapshotError`, not the cause. The "no substitution" case, where the index file was absent when the hash was computed, is not tested | HEAD tests run against the `20c3bbe` versions of `py/snapshot_manifest.py`, `py/snapshot_registry.py` and `py/allocation_data.py`: `test_validate_missing_index_file_exits_4` OK and `test_missing_premium_index_input_raises_without_substitution` OK. Own example at HEAD: delete `ifix.csv` (or `ibovespa.csv`), recompute the allocation sha256 and keep `caio_fiis.csv`. `validate_manifest` then raises `source fiis not found`, and the loader raises `SnapshotError` with the same message (correct) | recompute `allocation_source.sha256` after the deletion in the C6 tests (or assert stderr / the message contains `source fiis not found` / `source brazilian_stocks not found`), so the proof fails when a fallback is substituted and not only when the hash diverges | |

## Mutations injected

Narrowest proof per mutation. These are the spec entries passed to `check_proofs.py --mutations`; the builder re-creates `mutations.json` from them:

```json
[
 {"file": "py/snapshot_manifest.py",
  "old": "                declared == expected\n                and resolved.name == expected\n",
  "new": "                resolved.name == expected\n",
  "proof": "/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_snapshot_manifest -k test_derived_index_series_must_live_in_allocation_dir",
  "label": "C4: derived series no longer has to be declared character for character (drop declared == expected)"},
 {"file": "py/snapshot_manifest.py",
  "old": "            valid = resolved.name == expected\n",
  "new": "            valid = True\n",
  "proof": "/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_snapshot_manifest -k test_series_identity_rule_rejects_non_index_files",
  "label": "C2: explicit per-class source skips the identity rule (valid = True)"},
 {"file": "py/snapshot_registry.py",
  "old": "        \"fiis\": single(\"fiis\"),\n",
  "new": "        \"fiis\": single(\"brazilian_stocks\"),\n",
  "proof": "/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_snapshot_registry -k test_build_common_dates_follow_index_history",
  "label": "C11: build common_dates reads IBOV in place of IFIX for fiis"}
]
```

| Mutation | Location | Killed | Blocking |
| --- | --- | --- | --- |
| C4: drop `declared == expected` | `py/snapshot_manifest.py` `_validate_index_series` | no (full suites stay green too) | yes - V-1 |
| C2: explicit source `valid = True` | `py/snapshot_manifest.py` `_validate_index_series` | yes | - |
| C11: `fiis` reads `brazilian_stocks` | `py/snapshot_registry.py` `premium_level_series` | yes | - |

## Own examples (HEAD, throwaway, not committed)

- AC 3 / door 1: `metadata.files.fiis` set to `./ifix.csv`, `sub/../ifix.csv` and the absolute allocation path are all rejected. The code is correct, but no test pins this (V-1).
- AC 5: the index file was absent when the allocation hash was computed and the basket is present. `validate_manifest` and `load_premium_snapshot_bundle` raise `source <class> not found` for both `ifix.csv` and `ibovespa.csv`, with no substitution (correct; untested, V-2).
- AC 14 boundary: an `ifix.csv` with one column not named `value` (`IFIX`) loads, and one with `value,AAA11` loads, reading `value` (fiis return 11 -> 12 = 0.0909). This is consistent with "more than one column and none named `value`".

## Notes

- C7 also passes on the base code. This is not a gap: AC 6 states that hash verification is *unchanged*, so C7 is a regression guard. It asserts the exact `SHA-256 mismatch for source allocation` message and that `validate_manifest` raises for both files.
- C14 holds the criterion's values: every class has two returns within `delta=1e-12`, the rows' dates equal `common_dates`, and basket files that would give other numbers are present.
- AC 20 (the part C22 leaves to the Verifier): `docs/snapshot_registry.md` states the series map, that both indices are total-return, that `build` requires the six `ALLOCATION_FILES` (exit 3), that `validate` exits 4, and the five replacement steps in the plan's order, including the `metadata.json` and in-place warnings. It matches the plan.
- Full suites at HEAD: `tests` 282 OK (1 skipped); `api/tests` with `PYTHONPATH=../py` 110 OK.
