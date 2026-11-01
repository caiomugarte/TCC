# IBOV and IFIX as the stock and FII series of the Premium allocation snapshot - checks

Mutations: 3
Build: none - the repo has no CI, build, lint or type-check configuration (`pyproject.toml` declares no tools); the stdlib `unittest` suites are the only gate
Plan: `.specs/features/ibov-ifix-allocation-series/plan.md`

24 checks in 6 slices · 3 one-way doors · 0 open, of which 0 block

`PY` below is `/home/caio/Documents/projetos/TCC/.venv/bin/python`, written out in full in every
proof; each proof runs from the worktree root and selects its tests with `unittest -k`.

## Checks

### S1 - The manifest contract names IBOV and IFIX

**C1** - `ALLOCATION_FILES` equals exactly `{"brazilian_stocks": "ibovespa.csv", "fiis": "ifix.csv", "international_equity": "sp500_total_return_usd.csv", "fixed_income": "di.csv", "crypto": "btc_usd.csv", "ptax": "ptax.csv"}`, and `source_for("brazilian_stocks")` on a manifest with no `metadata.files` resolves to `<allocation>/ibovespa.csv` (AC 1)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_snapshot_manifest -k test_allocation_files_name_ibov_and_ifix`
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_snapshot_manifest -k test_selector_role_does_not_resolve_to_stock_class_benchmark`

**C2** - `validate_manifest` raises `SnapshotManifestError` for each of 3 manifests whose hashes all match: `metadata.files.fiis == "caio_fiis.csv"` (message contains `fiis`, `ifix.csv` and `caio_fiis.csv`), `metadata.files.brazilian_stocks == "caio_stocks.csv"` (message contains `brazilian_stocks`, `ibovespa.csv` and `caio_stocks.csv`), and an explicit `sources.fiis` pointing at `<dir>/caio_fiis.csv` (AC 2)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_snapshot_manifest -k test_series_identity_rule_rejects_non_index_files`

**C3** - `validate_manifest` returns the manifest for `metadata.files` naming `ibovespa.csv`/`ifix.csv` explicitly and for a manifest with no `metadata.files` and both files present (AC 2)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_snapshot_manifest -k test_series_identity_rule_accepts_index_files`

**C4** - With the allocation directory hash matching, `validate_manifest` raises `SnapshotManifestError` for `metadata.files.fiis == "../snap-old/ifix.csv"` (file present outside the directory; message contains `fiis`, `ifix.csv` and `../snap-old/ifix.csv`), for `metadata.files.brazilian_stocks == "sub/ibovespa.csv"`, and for an `ifix.csv` inside the directory that is a symlink to a file outside it, and for `metadata.files.fiis == "./ifix.csv"` (resolves to `<allocation>/ifix.csv`, hash matching; message contains `./ifix.csv`); `metadata.files.fiis == "ifix.csv"` with the file inside is accepted (AC 3)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_snapshot_manifest -k test_derived_index_series_must_live_in_allocation_dir`

**C5** - `snapshot_registry.py validate` on a manifest naming `caio_fiis.csv` exits 4 in `draft` (state stays `draft`) and in `published` with `--check`, prints the C2 message on stderr, and leaves the file byte-identical; `published` without `--check` exits 5 (AC 4)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_snapshot_registry -k test_validate_rejects_caio_series_manifest`

**C6** - With `ifix.csv` deleted and `caio_fiis.csv` present, and separately with `ibovespa.csv` deleted and `caio_stocks.csv` present, each with the allocation hash recomputed after the deletion so it matches, `validate` exits 4 with stderr containing `source fiis not found` / `source brazilian_stocks not found` and `load_premium_snapshot_bundle` raises `SnapshotError` with the same text (AC 5)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_snapshot_registry -k test_validate_missing_index_file_exits_4`
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_allocation_data -k test_missing_premium_index_input_raises_without_substitution`

**C7** - One byte changed in `ibovespa.csv`, and separately in `ifix.csv`, after `build` makes `validate` exit 4 with stderr containing `SHA-256 mismatch for source allocation`, and `validate_manifest` raise (AC 6)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_snapshot_registry -k test_validate_detects_changed_index_bytes`

**C8** - A registry with a valid IBOV/IFIX manifest (earlier `cutoff_date`) and a newer published `caio_*` manifest makes `SnapshotRegistry.latest_compatible` raise `SnapshotManifestError`, not return the older manifest; after the `caio_*` manifest is retired it returns the IBOV/IFIX manifest (AC 7)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_snapshot_manifest -k test_selection_raises_when_newest_manifest_names_caio_series`

### S2 - build requires IBOV and IFIX and measures ~10 years

**C9** - `build` exits 3 and writes no manifest when `ifix.csv` is missing with `caio_fiis.csv` present (stderr contains `ifix.csv`), when `ibovespa.csv` is missing with `caio_stocks.csv` present (stderr contains `ibovespa.csv`), and when both are missing (stderr contains both) (AC 8)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_snapshot_registry -k test_build_missing_index_file_exits_3`

**C10** - `build` on a directory with the six `ALLOCATION_FILES` and no `caio_stocks.csv`/`caio_fiis.csv` exits 0 and writes a `draft` manifest whose `sources.allocation.metadata.files` equals the C1 map exactly (AC 9)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_snapshot_registry -k test_build_without_caio_files_records_index_map`

**C11** - With all six series spanning 2016-07-21..2026-07-21 plus a `caio_fiis.csv` starting 2024-06-13, `build` writes `common_dates[0] == "2016-07-21"` and `cutoff_date == "2026-07-21"`; with `ifix.csv` starting 2018-01-02, `common_dates[0] == "2018-01-02"` (AC 10)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_snapshot_registry -k test_build_common_dates_follow_index_history`

**C12** - `build` exits 4 and writes no manifest when `ibovespa.csv` holds the ticker columns of `caio_stocks.csv`, and when `ifix.csv` has the columns `AAA11,BBB11` (AC 11)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_snapshot_registry -k test_build_multi_series_index_file_exits_4`

**C13** - `class_series_map` maps `brazilian_stocks` to `<snapshot>/ibovespa.csv` and `fiis` to `<snapshot>/ifix.csv`, each with `file_sha256` equal to `sha256_path` of that file, and maps no `caio_*` file (AC 1, AC 20)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_snapshot_registry -k test_class_series_map_resolves_index_files`

### S3 - The Premium loader reads the indices as single series

**C14** - For a manifest with IBOV 100, 110, 99; IFIX 50, 55, 44; S&P 100, 102, 102; PTAX 5.0, 5.0, 5.5; DI 1.0, 1.0004, 1.0008; BTC 1000, 900, 900, `load_premium_snapshot_bundle` returns `brazilian_stocks` 0.10, -0.10; `fiis` 0.10, -0.20; `international_equity` 0.02, 0.10; `fixed_income` 0.0004, 0.0004/1.0004; `crypto` -0.10, 0.10, each within 1e-12, on the manifest's `common_dates` (AC 12)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_allocation_data -k test_premium_returns_follow_index_and_class_formulas`

**C15** - `load_premium_snapshot_bundle` raises `SnapshotError` whose message contains the C2 message when the manifest names `caio_fiis.csv`, and when it names `caio_stocks.csv` (AC 13)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_allocation_data -k test_premium_loader_rejects_caio_series_manifest`

**C16** - `load_premium_snapshot_bundle` raises `SnapshotError` when `ibovespa.csv` has two series columns none named `value`, and when `ifix.csv` does (AC 14)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_allocation_data -k test_premium_loader_rejects_multi_series_index_file`

### S4 - The Basic path does not change

**C17** - `load_snapshot_bundle` loads a Basic snapshot directory that has no `ibovespa.csv`, and the two existing Basic loader tests pass with their assertions unchanged (AC 15)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_allocation -k test_basic_snapshot_loader_needs_no_ibovespa_file`
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_allocation -k test_snapshot_loader_aligns_and_converts_series_without_filling`
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_allocation -k test_snapshot_loader_uses_equal_weight_fii_sleeve_without_ifix`

### S5 - fetch always writes IFIX

**C18** - `fetch_snapshot` with every provider patched writes `ifix.csv` and `ibovespa.csv` under the snapshot dir, each with the first line `date,value`, and calls `fetch_ifix_levels` once (AC 16)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_allocation -k test_snapshot_fetch_always_writes_ifix_and_ibovespa`

**C19** - `main(["--skip-ifix"])` and `main(["--no-ifix"])` exit 2 with no provider function called, and `fetch_snapshot(..., include_ifix=False)` raises `TypeError` (AC 17)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_allocation -k test_snapshot_cli_rejects_removed_ifix_flags`

**C20** - With `fetch_ifix_levels` raising `SnapshotFetchError`, `main([...])` returns 2 and the `--snapshot-dir` path does not exist afterwards (AC 18)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_allocation -k test_snapshot_cli_ifix_failure_exits_2_without_files`

**C21** - The `metadata.json` fetch writes has `sources.ibovespa.role == "allocation:brazilian_stocks"`, `sources.ibovespa.treatment == "total return index (B3 methodology); dividends included"`, `sources.ifix.role == "allocation:fiis"`, and `notes` without `does not include dividends` (AC 19)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_allocation -k test_snapshot_fetch_metadata_describes_index_series`

### S6 - The registry doc and the code comments show the new map

**C22** - `docs/snapshot_registry.md` contains `data/snapshot-2020/ibovespa.csv` and `data/snapshot-2020/ifix.csv` (the class-to-series map example), `total-return`, and the replacement steps' `metadata.json` and `in place` warning; the Verifier reads the rest of AC 20 (the series map sentence, the exit codes, the step order) against the plan (AC 20)
Proof: `grep -q -e 'data/snapshot-2020/ibovespa.csv' docs/snapshot_registry.md && grep -q -e 'data/snapshot-2020/ifix.csv' docs/snapshot_registry.md && grep -q -e 'total-return' docs/snapshot_registry.md && grep -q -e 'in place' docs/snapshot_registry.md && grep -q -e 'metadata.json' docs/snapshot_registry.md`

**C23** - `docs/snapshot_registry.md` no longer matches `caio_fiis.csv. is required`, `are never mapped`, `resolves to .caio_fiis.csv.` or `no IFIX substitution, ever` (AC 21)
Proof: `! grep -n -e 'caio_fiis.csv. is required' -e 'are never mapped' -e 'resolves to .caio_fiis.csv.' -e 'no IFIX substitution, ever' docs/snapshot_registry.md`

**C24** - `py/allocation_data.py` no longer matches `never reads IFIX`, `py/snapshot_registry.py` no longer matches `multi-ticker sleeves rebuilt`, and the comment above `ALLOCATION_FILES` in `py/snapshot_manifest.py` no longer matches `may override them` (AC 22)
Proof: `! grep -n -e 'never reads IFIX' py/allocation_data.py && ! grep -n -e 'multi-ticker sleeves rebuilt' py/snapshot_registry.py && ! grep -n -e 'may override them' py/snapshot_manifest.py`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| series identity rule, AC 2 (5 examples) | `metadata.files.fiis=caio_fiis.csv` C2 · `metadata.files.brazilian_stocks=caio_stocks.csv` C2 · explicit `sources.fiis` -> `caio_fiis.csv` C2 · explicit index names accepted C3 · defaults accepted C3 | - |
| derived series containment, AC 3 (5 examples) | `../snap-old/ifix.csv` C4 · `sub/ibovespa.csv` C4 · symlink escaping the directory C4 · `./ifix.csv` not declared character for character C4 · `ifix.csv` inside accepted C4 | - |
| `validate` on a rejected manifest, AC 4 (3 states) | `draft` -> 4 C5 · `published --check` -> 4 C5 · `published` -> 5 C5 | - |
| missing index file, AC 5, 2 files x 2 surfaces (4) | `ifix.csv` validate C6 · `ifix.csv` loader C6 · `ibovespa.csv` validate C6 · `ibovespa.csv` loader C6 | - |
| changed bytes, AC 6 (2 files) | `ibovespa.csv` C7 · `ifix.csv` C7 | - |
| selection, AC 7 (2 examples) | newer `caio_*` raises C8 · after retire returns IBOV/IFIX C8 | - |
| `build` missing input, AC 8 (3 examples) | `ifix.csv` C9 · `ibovespa.csv` C9 · both C9 | - |
| `common_dates` window, AC 10 (2 examples) | full 10 years despite `caio_fiis.csv` C11 · late `ifix.csv` start C11 | - |
| multi-series index file, AC 11 + AC 14, 2 files x 2 surfaces (4) | `ibovespa.csv` build C12 · `ifix.csv` build C12 · `ibovespa.csv` loader C16 · `ifix.csv` loader C16 | - |
| Premium classes, AC 12 (5) | `brazilian_stocks` C14 · `fiis` C14 · `international_equity` C14 · `fixed_income` C14 · `crypto` C14 | - |
| caio manifest at the loader, AC 13 (2) | `caio_fiis.csv` C15 · `caio_stocks.csv` C15 | - |
| doors (3) | door 1 identity rule C2 C3 C4 · door 2 `ALLOCATION_FILES` and persisted `metadata.files` C1 C10 · door 3 fetch flags and keyword removed C19 | - |
| removed fetch inputs, door 3 (3) | `--skip-ifix` C19 · `--no-ifix` C19 · `include_ifix` C19 | - |
| fetch metadata fields, AC 19 (4) | `sources.ibovespa.role` C21 · `sources.ibovespa.treatment` C21 · `sources.ifix.role` C21 · `notes` C21 | - |
| `ALLOCATION_FILES` readers (4) | `SnapshotManifest.source_for` C1 · `_manifest_level_series` C14 · `snapshot_registry` build C9 C10 · `class_series_map` C13 | - |
| `validate_manifest` callers (4) | `snapshot_registry.py validate` C5 · `SnapshotRegistry.latest_compatible` C8 · `load_premium_snapshot_bundle` C15 · API `premium_recommendation`/`premium_executor` C15, through the same `validate_manifest` call they make | - |
| false statements removed, AC 21 + AC 22 (8) | doc `caio_fiis.csv` required C23 · doc never mapped C23 · doc `fiis` -> `caio_fiis.csv` C23 · doc no IFIX substitution C23 · `snapshot_manifest.py` may override C24 · `allocation_data.py` never reads IFIX C24 · `snapshot_registry.py` multi-ticker sleeves C24 · class map example rewritten C22 | - |
| `build` exit codes this feature adds causes for (3) | 0 C10 · 3 C9 · 4 C12 | - |
| `validate` exit codes this feature adds causes for (2) | 4 C5 C6 C7 · 5 C5 | - |

- Claims naming an exit code: C5, C6, C7, C9, C10, C12, C19, C20 - each proof runs the CLI entry point (`snapshot_registry.main` or `fetch_allocation_snapshot.main`), not the helper underneath
- The API services are not exercised directly: they reach the rule only through `validate_manifest`/`latest_compatible_manifest`, proven at C5, C8 and C15; queued runs on a `caio_*` manifest are an `Out of scope` row in the plan

## Swept

- validation: C2, C3, C4, C12, C16
- failure modes: C9, C12, C20 - no manifest or snapshot file is written on a failed `build` or fetch
- idempotency: C5 - a rejected `validate` leaves the manifest byte-identical; existing - `test_build_existing_target_or_missing_source_exits_3` keeps a second `build` from overwriting
- authorization: n/a - operator CLIs and a library rule; no route or caller identity changes (plan `Surface`: none)
- concurrency: n/a - no new writer or shared state; registry writes stay single-operator with atomic `os.replace` (existing, `docs/snapshot_registry.md` Prohibitions)
- data lifecycle: C7, C8 - old manifests stay immutable and are retired, not rewritten; republishing `dev-2026-10-29` is an `Out of scope` operator action documented by C22
- dependency failure: C20
- state transitions: C5, C8
- observability: C2, C5, C9 - the error names the class, the expected and the declared file; `build` names each missing file on stderr

## Handoff

- **Boundary:** C1-C17 closed at `7230940` (manifest identity rule, `ALLOCATION_FILES`, `build`, Premium loader, Basic guard); C18-C21 closed at `fe266b0` (fetch); C22-C24 closed at `7230940` (code comments, C24) and `c61440b` (registry doc, C22-C23). Every proof exits 0 at `c61440b`, each `-k` filter selecting exactly one test. The four suites ran 115 tests OK (baseline 96); `unittest discover -s tests` ran 282 OK, 1 skipped (baseline 263 OK, 1 skipped); `api/tests` with `PYTHONPATH=../py` ran 110 OK. Without `PYTHONPATH`, `api/tests/test_allocation_engine.py` fails to import `allocation_data`. That failure is pre-existing: it is identical at base `20c3bbe` (108 run, 1 error).
- **Settled mid-build:** nothing asked. Builder choices inside the approved doors: the rule lives in `_validate_index_series`, called from `validate_manifest` after `_validate_dates` and before any existence or hash check, so a `caio_*` manifest always gets the identity message. It is applied to `INDEX_SERIES_CLASSES = ("brazilian_stocks", "fiis")` whatever `required_classes` holds, as door 1 says. When a derived series is declared as exactly `ifix.csv`/`ibovespa.csv` but resolves outside the allocation directory (the symlink case), the message reads `manifest declares <name> (resolves to <path>)`. For an explicit per-class source, `<declared>` is the source's `path`. `load_premium_snapshot_bundle` keeps its `rebalance_years` parameter for signature compatibility, but it no longer affects the Premium path. The `load_snapshot_bundle` docstring "never substitutes IFIX" became "never substitutes a fallback series", because the Premium path now reads IFIX. Fetch `notes` now say Ibovespa (Yahoo `^BVSP`) and IFIX are total-return indices with dividends included.
- **Abandoned:** nothing.
- **Observed (non-gating, plan `Assumptions`):** `premium_level_series` + `levels_to_returns` over the real `/home/caio/Documents/projetos/TCC/snapshots/data/snap-2026-10-29/` (read only, nothing written) yield 2416 common dates from 2016-10-31 to 2026-10-28. The published `dev-2026-10-29` manifest was not touched; republishing it is out of scope.
