# v0-legacy baseline bundle - checks

Profile: standard
Plan: `.specs/features/p0-01-v0-legacy-baseline/plan.md`

35 checks in 4 slices · 4 one-way doors · 0 open, of which 0 block

## Intent

The thesis' v0 baseline lives only in `outputs/`, which no longer matches the commit the numbers
came from, and nothing in the repository says how the numbers were produced. When this ships, a
reader can point at `baselines/v0-legacy/`, a commit SHA and a manifest, verify that the archived
bytes are the archived bytes, read the protocol the numbers came from, and run one command that
either reproduces them, names the exact fields where they do not reproduce, or states
`UNVERIFIABLE` with a cause code.

## Checks

### S1 - The frozen bundle and its manifest · 4 authored files · 23 KB (+633 KB archived) · ~6k

**C1** - `baselines/v0-legacy/baseline.json` lists exactly 17 covered files and their combined `bytes` is 632,981 (AC 1)
Proof: `python3 -m unittest discover -s tests -p 'test_v0_legacy_manifest.py' -k test_bundle_holds_seventeen_files_and_632981_bytes`

**C2** - Each covered file's `sha256`, `git_blob_sha1`, `bytes` and `mode` equal the blob at `97ea3275293da4e1902fcfeb684c814f04f5368f` for its `source_path` (AC 1)
Proof: `python3 -m unittest discover -s tests -p 'test_v0_legacy_manifest.py' -k test_every_file_matches_its_commit_blob`

**C3** - Every covered file records `path`, `source_path`, `sha256`, `git_blob_sha1`, `bytes`, `mode` and `role`, with `role` taking `executed`, `comparison` or `provenance_only`, `data/processed/eligibility_filters.parquet` is recorded as `provenance_only`, and `tools/build_manifest.py` regenerates exactly the committed `baseline.json` from those bytes (AC 2)
Proof: `python3 -m unittest discover -s tests -p 'test_v0_legacy_manifest.py' -k test_every_file_records_the_seven_manifest_fields`
Proof: `python3 -m unittest discover -s tests -p 'test_v0_legacy_manifest.py' -k test_roles_and_the_provenance_only_parquet`
Proof: `python3 -m unittest discover -s tests -p 'test_v0_legacy_manifest.py' -k test_builder_reproduces_the_committed_manifest`

**C4** - A covered file deleted from a bundle copy makes `tools/verify_manifest.py` exit 1 and report that `path` with kind `missing` (AC 3)
Proof: `python3 -m unittest discover -s tests -p 'test_v0_legacy_manifest.py' -k test_missing_file_is_reported_by_path_and_kind`

**C5** - A covered file appended to in a bundle copy makes `tools/verify_manifest.py` exit 1 and report that `path` with kind `modified` (AC 3)
Proof: `python3 -m unittest discover -s tests -p 'test_v0_legacy_manifest.py' -k test_modified_file_is_reported_by_path_and_kind`

**C6** - An unlisted file written under `code/` in a bundle copy makes `tools/verify_manifest.py` exit 1 and report that `path` with kind `unexpected` (AC 3)
Proof: `python3 -m unittest discover -s tests -p 'test_v0_legacy_manifest.py' -k test_unexpected_file_is_reported_by_path_and_kind`

**C7** - `coverage.included_subtrees` is exactly `code/`, `data/`, `reference/` and `coverage.excluded` names `baseline.json`, `protocol.json`, `README.md` and `tools/`, each with a reason (AC 4)
Proof: `python3 -m unittest discover -s tests -p 'test_v0_legacy_manifest.py' -k test_coverage_covers_three_subtrees_and_names_four_exclusions`

**C8** - `tools/verify_manifest.py` on the committed bundle exits 0 and writes exactly one JSON object to stdout with `status` `VERIFIED` and `files` 17 (AC 5)
Proof: `python3 -m unittest discover -s tests -p 'test_v0_legacy_manifest.py' -k test_verified_bundle_exits_zero_with_seventeen_files`

### S2 - The protocol as a machine-readable contract · 1 file · 2 KB (reads 3 archived modules) · ~0.5k

**C9** - `protocol.json` declares profile `caio` with `n_assets` 10, `lambda` 0.49, `generations` 460 and `pop_size` 276, equal to the literals `ast` extracts from `code/ga.py` (AC 6)
Proof: `python3 -m unittest discover -s tests -p 'test_v0_legacy_protocol.py' -k test_ga_parameters_equal_the_archived_literals`

**C10** - `protocol.json` declares the five `caio` group weights `liquidez` 0.22, `rent` 0.35, `value` 0.08, `growth` 0.05, `div` 0.30, equal to the literals `ast` extracts from `code/profiles.py` (AC 6)
Proof: `python3 -m unittest discover -s tests -p 'test_v0_legacy_protocol.py' -k test_group_weights_equal_the_archived_literals`

**C11** - `protocol.json` declares 30 runs over `run_id` 0 to 29 with `seed` = 42 + `run_id`, seeded on both `random` and `numpy.random` in `code/ga_multiple_runs.py` (AC 7)
Proof: `python3 -m unittest discover -s tests -p 'test_v0_legacy_protocol.py' -k test_runs_and_seed_formula_equal_the_archived_literals`
Proof: `python3 -m unittest discover -s tests -p 'test_v0_legacy_protocol.py' -k test_wrong_seed_formula_is_rejected_naming_the_field`

**C12** - `protocol.json` declares consensus by frequency over run membership, the `ga.n_assets` most frequent tickers, ties broken by first appearance order, and early stopping disabled, each matching `ast` structure in `code/ga_multiple_runs.py` and `code/ga.py` (AC 8)
Proof: `python3 -m unittest discover -s tests -p 'test_v0_legacy_protocol.py' -k test_consensus_rule_and_disabled_early_stop`

**C13** - `protocol.json` declares `dependencies` with `status` `UNVERIFIABLE`, `declared_at_commit` `7d8d6a0`, `evidence_path` `py/requirements.txt` and a `reason` naming that `97ea327` declares no dependency set (AC 9)
Proof: `python3 -m unittest discover -s tests -p 'test_v0_legacy_protocol.py' -k test_dependencies_are_declared_unverifiable`

**C14** - The comparison reads the three archived modules with `ast` and imports none of them, so `ga`, `profiles` and `ga_multiple_runs` are absent from `sys.modules` afterwards (AC 10)
Proof: `python3 -m unittest discover -s tests -p 'test_v0_legacy_protocol.py' -k test_ast_comparison_imports_no_legacy_module`

**C15** - `runs.n_runs` set to 29 in a protocol copy is rejected with a violation naming the field `runs.n_runs` (AC 11)
Proof: `python3 -m unittest discover -s tests -p 'test_v0_legacy_protocol.py' -k test_wrong_run_count_is_rejected_naming_the_field`

**C16** - A protocol copy with the whole `dependencies` block deleted is rejected with a violation naming `dependencies` (AC 11)
Proof: `python3 -m unittest discover -s tests -p 'test_v0_legacy_protocol.py' -k test_missing_dependencies_block_is_rejected_naming_the_field`

**C17** - A protocol copy declaring an early stop, or a GA parameter the archived code does not have, is rejected with a violation naming `early_stopping.enabled` or `ga.<name>` (AC 11)
Proof: `python3 -m unittest discover -s tests -p 'test_v0_legacy_protocol.py' -k test_declared_early_stop_is_rejected_naming_the_field`
Proof: `python3 -m unittest discover -s tests -p 'test_v0_legacy_protocol.py' -k test_unknown_ga_parameter_is_rejected_naming_the_field`

### S3 - Replay and its status · 2 files · 21 KB · ~5k

**C18** - One byte appended to `code/ga.py` in a bundle copy makes `tools/replay.py` report `UNVERIFIABLE` with cause `hash_mismatch`, exit 2, and spawn no driver (AC 12)
Proof: `python3 -m unittest discover -s tests -p 'test_v0_legacy_replay.py' -k test_manifest_drift_stops_before_the_driver_runs`

**C19** - With the archived driver and `numpy` and `pandas` unimportable, `tools/replay.py` reports `UNVERIFIABLE` with cause `dependency_unavailable`, `missing_modules` equal to `["numpy", "pandas"]`, and exit 2 (AC 13)
Proof: `python3 -m unittest discover -s tests -p 'test_v0_legacy_replay.py' -k test_missing_dependencies_are_named_and_exit_two`

**C20** - With `comparison.reference_metrics` naming a file the manifest does not list, `tools/replay.py` reports `UNVERIFIABLE` with cause `dataset_unavailable` and exit 2 (AC 13, AD-001 vocabulary)
Proof: `python3 -m unittest discover -s tests -p 'test_v0_legacy_replay.py' -k test_uncovered_dataset_is_reported_unavailable`

**C21** - A driver whose `runs.json` matches `reference/metrics_stability_caio.csv` on every compared field makes `tools/replay.py` report `REPRODUCED` and exit 0 (AC 14)
Proof: `python3 -m unittest discover -s tests -p 'test_v0_legacy_replay.py' -k test_matching_runs_report_reproduced`

**C22** - A driver whose `runs.json` differs from that CSV in one `fitness` cell makes `tools/replay.py` report `MISMATCH`, exit 1, and one `fields` entry carrying that `field`, its `recorded` and its `recomputed` value (AC 15)
Proof: `python3 -m unittest discover -s tests -p 'test_v0_legacy_replay.py' -k test_one_differing_field_reports_mismatch`

**C23** - Every report carries the protocol values the replay used - `profile`, `n_assets`, `lambda`, `generations`, `pop_size`, `n_runs`, `run_id_start`, `rtol`, `atol`, `fields` - and a protocol copy without `runs.n_runs` yields `UNVERIFIABLE` with cause `protocol_mismatch` naming `runs.n_runs` (AC 16)
Proof: `python3 -m unittest discover -s tests -p 'test_v0_legacy_replay.py' -k test_report_carries_the_protocol_values_used`
Proof: `python3 -m unittest discover -s tests -p 'test_v0_legacy_replay.py' -k test_missing_protocol_value_is_reported_as_protocol_mismatch`

**C24** - No `TICKER` from `reference/carteira_caio_consensus.json` occurs literally in `tools/replay.py` or in `protocol.json` (AC 17)
Proof: `python3 -m unittest discover -s tests -p 'test_v0_legacy_replay.py' -k test_no_reference_ticker_is_hardcoded`

**C25** - After a `REPRODUCED` run the nine files under `reference/` have the same `sha256` as before it and the report says `golden_output_written` false (AC 18)
Proof: `python3 -m unittest discover -s tests -p 'test_v0_legacy_replay.py' -k test_reference_tree_is_untouched_and_no_golden_output`

**C26** - A driver exiting 3 after writing 5000 characters to stderr makes `tools/replay.py` report `UNVERIFIABLE` with cause `runtime_error`, a `stderr_tail` of at most 4096 characters, and exit 2 (AC 19)
Proof: `python3 -m unittest discover -s tests -p 'test_v0_legacy_replay.py' -k test_nonzero_driver_exit_reports_runtime_error`

**C27** - A driver still running at `--budget-seconds` 1 is terminated and reported `UNVERIFIABLE` with cause `runtime_budget_exceeded` and exit 2 (AC 20)
Proof: `python3 -m unittest discover -s tests -p 'test_v0_legacy_replay.py' -k test_budget_exceeded_terminates_the_executor`

**C28** - A covered byte moved while the driver runs makes `tools/replay.py` report `bundle_stable` false, cause `hash_mismatch`, and exit 3 (AC 21)
Proof: `python3 -m unittest discover -s tests -p 'test_v0_legacy_replay.py' -k test_bundle_moved_during_the_run_exits_three`

**C29** - `--driver` is resolved against the bundle and echoed in the report's `driver` field, the child runs with the scratch root as cwd and no `reference/` beside it, `<scratch_root>/runs.json` is the only datum source, a substitute driver needs neither `numpy` nor `pandas`, and a child that writes no `runs.json` yields `UNVERIFIABLE` with cause `runtime_error` (AC 22)
Proof: `python3 -m unittest discover -s tests -p 'test_v0_legacy_replay.py' -k test_substitute_driver_is_resolved_and_recorded`
Proof: `python3 -m unittest discover -s tests -p 'test_v0_legacy_replay.py' -k test_child_without_runs_json_reports_runtime_error`

**C30** - Each report row carries `reported_fitness` from the CSV cell and `recomputed_fitness` from `runs.json`, and a substituted `recomputed_fitness` that matches nothing leaves the status `REPRODUCED` (AC 23)
Proof: `python3 -m unittest discover -s tests -p 'test_v0_legacy_replay.py' -k test_reported_and_recomputed_fitness_stay_independent`

### S4 - CLI contract and production isolation · 1 file · 8 KB · ~2k

**C31** - `tools/verify_manifest.py` run with no arguments from the repository root verifies `baselines/v0-legacy` (AC 24)
Proof: `python3 -m unittest discover -s tests -p 'test_v0_legacy_isolation.py' -k test_verifier_defaults_to_the_repository_bundle`

**C32** - `tools/replay.py --help` names the defaults `baselines/v0-legacy`, `outputs/baseline_v0_legacy/replay_report.json`, `1800` and `tools/_legacy_driver.py`, and `--report` writes the file it prints (AC 25)
Proof: `python3 -m unittest discover -s tests -p 'test_v0_legacy_isolation.py' -k test_replay_flags_and_defaults`

**C33** - Each CLI prints exactly one JSON object on stdout with diagnostics on stderr, and exits 0 for `VERIFIED`/`REPRODUCED`, 1 for drift/`MISMATCH`, 2 for `UNVERIFIABLE`, 3 for a manifest it cannot trust (AC 26)
Proof: `python3 -m unittest discover -s tests -p 'test_v0_legacy_isolation.py' -k test_exit_codes_and_single_json_object_on_stdout`

**C34** - The isolation scan reports zero offenders under `py/`, `api/` and the repo-root `*.py` for both a bundle-module import and the path segment `baselines/`, and names `py/config.py:1` when that line is injected into a copy (AC 27)
Proof: `python3 -m unittest discover -s tests -p 'test_v0_legacy_isolation.py' -k test_no_production_file_reaches_the_bundle`
Proof: `python3 -m unittest discover -s tests -p 'test_v0_legacy_isolation.py' -k test_isolation_scan_flags_an_injected_bundle_path`

**C35** - No `__init__.py`, `setup.py` or `pyproject.toml` exists at any depth under `baselines/` (AC 28)
Proof: `python3 -m unittest discover -s tests -p 'test_v0_legacy_isolation.py' -k test_no_package_marker_under_baselines`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| covered files (17) | C1, C2 - table-driven over all 17 | - |
| finding kinds (3) | `missing` C4 · `modified` C5 · `unexpected` C6 | - |
| file roles (3) | `executed` C3 · `comparison` C3 · `provenance_only` C3 | - |
| covered subtrees (3) | `code/` C7 · `data/` C7 · `reference/` C7 | - |
| excluded authored paths (4) | `baseline.json` C7 · `protocol.json` C7 · `README.md` C7 · `tools/` C7 | - |
| protocol GA parameters (4) | `n_assets` C9 · `lambda` C9 · `generations` C9 · `pop_size` C9 | - |
| profile group weights (5) | `liquidez` C10 · `rent` C10 · `value` C10 · `growth` C10 · `div` C10 | - |
| seed targets (2) | `random` C11 · `numpy.random` C11 | - |
| dependencies block fields (4) | `status` C13 · `declared_at_commit` C13 · `evidence_path` C13 · `reason` C13 | - |
| replay statuses (3) | `REPRODUCED` C21 · `MISMATCH` C22 · `UNVERIFIABLE` C19 | - |
| cause codes (6) | `hash_mismatch` C18 · `dependency_unavailable` C19 · `dataset_unavailable` C20 · `protocol_mismatch` C23 · `runtime_error` C26 · `runtime_budget_exceeded` C27 | - |
| CLI exit codes (4) | 0 C33 · 1 C33 · 2 C33 · 3 C33 | - |
| replay flags (4) | `--bundle` C32 · `--report` C32 · `--budget-seconds` C32 · `--driver` C32 | - |
| scanned production groups (3) | `py/` C34 · `api/` C34 · repo-root `*.py` C34 | - |
| package markers (3) | `__init__.py` C35 · `setup.py` C35 · `pyproject.toml` C35 | - |

- Claims naming a status, a cause code or a process exit: C18-C22, C26-C28, C33 - each has a proof that runs the CLI and reads the exit code.
- Claims naming a manifest field or a protocol value: C3, C7, C9-C13, C23 - each has a proof that asserts that field against the archived blob or the `ast` extraction.
- `bundle_stable` takes two values and both are proven: true on every terminal path in C21-C30, false with exit 3 in C28.

## Test policy

The repository's `.specs/codebase/TESTING.md` names the runner and where tests live, and nothing
about which level proves which code, so these rows are the bar this build runs under.

| Code | Required proofs | Coverage expectation |
| --- | --- | --- |
| Decides, reached across a boundary | one at the boundary **and** one at its own layer | the contract at the boundary; one asserted case per row of the decision table at its own layer |
| Decides, not reached across a boundary | one at its own layer | one asserted case per row of the decision table |
| Entry point that decides nothing | one at the boundary | accepted input, each rejected input, each error path |
| Instrumentation, pass-throughs | none of its own | covered by its consumer's proof |

Evidence:

- `baselines/v0-legacy/tools/verify_manifest.py`: three finding kinds, two terminal statuses, one manifest-trust guard -> decides, reached across a boundary (C4, C5, C6, C8)
- `baselines/v0-legacy/tools/replay.py`: six cause codes, three statuses, four exit paths, one driver seam -> decides, reached across a boundary (C18-C30)
- `baselines/v0-legacy/tools/build_manifest.py`: walks three subtrees and writes one file, no conditional on an outcome -> instrumentation, covered by C2 and C3
- `baselines/v0-legacy/tools/_legacy_driver.py`: forwards the archived call sequence, writes one file -> instrumentation, covered by C29 and C30
- closest analogue in the repo: `tests/test_snapshot_manifest.py`, same shape - a hashing CLI asserted through its exit code - already proven at its own layer with a missing/modified pair

Cost: 10 proofs at their own layer across two deciding files, and 6 boundary proofs through the
two CLIs. Without these rows the cause-code table and the three finding kinds would be proven only
by a path that happens to traverse them.

## Swept

- validation: C3, C7 - a manifest missing a field or naming an unlisted subtree is rejected, C23 - a protocol missing a used value is rejected
- failure modes: C18, C26, C27, C28 - every terminal path names a status and a cause code
- idempotency: C25 - the replay writes nothing under `reference/` and re-verifies, so a second run reads the same bytes
- authorization: n/a - in-repo CLIs with no auth boundary and no caller to authorize
- concurrency: n/a - single-process CLIs; the only child is the driver, bounded by one timeout
- data lifecycle: C1, C2 - the archived bytes are frozen and `data/` and `outputs/` are not rewritten, so no migration exists
- dependency failure: C19, C27 - absent `numpy`/`pandas` and an over-budget executor each become a cause code, never a hang
- state transitions: C18, C21, C22, C23 - one status out of three plus one cause code per terminal path, per AD-001
- observability: C26, C33 - one JSON object on stdout, diagnostics on stderr, `stderr_tail` bounded at 4096

## Handoff

Slices: S1 bundle + manifest 23 KB authored · S2 protocol 2 KB · S3 replay 21 KB · S4 isolation
8 KB, plus 47 KB of tests and this file.

Arithmetic: the archived surface is 632,981 bytes copied with `git show` - 158k tokens if counted
as build surface, but not one line of it is authored or edited, so it is data and not work. The
authored surface is the 12 new files at 112,217 bytes = ~28k tokens. S2 enters S3 at ~7k and S4
at ~24k, both under the 150k budget - one builder, no ask.

- Mechanism: one builder (this delegation). No handoff cut.
- **Boundary:** closed in five commits - `061326c` for the first eight, `01cc288` for S2,
  `0a517ac` for S3, `f56809c` for S4, and `4713100` for the builder proof added to C3.
- **Settled mid-build:** the dependency probe applies to the archived driver only. AC 13 demands
  `dependency_unavailable` when `numpy`/`pandas` are absent, and AC 22 demands a substitute
  driver that needs neither; probing before every driver made those two contradict each other on
  a machine without `numpy`, so the probe reads the import set from `code/` only when the
  resolved driver is `tools/_legacy_driver.py`. C19 and C29 were reworded before the branch that
  closes them and the reason is in `README.md`.
- **Settled mid-build:** `build_manifest.py` takes its hash definitions from
  `verify_manifest.py`, in that direction only, so a builder edit cannot define a manifest its
  verifier disagrees with; C3 gained a proof that the committed `baseline.json` is regenerated
  from the archived bytes.
- **Abandoned:** none.
