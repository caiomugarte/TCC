# p0-01-v0-legacy-baseline proofs

**HEAD**: 0e6a961
**Tree**: clean
**Result**: 52 pass, 0 fail; mutations 18 killed, 0 not

| Check | Proof | Ran | Passed | Result |
| --- | --- | --- | --- | --- |
| C1 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_manifest.py' -k test_bundle_holds_seventeen_files_and` | 1 | - | pass |
| C2 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_manifest.py' -k test_every_file_matches_its_commit_bl` | 1 | - | pass |
| C3 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_manifest.py' -k test_every_file_records_the_seven_man` | 1 | - | pass |
| C3 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_manifest.py' -k test_roles_and_the_provenance_only_pa` | 1 | - | pass |
| C3 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_manifest.py' -k test_builder_reproduces_the_committed` | 1 | - | pass |
| C4 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_manifest.py' -k test_missing_file_is_reported_by_path` | 1 | - | pass |
| C5 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_manifest.py' -k test_modified_file_is_reported_by_pat` | 1 | - | pass |
| C6 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_manifest.py' -k test_unexpected_file_is_reported_by_p` | 1 | - | pass |
| C7 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_manifest.py' -k test_coverage_covers_three_subtrees_a` | 1 | - | pass |
| C8 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_manifest.py' -k test_verified_bundle_exits_zero_with_` | 1 | - | pass |
| C9 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_protocol.py' -k test_ga_parameters_equal_the_archived` | 1 | - | pass |
| C10 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_protocol.py' -k test_group_weights_equal_the_archived` | 1 | - | pass |
| C11 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_protocol.py' -k test_runs_and_seed_formula_equal_the_` | 1 | - | pass |
| C11 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_protocol.py' -k test_wrong_seed_formula_is_rejected_n` | 1 | - | pass |
| C12 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_protocol.py' -k test_consensus_rule_and_disabled_earl` | 1 | - | pass |
| C13 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_protocol.py' -k test_dependencies_are_declared_unveri` | 1 | - | pass |
| C14 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_protocol.py' -k test_ast_comparison_imports_no_legacy` | 1 | - | pass |
| C15 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_protocol.py' -k test_wrong_run_count_is_rejected_nami` | 1 | - | pass |
| C16 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_protocol.py' -k test_missing_dependencies_block_is_re` | 1 | - | pass |
| C17 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_protocol.py' -k test_declared_early_stop_is_rejected_` | 1 | - | pass |
| C17 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_protocol.py' -k test_unknown_ga_parameter_is_rejected` | 1 | - | pass |
| C18 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_replay.py' -k test_manifest_drift_stops_before_the_dr` | 1 | - | pass |
| C19 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_replay.py' -k test_missing_dependencies_are_named_and` | 1 | - | pass |
| C20 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_replay.py' -k test_uncovered_dataset_is_reported_unav` | 1 | - | pass |
| C21 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_replay.py' -k test_matching_runs_report_reproduced` | 1 | - | pass |
| C22 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_replay.py' -k test_one_differing_field_reports_mismat` | 1 | - | pass |
| C23 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_replay.py' -k test_report_carries_the_protocol_values` | 1 | - | pass |
| C23 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_replay.py' -k test_missing_protocol_value_is_reported` | 1 | - | pass |
| C24 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_replay.py' -k test_no_reference_ticker_is_hardcoded` | 1 | - | pass |
| C25 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_replay.py' -k test_reference_tree_is_untouched_and_no` | 1 | - | pass |
| C26 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_replay.py' -k test_nonzero_driver_exit_reports_runtim` | 1 | - | pass |
| C27 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_replay.py' -k test_budget_exceeded_terminates_the_exe` | 1 | - | pass |
| C28 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_replay.py' -k test_bundle_moved_during_the_run_exits_` | 1 | - | pass |
| C29 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_replay.py' -k test_substitute_driver_is_resolved_and_` | 1 | - | pass |
| C29 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_replay.py' -k test_child_without_runs_json_reports_ru` | 1 | - | pass |
| C29 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_replay.py' -k test_archived_driver_compiles_and_names` | 1 | - | pass |
| C30 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_replay.py' -k test_reported_and_recomputed_fitness_st` | 1 | - | pass |
| C31 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_isolation.py' -k test_verifier_defaults_to_the_reposi` | 1 | - | pass |
| C32 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_isolation.py' -k test_replay_flags_and_defaults` | 1 | - | pass |
| C33 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_isolation.py' -k test_exit_codes_and_single_json_obje` | 1 | - | pass |
| C34 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_isolation.py' -k test_no_production_file_reaches_the_` | 1 | - | pass |
| C34 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_isolation.py' -k test_isolation_scan_flags_an_injecte` | 1 | - | pass |
| C35 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_isolation.py' -k test_no_package_marker_under_baselin` | 1 | - | pass |
| C36 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_manifest.py' -k test_structurally_broken_manifest_is_` | 1 | - | pass |
| C36 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_manifest.py' -k test_manifest_that_is_not_a_json_obje` | 1 | - | pass |
| C37 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_manifest.py' -k test_mode_change_is_reported_by_path_` | 1 | - | pass |
| C38 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_isolation.py' -k test_isolation_scan_flags_a_repo_roo` | 1 | - | pass |
| C39 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_protocol.py' -k test_break_in_the_generation_loop_is_` | 1 | - | pass |
| C40 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_replay.py' -k test_reference_without_run_id_column_is` | 1 | - | pass |
| C40 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_replay.py' -k test_reference_with_a_non_numeric_run_i` | 1 | - | pass |
| C41 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_replay.py' -k test_report_inside_the_frozen_tree_is_r` | 1 | - | pass |
| C42 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_protocol.py' -k test_declared_weights_reproduce_the_c` | 1 | - | pass |

## Mutations

| Mutation | Location | Killed |
| --- | --- | --- |
| one recorded sha256 digit moved | `baselines/v0-legacy/baseline.json:81` | yes |
| protocol declares 29 runs | `baselines/v0-legacy/protocol.json:23` | yes |
| protocol declares seed base 43 | `baselines/v0-legacy/protocol.json:26` | yes |
| archived caio generation count moved | `baselines/v0-legacy/code/ga.py:33` | yes |
| reference source_path prefix loses its s | `baselines/v0-legacy/tools/build_manifest.py:30` | yes |
| drift exits zero | `baselines/v0-legacy/tools/verify_manifest.py:167` | yes |
| dependency probe skipped | `baselines/v0-legacy/tools/replay.py:303` | yes |
| stderr_tail bound widened to 8192 | `baselines/v0-legacy/tools/replay.py:57` | yes |
| report claims a golden output was written | `baselines/v0-legacy/tools/replay.py:261` | yes |
| a bundle that moved mid-run exits 1 | `baselines/v0-legacy/tools/replay.py:55` | yes |
| verifier: mode re-check dropped from verify_bundle | `baselines/v0-legacy/tools/verify_manifest.py:149` | yes |
| verifier: duplicate-entry guard removed from load_manifest | `baselines/v0-legacy/tools/verify_manifest.py:102` | yes |
| verifier: replay swallows differing fields, forces REPRODUCED | `baselines/v0-legacy/tools/replay.py:462` | yes |
| verifier: recorded/recomputed swapped in fields entries | `baselines/v0-legacy/tools/replay.py:428` | yes |
| verifier: isolation scan drops the repo-root *.py arm | `tests/test_v0_legacy_isolation.py:44` | yes |
| verifier: protocol checker drops the ast early-stop (break) arm | `tests/test_v0_legacy_protocol.py:222` | yes |
| judge r1 F1: reference reader stops naming the missing run_id column | `baselines/v0-legacy/tools/replay.py:204` | yes |
| judge r1 F3: report path inside a covered subtree is allowed again | `baselines/v0-legacy/tools/replay.py:557` | yes |
