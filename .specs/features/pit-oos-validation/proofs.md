# pit-oos-validation proofs

**HEAD**: 088e9ae
**Tree**: clean
**Result**: 50 pass, 0 fail; mutations 11 killed, 0 not

| Check | Proof | Ran | Passed | Result |
| --- | --- | --- | --- | --- |
| C1 | `python3 -m unittest tests.test_pit_validation -k ProtocolFileTests.test_committed_protocol_values` | 1 | - | pass |
| C2 | `python3 -m unittest tests.test_pit_validation -k ReadmeTests.test_readme_states_protocol` | 1 | - | pass |
| C3 | `python3 -m unittest tests.test_pit_validation -k ProtocolValidationTests.test_invalid_protocol_exits_2` | 1 | - | pass |
| C3 | `python3 -m unittest tests.test_pit_validation -k ProtocolValidationTests.test_zero_cost_is_accepted` | 1 | - | pass |
| C4 | `python3 -m unittest tests.test_pit_validation -k CliPathTests.test_missing_paths_exit_3` | 1 | - | pass |
| C46 | `python3 -m unittest tests.test_pit_validation -k ProtocolValidationTests.test_series_must_be_loader_files` | 1 | - | pass |
| C5 | `python3 -m unittest tests.test_pit_validation -k ScheduleTests.test_weekday_fixture_offsets` | 1 | - | pass |
| C6 | `python3 -m unittest tests.test_pit_validation -k ScheduleTests.test_real_snapshot_schedule` | 1 | - | pass |
| C7 | `python3 -m unittest tests.test_pit_validation -k ScheduleTests.test_period_end_and_union` | 1 | - | pass |
| C8 | `python3 -m unittest tests.test_pit_validation -k IntegrityTests.test_overlap_gap_dropped_exit_4` | 1 | - | pass |
| C9 | `python3 -m unittest tests.test_pit_validation -k VerdictTests.test_insufficient_history_examples` | 1 | - | pass |
| C10 | `python3 -m unittest tests.test_pit_validation -k LeakTests.test_spy_sees_rows_up_to_decision` | 1 | - | pass |
| C11 | `python3 -m unittest tests.test_pit_validation -k LeakTests.test_future_perturbation_is_inert` | 1 | - | pass |
| C12 | `python3 -m unittest tests.test_pit_validation -k LeakTests.test_leaky_slice_raises_and_exits_4` | 1 | - | pass |
| C13 | `python3 -m unittest tests.test_pit_validation -k CutoffTests.test_cutoff_mismatch_exits_4` | 1 | - | pass |
| C14 | `python3 -m unittest tests.test_pit_validation -k TrimTests.test_run_uses_verified_range_only` | 1 | - | pass |
| C15 | `python3 -m unittest tests.test_pit_validation -k TrimTests.test_real_run_period` | 1 | - | pass |
| C41 | `python3 -m unittest tests.test_pit_validation -k TrimTests.test_run_stops_at_verified_end` | 1 | - | pass |
| C16 | `python3 -m unittest tests.test_pit_validation -k TrimTests.test_committed_report_run_period` | 1 | - | pass |
| C17 | `python3 -m unittest tests.test_pit_validation -k SimulationTests.test_execution_lag` | 1 | - | pass |
| C18 | `python3 -m unittest tests.test_pit_validation -k SimulationTests.test_cost_factor_examples` | 1 | - | pass |
| C19 | `python3 -m unittest tests.test_pit_validation -k SimulationTests.test_benchmarks` | 1 | - | pass |
| C20 | `python3 -m unittest tests.test_pit_validation -k DistributionTests.test_quantiles_example` | 1 | - | pass |
| C21 | `python3 -m unittest tests.test_pit_validation -k DistributionTests.test_share_positive` | 1 | - | pass |
| C22 | `python3 -m unittest tests.test_pit_validation -k ReportTests.test_paths_allow_recompute` | 1 | - | pass |
| C23 | `python3 -m unittest tests.test_pit_validation -k RealDataTests.test_years_used_real` | 1 | - | pass |
| C24 | `python3 -m unittest tests.test_pit_validation -k MarkdownTests.test_markdown_statements` | 1 | - | pass |
| C25 | `python3 -m unittest tests.test_pit_validation -k SensitivityTests.test_sensitivity_rows` | 1 | - | pass |
| C26 | `python3 -m unittest tests.test_pit_validation -k SensitivityTests.test_invalid_sensitivity_exits_2` | 1 | - | pass |
| C27 | `python3 -m unittest tests.test_pit_validation -k AuditTests.test_rule_examples` | 1 | - | pass |
| C43 | `python3 -m unittest tests.test_pit_validation -k AuditTests.test_non_finite_level_is_revision` | 1 | - | pass |
| C45 | `python3 -m unittest tests.test_pit_validation -k AuditTests.test_infinite_or_subnormal_level_is_revision` | 1 | - | pass |
| C28 | `python3 -m unittest tests.test_pit_validation -k AuditTests.test_real_audit` | 1 | - | pass |
| C29 | `python3 -m unittest tests.test_pit_validation -k AuditTests.test_audit_record_fields` | 1 | - | pass |
| C30 | `python3 -m unittest tests.test_pit_validation -k AuditTests.test_selection_layers_recorded` | 1 | - | pass |
| C31 | `python3 -m unittest tests.test_pit_validation -k VerdictTests.test_verdict_rule` | 1 | - | pass |
| C32 | `python3 -m unittest tests.test_pit_validation -k VerdictTests.test_selection_overlap_examples` | 1 | - | pass |
| C33 | `python3 -m unittest tests.test_pit_validation -k ReportTests.test_unverifiable_outputs` | 1 | - | pass |
| C34 | `python3 -m unittest tests.test_pit_validation -k IntegrityTests.test_audit_hash_mismatch_exits_4` | 1 | - | pass |
| C47 | `python3 -m unittest tests.test_pit_validation -k IntegrityTests.test_paired_retrieval_revised_exits_4` | 1 | - | pass |
| C35 | `d=$(mktemp -d) && python3 py/pit_validation.py audit --protocol docs/validation/protocol.json --out "$d/pit-au` | count unknown | - | pass |
| C35 | `python3 -m unittest tests.test_pit_validation -k ReportTests.test_committed_environment_matches` | 1 | - | pass |
| C36 | `python3 -m unittest tests.test_pit_validation -k CliTests.test_failed_write_leaves_previous_outputs` | 1 | - | pass |
| C42 | `python3 -m unittest tests.test_pit_validation -k IntegrityTests.test_audit_protocol_mismatch_exits_4` | 1 | - | pass |
| C44 | `python3 -m unittest tests.test_pit_validation -k IntegrityTests.test_audit_layer_mismatch_exits_4` | 1 | - | pass |
| C37 | `python3 -m unittest tests.test_pit_validation -k NoticeTests.test_notices_match_report` | 1 | - | pass |
| C38 | `d=$(mktemp) && git diff 4a2481a..HEAD -- docs/adr docs/research web api landing.html > "$d" && ! grep -E '^\+[` | count unknown | - | pass |
| C39 | `git diff --quiet 4a2481a HEAD -- data/allocation` | count unknown | - | pass |
| C40 | `python3 -m unittest tests.test_pit_validation -k DataCopyTests.test_retrieval_copy_hashes` | 1 | - | pass |
| Build | `python3 -m py_compile py/pit_validation.py tests/test_pit_validation.py` | - | - | pass |

## Mutations

| Mutation | Location | Killed |
| --- | --- | --- |
| C10-C12 leak guard tolerates a row up to 3 days after t (weekend leak) | `py/pit_validation.py:693` | yes |
| C17 exec_j row earned by the new weights (same-close execution) | `py/pit_validation.py:750` | yes |
| C9/C31 verdict counts offsets without a window as paths | `py/pit_validation.py:1023` | yes |
| C14 trim drops the run_period.end bound (unverified tail rows kept) | `py/pit_validation.py:543` | yes |
| C14/C15 run_period.end is the latest verified end, not the intersection | `py/pit_validation.py:532` | yes |
| C42 report ignores a series_kind change since the audit (V-2 round 2) | `py/pit_validation.py:908` | yes |
| C42 report compares only the snapshot retrieval, not the paired ones (V-2 round 2) | `py/pit_validation.py:910` | yes |
| V-2 report ignores a selection-layer missing-list change since the audit (round 2) | `py/pit_validation.py:915` | yes |
| C44 report ignores an excluded_lookahead change since the audit (round 3) | `py/pit_validation.py:918` | yes |
| C45 a zero or non-finite factor is skipped, not read as a revision (round 3) | `py/pit_validation.py:302` | yes |
| C45 zero factor (subnormal level) reaches the division again (round 3) | `py/pit_validation.py:302` | yes |
