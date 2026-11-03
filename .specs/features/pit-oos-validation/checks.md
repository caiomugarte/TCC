# Point-in-time out-of-sample validation - checks

Mutations: 4
Build: `python3 -m py_compile py/pit_validation.py tests/test_pit_validation.py`
Plan: `.specs/features/pit-oos-validation/plan.md`

47 checks in 8 slices · 4 one-way doors · 0 open

Test module: `tests/test_pit_validation.py` (stdlib `unittest`, run from the repo root). The four
mutations aim at the per-decision slice (C10-C12), the execution lag (C17), the verdict rule (C31)
and the trim to verified dates (C14).

## Checks

### S1 - The protocol is written and its declarations are enforced

**C1** - `docs/validation/protocol.json` holds exactly the door-1 shape and values: `series_kind` with `btc_usd` `spot_close`, `execution_lag_common_dates` 1, `min_lookback_years` 3, `max_lookback_years` 10, `test_months` 12, `offset_months` 0..11, the three profiles, the three benchmarks, `cost_bps` 10, `sensitivity_bps` `[0, 10, 50]`, `basis` `one_way_on_traded_value_all_classes`, `taxes` `excluded`, `adopted_on` `2026-11-01` (AC 1)
Proof: `python3 -m unittest tests.test_pit_validation -k ProtocolFileTests.test_committed_protocol_values`

**C2** - `docs/validation/README.md` states the data rule, execution rule, rebalance rule, 12 offsets, lookback from 3 years, path period, three benchmarks, cost basis, every series' `series_kind` with its source, the PIT thresholds and trim, the evidence sentence "não revisão observada entre duas captações (`days_apart` dias); nenhum vintage as-of", the verdict rule and both CLI commands (AC 2)
Proof: `python3 -m unittest tests.test_pit_validation -k ReadmeTests.test_readme_states_protocol`

**C3** - `report` exits 2, prints one line `pit_validation: <field>: <reason>` naming the field, and leaves `--out-dir` byte-identical for each invalid protocol: no `ibov` (stderr has `ibov`), `sp500` added (`sp500`), `cost` removed (`cost_bps`), `cost_bps` `"10"`, `cost_bps` -1, `cost_bps` NaN, `basis` removed (`basis`), `taxes` `""`, `series_kind.btc_usd` `"close"` (`btc_usd`); `cost_bps` 0 is accepted (AC 3)
Proof: `python3 -m unittest tests.test_pit_validation -k ProtocolValidationTests.test_invalid_protocol_exits_2`
Proof: `python3 -m unittest tests.test_pit_validation -k ProtocolValidationTests.test_zero_cost_is_accepted`

**C4** - `audit` and `report` exit 3 naming the path and write nothing when `--protocol`, `--audit`, `snapshot_dir` or one retrieval does not exist (AC 4)
Proof: `python3 -m unittest tests.test_pit_validation -k CliPathTests.test_missing_paths_exit_3`

**C46** - A protocol whose `series.ibovespa` names `ibovespa_copy.csv` (a byte-identical copy present in every retrieval) instead of `ibovespa.csv`, the file `load_snapshot_bundle` reads, makes `audit` and `report` each exit 2 with one line naming `series.ibovespa`, writing nothing and leaving `--out-dir` byte-identical (AC 3, AC 25; review F2)
Proof: `python3 -m unittest tests.test_pit_validation -k ProtocolValidationTests.test_series_must_be_loader_files`

### S2 - The walk-forward covers its declared period without overlap

**C5** - On the weekday fixture 2016-07-21..2026-07-21, offset 0's first decision is 2019-07-19, exec 2019-07-22, 6 windows from `(2019-07-22, 2020-07-22]` to `(2024-07-22, 2025-07-22]`, no window 7; offset 11's first decision is 2020-06-19 with 6 windows, last `(2025-06-23, 2026-06-22]` (AC 5)
Proof: `python3 -m unittest tests.test_pit_validation -k ScheduleTests.test_weekday_fixture_offsets`

**C6** - On `data/allocation` trimmed to `run_period`, offset 0's first decision is 2019-10-31 with exec 2019-11-01 and 6 windows; offset 4's anchor is 2020-02-29, decision 2020-02-28, exec 2020-03-02, and its j=5 decision 2025-02-28 executes on 2025-03-05; offset 9 has 5 windows, last `(2024-08-01, 2025-08-01]` (AC 5)
Proof: `python3 -m unittest tests.test_pit_validation -k ScheduleTests.test_real_snapshot_schedule`

**C7** - Every path's windows are pairwise disjoint, contiguous and their union is the common rows in `(period_start, period_end]`; on `data/allocation` offset 8's `period_end` is 2026-07-01 and offset 9's is 2025-08-01 (AC 6)
Proof: `python3 -m unittest tests.test_pit_validation -k ScheduleTests.test_period_end_and_union`

**C8** - A schedule whose window 2 starts one row early, one that skips a row between windows, and one that drops offset 8's last window each make the run raise the integrity error and `report` exit 4 naming the offset, with no output written (AC 6)
Proof: `python3 -m unittest tests.test_pit_validation -k IntegrityTests.test_overlap_gap_dropped_exit_4`

**C9** - Weekday fixture to 2020-08-31 gives offsets 0 and 1 one window each (exec_1 2020-07-22 and 2020-08-24), 2 paths, not `insufficient_history`; ending 2020-07-31 gives 1 path and ending 2020-07-21 gives 0, both `UNVERIFIABLE` with `insufficient_history` (AC 7)
Proof: `python3 -m unittest tests.test_pit_validation -k VerdictTests.test_insufficient_history_examples`

### S3 - No date after the cutoff, and no unverified date, enters a decision

**C10** - A spy strategy records `max(row.date)` equal to the decision date on every call, for all 6 decisions x 12 offsets x 3 profiles on the weekday fixture (AC 8)
Proof: `python3 -m unittest tests.test_pit_validation -k LeakTests.test_spy_sees_rows_up_to_decision`

**C11** - Replacing every return after `t` with +50% per class leaves the real strategy's weights at `t` bit-identical, and changing the return dated `t` so the risky share moves changes them (AC 9)
Proof: `python3 -m unittest tests.test_pit_validation -k LeakTests.test_future_perturbation_is_inert`

**C12** - A slice swapped for `rows <= exec_date` raises `PITLeakError` naming the decision and the offending date and makes `report` exit 4 with no output; a slice swapped for `rows < t` is rejected because its last row is not `t` (AC 8, AC 10)
Proof: `python3 -m unittest tests.test_pit_validation -k LeakTests.test_leaky_slice_raises_and_exits_4`

**C13** - `cutoff_date` equal to the last common date runs; 2026-07-25 and 2026-07-18 against rows to 2026-07-21 each exit 4 naming both dates (AC 11)
Proof: `python3 -m unittest tests.test_pit_validation -k CutoffTests.test_cutoff_mismatch_exits_4`

**C14** - With a paired retrieval starting on 2016-08-01, `run_period.start` is 2016-08-01, no strategy call, path or benchmark sees an earlier row, the first row carries 0.0 in every class, and `unverified_common_rows` is 7 (AC 12)
Proof: `python3 -m unittest tests.test_pit_validation -k TrimTests.test_run_uses_verified_range_only`

**C15** - On `data/allocation` with the committed audit, `run_period` is 2016-10-31..2026-07-21 and `unverified_common_rows` 69 (AC 12)
Proof: `python3 -m unittest tests.test_pit_validation -k TrimTests.test_real_run_period`

**C41** - On the weekday fixture to 2022-07-21 whose paired `btc_usd` retrieval ends on 2022-06-30 (the other five end later), `run_period.end` is 2022-06-30, `unverified_common_rows` is 15, and no strategy call, path or benchmark sees a row dated after 2022-06-30 (AC 12; Verifier V-1)
Proof: `python3 -m unittest tests.test_pit_validation -k TrimTests.test_run_stops_at_verified_end`

**C16** - The committed `class-allocation-oos.json` records `run_period` 2016-10-31..2026-07-21, and every path's first window starts after `run_period.start` (AC 12)
Proof: `python3 -m unittest tests.test_pit_validation -k TrimTests.test_committed_report_run_period`

**C17** - Decision Friday, exec Monday, previous weights 100% `fixed_income` (0% Monday) and new weights 100% `crypto` (+10% Monday, +5% Tuesday): the path moves 0% on Monday before the execution cost and +5% on Tuesday; at j = 0 the value starts at 1.0 on exec_0 and that row earns nothing (AC 13)
Proof: `python3 -m unittest tests.test_pit_validation -k SimulationTests.test_execution_lag`

### S4 - Costs and benchmarks under the same rules

**C18** - The execution factor is `1 - cost_bps/10000 * turnover`: first execution at 10 bps gives 0.999, held 50/50 to 50/50 gives 1.0, held 60/40 to 50/50 gives 0.9998, and 0 bps always gives 1.0 (AC 14)
Proof: `python3 -m unittest tests.test_pit_validation -k SimulationTests.test_cost_factor_examples`

**C19** - `cdi` is 100% `fixed_income` charged only at exec_0, `ibov` is 100% `brazilian_stocks`, `naive_equal_weight` is 0.2 per class with turnover 0.2667 after a year where crypto doubled, each on the strategy's schedule (AC 15)
Proof: `python3 -m unittest tests.test_pit_validation -k SimulationTests.test_benchmarks`

### S5 - A distribution, not a single run

**C20** - 12 paths with annualized returns 0.01..0.12 give `min` 0.01, `p25` 0.0375, `median` 0.065, `p75` 0.0925, `max` 0.12, `paths` 12 (AC 16)
Proof: `python3 -m unittest tests.test_pit_validation -k DistributionTests.test_quantiles_example`

**C21** - Excesses -0.01, 0.0, 0.02, 0.03 give `share_positive` 0.5 (AC 17)
Proof: `python3 -m unittest tests.test_pit_validation -k DistributionTests.test_share_positive`

**C22** - A `VALIDATED` fixture report lists per path the offset, one decision date per window, one execution per decision plus `period_end`, per-window weights, `years_used`, `full_history` and per-path metrics, and every distribution and excess figure recomputes from those paths for `strategy`, `cdi`, `ibov` and `naive_equal_weight` (AC 16, AC 17, AC 18)
Proof: `python3 -m unittest tests.test_pit_validation -k ReportTests.test_paths_allow_recompute`

**C23** - On `data/allocation` the real strategy gives `years_used` 2.9979 at 2019-10-31 and 8.6626 at 2025-06-30, both `full_history` true, and no decision in the committed report has `full_history` false (AC 18)
Proof: `python3 -m unittest tests.test_pit_validation -k RealDataTests.test_years_used_real`

**C24** - The Markdown report has per-profile tables with columns `min`, `p25`, `mediana`, `p75`, `max`, `caminhos` and states the method-choice date, the `selection_overlap` value, the precedence sentence only when `full`, the offsets-overlap sentence whatever the value, the 10-year-window sentence with the `full_history` definition and ADR 0003's rejection of 3-year windows, the evidence sentence with each series' `days_apart`, the IBOV label and both conflicting declarations, and the cost and tax statement (AC 19)
Proof: `python3 -m unittest tests.test_pit_validation -k MarkdownTests.test_markdown_statements`

**C25** - A `VALIDATED` report gives per profile a `cost_sensitivity` row for each of 0, 10 and 50 bps over the same paths with no new strategy call; the 10 bps row equals the headline `annualized_return` statistics; per path the strategy's annualized return is strictly decreasing from 0 to 10 to 50 bps; with `sensitivity_bps` absent there is no block (AC 29)
Proof: `python3 -m unittest tests.test_pit_validation -k SensitivityTests.test_sensitivity_rows`

**C26** - `sensitivity_bps` `[0, "x"]`, `[]` and `[-1]` each make `report` exit 2 with `sensitivity_bps` on stderr and no file written (AC 29)
Proof: `python3 -m unittest tests.test_pit_validation -k SensitivityTests.test_invalid_sensitivity_exits_2`

### S6 - PIT availability is audited and recorded

**C27** - The door-3 rule on synthetic retrievals: identical values over 300 dates -> `PIT_QUALIFIED` `non_revision`; B day 0, A (snapshot) day 20, C day 40 -> pairs A-B and A-C only, `no_retrieval_pair_30d`; one value off by 1e-6 -> `values_revised`; `di` rebased at another start -> `PIT_QUALIFIED`; `^BVSP` vs `^IBOV` -> `provider_mismatch`; 249 shared -> `overlap_below_250`; one interior date missing -> `overlap_incomplete`; retrieval without the file -> `series_absent`; `vendor_composite_close` and `vendor_adjusted_close` -> `kind_revisable`; two qualifying pairs -> the one with more shared dates wins (AC 20, AC 12)
Proof: `python3 -m unittest tests.test_pit_validation -k AuditTests.test_rule_examples`

**C43** - The door-3 rule on a NaN level: the paired retrieval (or the snapshot) identical except one interior level `nan` -> `PIT_MISSING` `values_revised`, with `max_rel_diff_daily_factor` null (AC 20; Verifier V-3)
Proof: `python3 -m unittest tests.test_pit_validation -k AuditTests.test_non_finite_level_is_revision`

**C45** - The door-3 rule on an `inf` or subnormal (`5e-324`) level: the paired retrieval (or the snapshot) identical except one interior level `inf` (resp. `5e-324`) -> `PIT_MISSING` `values_revised`, with `max_rel_diff_daily_factor` null, and no exception (AC 20; Verifier V-7)
Proof: `python3 -m unittest tests.test_pit_validation -k AuditTests.test_infinite_or_subnormal_level_is_revision`

**C28** - The real audit groups the three repo copies of `ibovespa.csv` with `retrieved_at` 2026-07-24 and each member's `same_bytes_as` listing the other two; the pair with `data/allocation_retrieval_2026-10-29` has `days_apart` 97 and 2413 shared dates; all six series are `PIT_QUALIFIED` with `unverified_dates` 70 from 2016-07-21 to 2016-10-28 (S&P TR 71, BTC 100) (AC 20)
Proof: `python3 -m unittest tests.test_pit_validation -k AuditTests.test_real_audit`

**C29** - Every series record in `pit-audit.json` carries per retrieval `provider`, identifier, `retrieved_at`, `sha256`, `same_bytes_as`, first and last date and row count; per pair both `retrieved_at`, `days_apart`, `shared_dates` and `max_rel_diff_daily_factor`; and `status`, `reasons`, `evidence`, `days_apart`, verified range and `unverified_dates` (AC 20)
Proof: `python3 -m unittest tests.test_pit_validation -k AuditTests.test_audit_record_fields`

**C30** - `pit-audit.json` records `stock_selection` and `fii_selection` as `PIT_MISSING` with the protocol's missing items, and `caio_stocks`/`caio_fiis` as `excluded_lookahead` (AC 21)
Proof: `python3 -m unittest tests.test_pit_validation -k AuditTests.test_selection_layers_recorded`

### S7 - The verdict is computed, never inferred

**C31** - `ifix` `PIT_MISSING` (`no_retrieval_pair_30d`) gives `class_allocation` and `basic_recommendation` `UNVERIFIABLE` with `ifix: no_retrieval_pair_30d` in missing; all six qualified with 2+ paths gives `VALIDATED`; `stock_selection`, `fii_selection` and `premium_recommendation` are always `UNVERIFIABLE`; `evidence.class_allocation` is `non_revision` (AC 22)
Proof: `python3 -m unittest tests.test_pit_validation -k VerdictTests.test_verdict_rule`

**C32** - `selection_overlap.class_allocation`: `adopted_on` 2026-11-01 with end 2026-07-21 -> `full`; 2026-01-01 -> `partial`; 2019-01-01 with earliest exec_0 2019-11-01 -> `none`; a `run_period` under 3 years ending on or after `adopted_on` -> `none` (AC 23)
Proof: `python3 -m unittest tests.test_pit_validation -k VerdictTests.test_selection_overlap_examples`

**C33** - An `UNVERIFIABLE` run exits 0, its JSON has no `profiles` key, its Markdown holds the title, protocol id, both sha256, `run_period`, verdict, evidence and overlap per layer, missing items, the per-series audit table, thresholds `30`, `250` and `1e-9`, the criterion-19 statements and the regeneration commands, and no line matches `[0-9]+[.,][0-9]+|%` (AC 24)
Proof: `python3 -m unittest tests.test_pit_validation -k ReportTests.test_unverifiable_outputs`

**C34** - An audit whose `snapshot_dir` sha256 for a series differs from the current file makes `report` exit 4 naming the series (AC 25)
Proof: `python3 -m unittest tests.test_pit_validation -k IntegrityTests.test_audit_hash_mismatch_exits_4`

**C47** - After `audit`, revising one shared interior level of the paired (non-snapshot) retrieval's `ifix.csv` makes `report` exit 4 naming `audit.ifix`, and changing the paired retrieval's `metadata.json` `retrieved_at` makes it exit 4 naming `audit.`, each with no output written; the paired retrieval restored exits 0 (AC 25; review F3)
Proof: `python3 -m unittest tests.test_pit_validation -k IntegrityTests.test_paired_retrieval_revised_exits_4`

**C35** - Running `audit` then `report` at `HEAD` on the committed protocol reproduces `pit-audit.json`, `class-allocation-oos.json` and `class-allocation-oos.md` byte for byte, on the interpreter the JSON records (AC 26)
Proof: `d=$(mktemp -d) && python3 py/pit_validation.py audit --protocol docs/validation/protocol.json --out "$d/pit-audit.json" && cmp "$d/pit-audit.json" docs/validation/pit-audit.json && python3 py/pit_validation.py report --protocol docs/validation/protocol.json --audit docs/validation/pit-audit.json --out-dir "$d" && cmp "$d/class-allocation-oos.json" docs/validation/class-allocation-oos.json && cmp "$d/class-allocation-oos.md" docs/validation/class-allocation-oos.md`
Proof: `python3 -m unittest tests.test_pit_validation -k ReportTests.test_committed_environment_matches`

**C36** - A `report` that fails while writing its second output leaves both previous files in `--out-dir` byte-identical and no temporary file behind; `audit` does the same for its one file (Surface)
Proof: `python3 -m unittest tests.test_pit_validation -k CliTests.test_failed_write_leaves_previous_outputs`

**C42** - After `audit`, changing the protocol's `series_kind.btc_usd` to `vendor_composite_close`, or cutting `retrievals` to the snapshot alone, makes `report` exit 4 naming `series_kind.btc_usd` (resp. `retrievals`) with no output written; the audited protocol restored exits 0 (AC 22, AC 25; Verifier V-2)
Proof: `python3 -m unittest tests.test_pit_validation -k IntegrityTests.test_audit_protocol_mismatch_exits_4`

**C44** - After `audit`, adding one item to the protocol's `layers.stock_selection.missing`, changing `layers.stock_selection.excluded_lookahead.caio_stocks`, or removing `layers.fii_selection.excluded_lookahead` makes `report` exit 4 naming that field (`layers.<layer>.missing`, resp. `layers.<layer>.excluded_lookahead`) with no output written; the audited protocol restored exits 0 (AC 22, AC 25; Verifier V-8)
Proof: `python3 -m unittest tests.test_pit_validation -k IntegrityTests.test_audit_layer_mismatch_exits_4`

### S8 - Historical numbers already published are marked

**C37** - ADR 0003, `docs/research/class-allocation-methods-2026-11.md` and `docs/research/class-allocation-2026-11/README.md` each carry, right below the title, the notice line whose verdict and overlap equal the committed JSON's, with the right relative link (AC 27)
Proof: `python3 -m unittest tests.test_pit_validation -k NoticeTests.test_notices_match_report`

**C38** - The change adds no line matching `[0-9]+[,.][0-9]+ ?%` under `docs/adr`, `docs/research`, `web`, `api` or `landing.html` (AC 28)
Proof: `d=$(mktemp) && git diff 4a2481a..HEAD -- docs/adr docs/research web api landing.html > "$d" && ! grep -E '^\+[^+].*[0-9]+[,.][0-9]+ ?%' "$d"`

**C39** - `data/allocation/` is byte-identical to the base commit (Out of scope: its `metadata.json` is not rewritten) (Changes underneath)
Proof: `git diff --quiet 4a2481a HEAD -- data/allocation`

**C40** - `data/allocation_retrieval_2026-10-29/` holds exactly the six series and `metadata.json`, each with the SHA-256 of its `snapshots/data/snap-2026-10-29/` source recorded at copy time, and no `caio_*` file (door 2)
Proof: `python3 -m unittest tests.test_pit_validation -k DataCopyTests.test_retrieval_copy_hashes`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| CLI exit codes (4) | 0 C33 · 2 C3 · 3 C4 · 4 C8 | - |
| exit-4 integrity violations (9) | leak C12 · overlap C8 · gap C8 · dropped-window C8 · cutoff-late C13 · cutoff-early C13 · hash-mismatch C34 · audit-protocol-mismatch C42, C44 · paired-retrieval-revised C47 | - |
| invalid protocol examples, AC 3 (11) | no-ibov C3 · sp500-added C3 · cost-removed C3 · cost-string C3 · cost-negative C3 · cost-nan C3 · basis-removed C3 · taxes-empty C3 · kind-unknown C3 · cost-zero-accepted C3 · series-file-not-loaders C46 | - |
| missing paths, AC 4 (4) | `--protocol` C4 · `--audit` C4 · `snapshot_dir` C4 · retrieval C4 | - |
| `series_kind` vocabulary (7) | `exchange_index_close` C27 · `official_rate` C27 · `official_fixing` C28 · `spot_close` C28 · `vendor_composite_close` C27 · `vendor_adjusted_close` C27 · unknown C3 | - |
| `PIT_MISSING` reasons (7) | `series_absent` C27 · `kind_revisable` C27 · `provider_mismatch` C27 · `no_retrieval_pair_30d` C27 · `overlap_below_250` C27 · `overlap_incomplete` C27 · `values_revised` C27, C43, C45 | - |
| schedule examples, AC 5 (6) | fixture-offset-0 C5 · fixture-offset-11 C5 · real-offset-0 C6 · real-offset-4-clamp C6 · real-offset-4-j5 C6 · real-offset-9 C6 | - |
| path ends, AC 6 (2) | offset-8 C7 · offset-9 C7 | - |
| `insufficient_history` examples, AC 7 (3) | to-2020-08-31 C9 · to-2020-07-31 C9 · to-2020-07-21 C9 | - |
| decision slice, AC 8-10 (4) | spy-max-date C10 · future-perturbation C11 · exec-leak C12 · one-row-short C12 | - |
| verdict layers (5) | `class_allocation` C31 · `basic_recommendation` C31 · `stock_selection` C31 · `fii_selection` C31 · `premium_recommendation` C31 | - |
| `selection_overlap` examples (4) | full C32 · partial C32 · none-after C32 · none-no-exec C32 | - |
| benchmarks (3) | `cdi` C19 · `ibov` C19 · `naive_equal_weight` C19 | - |
| cost examples, AC 14 (4) | first-exec C18 · no-trade C18 · 60/40 C18 · zero-bps C18 | - |
| distribution series (4) | `strategy` C22 · `cdi` C22 · `ibov` C22 · `naive_equal_weight` C22 | - |
| distribution metrics (3) | `annualized_return` C22 · `annualized_volatility` C22 · `max_drawdown` C22 | - |
| profiles (3) | `conservador` C10 · `moderado` C10 · `arrojado` C10 | - |
| sensitivity bps (3) | 0 C25 · 10 C25 · 50 C25 | - |
| invalid `sensitivity_bps` (3) | non-number C26 · empty C26 · negative C26 | - |
| generated outputs (3) | `pit-audit.json` C35 · `class-allocation-oos.json` C35 · `class-allocation-oos.md` C35 | - |
| notice documents (3) | ADR 0003 C37 · research doc C37 · research README C37 | - |
| retrieval copy files (7) | `ibovespa.csv` C40 · `ifix.csv` C40 · `sp500_total_return_usd.csv` C40 · `di.csv` C40 · `btc_usd.csv` C40 · `ptax.csv` C40 · `metadata.json` C40 | - |
| Markdown statements, AC 19 (8) | tables C24 · method-date C24 · overlap-value C24 · offsets-overlap C24 · window-and-full-history C24 · evidence C24 · ibov-label C24 · costs C24 | - |

- Claims about the CLI's exit codes (C3, C4, C8, C12, C13, C26, C33, C34) are proven by calling `main()` with real files in a temporary directory, not by calling the inner function only
- C35 runs the committed CLI end to end; it takes about 3 minutes

## Swept

- validation: C3, C26, C46
- failure modes: C36
- idempotency: C35
- authorization: n/a - a local operator CLI over committed files; no user, network or credential
- concurrency: n/a - one offline process; the atomic rename (C36) keeps a reader from seeing a half-written output
- data lifecycle: C39, C40
- dependency failure: C4
- state transitions: n/a - no stateful entity; the verdict is recomputed from the audit on every run (C31)
- observability: C3

## Handoff

- **Boundary:** C1-C40 closed at `681def9` (code and outputs at `d0bdcd1`, notices at `681def9`, data copy at `95dbbf3`). Real run: `class_allocation` `VALIDATED`, `basic_recommendation` `VALIDATED`, `stock_selection`/`fii_selection`/`premium_recommendation` `UNVERIFIABLE`, `selection_overlap.class_allocation` `full`, `run_period` 2016-10-31..2026-07-21 (69 unverified common rows), 12 paths x 3 profiles, 207 decisions; `report` takes about 55 s on Python 3.14.4
- **Settled mid-build:** none asked. Decided: the decision guard also rejects a slice whose last row is not `t` (`IntegrityError`, exit 4), which is how C12's one-row-short case fails; `run_period` is `null` and `selection_overlap` `none` when a series has no verified range; an I/O failure while staging outputs exits 1 with one `pit_validation: output: <reason>` line and leaves previous files untouched (C36); `cost.sensitivity_bps` is optional (absent -> no `cost_sensitivity` block), validated when present; Markdown percentages use a decimal comma
- **Abandoned:** none
- **Walked:** n/a - no screen. Root suite: system `python3` 3.14.4 has no pandas, so 4 pre-existing modules fail to import there exactly as at the base commit; with `/home/caio/Documents/projetos/TCC/.venv/bin/python` the full root suite passes (336 tests, 1 skipped)
- **Verifier round (V-1..V-5):** C41-C43 appended for V-1, V-2 and V-3; C9's test now runs its three end dates through `run_report` (V-4); the mutation note names C31 (V-5). Decided: `report` compares the audit inputs (`snapshot_dir`, per-series `file`, `series_kind` and `retrievals`, the selection layers' `missing` and `excluded_lookahead`) rather than `protocol_sha256`, so a protocol edit that leaves the audit inputs unchanged (cost, profiles, decision) needs no re-audit; a non-finite daily factor records `max_rel_diff_daily_factor` null and so `values_revised`. Committed outputs are unchanged and byte-reproducible; real verdict unchanged. V-6 (C38's proof cell) left open: rewriting a frozen proof needs the human's go-ahead
- **Verifier round 2 (V-6..V-8):** C44 appended for V-8 (selection-layer half of the audit/protocol match) and C45 for V-7 (`inf`/subnormal level -> `values_revised`, no traceback), as new checks rather than widening C42's and C43's claims; C38's proof rewritten without a literal pipe under AD-021 (human-authorized, claim unchanged) so `proofs.md` parses (V-6). Real verdict unchanged
