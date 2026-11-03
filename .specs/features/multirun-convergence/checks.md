# Multi-run convergence experiment - checks

Mutations: 4
Build: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m py_compile scripts/multirun_convergence.py scripts/convergence_analysis.py tests/test_convergence_analysis.py tests/test_multirun_convergence.py`
Plan: `.specs/features/multirun-convergence/plan.md`

37 checks in 8 slices · 4 one-way doors · 0 open

Test modules: `tests/test_convergence_analysis.py` (pure analysis on hand-built lots, offline) and
`tests/test_multirun_convergence.py` (CLI, child runner and a fixture registry built from the
committed `data/allocation` and `data/raw/status_invest_*.csv` with a tiny GA: population 20,
5 generations, 12 runs, prefixes `[4, 8, 12]`, `curve_step` 4). Both run under stdlib `unittest`
with `/home/caio/Documents/projetos/TCC/.venv/bin/python` from the repo root (pytest is not
installed). Part (a) is S1-S6 and S8; S7 (C29, C30, C33b) is part (b) and stays red until the
measured lots and report are committed. The four mutations aim at the zone rule (C23), the
run_count rule (C25), the K and δ quantiles (C27) and the stop cadence (C19).

## Checks

### S1 - The decision criterion is registered before the experiment

**C1** - `docs/validation/multirun-convergence-protocol.json` equals the door-1 literal: `protocol_id` `multirun-convergence-v1`, `issue` 9, `capital_brl` 100000, seeds `[46641103, 42, 43, 44, 45]`, both profiles with their records and `expected_n_assets` 13/13 and 15/15, `ga_override` `{adaptive_mode: false, run_count: 150}`, prefixes `[30, 60, 100, 150]`, `curve_step` 10, `rank_window_max` 30, the stop rule (0.02, 0.75, 40, stocks 10, fiis 1), the decision block (ε 0.5, mean and median, `"max"`, target prefix 150, quantile 0.95 `inverted_cdf`, round up 0.01, `fraction_from_counts`, the 9-point δ grid) and the 13 + 13 production tickers of run `9e22cc16-cfda-495e-a8c9-c0d202ef32b6` (AC 1)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_convergence_analysis -k ProtocolFileTests.test_committed_protocol_values`

**C2** - `docs/validation/README.md` has a Portuguese section listing the experiment's files, defining "giro entre seeds" apart from month-to-month turnover, and stating ε 0.5 on mean and median, the leave-one-seed-out maximum, quantile 0.95 by inverted CDF, δ exact from counts rounded up to 0.01, and targets 0.75 and 0.02 judged together at 150 runs (AC 2)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_convergence_analysis -k ReadmeTests.test_readme_section`

**C3** - `run` and `report` each exit 2, print one line naming the field and write nothing for: `prefixes` `[30, 60, 100]` (`prefixes`), seeds `[42, 42]` (`seeds`), seeds `[42, 43]` with `leave_one_seed_out` `"max"` (`seeds`), `adaptive_mode` true (`adaptive_mode`), `curve_step` 7 (`curve_step`), a quantile of 0 and of 1.5 (`quantile`), candidates `[30, 45, 150]` (`candidates`), target prefix 120 (`prefix`), `leave_one_seed_out` `"min"` (`leave_one_seed_out`), `quantile_method` `"linear"` (`quantile_method`), `delta_arithmetic` `"float"` (`delta_arithmetic`), a missing `production_run` (`production_run`) (AC 3, R2-1)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_multirun_convergence -k ProtocolValidationTests.test_invalid_protocol_exits_2`

**C4** - Measured `run` exits 6 before spawning any child and writes nothing when the protocol is untracked, when it is committed but edited, and when a file under `py/` is modified; an untracked file under `scripts/` outside the listed paths is accepted; `--smoke` skips the check and marks every lot `mode: smoke` (AC 4)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_multirun_convergence -k PreregistrationTests`

**C5** - Lots record `protocol_commit` = `git log -1 --format=%H -- <protocol>` and `protocol_committed_at` = its `%cI`, and `report` copies both from the lots into `multirun-convergence.json` unchanged (AC 5)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_multirun_convergence -k ProvenanceTests.test_protocol_commit_of_committed_protocol`
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_convergence_analysis -k ReportTests.test_provenance_copied`

### S2 - A full lot per profile and seed, persisted reproducibly

**C6** - On the fixture registry each lot's `stocks` and `fiis` hold exactly 12 runs with `run_id` 0..11 in order, `selected_tickers` equal to the tickers `run_premium_optimization` returned for that class, and the selectors were called with `adaptive_mode` false and `n_runs` 12; with `SYSTEM_GA_CONFIG` patched so production's stop would fire from the first check (min_runs 1, target_cv 1.0, target_jaccard 0.0), both selectors still record 12 of 12 runs (AC 6)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_multirun_convergence -k FixtureRunTests.test_lot_shape_and_runs`
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_multirun_convergence -k AdaptiveOffTests.test_stop_disabled_records_all_runs`

**C7** - The child resolves `premium-2026-11-02` to 13 stocks and 13 FIIs and `arrojado-anchor` to 15 and 15 through `resolve_recommendation_policy(..., plan="premium")`; a record declared with `expected_n_assets.stocks` 12 makes the child exit 2 naming `expected_n_assets` with no lot written and no GA run (AC 7)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_multirun_convergence -k PolicyTests`

**C8** - `run` exits 3 without spawning a child when the registry's stock CSV, allocation series or `manifest_sha256` differs from the declaration (AC 8)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_multirun_convergence -k SnapshotTests.test_registry_mismatch_exits_3`

**C9** - Two fixture `run`s of the same protocol into different lot directories give byte-identical `seed-*.json` files, and `report` over each gives byte-identical `multirun-convergence.json` (AC 9)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_multirun_convergence -k FixtureRunTests.test_byte_identical_lots_and_reports`

**C10** - On a 12-run fixture lot the analysis consensus at k 8 equals `selected_tickers` of a fresh 8-run `run_multi_execution_profile` and a fresh 8-run `run_fii_selection` with the lot's `selection_config`, seed and adaptive off (AC 10)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_multirun_convergence -k FixtureRunTests.test_prefix_equals_fresh_execution`

**C11** - The scheduler keeps at most `--jobs` children alive (jobs 2 over 5 tasks: peak 2, all 5 run); an existing valid lot with the `HEAD` fingerprints is skipped without spawning; the same lot when `HEAD`'s `experiment_fingerprint` differs makes `run` exit 4 naming `experiment_fingerprint` and leaves the file byte-identical (AC 11)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_multirun_convergence -k SchedulerTests.test_at_most_jobs_alive`
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_multirun_convergence -k SkipTests.test_valid_lot_skipped_and_stale_lot_exit_4`

**C12** - With one profile whose child fails (`expected_n_assets` 12), `run` lets the other children finish and write their lots, writes no lot or cost file for the failed ones, prints `multirun_convergence: <profile_id>:<seed>: ...` for each and exits 1 (AC 12)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_multirun_convergence -k ChildFailureTests.test_failed_child_exit_1_others_written`

**C13** - Each written lot has `seed-<seed>.cost.json` with `started_at`, `finished_at`, `wall_s`, `cpu_s`, per-selector `stages` with `wall_s` and `cpu_s`, `rss_peak_kb`, `jobs`, `load_avg_start`, `load_avg_end` and `machine` with `libraries`; `lots/run-summary.json` has `jobs`, `experiment_wall_s` >= every lot's `wall_s`, `first_started_at`, `last_finished_at` and the lot list (AC 13)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_multirun_convergence -k FixtureRunTests.test_cost_and_summary_files`

**C34** - Two children of one `run` print distinct workspace paths and neither exists after `run` ends; a proxy that drops run 7 makes the child exit 1 naming `stocks` with no `seed-*.json` or cost file; captured 12 runs with a selector result `n_runs` 11 → exit 1; a selector result without `n_runs` → exit 1 (AC 34, R2-2)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_multirun_convergence -k FixtureRunTests.test_workspaces_distinct_and_removed`
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_multirun_convergence -k RunCountGuardTests`

### S3 - Every prefix is analysed from the stored lot

**C14** - `fitness_cv` gives `[1, 2, 3]` → 0.408248290463863 (within 1e-15), `[5, 5, 5]` → 0.0, `[-1, 1]` → 0.0; when the production `coefficient_of_variation` differs by more than 1e-12 on a lot, `report` exits 4 naming the lot and writes nothing (AC 14)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_convergence_analysis -k MetricTests.test_fitness_cv_examples`
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_convergence_analysis -k IntegrityTests.test_cv_production_mismatch_exits_4`

**C15** - `jaccard_mean` gives `{A,B},{A,C}` → 1/3, `{A,B},{A,B},{C,D}` → 1/3, `{A},{A}` → 1.0, one run → 0.0, each pair equal to `jaccard_similarity` (AC 15)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_convergence_analysis -k MetricTests.test_jaccard_examples`

**C16** - Every cell, seed and k in `{curve_step, ..., run_count}` has `fitness_cv`, `jaccard_mean`, fitness mean/min/max and their mean/min/max over seeds; at k = run_count they equal `analyze_stability` (stocks) and `_fii_stability` (FIIs) within 1e-12, and a production `jaccard_mean` off by 1e-9 makes `report` exit 4 (AC 16)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_convergence_analysis -k CurveTests`

**C17** - For every cell, seed and prefix the report lists each picked ticker's count and count / k (AC 17)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_convergence_analysis -k FrequencyTests.test_frequencies`

**C18** - Stocks n 2 with counts POMO4 10, POMO3 9, VALE3 8 → `R_k` [POMO4, VALE3]; equal counts → higher score first; equal count and score → ticker ascending; FIIs KNCR11 and KNRI11 both ranked; the consensus is the production function's output; a lot whose `selected_tickers` differ from the 150-run consensus, or whose candidates miss a picked ticker, makes `report` exit 4 naming the lot (AC 18)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_convergence_analysis -k RankingTests`
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_convergence_analysis -k IntegrityTests.test_consensus_mismatch_exits_4`

**C19** - `adaptive_stop.at` with targets first met from k 43 → FIIs 43, stocks 50; never met → null for both; met from k 1 (identical runs) → 40 for both; met only at k 145 with run_count 150 → stocks 150 (the run_count check), FIIs 145 (AC 19)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_convergence_analysis -k StopTests`

**C20** - `report` exits 4 naming the lot and writes nothing for: a missing run, a duplicated run, runs out of `run_id` order, a lot with another `protocol_sha256`, snapshot, `code_fingerprint`, `experiment_fingerprint` or `mode` than the others, and a declared (profile, seed) with no lot (AC 20)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_convergence_analysis -k IntegrityTests.test_bad_lots_exit_4`

### S4 - Turnover between seeds and the edge of the consensus

**C21** - n 13 sharing 10 tickers → swapped 3, Jaccard 0.625; identical → 0 and 1.0; 3 seeds give 3 unordered pairs with mean, median, min and max (AC 21)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_convergence_analysis -k TurnoverTests.test_seed_pairs`

**C22** - k 150, n 13, rank-13 count 60 and rank-14 count 54 → `f_13` 0.4, `f_14` 0.36, gap 0.04 (1e-12); `R_k` of 25 tickers → ranks 1-25 listed, `ranks_available` 25; the median gap over seeds is reported (AC 22)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_convergence_analysis -k RankGapTests.test_rank_gap`

**C35** - `within_seed_swapped` is 0 at k = run_count for every seed and 3 for a 13-ticker consensus at 30 sharing 10 with the one at 150, with mean, median and max per prefix (AC 35)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_convergence_analysis -k WithinSeedTests.test_within_seed`

### S5 - The protection zone is simulated

**C23** - The door-3 examples: K 5 δ 0.1 → {A, B, D} turnover 1; K 4 δ 0.1 → {A, B, D} 1; K 0 → {D, A, E} 2 (and K 0 equals the first n of `R` for every δ in the grid); E 15 → {A, D, E} 2; E 13 tied below B at δ 0 → {A, B, D} 1; a stock incumbent POMO4 with POMO3 ranked above it is unranked and unprotected (AC 23)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_convergence_analysis -k ZoneTests`

**C24** - For every cell and prefix the grid holds K in `{0} ∪ {n, ..., rank_window_max}` and every δ of `delta_grid`, each with mean and max turnover over the 20 ordered pairs of 5 seeds (6 for 3 seeds) (AC 24)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_convergence_analysis -k ZoneGridTests.test_grid_shape`

### S6 - The three recommendations follow the registered rules

**C25** - S {30: 3.1, 60: 2.4, 100: 2.0, 150: 1.8} with M = S → flattened, `k_min` 100; S(60) 2.3 → 60; S(60) 2.3, M(60) 3.0, M(150) 2.0 → 100; S(100) 2.4 → not flattened → `data_insufficient`; cells 60 and 100 → 100; full 60 with one left-out seed 100 → `value` 100; a left-out `data_insufficient` → `value` `data_insufficient`; the plateau uses the last two declared candidates (fixture `[4, 8, 12]`) (AC 25, R2-1)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_convergence_analysis -k RunCountTests`

**C26** - Jaccard 0.14, 0.13, 0.15, 0.12, 0.14 → `unreachable`; 0.80, 0.90, 0.76, 0.75, 0.99 → `reachable`; 0.74 with four above → `mixed`; Jaccard `reachable` everywhere but one CV `mixed` → `use_fixed_run_count` with `fixed_run_count` = the run_count value (150 when `data_insufficient`); both `reachable` everywhere → `keep`, `fixed_run_count` null (AC 26, R2-3)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_convergence_analysis -k TargetTests.test_reachability_examples`

**C27** - 260 ranks with the 247th smallest 17 → `K*` 17, offset 4; 14 or more unranked of 260 → null; 95 × 0.02 + 5 × 0.20 → 0.02; 94 + 6 → 0.20; 0.0333 → 0.04; 0.04 → 0.04; counts 8 and 5 at 150 → 0.02; 21 and 0 → 0.14; `k_offset` and `delta` are the largest over a selector's cells and both null when one cell's `K*` is null; `at_prefix` is the run_count value or 150 (AC 27)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_convergence_analysis -k ZoneRecommendationTests`

**C28** - `production_run_match.stocks` and `.fiis` are true when the declared lot's `selected_tickers` equal the declared tickers as sets and false on a mismatch, with `report` still exiting 0 (AC 28)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_convergence_analysis -k ProductionMatchTests.test_match_reported`

**C31** - The generated Markdown has, in Portuguese, per cell the curve table, the top-30 frequencies, the seed-turnover tables (stocks and FIIs apart) with within-seed turnover, the rank-gap table, the zone grid at `at_prefix` for K in {n, n+2, n+5, n+10, 30}, the three recommendations with their rules and leave-one-seed-out results, the production-run observation, the cost table, and every limitation of AC 31 including 0.697275 and the numpy/pandas/Python versions from the cost files (AC 31)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_convergence_analysis -k ReportTests.test_markdown_sections`

**C32** - `multirun-convergence-cost.json` and the Markdown cost table give per lot wall and CPU seconds for stocks, FIIs and total copied from the cost files, the CPU-hour sum, `experiment_wall_s` and `jobs` from `run-summary.json` and the machine fingerprint, labelled measured (AC 32)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_convergence_analysis -k CostTests.test_cost_report`

**C33** - Smoke lots give a report whose JSON `mode` is `smoke` and whose Markdown title says smoke (AC 33)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_convergence_analysis -k ReportTests.test_smoke_labels`

### S7 - The measured experiment and its report are committed (part b)

**C29** - The committed `multirun-convergence.json` has `mode` `measured`, all 10 lots of the committed protocol with 150 runs per selector, `protocol_sha256` equal to the committed protocol's, and `protocol_commit` holds the current protocol bytes and is an ancestor of every lot's `code.commit` (AC 29, AC 33)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_convergence_analysis -k CommittedReportTests.test_committed_report_is_measured`

**C30** - `report` over the committed protocol and lots reproduces the three committed files byte for byte without the registry (AC 30)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_convergence_analysis -k CommittedReportTests.test_report_regenerates_byte_for_byte`

### S8 - Production and the #8 gate are untouched

**C36** - Nothing under `py/`, `api/app/`, `scripts/benchmark_premium.py` or `scripts/bench_measure.py` changes between the plan commit and `HEAD` (Changes underneath, readers)
Proof: `git diff --quiet 751cafb HEAD -- py api/app scripts/benchmark_premium.py scripts/bench_measure.py`

**C37** - Every generated JSON file and lot is `json.dumps(sort_keys=True, indent=2, ensure_ascii=False)` + newline with no timestamp other than those copied from the lots, and the analysis floats come from stdlib arithmetic (`report` with numpy's `mean` patched to add 1e-14 gives the same report bytes) (door 1, door 2)
Proof: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_convergence_analysis -k ReportTests.test_serialization_is_stdlib`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| exit codes of `run` and `report` (6) | 0 C9 · 1 C12 · 2 C3 · 3 C8 · 4 C11 · 6 C4 | - |
| exit-4 integrity causes (12) | missing run C20 · duplicated run C20 · out of order C20 · `protocol_sha256` C20 · snapshot C20 · `code_fingerprint` C20 · `experiment_fingerprint` C20 · `mode` C20 · missing lot C20 · consensus mismatch C18 · candidate scores C18 · stdlib vs production metric C14 | - |
| protocol validation rules (12) | missing field C3 · `prefixes` C3 · `seeds` distinct C3 · `seeds` >= 3 with leave-one-out C3 · `adaptive_mode` C3 · `curve_step` C3 · quantile range C3 · `candidates` C3 · target `prefix` C3 · `leave_one_seed_out` enum C3 · `quantile_method` enum C3 · `delta_arithmetic` enum C3 | - |
| door 1 protocol and report contract (3) | protocol file C1 · report shape C9 · serialization C37 | - |
| door 2 lot format (3) | lot fields C6 · deterministic bytes C9 · cost file and run summary C13 | - |
| door 3 zone rule examples (6) | K 5 C23 · K 4 C23 · K 0 C23 · inclusive boundary C23 · δ 0 tie C23 · stock dedup C23 | - |
| door 4 pre-registration (3) | exit 6 C4 · `protocol_commit` recorded C5 · ancestry at `HEAD` C29 | - |
| stop cadence, AC 19 (4 examples) | from 43 C19 · never C19 · identical runs C19 · only at 145 C19 | - |
| run_count rule, AC 25 (7 examples) | flattened 100 C25 · 60 C25 · median binds C25 · not flattened C25 · max over cells C25 · leave-one-out max C25 · fixture candidates C25 | - |
| reachability, AC 26 (5 examples) | unreachable C26 · reachable C26 · mixed C26 · CV mixed C26 · keep C26 | - |
| K and δ quantiles, AC 27 (8 examples) | K* 17 C27 · null C27 · 0.02 C27 · 0.20 C27 · 0.0333 C27 · 0.04 C27 · 8 and 5 C27 · 21 and 0 C27 | - |
| selectors (2) | stocks C6 C10 C16 C18 · fiis C6 C10 C16 C18 | - |
| profiles of the protocol (2) | `premium-2026-11-02` C7 · `arrojado-anchor` C7 | - |
| report sections, AC 31 (10) | curves C31 · frequencies C31 · seed turnover C31 · within-seed C31 · rank gap C31 · zone grid C31 · recommendations C31 · production run C31 · cost C32 · limitations C31 | - |
| generated files (3) | `multirun-convergence.json` C9 · `multirun-convergence-cost.json` C32 · `multirun-convergence.md` C31 | - |
| production code paths kept (4) | `py` C36 · `api/app` C36 · `scripts/benchmark_premium.py` C36 · `scripts/bench_measure.py` C36 | - |

## Swept

- validation: C3, C7
- failure modes: C12, C34, C20
- idempotency: C11, C9
- authorization: n/a - local operator CLI, no user or network surface
- concurrency: C11, C34
- data lifecycle: C11, C30
- dependency failure: C8, C12
- state transitions: C4, C11
- observability: C12, C13

## Handoff

- **Boundary:** part (a) - C1-C28, C31-C37 closed at `bd144b4` (protocol committed first at `19ead15`, plan approval and checks at `369cc95`). C29 and C30 (S7, part b) are red by design until the measured lots and report are committed: their tests (`CommittedReportTests`) are the only 2 failures of the root suite. Root suite with `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest discover -s tests`: 429 tests, 2 errors (C29, C30), 1 skipped (pre-existing); `api` suite (`cd api && ... -m unittest discover -s tests -t .`): 125 tests OK. The new modules run in about 65 s (the fixture GA runs 3 + 3 + 2 lots of 12 + 12 runs in subprocesses).
- **Smoke wiring check (real registry, not committed):** `run --smoke --jobs 5` against `/home/caio/Documents/projetos/TCC/snapshots` into the session scratchpad: 10 lots of 150 + 150 runs at population 20 / 5 generations in 100 s wall, every lot valid; `report` took 3.5 s and passed every exit-4 cross-check (stdlib vs `coefficient_of_variation`, `jaccard_similarity`, `analyze_stability`, `_fii_stability`; first n of `R_k` vs the production consensus at every prefix; the 150-run consensus vs `selected_tickers`). Sizes at full scale: lots 2.0 MB, `multirun-convergence.json` 2.4 MB, `.md` 29 KB.
- **Settled mid-build (needs the human before part b):** door 5 in the plan. Production's `build_scores` leaves 45 of 208 stocks (`premium-2026-11-02`) and 73 of 272 (arrojado) with SCORE NaN on the real `stocks.csv`, and the GA picks them (the fixture run hit it on its first lot; FIIs have none). JSON has no NaN, so a lot stores `"score": null`, and the extended ranking reproduces production's order exactly (same `sorted` key over a `Counter` in production's order), where a NaN score leaves count ties in first-appearance order. AD-022 says "score desc" and is silent on NaN; door 2 says `score` is a float. The alternative (NaN as lowest score) would make criterion 18 exit 4 on real lots. That production selects NaN-score stocks at all looks like a production defect outside this issue's scope; it is surfaced, not fixed (`py/` is untouched, C36).
- **Settled mid-build (decided, no conflict with the plan):** criterion 22's "median of `f_(n+1)` to `f_30`" is reported per rank (`median_frequency_by_rank`, the median over seeds at each rank n+1..30, 0.0 where a seed has fewer ranks), and `f_(n+1)` is 0.0 when `R_k` holds n tickers or fewer; the report's `code.commit` is the first lot's, with every lot's commit under `lots`; `report` also requires every lot to carry the same `protocol_commit` (exit 4); "inconsistent candidate scores" is caught in the child when two runs disagree on a ticker's SCORE or SETOR (exit 1, no lot) and in `report` when `candidates` differs from the picked tickers or a score is neither a number nor null (exit 4); C6's adaptive-off proof patches `SYSTEM_GA_CONFIG` (min_runs 1, target_cv 1.0, target_jaccard 0.0) so production's stop would fire at the first check, instead of building an FII universe with identical runs; `run-summary.json` is written only by an invocation that started children, so after a resumed run `experiment_wall_s` covers that invocation only.
- **Abandoned:** none.
- **Walked:** n/a - no screen.
- **Part (b), after every part-(a) check passes at a clean `HEAD` and the tooling review closes its blocking findings (AD-024 Q7):** from `/home/caio/.orbita/worktrees/TCC/issue-9`, at a clean tree:
  `nohup /home/caio/Documents/projetos/TCC/.venv/bin/python scripts/multirun_convergence.py run --protocol docs/validation/multirun-convergence-protocol.json --registry /home/caio/Documents/projetos/TCC/snapshots --lots-dir docs/validation/multirun-convergence/lots --jobs 5 > /tmp/multirun-convergence-run.log 2>&1 &`
  Lots land in `docs/validation/multirun-convergence/lots/<profile_id>/seed-<seed>.json`, cost files beside them as `seed-<seed>.cost.json`, and `docs/validation/multirun-convergence/lots/run-summary.json`; about 22 CPU-hours, 5-6 h wall. A failed child exits the run with 1 and leaves the others written; re-running the same command skips valid lots. Then `report --protocol docs/validation/multirun-convergence-protocol.json --lots-dir docs/validation/multirun-convergence/lots --out-dir docs/validation` writes `docs/validation/multirun-convergence.json`, `-cost.json` and `.md`; commit lots and reports as new commits (no rebase, amend or force-push from the first lot on, AD-023), then C29 and C30 must pass.
- **For the tooling reviewer:** `scripts/multirun_convergence.py` - `capture_selector_runs` (module-attribute proxies on `pipelines.multi_run.run_single_execution` and `fii_selection._run_fii_execution`, returning the original objects), `run_child` (fresh `tempfile.mkdtemp` removed in `finally`, `apply_override` setting only `adaptive_mode` false and `run_count`, `check_selector_runs` against `result["n_runs"]`, `merge_candidates`), `write_files_atomically` (cost renamed before the lot), `preregistration_problems` and `CODE_PATHS`, `check_existing_lot` and `schedule`; `scripts/convergence_analysis.py` - `validate_protocol`, `check_lot`/`load_lots`, `SeedSeries` (cross-checks), `extended_ranking`/`production_consensus` (door 5), `zone_portfolio`, `adaptive_stop_at`, `run_count_rule`/`recommend_run_count`, `k_star`/`delta_star`/`round_up`, `recommend_targets`. Open for the human: door 5, and whether a 2.4 MB generated report JSON is acceptable beside the ~2 MB lots Q8 approved.
