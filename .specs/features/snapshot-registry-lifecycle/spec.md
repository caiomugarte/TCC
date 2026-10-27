# Immutable Snapshot Registry & CLI Specification

**Issue**: #16 (P0-09) — Registro e CLI de snapshots imutáveis
**Route**: driven · **Priority**: P0 · **Depends on**: #11 (P0-04), #15 (P0-08)

## Problem Statement

The snapshot registry described by `py/snapshot_manifest.py` has no lifecycle: manifests are loaded ad hoc, there is no `draft/validated/published/retired` state, no operational validation entry point, no retention/immutability rule, and no CLI. A published snapshot cannot be proven immutable today because nothing pins the manifest itself to a hash, and nothing stops an operator from overwriting or silently substituting data. This blocks auditable Premium runs (P0) and collides with open issues #11/#15 unless the contract changes stay additive.

## Goals

- [ ] A published snapshot is immutable and auditable by a canonical manifest hash, with per-source hashes retained.
- [ ] An offline CLI (`build`/`validate`/`publish`/`retire`) drives the lifecycle with documented, deterministic exit codes.
- [ ] Validation fails on hash divergence or date divergence (`cutoff_date` != last `common_date`), with no fallback series.
- [ ] Legacy manifests without `state` keep working unchanged (load, round-trip, selection).
- [ ] The registry layout is explicit and does not break `SnapshotRegistry.manifests()`'s top-level `*.json` loader.
- [ ] Which series feed each asset class is exposed in an auditable, deterministic form.

## Out of Scope

| Feature | Reason |
| ------- | ------ |
| Scheduling snapshot collection | Issue #16 "Fora de escopo": agendar coleta |
| Storage high-availability / replication | Issue #16 "Fora de escopo": criar HA de storage |
| Retention/deletion policy beyond `retire` | `retire` is state-only and never deletes; erasure policy is a separate decision |
| Real snapshot publication | No real data with `caio_fiis.csv` exists; proofs use synthetic fixtures only |
| `ALLOCATION_FILES` refactor, per-class metrics, pytest adoption | Collision guard with open issues #11 (P0-04) and #15 (P0-08) |
| Changes to `py/fetch_allocation_snapshot.py` | It already writes CSVs + `metadata.json`; `build` consumes its output as-is |

---

## Assumptions & Open Questions

Every ambiguity is resolved or recorded here - nothing is left silently unclear.

| Assumption / decision | Chosen default | Rationale | Confirmed? |
| --------------------- | -------------- | --------- | ---------- |
| Where lifecycle state lives | Optional `state` (+ `state_changed_at`, `retire_reason`) fields inside the manifest JSON itself; no sidecar state files | Issue text says "manifest legado sem `state`"; `SnapshotRegistry.manifests()` globs `*.json` at the registry root and only skips `metadata.json`, so loose state/index JSON at root would break the loader; sidecars in subdirectories add a second source of truth | n (agent default) |
| Determinism / `created_at` | `build` REQUIRES explicit `--created-at`; the canonical hash INCLUDES `created_at` | "Reproduzível" is satisfied by byte-identical rebuilds from identical inputs; keeping `created_at` inside the hash preserves full tamper evidence | n (agent default) |
| Canonical hash scope | `manifest_sha256` = SHA-256 over canonical JSON of the manifest's `as_dict()` projection (the normalization point: unknown keys dropped, absent optional fields omitted - but `as_dict()` does NOT reorder `supported_classes`; order is pinned by `build` deriving it from `ASSET_CLASSES`) minus `manifest_sha256`, `state`, `state_changed_at`, `retire_reason`; computed in ONE helper (`manifest_canonical_sha256`); constant across the whole lifecycle | Single canonicalization point makes the hash recomputable on the Premium rehydration path (`api/app/services/premium_executor.py:162` → `SnapshotManifest.from_dict(manifest_data, base_dir=…)` → `validate_manifest`) even when `api/app/services/premium_recommendation.py:87-88` injects `manifest_path`; self-reference and lifecycle envelope excluded by definition; the `as_dict()` projection means unknown extra keys are outside the audit contract (documented); "normalized" = schema projection only - `as_dict()` (`py/snapshot_manifest.py:333-346`) emits `supported_classes` in declaration order and does not sort it (review nit), so order determinism lives in `build`, not in `as_dict()` | n (agent default) |
| Manifest emit / on-disk form | `build` writes exactly the `SnapshotManifest.as_dict()` projection (single `allocation` source + `metadata.files`, explicit `stock`/`fii` selector sources and their `*_source` aliases); every manifest write uses the fixed serializer `json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)` + trailing `\n`; `supported_classes` is derived from `ASSET_CLASSES` (`py/allocation_config.py:14-20`) in fixed order; each source's `provider` is a fixed literal per source role and `metadata` is exactly `{"files": {...}}` (`allocation`) or `{}` (selector sources) - fetch `metadata.json` content is never copied into the manifest | A fixed serializer is what makes "reproduzível" testable; AC2's proof is building the same `manifest_id` in TWO distinct registries and comparing bytes (a repeated `build` in one registry is exit 3 by design - review F4); fetch `metadata.json` carries absolute paths (`portfolio_path`, `portfolio_artifacts` - `py/fetch_allocation_snapshot.py:538,584-590`), so copying it in would make two registries emit different bytes and break AC3 (review N5) | n (agent default) |
| Source path base | All source paths are registry-root-relative POSIX paths; `build` accepts `--snapshot-dir`/`--stock-universe`/`--fii-universe` only when they resolve INSIDE the registry root (any subdirectory - `data/<snapshot_name>/` is the conventional layout) and exits 2 on an escape detected after `Path.resolve()` | The necessary invariant is "inside the registry root", not "under `<registry>/data/`": `SnapshotSource.resolved_path` resolves relative to the manifest's directory (`py/snapshot_manifest.py:138-145`), so a path recorded outside the registry could never resolve during `validate` (review F5; the stronger `data/`-only rule relaxed per review nit) | n (agent default) |
| `common_dates` semantics | Sorted intersection of the five class series exactly as `load_premium_snapshot_bundle` consumes them (`international_equity`/`crypto` already multiplied by `ptax`); first date retained with return 0; `build` exits 4 only when the intersection is empty | `py/allocation_data.py:138-141` intersects and `:312-315` requires `row_dates == manifest.common_dates`, so a "full coverage" rule would reject nearly every real snapshot (review F3) | n (agent default) |
| `validate --check` audit path | Read-only flag on `validate` (not a subcommand) that runs the FULL `validate_manifest` verification set - everything plain `validate` checks except the lifecycle-state gate (hashes matching is necessary, not sufficient) - in ANY lifecycle state and exits 0 (all checks pass) / 4 (any verification failure) with zero writes; combines with `--json` (0/4 semantics, zero writes) | AC1 "auditável por hash" must hold in the final states, where plain `validate` refuses with exit 5 before recomputing anything (review F6); a green audit must never mean less than `validate` (review N2); P3 promises review of a *published* snapshot's inputs, which only `--check --json` can print (review N3) | n (agent default) |
| Backfill of `manifest_sha256` | CLI state transitions backfill `manifest_sha256` when the manifest (legacy) lacks it | Legacy manifests become auditable through the normal lifecycle without a separate migration step | n (agent default) |
| CLI semantics for legacy manifests (no `state`) | For transitions, absent `state` behaves as `draft` (so `validate` upgrades it); for selection, absent `state` stays selectable | Split semantics required by "retrocompatíveis com manifest legado sem `state`" plus "estado ausente (legado) continua selecionável" | n (agent default) |
| `validate` idempotency | `validate` on `validated` exits 0 with no write; `validate` on `published`/`retired` exits 5 (frozen) | Re-running operational checks must be safe; published content is immutable | n (agent default) |
| `retire` sources | Allowed from `published` and `validated`; `retired` is terminal; re-retire exits 5 | Covers "retire só muda estado" without inventing resurrection paths | n (agent default) |
| Double `publish` | Exits 3 (refused overwrite) and leaves the file byte-identical; other invalid transitions exit 5 | Directly encodes "publish nunca sobrescreve" / "Snapshot publicado não pode ser sobrescrito" | n (agent default) |
| Exit code table | 0 success; 2 usage/args (incl. `--manifest-id` failing `^[A-Za-z0-9][A-Za-z0-9._-]*$`, `--manifest-id` == `metadata`, loaded `manifest_id` != `--manifest-id`, `build` inputs escaping the registry root); 3 target missing ON DISK (registry dir, `<registry>/<manifest-id>.json`) or refused overwrite (immutability), plus `build` source inputs missing on disk (AC7); 4 contract divergence (hash/date divergence, a required source of a manifest under audit absent from the manifest OR missing/unreadable on disk - `validate` and `validate --check` alike - empty `common_dates` intersection at `build`, or any `validate --check` verification failure); 5 invalid state transition | Stable machine contract for automation; argparse already uses 2; the 3/4 boundary (review F10, ruling N1): 3 = "the operator pointed at something that does not exist" (CLI target or `build` input); 4 = "the artifact under audit does not hold up" - after `publish` the data files ARE part of the audited artifact, so losing them is tampering (coherent with AC16 = 0/4 and IT-2) | n (agent default) |
| Registry layout | `<registry>/<manifest_id>.json` at root (only manifest JSON tolerated at root); snapshot data in subdirectories of the registry; atomic writes via tmp + `os.replace` | Loader glob constraint (see design AD-001) | n (agent default) |
| Documentation location | `docs/snapshot_registry.md` + module docstring | AC3 "CLI documentada e reproduzível"; matches existing `docs/` usage | n (agent default) |
| Test strategy | 100% stdlib `unittest`, offline, synthetic `tempfile` fixtures; new tests in NEW file `tests/test_snapshot_registry.py`, CREATED MINIMAL BY T1 (so `tests.test_snapshot_registry` is runnable from T1 on - review F2); existing test files untouched | `pytest`/`pandas`/`api/.venv` absent; minimal diff avoids collisions with #11/#15 | n (agent default) |
| Proof boundary | Never publish/validate a "real" snapshot; no creation of `snapshots/` in the repo by tests or the spec stage | Blocker: no real data with `caio_fiis.csv` exists (issue-internal) | n (agent default) |
| Pre-existing suite failures | `tests/test_premium_research_contracts.py`, `test_fii_selection.py`, `test_multi_run_backtest.py`, `test_fii_backtest_analysis.py` fail today on missing `pandas`; out of scope | Environment fact verified 2026-10-27; gates target the stdlib-only subset | n (agent default) |

**Open questions:** none - all resolved or logged above (required before the spec is confirmed).

---

## User Stories

### P1: Lifecycle CLI with immutability and audit hash ⭐ MVP

**User Story**: As a research operator, I want an offline CLI that builds, validates, publishes, and retires snapshot manifests so that a published snapshot is provably immutable and auditable by hash.

**Why P1**: This is the issue's acceptance core (AC1-AC3) and unblocks auditable Premium runs.

**Acceptance Criteria** (each line is one EARS pattern):

1. WHEN an operator runs `build` with complete source inputs and an explicit `--created-at` THEN the CLI SHALL write `<registry>/<manifest_id>.json` in exactly the `SnapshotManifest.as_dict()` form (single `allocation` source carrying `metadata.files`, explicit `stock`/`fii` selector sources and their `*_source` aliases) with `state: "draft"`, per-source `sha256`, `common_dates`, `cutoff_date`, and `manifest_sha256`, and SHALL exit 0. <!-- event-driven -->
2. WHEN the required series declare different date coverage THEN `build` SHALL set `common_dates` to the sorted intersection of the five class series exactly as `load_premium_snapshot_bundle` consumes them (`international_equity`/`crypto` already multiplied by `ptax`), keeping the first date with zero return and `cutoff_date` = last date, and SHALL exit 4 only when that intersection is empty. <!-- event-driven -->
3. IF `build` is run with identical inputs and identical `--created-at` in two different registries THEN the two written manifest files SHALL be byte-identical (same `manifest_id`, fixed disk serializer). <!-- unwanted-behavior -->
4. IF `build` targets a manifest file that already exists OR references a source file missing on disk THEN the CLI SHALL exit 3 and SHALL NOT modify any existing file. <!-- unwanted-behavior -->
5. WHEN `validate` runs on a manifest whose source hashes match and whose `cutoff_date` equals the last `common_dates` entry THEN the CLI SHALL transition `draft` to `validated` and exit 0. <!-- event-driven -->
6. IF `validate` finds a source hash that diverges from the manifest OR a `cutoff_date` different from the last `common_dates` entry THEN the CLI SHALL exit 4 and SHALL NOT modify any file. <!-- unwanted-behavior -->
7. IF `build` references a required source file that is missing on disk THEN the CLI SHALL exit 3 and SHALL NOT substitute any fallback series (including IFIX). <!-- unwanted-behavior -->
8. IF a required class series is absent from the manifest (for example `caio_fiis.csv`) THEN `validate` SHALL exit 4 and SHALL NOT substitute any fallback series (including IFIX). <!-- unwanted-behavior -->
9. WHEN `publish` runs on a manifest in state `validated` THEN the CLI SHALL transition it to `published`, exit 0, and leave every field covered by `manifest_sha256` unchanged. <!-- event-driven -->
10. IF `publish` runs on a manifest that is already `published` THEN the CLI SHALL exit 3 and leave the manifest file byte-identical. <!-- unwanted-behavior -->
11. WHILE a manifest is `published` the CLI SHALL refuse every content modification; only `retire` SHALL change it, and only lifecycle fields (`state`, `state_changed_at`, `retire_reason`). <!-- state-driven -->
12. WHEN `retire` runs on a manifest in state `published` or `validated` THEN the CLI SHALL set `state: "retired"` (recording the optional reason) and exit 0 without deleting any file. <!-- event-driven -->
13. IF a CLI command requests an invalid state transition (including retiring an already `retired` manifest) THEN the CLI SHALL exit 5 and modify nothing. <!-- unwanted-behavior -->
14. The manifest payload SHALL carry `manifest_sha256`, computed as SHA-256 over canonical JSON (`sort_keys=True`, `separators=(",", ":")`, UTF-8) of the normalized projection `SnapshotManifest.from_dict(raw).as_dict()` minus `manifest_sha256`, `state`, `state_changed_at`, and `retire_reason`. <!-- ubiquitous -->
15. IF a manifest declares `manifest_sha256` and its recomputed normalized hash differs THEN `validate_manifest` and CLI `validate` SHALL reject it. <!-- unwanted-behavior -->
16. WHEN `validate --check` runs on a manifest in any lifecycle state (optionally with `--json`) THEN the CLI SHALL execute the full `validate_manifest` verification set - required classes and selector sources, every source hash, `manifest_sha256` recomputation, `cutoff_date` coherence - skipping only the lifecycle-state gate, and SHALL exit 0 when every check passes or 4 on any verification failure without writing any file; matching hashes alone are necessary but not sufficient. <!-- event-driven -->
17. IF `--manifest-id` fails `^[A-Za-z0-9][A-Za-z0-9._-]*$`, equals `metadata`, or differs from the loaded `manifest_id` THEN the CLI SHALL exit 2 and write nothing. <!-- unwanted-behavior -->
18. The CLI SHALL perform no network access during any command (verified by a stdlib source scan of the new module - `test_cli_source_has_no_network_imports`). <!-- ubiquitous -->
19. WHEN any command writes a manifest file THEN the write SHALL be atomic (temporary file created in the target's own directory plus `os.replace`) so no partial manifest is observable. <!-- event-driven -->
20. WHERE a manifest has no `state` field (legacy) the CLI SHALL treat it as `draft` for transitions and backfill `manifest_sha256` on its first transition. <!-- optional-feature -->
21. The registry root SHALL contain only manifest `*.json` files (plus the tolerated `metadata.json`); snapshot data SHALL live in subdirectories, and no loose state or index JSON SHALL be written at the root. <!-- ubiquitous -->

**Independent Test**: Run the four commands against synthetic fixture registries in `tempfile` and observe exit codes 0/2/3/4/5 as specified; build the same `manifest_id` in two distinct registries and `diff` the outputs; run `validate --check` on a published fixture before and after tampering with a content field.

---

### P2: State-aware selection with legacy compatibility

**User Story**: As the Premium API, I want `latest_compatible` to respect lifecycle state so that drafts and retired snapshots are never silently used in a run.

**Why P2**: Selection is consumed by `api/app/services/premium_recommendation.py`; wrong selection corrupts runs, but the CLI is the acceptance core.

**Acceptance Criteria**:

1. WHEN `SnapshotRegistry.latest_compatible` selects a manifest THEN it SHALL never return one whose `state` is `draft` or `retired`. <!-- event-driven -->
2. WHERE the caller passes `include_validated=True` the selector SHALL also accept manifests whose `state` is `validated`. <!-- optional-feature -->
3. The selector SHALL accept manifests without a `state` field (legacy) exactly as today, and SHALL return the same result as the current implementation for legacy-only registries. <!-- ubiquitous -->
4. IF no candidate manifest qualifies THEN the selector SHALL raise and SHALL NOT fall back to an older artifact. <!-- unwanted-behavior -->

**Independent Test**: Build a fixture registry with one manifest per state plus one legacy manifest; assert selection results with and without `include_validated`.

---

### P3: Auditable class-to-series mapping

**User Story**: As an auditor, I want a deterministic view of which series feed each asset class so that a published snapshot's inputs are reviewable.

**Why P3**: Required by the issue scope ("Registrar quais séries alimentam cada classe") but not part of the three acceptance criteria.

**Acceptance Criteria**:

1. WHEN `validate --json` (or `validate --check --json`) runs THEN the CLI SHALL report, per supported class, the resolved series path (registry-root-relative POSIX), the `file_sha256` of the series file itself (never a directory digest), and its provider, derived from the manifest sources; `--check --json` prints the same map in every lifecycle state with `--check` semantics (0/4, zero writes). <!-- event-driven -->
2. The mapping SHALL be deterministic across runs on the same manifest, SHALL resolve `fiis` to `caio_fiis.csv` with no fallback, and SHALL cover only the class inputs plus `ptax` and the explicit selector sources - non-class files that `py/fetch_allocation_snapshot.py` always writes (`ifix.csv`, `ibovespa.csv`) SHALL NOT be mapped. <!-- ubiquitous -->
3. The registry documentation SHALL state the registry layout, every command, the exit code table, and the prohibitions (no network, no overwrite of published artifacts, no deletion on `retire`, no series fallback). <!-- ubiquitous -->

**Independent Test**: Run `validate --json` on a fixture manifest and compare the printed class-to-series map to the manifest sources.

---

## Edge Cases

- IF the registry directory does not exist THEN the CLI SHALL exit 3 with a message naming the missing path.
- IF `--created-at` is missing or not an ISO datetime THEN `build` SHALL exit 2 and write nothing.
- IF `--manifest-id` fails `^[A-Za-z0-9][A-Za-z0-9._-]*$` or equals `metadata` (the loader skip in `py/snapshot_manifest.py:463`) THEN the CLI SHALL exit 2 and write nothing; a loaded `manifest_id` different from `--manifest-id` SHALL also exit 2.
- IF a `build` input (`--snapshot-dir`, `--stock-universe`, `--fii-universe`) resolves outside the registry root (escape detected after `Path.resolve()`) THEN `build` SHALL exit 2 and write nothing (the recorded manifest-relative path would never resolve during `validate`).
- IF a required source of a manifest under audit is absent - missing from the manifest OR its file/derived artifact missing or unreadable on disk - THEN `validate` and `validate --check` SHALL exit 4 and SHALL NOT substitute any fallback series (after `publish` the data files are part of the audited artifact; review ruling N1).
- IF a manifest declares an unknown `state` string THEN loading SHALL fail with a validation error (never a silent default).
- IF the manifest JSON is modified after `validated` (fields covered by `manifest_sha256`) THEN the next `validate` SHALL exit 4, and `validate --check` SHALL exit 4 in ANY state (including `published`/`retired`).
- WHEN `validate` succeeds on a legacy manifest THEN only `state`, `state_changed_at`, and `manifest_sha256` SHALL change in the file (unknown fields preserved; those fields are outside the normalized hash projection by definition).
- WHEN two required series declare overlapping but different date coverage THEN `build` SHALL narrow `common_dates` to the intersection consumed by `load_premium_snapshot_bundle` (this is the intended semantics of `py/allocation_data.py:138-141`) and SHALL exit 4 only when the intersection is empty.
- IF the final `os.replace` of an atomic write fails THEN the original manifest SHALL remain byte-identical and no partial manifest SHALL be observable (the temporary file lives in the target's own directory).

## Implicit-Requirement Dimension Sweep

| Dimension | Resolution |
| --------- | ---------- |
| Input validation & bounds | SNAP-03/SNAP-04: CLI arg formats (ISO datetime/date, `--manifest-id` pattern/reserved name), manifest schema validation + `manifest_sha256` recomputation via `validate_manifest`; unknown `state` rejected at load |
| Failure / partial-failure states | SNAP-07/SNAP-03: atomic tmp (in the target's own directory) + `os.replace` writes; any failure leaves files unmodified |
| Idempotency / retry / duplicate handling | SNAP-04 (validate idempotent on `validated`), SNAP-05 (double publish refused, exit 3), SNAP-03 (build no-overwrite) |
| Auth boundaries & rate limits | N/A because the CLI is a single-operator offline tool with no auth surface or remote calls |
| Concurrency / ordering | Atomic writes prevent torn manifests; concurrent writers on the same manifest are outside the contract and documented as prohibited (single-operator CLI) |
| Data lifecycle / expiry | SNAP-06: `retire` is state-only and never deletes; erasure/retention policy is explicitly out of scope |
| Observability | Exit codes 0/2/3/4/5 plus stderr diagnostics and `--json` machine-readable output (SNAP-03..SNAP-06, SNAP-09) |
| External-dependency failure | N/A because the CLI is offline by requirement (SNAP-10 prohibition) and touches only the local filesystem |
| State-transition integrity | SNAP-01: explicit transition table with terminal `retired`; invalid transitions exit 5 |

---

## Requirement Traceability

| Requirement ID | Obligation | Story | Phase | Status |
| -------------- | ---------- | ----- | ----- | ------ |
| SNAP-01 | Lifecycle envelope fields + state machine (`draft`/`validated`/`published`/`retired`; legacy behaves as `draft` for transitions) | P1: Lifecycle CLI | Design | In Design |
| SNAP-02 | Canonical `manifest_sha256` over the normalized `as_dict()` projection; recomputation, rejection, and legacy backfill | P1: Lifecycle CLI | Design | In Design |
| SNAP-03 | CLI contract: `build` arguments (`--created-at`, `--manifest-id` validation, input paths inside the registry root), exit code table, no network | P1: Lifecycle CLI | Design | In Design |
| SNAP-04 | `validate` semantics: divergence checks (hash/date/series, on manifest and on disk), idempotency, read-only `validate --check` audit path (full `validate_manifest` set minus the state gate; combinable with `--json`) | P1: Lifecycle CLI | Design | In Design |
| SNAP-05 | `publish` immutability (published artifacts are never overwritten) | P1: Lifecycle CLI | Design | In Design |
| SNAP-06 | `retire` is state-only, never deletes; `retired` is terminal | P1: Lifecycle CLI | Design | In Design |
| SNAP-07 | Registry layout, emitted `as_dict()` manifest form, atomic writes (tmp in target directory), byte-reproducibility, `common_dates` intersection | P1: Lifecycle CLI | Design | In Design |
| SNAP-08 | State-aware selection with legacy compatibility and no fallback | P2: State-aware selection | Design | In Design |
| SNAP-09 | Deterministic class-to-series mapping (class inputs + `ptax` + selector sources only; registry-relative paths, file-level `sha256`) | P3: Class-to-series mapping | Design | In Design |
| SNAP-10 | Registry documentation (layout, commands, exit codes, prohibitions) | P3: Class-to-series mapping | Design | In Design |

**ID format:** `[CATEGORY]-[NUMBER]` (e.g., `SNAP-01`)

**Coverage:** 10 total, 10 mapped to tasks (see `tasks.md`), 0 unmapped.

### Issue #16 Traceability (ACs and tests of the issue)

| Issue item | Requirement | Task | Concrete test |
| ---------- | ----------- | ---- | ------------- |
| AC1 "Snapshot publicado é imutável e auditável por hash" | SNAP-02, SNAP-05 | T2, T6 | `tests/test_snapshot_registry.py::test_manifest_hash_survives_transitions`, `::test_publish_never_overwrites_published_manifest` |
| AC2 "Validação falha se data ou hash divergirem" | SNAP-04 | T5 | `tests/test_snapshot_registry.py::test_validate_rejects_divergent_source_hash`, `::test_validate_rejects_cutoff_date_divergence` |
| AC3 "CLI documentada e reproduzível" | SNAP-03, SNAP-07, SNAP-10 | T4, T9 | `tests/test_snapshot_registry.py::test_build_is_byte_reproducible_across_registries` (+ executable doc-content gate) |
| Test IT-1 "Hash divergente rejeitado" | SNAP-04 | T5 | New: `tests/test_snapshot_registry.py::test_validate_rejects_divergent_source_hash`; existing library-level: `tests/test_snapshot_manifest.py::test_changed_source_hash_is_rejected` |
| Test IT-2 "Manifest sem caio_fiis.csv falha, sem fallback" | SNAP-04, SNAP-09 | T5, T8 | New: `tests/test_snapshot_registry.py::test_validate_requires_caio_fiis_without_ifix_fallback`; existing loader-level: `tests/test_allocation_data.py::test_missing_premium_fii_input_does_not_use_ifix` |
| Test IT-3 "Snapshot publicado não pode ser sobrescrito" | SNAP-05 | T6 | `tests/test_snapshot_registry.py::test_publish_never_overwrites_published_manifest` |
| Spec review F1 (canonical hash implementable on the Premium path) | SNAP-02 | T2 | `tests/test_snapshot_registry.py::test_manifest_hash_survives_premium_rehydration` |
| Spec review F3 (`common_dates` = consumer intersection) | SNAP-07 | T4 | `tests/test_snapshot_registry.py::test_build_common_dates_match_premium_bundle_rows` |
| Spec review F6 (audit path for a published manifest) | SNAP-04 | T5 | `tests/test_snapshot_registry.py::test_verify_published_detects_tampering` |
| Spec review F7 (`--manifest-id` hardening) | SNAP-03 | T4 | `tests/test_snapshot_registry.py::test_manifest_id_rejects_traversal_and_reserved_name` |
| Spec review F8 (test gaps) | SNAP-04, SNAP-06, SNAP-07 | T5, T7, T4 | `tests/test_snapshot_registry.py::test_validate_on_published_or_retired_exits_5_without_writes`, `::test_validate_backfills_manifest_sha256_on_legacy_manifest`, `::test_retire_preserves_content_fields_byte_for_byte`, `::test_failed_replace_keeps_original_manifest` |
| Spec review N1 (required-source failure = exit 4 in `validate`/`validate --check`) | SNAP-04 | T5 | `tests/test_snapshot_registry.py::test_validate_missing_source_file_exits_4` |
| Spec review N2 (`--check` runs the full `validate_manifest` set) | SNAP-04 | T5 | `tests/test_snapshot_registry.py::test_check_runs_full_validation_not_just_hashes` |
| Spec review N3 (`validate --check --json` usable in every state) | SNAP-09 | T8 | `tests/test_snapshot_registry.py::test_check_json_prints_map_in_published_state` |
| Spec review N4 (map exposes file-level `sha256`, registry-relative paths) | SNAP-09 | T8 | `tests/test_snapshot_registry.py::test_class_series_map_hashes_the_file_not_the_directory`, `::test_class_series_map_uses_registry_relative_paths` |
| Spec review N5 (fixed `provider`/`metadata` literals in `build`) | SNAP-07 | T4 | `tests/test_snapshot_registry.py::test_build_source_metadata_is_fixed_and_absolute_path_free` |
| Spec review N7 (AC18 verification method) | SNAP-03 | T4 | `tests/test_snapshot_registry.py::test_cli_source_has_no_network_imports` |

---

## Success Criteria

- [ ] All four CLI commands behave per the exit code table against synthetic fixtures, offline, in a clean checkout with only the stdlib.
- [ ] `manifest_sha256` of a fixture manifest is unchanged after `validate`, `publish`, and `retire`, and recomputes identically after an `as_dict() → json → from_dict` round-trip that injects `manifest_path`.
- [ ] Building the same `manifest_id` in two distinct registries yields byte-identical manifest files.
- [ ] `validate --check` exits 0 on a published fixture and 4 after a content field is tampered with, in every lifecycle state.
- [ ] `validate --check` rejects a manifest whose hashes are consistent but which violates the `validate_manifest` contract (e.g. `fiis` absent), exactly as `validate` does, and `validate --check --json` prints the class-to-series map for a published fixture with zero writes.
- [ ] Legacy fixture manifests (no `state`) produce identical selection results before and after the change.
- [ ] The full stdlib test subset passes (`tests/test_snapshot_registry.py` created by T1, `tests/test_snapshot_manifest.py`, `tests/test_allocation_data.py`, `tests/test_allocation.py`, `tests/test_fetch_status_invest*.py`).
