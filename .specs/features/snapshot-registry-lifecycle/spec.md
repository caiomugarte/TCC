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
| Canonical hash scope | `manifest_sha256` = SHA-256 over canonical JSON of the manifest payload EXCLUDING `manifest_sha256` itself and the lifecycle envelope (`state`, `state_changed_at`, `retire_reason`); constant across the whole lifecycle | Self-reference excluded by definition; a hash that survives `retire` is what makes "conteúdo imutável e auditável por hash" true across transitions | n (agent default) |
| Backfill of `manifest_sha256` | CLI state transitions backfill `manifest_sha256` when the manifest (legacy) lacks it | Legacy manifests become auditable through the normal lifecycle without a separate migration step | n (agent default) |
| CLI semantics for legacy manifests (no `state`) | For transitions, absent `state` behaves as `draft` (so `validate` upgrades it); for selection, absent `state` stays selectable | Split semantics required by "retrocompatíveis com manifest legado sem `state`" plus "estado ausente (legado) continua selecionável" | n (agent default) |
| `validate` idempotency | `validate` on `validated` exits 0 with no write; `validate` on `published`/`retired` exits 5 (frozen) | Re-running operational checks must be safe; published content is immutable | n (agent default) |
| `retire` sources | Allowed from `published` and `validated`; `retired` is terminal; re-retire exits 5 | Covers "retire só muda estado" without inventing resurrection paths | n (agent default) |
| Double `publish` | Exits 3 (refused overwrite) and leaves the file byte-identical; other invalid transitions exit 5 | Directly encodes "publish nunca sobrescreve" / "Snapshot publicado não pode ser sobrescrito" | n (agent default) |
| Exit code table | 0 success; 2 usage/args; 3 target missing or refused overwrite (immutability); 4 validation divergence (hash/date/series); 5 invalid state transition | Stable machine contract for automation; argparse already uses 2 | n (agent default) |
| Registry layout | `<registry>/<manifest_id>.json` at root (only manifest JSON tolerated at root); snapshot data in subdirectories of the registry; atomic writes via tmp + `os.replace` | Loader glob constraint (see design AD-001) | n (agent default) |
| Documentation location | `docs/snapshot_registry.md` + module docstring | AC3 "CLI documentada e reproduzível"; matches existing `docs/` usage | n (agent default) |
| Test strategy | 100% stdlib `unittest`, offline, synthetic `tempfile` fixtures; new tests in NEW file `tests/test_snapshot_registry.py` (existing test files untouched) | `pytest`/`pandas`/`api/.venv` absent; minimal diff avoids collisions with #11/#15 | n (agent default) |
| Proof boundary | Never publish/validate a "real" snapshot; no creation of `snapshots/` in the repo by tests or the spec stage | Blocker: no real data with `caio_fiis.csv` exists (issue-internal) | n (agent default) |
| Pre-existing suite failures | `tests/test_premium_research_contracts.py`, `test_fii_selection.py`, `test_multi_run_backtest.py`, `test_fii_backtest_analysis.py` fail today on missing `pandas`; out of scope | Environment fact verified 2026-10-27; gates target the stdlib-only subset | n (agent default) |

**Open questions:** none - all resolved or logged above (required before the spec is confirmed).

---

## User Stories

### P1: Lifecycle CLI with immutability and audit hash ⭐ MVP

**User Story**: As a research operator, I want an offline CLI that builds, validates, publishes, and retires snapshot manifests so that a published snapshot is provably immutable and auditable by hash.

**Why P1**: This is the issue's acceptance core (AC1-AC3) and unblocks auditable Premium runs.

**Acceptance Criteria** (each line is one EARS pattern):

1. WHEN an operator runs `build` with complete source inputs and an explicit `--created-at` THEN the CLI SHALL write `<registry>/<manifest_id>.json` with `state: "draft"`, per-source `sha256`, `common_dates`, `cutoff_date`, and `manifest_sha256`, and SHALL exit 0. <!-- event-driven -->
2. IF `build` is invoked again with identical inputs and identical `--created-at` THEN the CLI SHALL produce a byte-identical manifest. <!-- unwanted-behavior -->
3. IF `build` targets a manifest file that already exists OR references a missing source file THEN the CLI SHALL exit 3 and SHALL NOT modify any existing file. <!-- unwanted-behavior -->
4. WHEN `validate` runs on a manifest whose source hashes match and whose `cutoff_date` equals the last `common_dates` entry THEN the CLI SHALL transition `draft` to `validated` and exit 0. <!-- event-driven -->
5. IF `validate` finds a source hash that diverges from the manifest OR a `cutoff_date` different from the last `common_dates` entry THEN the CLI SHALL exit 4 and SHALL NOT modify any file. <!-- unwanted-behavior -->
6. IF a required series is absent from the manifest (for example `caio_fiis.csv`) THEN `build` and `validate` SHALL fail and SHALL NOT substitute any fallback series (including IFIX). <!-- unwanted-behavior -->
7. WHEN `publish` runs on a manifest in state `validated` THEN the CLI SHALL transition it to `published`, exit 0, and leave every field covered by `manifest_sha256` unchanged. <!-- event-driven -->
8. IF `publish` runs on a manifest that is already `published` THEN the CLI SHALL exit 3 and leave the manifest file byte-identical. <!-- unwanted-behavior -->
9. WHILE a manifest is `published` the CLI SHALL refuse every content modification; only `retire` SHALL change it, and only lifecycle fields (`state`, `state_changed_at`, `retire_reason`). <!-- state-driven -->
10. WHEN `retire` runs on a manifest in state `published` or `validated` THEN the CLI SHALL set `state: "retired"` (recording the optional reason) and exit 0 without deleting any file. <!-- event-driven -->
11. IF a CLI command requests an invalid state transition (including retiring an already `retired` manifest) THEN the CLI SHALL exit 5 and modify nothing. <!-- unwanted-behavior -->
12. The manifest payload SHALL carry `manifest_sha256`, computed over canonical JSON of the payload excluding `manifest_sha256`, `state`, `state_changed_at`, and `retire_reason`. <!-- ubiquitous -->
13. IF a manifest declares `manifest_sha256` and its recomputed canonical hash differs THEN `validate_manifest` and CLI `validate` SHALL reject it. <!-- unwanted-behavior -->
14. The CLI SHALL perform no network access during any command. <!-- ubiquitous -->
15. WHEN any command writes a manifest file THEN the write SHALL be atomic (temporary file plus `os.replace`) so no partial manifest is observable. <!-- event-driven -->
16. WHERE a manifest has no `state` field (legacy) the CLI SHALL treat it as `draft` for transitions and backfill `manifest_sha256` on its first transition. <!-- optional-feature -->
17. The registry root SHALL contain only manifest `*.json` files (plus the tolerated `metadata.json`); snapshot data SHALL live in subdirectories, and no loose state or index JSON SHALL be written at the root. <!-- ubiquitous -->

**Independent Test**: Run the four commands against a synthetic fixture registry in `tempfile` and observe exit codes 0/3/4/5 as specified; re-run `build` twice and `diff` the outputs.

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

1. WHEN `build` or `validate` completes THEN the CLI SHALL report, per supported class, the resolved series path, its `sha256`, and its provider, derived from the manifest sources. <!-- event-driven -->
2. The mapping SHALL be deterministic across runs on the same manifest and SHALL resolve `fiis` to `caio_fiis.csv` with no fallback. <!-- ubiquitous -->
3. The registry documentation SHALL state the registry layout, every command, the exit code table, and the prohibitions (no network, no overwrite of published artifacts, no deletion on `retire`, no series fallback). <!-- ubiquitous -->

**Independent Test**: Run `validate --json` on a fixture manifest and compare the printed class-to-series map to the manifest sources.

---

## Edge Cases

- IF the registry directory does not exist THEN the CLI SHALL exit 3 with a message naming the missing path.
- IF `--created-at` is missing or not an ISO datetime THEN `build` SHALL exit 2 and write nothing.
- IF a manifest declares an unknown `state` string THEN loading SHALL fail with a validation error (never a silent default).
- IF the manifest JSON is modified after `validated` (fields covered by `manifest_sha256`) THEN the next `validate` SHALL exit 4.
- WHEN `validate` succeeds on a legacy manifest THEN only `state`, `state_changed_at`, and `manifest_sha256` SHALL change in the file (unknown fields preserved).
- IF two required series declare overlapping but different date coverage THEN `build` SHALL fail rather than narrowing `common_dates` silently below full coverage of the required series.

## Implicit-Requirement Dimension Sweep

| Dimension | Resolution |
| --------- | ---------- |
| Input validation & bounds | SNAP-03/SNAP-04: CLI arg formats (ISO datetime/date), manifest schema validation via `validate_manifest`; unknown `state` rejected at load |
| Failure / partial-failure states | SNAP-02/SNAP-03: atomic tmp + `os.replace` writes; any failure leaves files unmodified |
| Idempotency / retry / duplicate handling | SNAP-04 (validate idempotent on `validated`), SNAP-05 (double publish refused, exit 3), SNAP-03 (build no-overwrite) |
| Auth boundaries & rate limits | N/A because the CLI is a single-operator offline tool with no auth surface or remote calls |
| Concurrency / ordering | Atomic writes prevent torn manifests; concurrent writers on the same manifest are outside the contract and documented as prohibited (single-operator CLI) |
| Data lifecycle / expiry | SNAP-06: `retire` is state-only and never deletes; erasure/retention policy is explicitly out of scope |
| Observability | Exit codes 0/2/3/4/5 plus stderr diagnostics and `--json` machine-readable output (SNAP-03..SNAP-06, SNAP-09) |
| External-dependency failure | N/A because the CLI is offline by requirement (SNAP-10 prohibition) and touches only the local filesystem |
| State-transition integrity | SNAP-01: explicit transition table with terminal `retired`; invalid transitions exit 5 |

---

## Requirement Traceability

| Requirement ID | Story | Phase | Status |
| -------------- | ----- | ----- | ------ |
| SNAP-01 | P1: Lifecycle CLI | Design | In Design |
| SNAP-02 | P1: Lifecycle CLI | Design | In Design |
| SNAP-03 | P1: Lifecycle CLI | Design | In Design |
| SNAP-04 | P1: Lifecycle CLI | Design | In Design |
| SNAP-05 | P1: Lifecycle CLI | Design | In Design |
| SNAP-06 | P1: Lifecycle CLI | Design | In Design |
| SNAP-07 | P1: Lifecycle CLI | Design | In Design |
| SNAP-08 | P2: State-aware selection | Design | In Design |
| SNAP-09 | P3: Class-to-series mapping | Design | In Design |
| SNAP-10 | P3: Class-to-series mapping | Design | In Design |

**ID format:** `[CATEGORY]-[NUMBER]` (e.g., `SNAP-01`)

**Coverage:** 10 total, 10 mapped to tasks (see `tasks.md`), 0 unmapped.

### Issue #16 Traceability (ACs and tests of the issue)

| Issue item | Requirement | Task | Concrete test |
| ---------- | ----------- | ---- | ------------- |
| AC1 "Snapshot publicado é imutável e auditável por hash" | SNAP-02, SNAP-05 | T2, T6 | `tests/test_snapshot_registry.py::test_manifest_hash_survives_transitions`, `::test_publish_never_overwrites_published_manifest` |
| AC2 "Validação falha se data ou hash divergirem" | SNAP-04 | T5 | `tests/test_snapshot_registry.py::test_validate_rejects_divergent_source_hash`, `::test_validate_rejects_cutoff_date_divergence` |
| AC3 "CLI documentada e reproduzível" | SNAP-03, SNAP-10 | T4, T9 | `tests/test_snapshot_registry.py::test_build_is_byte_reproducible` (+ doc review gate) |
| Test IT-1 "Hash divergente rejeitado" | SNAP-04 | T5 | New: `tests/test_snapshot_registry.py::test_validate_rejects_divergent_source_hash`; existing library-level: `tests/test_snapshot_manifest.py::test_changed_source_hash_is_rejected` |
| Test IT-2 "Manifest sem caio_fiis.csv falha, sem fallback" | SNAP-04, SNAP-09 | T5, T8 | New: `tests/test_snapshot_registry.py::test_validate_requires_caio_fiis_without_ifix_fallback`; existing loader-level: `tests/test_allocation_data.py::test_missing_premium_fii_input_does_not_use_ifix` |
| Test IT-3 "Snapshot publicado não pode ser sobrescrito" | SNAP-05 | T6 | `tests/test_snapshot_registry.py::test_publish_never_overwrites_published_manifest` |

---

## Success Criteria

- [ ] All four CLI commands behave per the exit code table against synthetic fixtures, offline, in a clean checkout with only the stdlib.
- [ ] `manifest_sha256` of a fixture manifest is unchanged after `validate`, `publish`, and `retire`.
- [ ] Legacy fixture manifests (no `state`) produce identical selection results before and after the change.
- [ ] The full stdlib test subset passes (`tests/test_snapshot_registry.py`, `tests/test_snapshot_manifest.py`, `tests/test_allocation_data.py`, `tests/test_allocation.py`, `tests/test_fetch_status_invest*.py`).
