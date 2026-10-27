# STATE

## Decisions

### AD-001
- **Decision**: Snapshot registry layout is fixed: manifest files are the only `*.json` at the registry root (`<registry>/<manifest_id>.json`, plus the loader-tolerated `metadata.json`); snapshot data lives in subdirectories; lifecycle state lives inside the manifest envelope - never in loose state/index files at the root.
- **Reason**: `SnapshotRegistry.manifests()` globs `*.json` at the registry root and skips only `metadata.json`, so any loose JSON there would be parsed as a manifest and break `api/app/services/premium_recommendation.py` selection; a single source of truth also matches the issue wording "manifest legado sem `state`".
- **Trade-off**: `retire` rewrites the manifest file (envelope fields only), so immutability is defined over content fields, not over file bytes.
- **Scope**: `py/snapshot_manifest.py`, `py/snapshot_registry.py`, `snapshots/` consumers (Premium API), future snapshot features (#11, #15).
- **Date**: 2026-10-27
- **Status**: active

### AD-002
- **Decision**: The manifest audit hash `manifest_sha256` is SHA-256 over canonical JSON (`sort_keys=True`, `separators=(",", ":")`, UTF-8) of the NORMALIZED manifest projection `SnapshotManifest.from_dict(raw).as_dict()` minus `manifest_sha256` itself and the lifecycle envelope (`state`, `state_changed_at`, `retire_reason`); `created_at` is included, `build` requires an explicit `--created-at`, and the canonicalization lives in one helper (`manifest_canonical_sha256`).
- **Reason**: The hash must be recomputable wherever the manifest is rehydrated. The Premium path (`api/app/services/premium_executor.py:158` → `validate_manifest`, fed by `api/app/services/premium_recommendation.py:87-88` which persists `as_dict()` with `manifest_path` injected) only sees the dataclass, so a hash over raw JSON bytes would reject every rehydrated run; hashing the normalized projection recomputes identically for file loads, dataclass instances, and persisted provenance (review finding F1).
- **Trade-off**: Tampering with `state`, or with keys outside the `as_dict()` projection (unknown extras in legacy files), is not hash-detected - guarded by the CLI transition table, selection semantics, and documentation; operators must supply `--created-at` explicitly.
- **Scope**: `py/snapshot_manifest.py` hash/validation surface, `py/snapshot_registry.py` CLI, audit tooling over manifests.
- **Date**: 2026-10-27 (revised in spec review round 2)
- **Status**: active

### AD-003
- **Decision**: Registry selection semantics by lifecycle state: `draft`/`retired` are never selectable; `validated` is selectable only with an explicit `include_validated=True`; manifests without `state` (legacy) remain selectable and produce identical results to the pre-change implementation.
- **Reason**: Requirement of issue #16 (P0-09) plus strict backward compatibility for existing manifests consumed through `latest_compatible_manifest`.
- **Trade-off**: A legacy manifest is treated as `draft` for CLI transitions but as selectable for API runs until an operator runs `validate` - a deliberate split to avoid breaking live selection.
- **Scope**: `SnapshotRegistry.latest_compatible` / `latest_compatible_manifest`, `api/app/services/premium_recommendation.py`.
- **Date**: 2026-10-27
- **Status**: active

### AD-004
- **Decision**: The manifest emit/canonical on-disk form is fixed: `build` writes exactly the `SnapshotManifest.as_dict()` projection (single `allocation` source with `metadata.files`, explicit `stock`/`fii` selector sources and their `*_source` aliases); every manifest write uses `json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)` + trailing `\n`; source paths are registry-root-relative POSIX paths and every `build` input must resolve under `<registry>/data/` (exit 2 otherwise); byte-reproducibility is proven by building the same `manifest_id` in two distinct registries and comparing bytes.
- **Reason**: `SnapshotSource.resolved_path` resolves relative to the manifest's directory, so a path recorded outside the registry produces a manifest that fails its own `validate`; the consumer contract (`py/allocation_data.py:238-250`, fixture `tests/test_snapshot_manifest.py:54-68`) expects the `allocation` + `metadata.files` shape; a repeated `build` in one registry is exit 3 by design (immutability), so the reproducibility proof needs two registries (review findings F4, F5).
- **Trade-off**: Operators must place inputs under `<registry>/data/` before building; envelope rewrites preserve unknown keys but those are not covered by `manifest_sha256`.
- **Scope**: `py/snapshot_registry.py` (`build`, `write_manifest_atomic`), `docs/snapshot_registry.md`, fixture strategy for tests.
- **Date**: 2026-10-27
- **Status**: active

## Handoff

- **Feature**: snapshot-registry-lifecycle (Issue #16 / P0-09) - `.specs/features/snapshot-registry-lifecycle/`
- **Phase / Task**: SPEC stage complete, including the round-2 revision after independent review (`judge_spec` → `CHANGES_REQUIRED`, findings F1-F15 resolved). IMPLEMENT not started.
- **Completed**: spec/design/tasks authored at `cf4e8ad` and revised for review findings; gates green: `validate_spec.py`, `validate_tasks.py`, `check_commit.py` (skill scripts at `/home/caio/.agents/skills/tlc-spec-driven/scripts/...` - NOT in this repo) and the stdlib quick-gate baseline (`tests.test_snapshot_manifest`, `tests.test_allocation_data`).
- **In-progress**: none
- **Next step**: IMPLEMENT stage executes tasks T1-T9 per `tasks.md` (suggested batches: A = T1-T7, B = T8-T9); start with T1 (lifecycle envelope fields in `py/snapshot_manifest.py` + creation of `tests/test_snapshot_registry.py`).
- **Blockers**: none blocking; constraints: no real `caio_fiis.csv` data (fixture-only proofs), `pytest`/`pandas` absent (stdlib `unittest` only), open #11/#15 forbid touching `ALLOCATION_FILES`, per-class metrics, or test infra.
- **Uncommitted files**: none - the four spec artifacts are committed on `t3/issue-16` (`cf4e8ad` = spec stage; the follow-up commit carries this review revision).
- **Branch**: t3/issue-16
