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
- **Decision**: The manifest audit hash `manifest_sha256` is SHA-256 over canonical JSON (`sort_keys=True`, `separators=(",", ":")`, UTF-8) of the manifest payload excluding `manifest_sha256` itself and the lifecycle envelope (`state`, `state_changed_at`, `retire_reason`); `created_at` is included, and `build` requires an explicit `--created-at`.
- **Reason**: The hash must survive lifecycle transitions to serve as a constant content identity, must not be self-referential, and reproducibility is achieved by byte-identical rebuilds from identical inputs instead of weakening tamper evidence.
- **Trade-off**: Tampering with `state` alone is not hash-detected (guarded by the CLI transition table and selection semantics); operators must supply `--created-at` explicitly.
- **Scope**: `py/snapshot_manifest.py` hash/validation surface, `py/snapshot_registry.py` CLI, audit tooling over manifests.
- **Date**: 2026-10-27
- **Status**: active

### AD-003
- **Decision**: Registry selection semantics by lifecycle state: `draft`/`retired` are never selectable; `validated` is selectable only with an explicit `include_validated=True`; manifests without `state` (legacy) remain selectable and produce identical results to the pre-change implementation.
- **Reason**: Requirement of issue #16 (P0-09) plus strict backward compatibility for existing manifests consumed through `latest_compatible_manifest`.
- **Trade-off**: A legacy manifest is treated as `draft` for CLI transitions but as selectable for API runs until an operator runs `validate` - a deliberate split to avoid breaking live selection.
- **Scope**: `SnapshotRegistry.latest_compatible` / `latest_compatible_manifest`, `api/app/services/premium_recommendation.py`.
- **Date**: 2026-10-27
- **Status**: active

## Handoff

- **Feature**: snapshot-registry-lifecycle (Issue #16 / P0-09) - `.specs/features/snapshot-registry-lifecycle/`
- **Phase / Task**: SPEC stage complete (spec.md, design.md, tasks.md written; gates green). IMPLEMENT not started.
- **Completed**: spec/design/tasks authored and validated (`validate_spec.py`, `validate_tasks.py`, `check_commit.py` all exit 0).
- **In-progress**: none
- **Next step**: IMPLEMENT stage executes tasks T1-T9 per `tasks.md` (suggested batches: A = T1-T7, B = T8-T9); start with T1 (lifecycle envelope fields in `py/snapshot_manifest.py`).
- **Blockers**: none blocking; constraints: no real `caio_fiis.csv` data (fixture-only proofs), `pytest`/`pandas` absent (stdlib `unittest` only), open #11/#15 forbid touching `ALLOCATION_FILES`, per-class metrics, or test infra.
- **Uncommitted files**: `.specs/features/snapshot-registry-lifecycle/{spec,design,tasks}.md`, `.specs/STATE.md`
- **Branch**: t3/issue-16
