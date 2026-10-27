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
- **Reason**: The hash must be recomputable wherever the manifest is rehydrated. The Premium path (`api/app/services/premium_executor.py:162` → `SnapshotManifest.from_dict(manifest_data, base_dir=…)` → `validate_manifest`, fed by `api/app/services/premium_recommendation.py:87-88` which persists `as_dict()` with `manifest_path` injected) only sees the dataclass, so a hash over raw JSON bytes would reject every rehydrated run; hashing the `as_dict()` projection recomputes identically for file loads, dataclass instances, and persisted provenance (review finding F1).
- **Trade-off**: Tampering with `state`, or with keys outside the `as_dict()` projection (unknown extras in legacy files), is not hash-detected - guarded by the CLI transition table, selection semantics, and documentation; operators must supply `--created-at` explicitly. "Normalized" means the schema projection only: `as_dict()` does not sort `supported_classes` - its order determinism comes from `build` deriving it from `ASSET_CLASSES` (AD-004).
- **Scope**: `py/snapshot_manifest.py` hash/validation surface, `py/snapshot_registry.py` CLI, audit tooling over manifests.
- **Date**: 2026-10-27 (revised spec review round 2; round 3 fixes the `premium_executor.py:162` citation and the `as_dict()`/`supported_classes` wording)
- **Status**: active

### AD-003
- **Decision**: Registry selection semantics by lifecycle state: `draft`/`retired` are never selectable; `validated` is selectable only with an explicit `include_validated=True`; manifests without `state` (legacy) remain selectable and produce identical results to the pre-change implementation.
- **Reason**: Requirement of issue #16 (P0-09) plus strict backward compatibility for existing manifests consumed through `latest_compatible_manifest`.
- **Trade-off**: A legacy manifest is treated as `draft` for CLI transitions but as selectable for API runs until an operator runs `validate` - a deliberate split to avoid breaking live selection.
- **Scope**: `SnapshotRegistry.latest_compatible` / `latest_compatible_manifest`, `api/app/services/premium_recommendation.py`.
- **Date**: 2026-10-27
- **Status**: active

### AD-004
- **Decision**: The manifest emit/canonical on-disk form is fixed: `build` writes exactly the `SnapshotManifest.as_dict()` projection (single `allocation` source with `metadata.files`, explicit `stock`/`fii` selector sources and their `*_source` aliases); every manifest write uses `json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)` + trailing `\n`; `supported_classes` is derived from `ASSET_CLASSES` (`py/allocation_config.py:14-20`) in fixed order; each source's `provider` is a fixed literal per source role and `metadata` is exactly `{"files": {...}}` (`allocation`) or `{}` (selector sources) - fetch `metadata.json` content is never copied into the manifest; source paths are registry-root-relative POSIX paths and every `build` input must resolve INSIDE the registry root (escape after `Path.resolve()` ⇒ exit 2; `data/<snapshot_name>/` is the conventional layout); byte-reproducibility is proven by building the same `manifest_id` in two distinct registries and comparing bytes.
- **Reason**: `SnapshotSource.resolved_path` resolves relative to the manifest's directory, so a path recorded outside the registry produces a manifest that fails its own `validate` (the necessary invariant is "inside the registry root", not "under `data/`" - review nit); the consumer contract (`py/allocation_data.py:238-250`, fixture `tests/test_snapshot_manifest.py:54-68`) expects the `allocation` + `metadata.files` shape; fetch `metadata.json` carries absolute paths (`portfolio_path`, `portfolio_artifacts` - `py/fetch_allocation_snapshot.py:538,584-590`) that would break AC3 byte-reproducibility across registries (review N5); `as_dict()` does not sort `supported_classes`, so order determinism lives here (review nit); a repeated `build` in one registry is exit 3 by design (immutability), so the reproducibility proof needs two registries (review findings F4, F5).
- **Trade-off**: Operators must place inputs inside the registry root (conventionally `data/<snapshot_name>/`) before building; envelope rewrites preserve unknown keys but those are not covered by `manifest_sha256`; fetch-side provenance beyond `metadata.files` stays in the snapshot directory's own `metadata.json` and is not part of the manifest.
- **Scope**: `py/snapshot_registry.py` (`build`, `write_manifest_atomic`), `docs/snapshot_registry.md`, fixture strategy for tests.
- **Date**: 2026-10-27 (revised spec review round 3)
- **Status**: active

### AD-005
- **Decision**: Validation exit-code and audit-path contract: every required-source verification failure of a manifest under audit - a required series absent from the manifest OR its file/derived artifact missing or unreadable on disk - exits **4** in `validate` AND `validate --check` (review ruling N1); exit **3** is reserved for missing CLI targets (registry dir, `<registry>/<manifest-id>.json`), refused overwrite, and `build` source inputs (AC7). `validate --check` runs the FULL `validate_manifest` verification set minus the lifecycle-state gate (hashes matching is necessary, not sufficient) and combines with `--json` (0/4 semantics, zero writes), which is the only way to print the class-to-series map in `published`/`retired` states.
- **Reason**: 3 = "the operator pointed at something that does not exist"; 4 = "the artifact under audit does not hold up" - after `publish` the data files are part of the audited artifact, so losing them is tampering (coherent with AC16 = 0/4 and issue IT-2, whose fixture unlinks `caio_fiis.csv` from disk); a `--check` that verifies less than `validate` could report a green audit on an invalid manifest (review N2); P3 promises review of a *published* snapshot's inputs, which plain `validate` refuses with exit 5 (review N3).
- **Trade-off**: The 3/4 split is no longer "missing file vs contract divergence" at the file level - a missing audited source file is exit 4, which operators must learn from the docs; `--check` becomes as strict as `validate` for reads.
- **Scope**: `py/snapshot_registry.py` (`validate`, `validate --check`, `validate --json`), `docs/snapshot_registry.md` exit table.
- **Date**: 2026-10-27
- **Status**: active

### AD-006
- **Decision**: Provenance honesty in emitted and reported data: `class_series_map` reports each class's `file_sha256` (the digest of the series FILE itself) and a registry-root-relative POSIX `path`, adding `derived_from: "allocation"` for derived series - it must never expose the `allocation` directory digest (`source_for` returns `sha256=allocation.sha256` + absolute `path` for derived series, `py/snapshot_manifest.py:312-318`) as a file hash; `build` fixes `provider` as a literal constant per source role and `metadata` to `{"files": {...}}`/`{}`, never copying fetch `metadata.json`.
- **Reason**: An audit artifact must say what it means: labeling a directory digest as a series' `sha256` is verifiable but misleading (review N4), and absolute paths or fetch-side metadata make two registries produce different manifest bytes, breaking AC3 reproducibility (review N5).
- **Trade-off**: The map's `sha256` no longer coincides with the `allocation` source hash the manifest stores; auditors cross-check derived files via the allocation directory hash recorded in `sources.allocation` (kept authoritative in `validate_manifest`, `py/snapshot_manifest.py:425-434`).
- **Scope**: `py/snapshot_registry.py` (`class_series_map`, `build`), `docs/snapshot_registry.md`, P3 contract.
- **Date**: 2026-10-27
- **Status**: active

## Handoff

- **Feature**: snapshot-registry-lifecycle (Issue #16 / P0-09) - `.specs/features/snapshot-registry-lifecycle/`
- **Phase / Task**: SPEC stage complete, through independent review round 3 (`judge_spec` round 2 → `CHANGES_REQUIRED` on N1-N7 + nits; all closed in the round-3 revision). IMPLEMENT not started.
- **Completed**: spec/design/tasks authored at `cf4e8ad`, review findings F1-F15 resolved at `b736875`, review findings N1-N7 + nits resolved in the follow-up commit (exit-code contract ruling N1 = AD-005, `--check` full-set + `--check --json` = AD-005, provenance literals + map honesty = AD-004/AD-006). Gates green: `validate_spec.py`, `validate_tasks.py`, `check_commit.py` (skill scripts at `/home/caio/.agents/skills/tlc-spec-driven/scripts/...` - NOT in this repo) and the stdlib quick-gate baseline (`tests.test_snapshot_manifest`, `tests.test_allocation_data`, 9 tests OK).
- **In-progress**: none
- **Next step**: IMPLEMENT stage executes tasks T1-T9 per `tasks.md` (suggested batches: A = T1-T7, B = T8-T9); start with T1 (lifecycle envelope fields in `py/snapshot_manifest.py` + creation of `tests/test_snapshot_registry.py`); honor AD-005 (all required-source failures in `validate`/`--check` = exit 4) and AD-006 (file-level `sha256` + registry-relative paths in `class_series_map`; fixed `provider`/`metadata` literals in `build`).
- **Blockers**: none blocking; constraints: no real `caio_fiis.csv` data (fixture-only proofs), `pytest`/`pandas` absent (stdlib `unittest` only), open #11/#15 forbid touching `ALLOCATION_FILES`, per-class metrics, or test infra.
- **Uncommitted files**: none - the four spec artifacts are committed on `t3/issue-16` (`cf4e8ad` = spec stage, `b736875` = review round 2; the follow-up commit carries this round-3 revision).
- **Branch**: t3/issue-16
