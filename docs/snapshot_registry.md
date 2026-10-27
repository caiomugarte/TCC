# Snapshot Registry & Lifecycle CLI

Offline lifecycle tool for immutable snapshot manifests (issue #16, P0-09;
spec `.specs/features/snapshot-registry-lifecycle/spec.md`). One module drives
the whole lifecycle: `py/snapshot_registry.py` with the subcommands `build`,
`validate`, `publish`, and `retire`. The CLI is a **single-operator** tool: it
has **no network** access, touches only the local filesystem, and never
substitutes a fallback series.

## Registry layout

```
<registry>/                       # e.g. <repo>/snapshots
  <manifest_id>.json              # manifest files ONLY at the root
  metadata.json                   # tolerated legacy skip in SnapshotRegistry.manifests()
  data/<snapshot_name>/           # CSVs + metadata.json written by py/fetch_allocation_snapshot.py
  stocks.csv, fiis.csv            # explicit selector universes (any path inside the root)
```

- The root carries only manifest `*.json` files (plus the tolerated
  `metadata.json`): `SnapshotRegistry.manifests()` globs `*.json` at the root,
  so no state, index, or audit JSON may ever be written there. Snapshot data
  lives in subdirectories.
- All source paths inside a manifest are registry-root-relative POSIX paths
  (the registry root is the manifest's directory).
- Every manifest write is atomic: a temporary file is created in the target's
  own directory and moved into place with `os.replace`, so no partial manifest
  is ever observable. A failing `os.replace` leaves the original file
  byte-identical.

## Manifest form

`build` writes exactly the `SnapshotManifest.as_dict()` projection: one
`allocation` source carrying `metadata.files`, explicit `stock`/`fii` selector
sources, and the `allocation_source`/`stock_source`/`fii_source` aliases. Plus
the optional lifecycle envelope:

```jsonc
{
  "manifest_id": "p0-09-2026-07-21",
  "manifest_version": "1",
  "created_at": "2026-10-27T12:00:00+00:00",   // REQUIRED at build (--created-at)
  "cutoff_date": "2026-07-21",
  "common_dates": ["..."],                      // sorted, unique, last == cutoff_date
  "supported_classes": ["brazilian_stocks", "fiis", "international_equity", "fixed_income", "crypto"],
  "sources": { "…": { "path": "...", "sha256": "…", "provider": "…", "metadata": {} } },
  "state": "published",                         // draft | validated | published | retired
  "state_changed_at": "2026-10-27T12:05:00+00:00",
  "retire_reason": null,                        // only set on retired
  "manifest_sha256": "…"                        // canonical content hash (below)
}
```

- `state`, `state_changed_at`, `retire_reason`, and `manifest_sha256` are
  optional; a manifest without `state` is a **legacy** artifact and stays fully
  usable (see "Legacy compatibility"). An unknown `state` string is rejected at
  load - never silently defaulted.
- `supported_classes` is derived from `ASSET_CLASSES`
  (`py/allocation_config.py`) in that fixed order; the order is pinned by
  `build`, not by `as_dict()`.
- Provenance honesty: each source's `provider` is a fixed literal per role
  (`snapshot-build:allocation`, `snapshot-build:selector`) and `metadata` is
  exactly `{"files": {...}}` (`allocation`) or `{}` (selectors). The fetch
  `metadata.json` content is never copied into a manifest (it carries absolute
  paths that would break byte-reproducibility).

## Canonical manifest hash

`manifest_sha256` = SHA-256 over the canonical JSON
(`sort_keys=True`, `separators=(",", ":")`, UTF-8) of the normalized projection
`SnapshotManifest.from_dict(raw).as_dict()` minus `manifest_sha256`, `state`,
`state_changed_at`, and `retire_reason`. The single canonicalization point is
`manifest_canonical_sha256` in `py/snapshot_manifest.py`; `validate_manifest`
recomputes and rejects any divergence. Consequences:

- The hash is constant across the whole lifecycle (`build` → `retire`) and
  recomputes identically for file loads, dataclass instances, and persisted
  provenance with `manifest_path` injected.
- `state` tampering and unknown keys outside the `as_dict()` projection are
  not hash-covered - guarded by the transition table, the selection semantics,
  and this documentation. Envelope rewrites still preserve unknown keys in the
  file byte-for-byte.
- Legacy manifests without `manifest_sha256` get it backfilled on their first
  lifecycle transition.

## Commands

```
python3 py/snapshot_registry.py build    --registry DIR --manifest-id ID --created-at ISO
                                         --snapshot-dir DIR --stock-universe CSV --fii-universe CSV
python3 py/snapshot_registry.py validate --registry DIR --manifest-id ID [--check] [--json]
python3 py/snapshot_registry.py publish  --registry DIR --manifest-id ID
python3 py/snapshot_registry.py retire   --registry DIR --manifest-id ID [--reason TEXT]
```

Common flags: `--registry` (registry root directory), `--manifest-id` (must
match `^[A-Za-z0-9][A-Za-z0-9._-]*$`, never `metadata`, and must equal the
loaded `manifest_id`; violations are exit 2).

### `build`

Compose a `draft` manifest from local fetch output. `--snapshot-dir` (the
`data/<snapshot_name>/` directory with the six CSVs), `--stock-universe`, and
`--fii-universe` must all resolve INSIDE the registry root (escape after
resolution is exit 2); `--created-at` is a required explicit ISO datetime.
`--snapshot-dir` must also NOT be an ancestor of (or equal to) the destination
manifest path - in particular `--snapshot-dir <registry>` itself is exit 2 with
nothing written: the manifest would land inside the very tree whose
`sha256_path` digest the manifest records, and `validate` could never pass. The
documented layout `<registry>/data/<snapshot_name>/` is fine because it is not
an ancestor of `<registry>/<manifest-id>.json`.
`common_dates` is the sorted intersection of the five class series exactly as
`load_premium_snapshot_bundle` consumes them (`international_equity`/`crypto`
already multiplied by `ptax`, multi-ticker sleeves intersected like
`levels_to_returns`), keeping the first date with zero return;
`cutoff_date` is the last date. The intersection is computed by reusing
`py/allocation_data.py` (`read_levels_csv`, `multiply_levels`,
`levels_to_returns`) - never reimplemented. The same `manifest_id` built from
identical inputs and the same `--created-at` in two different registries yields
byte-identical manifest files. A target that already exists is exit 3 and is
never modified; a source input missing on disk is exit 3 with no series
substitution (`caio_fiis.csv` is required; `ifix.csv` is never used in its
place).

### `validate`

Run the full verification set (required classes and selector sources, every
source hash, `manifest_sha256` recomputation, `cutoff_date` coherence) and on
success transition `draft` → `validated` (legacy: backfill `manifest_sha256`).
Re-running on `validated` exits 0 with no write. On `published`/`retired` the
state gate fires first and the command exits 5 without recomputing anything.
Any required-source verification failure - a required series absent from the
manifest OR its file/derived artifact missing or unreadable on disk - is exit 4
with zero writes and no fallback.

- `validate --check`: read-only audit in ANY lifecycle state (the full
  verification set minus the lifecycle-state gate; matching hashes is
  necessary, not sufficient) with 0/4 semantics and zero writes.
- `validate --json`: print the deterministic class-to-series map (below) to
  stdout (status chatter moves to stderr), then proceed with normal `validate`
  semantics.
- `validate --check --json`: the same map under `--check` semantics (0/4, zero
  writes) - the only way to print the map for a `published`/`retired`
  manifest.

### `publish`

Transition `validated` → `published` via an envelope-only rewrite; content
fields covered by `manifest_sha256` never change. An already-published
manifest is exit 3 and the file stays byte-identical: the CLI never overwrites
a published artifact (never overwrite semantics). Any other source state is
exit 5.

### `retire`

Transition `published`|`validated` → `retired`, recording the optional
`--reason`. Terminal: re-retiring is exit 5, and retiring a `draft` is exit 5.
`retire` never deletes anything - the CLI never deletes files (manifest and
data stay on disk) - and never rewrites non-envelope fields.

## Exit codes

| Code | Meaning |
| ---- | ------- |
| exit 0 | success |
| exit 2 | usage/argument error: bad or missing `--created-at`, `--manifest-id` failing `^[A-Za-z0-9][A-Za-z0-9._-]*$` or equal to `metadata` or differing from the loaded `manifest_id`, a `build` input escaping the registry root, a `build` `--snapshot-dir` containing the destination manifest path, unknown flag |
| exit 3 | the operator pointed at something that does not exist: registry directory or `<registry>/<manifest-id>.json` missing on disk, refused overwrite of an existing/published artifact, or a `build` source input missing on disk |
| exit 4 | the artifact under audit does not hold up: hash divergence, `cutoff_date` != last `common_date`, empty `common_dates` intersection at `build`, or any required-source verification failure in `validate`/`validate --check` (series absent from the manifest OR file missing/unreadable on disk - after `publish` the data files are part of the audited artifact) |
| exit 5 | invalid lifecycle transition |

The 3/4 split: a missing audited source file is exit 4 (tampering), while exit 3
stays reserved for missing CLI targets, refused overwrite, and `build` inputs.

## State machine

| From \ To | draft | validated | published | retired |
| --------- | ----- | --------- | --------- | ------- |
| draft | - | `validate` | exit 5 | exit 5 |
| validated | exit 5 | `validate` = no-op, exit 0 | `publish` | `retire` |
| published | exit 5 | exit 5 | `publish` = exit 3 (never overwrite) | `retire` |
| retired | exit 5 | exit 5 | exit 5 | exit 5 (re-retire) |
| *(absent / legacy)* | - | `validate` (backfills `manifest_sha256`) | via `validate` then `publish` | via `validate` → `retire` (or `validate` → `publish` → `retire`) |

All transitions rewrite only `state`, `state_changed_at`, `retire_reason` (and
the backfilled `manifest_sha256`); fields covered by the content hash never
change.

## Class-to-series map (auditability)

`validate --json` / `validate --check --json` print one entry per supported
class plus `ptax` and the explicit `stock`/`fii` selector sources - non-class
files that `py/fetch_allocation_snapshot.py` always writes (`ifix.csv`,
`ibovespa.csv`) are never mapped:

```json
{
  "brazilian_stocks": {"path": "data/snapshot-2020/caio_stocks.csv", "sha256": "…", "provider": "snapshot-build:allocation", "derived_from": "allocation"},
  "fiis": {"path": "data/snapshot-2020/caio_fiis.csv", "sha256": "…", "provider": "snapshot-build:allocation", "derived_from": "allocation"},
  "ptax": {"path": "data/snapshot-2020/ptax.csv", "sha256": "…", "provider": "snapshot-build:allocation", "derived_from": "allocation"},
  "stock": {"path": "stocks.csv", "sha256": "…", "provider": "snapshot-build:selector"}
}
```

`path` is registry-root-relative POSIX even for derived series and `sha256` is
the `file_sha256` of the series FILE itself - never the `allocation` directory
digest. The map is deterministic and byte-stable across runs on the same
manifest. `fiis` resolves to `caio_fiis.csv` with no fallback.

## Prohibitions

- **no network** access during any command (verified by a stdlib source scan,
  `test_cli_source_has_no_network_imports`).
- A published artifact is immutable: the CLI never overwrites it (never
  overwrite), and `publish` twice is exit 3 with a byte-identical file.
- `retire` never deletes files - it is state-only; erasure policy is a separate
  decision.
- No series fallback (no IFIX substitution, ever): a required class series is
  used exactly as declared or the command fails.
- Single-operator writes: concurrent writers on the same manifest are
  prohibited; atomic `os.replace` prevents torn files.

## Legacy compatibility

Manifests without `state`/`manifest_sha256` load, round-trip, and are selected
exactly as before. For transitions a legacy manifest behaves as `draft`
(`validate` upgrades it and backfills the hash); for selection it stays
selectable. `SnapshotRegistry.latest_compatible` never selects `draft` or
`retired`, selects `validated` only with `include_validated=True`, and raises
when no candidate qualifies - it never falls back to an older artifact.

## Walkthrough (manual)

```
python3 py/snapshot_registry.py build    --registry ./snapshots --manifest-id demo-2020 \
    --created-at 2020-01-04T12:00:00+00:00 --snapshot-dir ./snapshots/data/demo \
    --stock-universe ./snapshots/stocks.csv --fii-universe ./snapshots/fiis.csv
python3 py/snapshot_registry.py validate --registry ./snapshots --manifest-id demo-2020
python3 py/snapshot_registry.py publish  --registry ./snapshots --manifest-id demo-2020
python3 py/snapshot_registry.py retire   --registry ./snapshots --manifest-id demo-2020 --reason "superseded"
```

Each step prints its outcome and exits 0; `validate --check --json` works at
any point for a read-only audit. Reader walkthrough of this sequence from the
doc alone is a manual check (explicitly outside the automated gate).
