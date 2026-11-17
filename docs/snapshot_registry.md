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
- The one exception is a stock CSV built from CVM and B3 (issue #78) and passed
  with `--stock-provenance`: the `stock` source then has `provider: "cvm-b3"`
  and `metadata` = `{"as_of": <cutoff of the CSV>, "raw_files": {<kind>:
  {<cache file>: <sha256>}}}`, copied from the collector's provenance file.
  The kinds are `dfp_itr`, `cotahist`, `proventos`, `eventos`,
  `classificacao_setorial` and `cadastro` (FCA and the CVM cadastro). The
  metadata holds file names and hashes only, no paths.

## Class series

`metadata.files` maps each class to its file in the snapshot directory:

| Class | File | Series |
| ----- | ---- | ------ |
| `brazilian_stocks` | `ibovespa.csv` | IBOV, total-return |
| `fiis` | `ifix.csv` | IFIX, total-return |
| `international_equity` | `sp500_total_return_usd.csv` | S&P 500 total return, USD (converted with `ptax`) |
| `fixed_income` | `di.csv` | CDI/DI daily factor |
| `crypto` | `btc_usd.csv` | BTC/USD (converted with `ptax`) |
| `ptax` | `ptax.csv` | USD/BRL PTAX |

`brazilian_stocks` → `ibovespa.csv` (IBOV) and `fiis` → `ifix.csv` (IFIX) are
fixed: both are total-return indices read as single series, and no manifest
can point those two classes anywhere else. `validate_manifest` rejects a
manifest whose stock or FII series resolves to another file - through
`metadata.files`, the defaults, or an explicit per-class source - with
`source <class> must be <expected>; manifest declares <declared>`. A series
derived from `allocation` must also be declared by exactly that name and live
directly inside the hashed allocation directory (`../snap-old/ifix.csv`,
`sub/ifix.csv` and a symlink leaving the directory are rejected). Manifests
that name the fixed baskets `caio_stocks.csv`/`caio_fiis.csv` are therefore
invalid for Premium: `validate` exits 4 and Premium selection raises with no
fallback to an older manifest. The baskets stay in the fetch output for the
Basic loader only.

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
                                         [--stock-provenance JSON]
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
`load_premium_snapshot_bundle` consumes them (IBOV and IFIX as single series,
`international_equity`/`crypto` already multiplied by `ptax`, intersected like
`levels_to_returns`), keeping the first date with zero return;
`cutoff_date` is the last date. The intersection is computed by reusing
`py/allocation_data.py` (`read_levels_csv`, `multiply_levels`,
`levels_to_returns`) - never reimplemented. The same `manifest_id` built from
identical inputs and the same `--created-at` in two different registries yields
byte-identical manifest files. A target that already exists is exit 3 and is
never modified. `build` requires the six `ALLOCATION_FILES` (`ibovespa.csv`,
`ifix.csv`, `sp500_total_return_usd.csv`, `di.csv`, `btc_usd.csv`, `ptax.csv`)
and no longer `caio_stocks.csv`/`caio_fiis.csv`: a missing one is exit 3, each
missing file named on stderr, with no series substitution; any `caio_*` file in
the directory is ignored. An `ibovespa.csv`/`ifix.csv` with more than one
series column and none named `value` is exit 4.
`--stock-provenance` names the `<stem>.provenance.json` that
`python3 py/pit_rebuild collect` writes next to the stock CSV. A missing file is
exit 3; a provenance whose `csv_sha256` is not the hash of `--stock-universe`,
or that lacks `provider`, `as_of`, `csv_sha256` or `raw_files`, is exit 4 with
nothing written.

### `validate`

Run the full verification set (required classes and selector sources, the
IBOV/IFIX series identity rule, every source hash, `manifest_sha256`
recomputation, `cutoff_date` coherence) and on
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
| exit 2 | usage/argument error: bad or missing `--created-at`, `--manifest-id` failing `^[A-Za-z0-9][A-Za-z0-9._-]*$` or equal to `metadata` or differing from the loaded `manifest_id`, a `build` input escaping the registry root, a `build` `--snapshot-dir` containing the destination manifest path, a registry that is not readable/writable (`OSError` mapped here - never a traceback), unknown flag |
| exit 3 | the operator pointed at something that does not exist: registry directory or `<registry>/<manifest-id>.json` missing on disk, refused overwrite of an existing/published artifact, or a `build` source input missing on disk |
| exit 4 | the artifact under audit does not hold up: hash divergence, `cutoff_date` != last `common_date`, empty `common_dates` intersection at `build`, a multi-series `ibovespa.csv`/`ifix.csv` at `build`, a stock or FII series that is not `ibovespa.csv`/`ifix.csv` in `validate`/`validate --check`, or any required-source verification failure in `validate`/`validate --check` (series absent from the manifest OR file missing/unreadable on disk - after `publish` the data files are part of the audited artifact) |
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
class plus `ptax` and the explicit `stock`/`fii` selector sources - files that
are not class inputs, such as the Basic baskets `caio_stocks.csv`/`caio_fiis.csv`
that `py/fetch_allocation_snapshot.py` still writes, do not appear:

```json
{
  "brazilian_stocks": {"path": "data/snapshot-2020/ibovespa.csv", "sha256": "…", "provider": "snapshot-build:allocation", "derived_from": "allocation"},
  "fiis": {"path": "data/snapshot-2020/ifix.csv", "sha256": "…", "provider": "snapshot-build:allocation", "derived_from": "allocation"},
  "ptax": {"path": "data/snapshot-2020/ptax.csv", "sha256": "…", "provider": "snapshot-build:allocation", "derived_from": "allocation"},
  "stock": {"path": "stocks.csv", "sha256": "…", "provider": "snapshot-build:selector"}
}
```

`path` is registry-root-relative POSIX even for derived series and `sha256` is
the `file_sha256` of the series FILE itself - never the `allocation` directory
digest. The map is deterministic and byte-stable across runs on the same
manifest. `brazilian_stocks` resolves to `ibovespa.csv` and `fiis` to
`ifix.csv`, with no fallback.

## Prohibitions

- **no network** access during any command (verified by a stdlib source scan,
  `test_cli_source_has_no_network_imports`).
- A published artifact is immutable: the CLI never overwrites it (never
  overwrite), and `publish` twice is exit 3 with a byte-identical file.
- `retire` never deletes files - it is state-only; erasure policy is a separate
  decision.
- No series fallback: a required class series is used exactly as declared or
  the command fails, and the stock and FII series can only be IBOV and IFIX.
- Single-operator writes: concurrent writers on the same manifest are
  prohibited; atomic `os.replace` prevents torn files. Trade-off: there is no
  OS-level exclusivity on the target (no `O_CREAT|O_EXCL` link/rename dance) -
  an existing target is refused via a `Path.exists()` check and the temporary
  file plus `os.replace` carries the atomicity. Writers racing on the SAME
  manifest are out of scope for this single-operator tool (last `os.replace`
  wins); published immutability is enforced by the transition table and the
  overwrite refusal, not by OS-level file locking.

## Legacy compatibility

Manifests without `state`/`manifest_sha256` load, round-trip, and are selected
exactly as before. For transitions a legacy manifest behaves as `draft`
(`validate` upgrades it and backfills the hash); for selection it stays
selectable. `SnapshotRegistry.latest_compatible` never selects `draft` or
`retired`, selects `validated` only with `include_validated=True`, and raises
when no candidate qualifies - it never falls back to an older artifact.

## Consumption requires `publish`

Fail-closed selection is deliberate: only `published` (and legacy) manifests
are consumable, so `publish` is a **prerequisite for consumption** - including
the Premium path (`api/app/services/premium_recommendation.py` loads via
`latest_compatible_manifest`, which defaults to `include_validated=False`).
`draft` and `validated` are pre-publication states and are NOT selectable by
default: after `build` + `validate` the registry still has no consumable
candidate until `publish` runs. The selection error is state-aware and
actionable: when compatible manifests exist but none is selectable it reports
the per-state counts and instructs the operator to publish, e.g.
`no compatible snapshot manifest found; registry has 2 validated manifest(s)
in non-selectable state(s); publish them to make them selectable`.

## Premium runs read only the snapshot

The Premium executor (`api/app/services/premium_executor.py`) makes no network
call. It loads the `stock` and `fii` selector CSVs named by the run's manifest
(`load_snapshot_inputs` in `api/app/services/status_invest_inputs.py`), checks
each SHA-256 against the manifest and against the `source_snapshot_hashes`
recorded in the run's policy, validates the CSV, and copies the verified bytes
into the run workspace before any calculation. Live collection (Status Invest,
COTAHIST) belongs to snapshot creation only, so a selector CSV must already be
filtered when it is registered. Failure codes:

| Cause | Code |
|---|---|
| hash divergence (manifest or recorded hash), invalid `manifest_sha256`, malformed selector CSV | `snapshot_changed` |
| manifest or source file missing | `snapshot_unavailable` |

The result's `provenance.selector_sources` keeps its shape (`provider`,
`retrieved_at`, `sha256`): `provider` is the manifest source's provider,
`retrieved_at` is the source's `metadata.retrieved_at` or else the manifest's
`created_at`, and `sha256` is the hash verified at read time.

## Replacing a `caio_*` manifest

A manifest built before the IBOV/IFIX series names `caio_stocks.csv`/
`caio_fiis.csv` and no longer validates. Manifests are immutable, so it is
replaced, never edited:

1. Prepare a NEW data directory `data/<new_name>/`: either re-fetch into it
   with `py/fetch_allocation_snapshot.py --snapshot-dir`, or copy the six
   `ALLOCATION_FILES` from the old directory into it together with a corrected
   `metadata.json` (old fetch metadata calls IFIX `benchmark-only` and
   Ibovespa a price index without dividends, and the Premium loader merges that
   file into the run's provenance). Never edit the old data directory
   in place: its directory hash is recorded by the old manifest, and
   changing any byte there breaks that manifest's hash.
2. `build` a new manifest from `data/<new_name>/`.
3. `validate` it.
4. `publish` it.
5. `retire` the old `caio_*` manifest (`validate --check` keeps reporting it
   as exit 4 in any state; retiring removes it from selection).

## Stock CSV from CVM and B3 (issue #78)

`python3 py/pit_rebuild collect --date D --out <registry>/data/<name>/stocks.csv
[--cache-dir DIR] [--offline]` builds the stock selector CSV of a snapshot from
CVM and B3, with the point-in-time rebuild of #59/#64/#77 at `d = D`
(`py/pit_rebuild/collector.py`). Without `--offline` it downloads the raw files
missing from the cache; with `--offline` it reads only the cache. Exit codes:
0 written, 1 output not written, 2 bad arguments or a forbidden directory,
3 cache directory absent in `--offline`, 4 a cached file differs from its hash
or a PIT leak, 5 a required raw file is missing.

- **Universe:** every COTAHIST cash-market ticker with a session in the 30 days
  up to `D` that maps to a CVM company with a usable filing. The tickers left
  out and their reasons are in the provenance file (`excluded`).
- **Columns:** those of the Status Invest export (`py/fetch_status_invest.py`
  `OUTPUT_COLUMNS`), in its scale (percent indicators in percent, money in BRL),
  plus `SUBSETOR`. `SETOR` and `SUBSETOR` are the B3 economic sector and
  subsector from GetDetail, as B3 writes them (B3 writes commas as periods:
  `Petróleo. Gás e Biocombustíveis`). A company without a classification has
  both empty.
- **Formulas:** the 19 rebuilt fields follow `py/pit_rebuild/indicators.py`,
  except `ROIC` = 100 × EBIT 12m × (1 − 0.34) ÷ (PL of the controllers + gross
  debt − cash), empty when the denominator is ≤ 0, and `PEG RATIO` = `P/L` ÷
  `CAGR LUCROS 5 ANOS`, empty when the CAGR is ≤ 0 or `P/L` is negative. The
  other Status Invest columns (`VPA`, `LPA`, `P/ATIVOS`, `P/EBIT`,
  `P/CAP. GIRO`, `P. AT CIR. LIQ.`, `MARGEM BRUTA`, `PATRIMONIO / ATIVOS`,
  `PASSIVOS / ATIVOS`, `GIRO ATIVOS`) come from the same filing. A filing older
  than 270 days leaves every accounting column empty.
- **Dividends:** B3 only. A company whose B3 history fails or is incomplete has
  `DY` empty, and the selector reads an empty `DY` as 0 (#52).
- **Financials:** banks and insurers are still computed like any company here;
  their own indicator set and peer group are issue #79.
- **`SUBSETOR` in the selector:** the selector does not read it, but its
  pre-processing treats every non-ID column as a metric candidate. With pandas 3
  the text column stays text and the robustness filter counts it as filled;
  with pandas 2 (`object` dtype) it becomes an all-NaN column and counts as
  missing. #79, which reads `SUBSETOR`, should keep it out of the metrics.

The provenance file `<stem>.provenance.json` goes next to the CSV; pass it to
`build --stock-provenance` to record the `cvm-b3` provider and the raw hashes in
the manifest.

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
