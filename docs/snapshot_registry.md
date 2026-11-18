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
  exactly `{"files": {...}}` (`allocation`) or `{}` (a FII CSV built without
  `--fii-provenance`). The fetch
  `metadata.json` content is never copied into a manifest (it carries absolute
  paths that would break byte-reproducibility).
- The stock source is the exception, and `build` always writes it this way
  (spec #75): a stock CSV built from CVM and B3 (issue #78) and passed with the
  required `--stock-provenance`. The `stock` source has `provider: "cvm-b3"`
  and `metadata` = `{"as_of": <cutoff of the CSV>, "raw_files": {<kind>:
  {<cache file>: <sha256>}}}`, copied from the collector's provenance file.
  The kinds are `dfp_itr`, `cotahist`, `proventos`, `eventos`,
  `classificacao_setorial` and `cadastro` (FCA and the CVM cadastro). The
  metadata holds file names and hashes only, no paths.
- Likewise a FII CSV made by `scripts/create_selector_snapshot.py` (issue #80)
  and passed with `--fii-provenance`: the `fii` source has `provider:
  "statusinvest"` and `metadata` = `{"retrieved_at": <download time>,
  "recent_trade_filter": {"collected_on", "window_days", "cotahist"}}`.

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
                                         --stock-provenance JSON [--fii-provenance JSON]
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
`--stock-provenance` is required: it names the `<stem>.provenance.json` that
`python3 py/pit_rebuild collect` (or `scripts/create_selector_snapshot.py`)
writes next to the stock CSV, and a stock CSV without it is exit 2 with nothing
written, so there is no generic `snapshot-build:selector` stock source (spec
#75: the stock CSV comes only from CVM and B3). A missing file is
exit 3; a provenance whose `csv_sha256` is not the hash of `--stock-universe`,
or that lacks `provider`, `as_of`, `csv_sha256` or `raw_files`, is exit 4 with
nothing written. `--fii-provenance` names the `fiis.provenance.json` of
`scripts/create_selector_snapshot.py`, with the same rules: missing is exit 3;
a `csv_sha256` that is not the hash of `--fii-universe`, or a provenance
without `provider`, `retrieved_at`, `csv_sha256` or `recent_trade_filter`, is
exit 4 with nothing written.

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
  "stock": {"path": "stocks.csv", "sha256": "…", "provider": "cvm-b3"}
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
(`load_snapshot_inputs` in `api/app/services/snapshot_inputs.py`), checks
each SHA-256 against the manifest and against the `source_snapshot_hashes`
recorded in the run's policy, validates the CSV, and copies the verified bytes
into the run workspace before any calculation. Live collection (Status Invest,
COTAHIST) belongs to snapshot creation only, so a selector CSV must already be
filtered when it is registered (`scripts/create_selector_snapshot.py`, below).
Failure codes:

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
  `Petróleo. Gás e Biocombustíveis`). A company whose GetDetail record has no
  classification, or whose GetDetail download fails online, has both empty.
  With `--offline` the GetDetail of every company in the universe must be in
  the cache: a missing one is exit 5, so an offline run never leaves the
  sectors empty because they were never fetched (#80). Every row with empty
  sectors is listed in the provenance file under `unclassified`, ticker →
  reason (`no_classification`, `detail_download_failed` or
  `detail_unparseable`); the selector z-scores by `SETOR`, so such a row leaves
  the selection by metric coverage.
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
- **Financials:** banks and insurers file other statement layouts, so the net
  income and the equity are found by name, not by a fixed account (#80). The
  net income is the level-2 DRE account named "Lucro/Prejuízo ... do Período"
  (3.11 in the common layout, 3.09 for banks, 3.13 for insurers, where 3.11 is
  the continuing operations), and its `.01` child when non-zero. The equity is
  the level-2 BPP account named "Patrimônio Líquido ..." (2.03, or 2.07/2.08 for
  banks) and the minority its child named "... Não Controladores". Without
  those names the accounts are 3.11 and 2.03 as before. The indicators that do
  not apply to a financial (#79) are left empty.
- **`SUBSETOR` in the selector:** the selector does not read it, but its
  pre-processing treats every non-ID column as a metric candidate. With pandas 3
  the text column stays text and the robustness filter counts it as filled;
  with pandas 2 (`object` dtype) it becomes an all-NaN column and counts as
  missing. #79, which reads `SUBSETOR`, should keep it out of the metrics.

The provenance file `<stem>.provenance.json` goes next to the CSV; pass it to
`build --stock-provenance` to record the `cvm-b3` provider and the raw hashes in
the manifest.

## Creating the selector CSVs (issue #80)

`python3 scripts/create_selector_snapshot.py --date D --out-dir
<registry>/data/selectors-<name> [--cache-dir DIR] [--offline]` is the snapshot
creation of the selector CSVs. It writes `stocks.csv`,
`stocks.provenance.json`, `fiis.csv` and `fiis.provenance.json` together, or
none of them.

- **Stocks:** the CVM/B3 collector above at `d = D`. Its universe already is
  the B3 recent-trade filter of #63. Status Invest is never called for stocks,
  and a failed stock collection never falls back to it (`py/fetch_status_invest.py`
  stays only as the reference of the measurement tool).
- **FIIs:** still Status Invest (`py/fetch_status_invest_fii.py`), downloaded
  live, then filtered by the B3 recent-trade rule of #63 at `D` (CODBDI 12,
  `filter_recent_trades`), reading the COTAHIST of the raw cache. The filter
  record goes into `fiis.provenance.json`.
- `--offline` applies to the stock raw files; the FII CSV is always downloaded.
- Exit codes: 0 written, 1 output not written, 2 bad arguments or a forbidden
  directory, 3 cache directory absent in `--offline`, 4 a cached stock raw file
  differs from its hash or a PIT leak, 5 a required stock raw file is missing,
  6 the FII CSV failed (Status Invest download or validation, or the B3 filter),
  7 a stock has no B3 classification because its GetDetail download failed or
  its body did not parse (the selector z-scores by `SETOR` and would drop it
  silently). A GetDetail record with no classification is not a failure: the
  stock keeps empty `SETOR`/`SUBSETOR` and is listed with the reason
  `no_classification` under `unclassified` in `stocks.provenance.json`, next to
  the failures `detail_download_failed` and `detail_unparseable` that
  `py/pit_rebuild collect` alone still writes.

Then `build` with `--stock-provenance <out>/stocks.provenance.json
--fii-provenance <out>/fiis.provenance.json`, `validate` and `publish`.

## Snapshot do mês num comando (issue #85)

`python3 scripts/publish_snapshot.py --date D --registry snapshots [--cache-root DIR] [--seed-cache DIR]
[--limits JSON]` faz as três etapas acima de uma vez e só publica se tudo passar. Para chamar de outro código (o
`ciclo-mensal` do #87), use `publish_snapshot.publish_snapshot(day, registry=..., cache_root=...)`: devolve
`PublishResult` (manifesto publicado, caminho, `reused`, checagens) ou levanta `PublishSnapshotError` com `stage`,
`check` e `exit_code`.

- **Etapas**, paradas na primeira falha: `series` (`fetch_snapshot` sem as cestas `caio_*`, dez anos até `D`),
  `selecao` (`create_selector_snapshot.create`), `build`, `validate`, `sanidade`, `publish`. O manifesto é
  `mensal-D` e os dados vão para `data/mensal-D/series/` e `data/mensal-D/selecao/`. Antes do `publish` o manifesto
  fica em `draft`/`validated`, que a API não seleciona: uma falha deixa o publicado anterior valendo.
- **Checagens de sanidade** (`SanityLimits`, padrões no código, sobrescritos por `--limits`): ações com score e FIIs
  elegíveis nos três perfis-âncora de `STOCK_RULES`/`FII_RULES` (o perfil de uma conta interpola entre eles);
  IBOV, IFIX e CDI com dado até o corte e no máximo `max_cutoff_lag_days` antes de `D`; corte no máximo
  `max_cutoff_lag_days` antes de `D` (pega SP500, BTC e PTAX parados); linhas de `stocks.csv` e `fiis.csv` sem queda
  maior que `max_universe_drop` contra o último publicado. Padrões: 40 ações com score, 8 FIIs elegíveis, 5 dias,
  20%. Referência: no `cvm-b3-2026-11-17` o perfil conservador tem 97 ações com score e 14 FIIs elegíveis.
- **Data de corte e idempotência**: o corte é a última data comum das séries, não `D`. Já publicado = há manifesto
  `published` com id `mensal-D` ou com corte igual a `D` (checado antes de buscar), ou com o mesmo corte efetivo que
  as séries recém-buscadas dão (checado depois do `build`, antes de gravar; os dados buscados são apagados). Nos três
  casos nada é refeito e o comando devolve esse manifesto com saída 0. Restos `draft`/`validated` de uma tentativa
  falha com o mesmo id são apagados e refeitos; um `retired` com o mesmo id é saída 3.
- **Cache de brutos**: o cache do pit_rebuild nunca renova um arquivo. Cada `D` usa `<cache-root>/D/` (padrão
  `~/.cache/tcc-pit-rebuild-ciclos/D/`), semeado por hard link só com os anuais de anos até `ano(D) - 2` (DFP, ITR,
  FCA, COTAHIST, informes de FII) do cache de ciclo anterior mais recente ou de `--seed-cache` no primeiro ciclo; o
  resto (COTAHIST do ano, B3, cadastro, listagens) é baixado de novo. Uma retomada da mesma `D` reaproveita o cache.
- Códigos de saída: 0 publicado ou já publicado, 1 cache não preparado, 2 argumentos, 3 registro ausente ou id
  `retired`, 4 `series`, 5 `selecao`, 6 `build`/`validate`/`publish`, 7 checagem de sanidade.

## Ciclo mensal num comando (issue #87)

`DATABASE_URL=... python3 scripts/ciclo_mensal.py --date D --registry snapshots [--account ID]` encadeia a
publicação acima, o run Premium da conta com o manifesto recém-publicado (`scripts/premium_run.py`) e o teste para
frente da carteira anterior (`py/teste_para_frente.py`), e grava `docs/validation/ciclos/mensal-D.json` e `.md`. Para
na primeira etapa que falha (`conta`, `publicacao`, `run`, `teste_para_frente`, `relatorio`) e diz qual. Rodar de novo
com a mesma `D` reaproveita o manifesto e o run e regrava o relatório, byte a byte igual se nada mudou. A docstring
do script tem as etapas, os caminhos configuráveis e os códigos de saída (0 concluído, 2 argumentos, 3 conta,
4 publicação, 5 run, 6 teste para frente, 7 relatório, 130 interrompido).

Aviso no Telegram (issue #88, `scripts/aviso_telegram.py`), ligado por padrão: no fim do ciclo, também quando ele
falha ou é interrompido, o bot do Prumo manda o resumo (manifesto, o que ficou/entrou/saiu da carteira por classe e o
teste para frente em uma linha por classe) ou a etapa, o código de saída e o erro. O bot e o chat vêm de duas
variáveis de ambiente (o comando não lê `api/.env`; exporte-as no shell que roda o ciclo):

- `PRUMO_TELEGRAM_BOT_TOKEN`: token do bot criado no BotFather.
- `PRUMO_TELEGRAM_CHAT_ID`: id do chat que recebe o aviso (o chat do Caio com o bot).

Sem uma delas, o ciclo segue e o stderr diz que o aviso não foi enviado. Uma falha no envio (rede, timeout de 10 s,
token ou chat inválido) vira só uma linha no stderr, sem o token, e não muda o código de saída do ciclo.

## Walkthrough (manual)

```
python3 py/snapshot_registry.py build    --registry ./snapshots --manifest-id demo-2020 \
    --created-at 2020-01-04T12:00:00+00:00 --snapshot-dir ./snapshots/data/demo \
    --stock-universe ./snapshots/stocks.csv --fii-universe ./snapshots/fiis.csv \
    --stock-provenance ./snapshots/stocks.provenance.json --fii-provenance ./snapshots/fiis.provenance.json
python3 py/snapshot_registry.py validate --registry ./snapshots --manifest-id demo-2020
python3 py/snapshot_registry.py publish  --registry ./snapshots --manifest-id demo-2020
python3 py/snapshot_registry.py retire   --registry ./snapshots --manifest-id demo-2020 --reason "superseded"
```

Each step prints its outcome and exits 0; `validate --check --json` works at
any point for a read-only audit. Reader walkthrough of this sequence from the
doc alone is a manual check (explicitly outside the automated gate).
