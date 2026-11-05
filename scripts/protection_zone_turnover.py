#!/usr/bin/env python3
"""Month-to-month turnover with and without the protection zone (issue #46, phase 2 of #9).

``measure`` runs the production Premium pipeline (``run_premium_optimization`` with
the adapter's default engines) once on a published November manifest, with the
profile record and seed of the October production run declared in
``docs/validation/multirun-convergence-protocol.json`` (``production_run``) and
the tickers that run delivered as incumbents. The zone runs after the consensus,
so the same GA runs give both numbers: the turnover of the zone portfolio
(``turnover``) and of the pure consensus (``consensus_turnover``), each counted
as entries against the October tickers.

Writes ``protection-zone-turnover.json`` and ``.md`` into ``--out-dir``. The
files hold no timestamps and no absolute paths: rerunning on the same manifest,
code and seed rewrites them byte for byte.

Exit codes:
    0  both files were written
    1  the pipeline failed or a selector did not apply the zone
    2  invalid arguments or protocol
    3  the manifest is missing, unpublished, invalid, the October manifest or cut
       off before 2026-11-01; nothing is written
"""

from __future__ import annotations

import argparse
from datetime import date
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT / "api", PROJECT_ROOT / "py", PROJECT_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import convergence_analysis as analysis  # noqa: E402

PREFIX = "protection_zone_turnover"
SCHEMA = "protection-zone-turnover-v1"
DEFAULT_PROTOCOL = PROJECT_ROOT / "docs" / "validation" / "multirun-convergence-protocol.json"
DEFAULT_REGISTRY = PROJECT_ROOT / "snapshots"
DEFAULT_OUT_DIR = PROJECT_ROOT / "docs" / "validation"
DEFAULT_SEED = 46641103
OCTOBER_MANIFEST_ID = "dev-2026-10-29-ibov"
MINIMUM_CUTOFF = date(2026, 11, 1)
JSON_NAME = "protection-zone-turnover.json"
MD_NAME = "protection-zone-turnover.md"
SELECTORS = ("stocks", "fiis")
APPLIED_STATUSES = ("applied", "fallback_hhi")

EXIT_OK = 0
EXIT_RUN_FAILED = 1
EXIT_USAGE = 2
EXIT_SNAPSHOT = 3

LIMITATIONS = (
    "The October incumbents are the tickers of production run 9e22cc16-cfda-495e-a8c9-c0d202ef32b6, "
    "scored before #52 (stocks with a NaN score could still be picked); a November run drops those "
    "stocks before the GA, so part of the stock exits can come from the scoring change and not from "
    "the new snapshot.",
    "The seed is fixed at the October run's seed. Production derives the GA seed from the policy "
    "version (derive_premium_seed), so a real premium-policy-v3 run on the same account, profile and "
    "manifest draws different GA runs than this measurement.",
    "One profile (premium-2026-11-02) and one month pair; K and delta come from the #9 calibration on "
    "lots that still held NaN-score stocks (spec A-9).",
)


class MeasureError(RuntimeError):
    def __init__(self, code: int, where: str, reason: str):
        super().__init__(f"{where}: {reason}")
        self.code = code
        self.where = where
        self.reason = reason


def _error(where: str, reason: str) -> None:
    print(f"{PREFIX}: {where}: {reason}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def load_baseline(protocol_path: Path) -> dict[str, Any]:
    """October production run and its profile record from the #9 protocol."""

    try:
        protocol, protocol_sha256 = analysis.load_protocol(protocol_path)
        production = protocol["production_run"]
        record = protocol["profiles"][production["profile"]]["record"]
        baseline = {
            "run_id": str(production["id"]),
            "profile_id": str(production["profile"]),
            "seed": int(production["seed"]),
            "stocks": [str(ticker) for ticker in production["stocks"]],
            "fiis": [str(ticker) for ticker in production["fiis"]],
            "record": record,
            "capital_brl": float(protocol["capital_brl"]),
            "protocol_sha256": protocol_sha256,
        }
    except (analysis.ProtocolError, KeyError, TypeError, ValueError) as exc:
        raise MeasureError(EXIT_USAGE, "protocol", str(exc)) from exc
    return baseline


def load_november_manifest(
    registry: Path,
    manifest_id: str,
    validator: Callable[[Any], Any] | None = None,
) -> Any:
    """The published November manifest; any other manifest is refused (PZ-33)."""

    from snapshot_manifest import SnapshotManifest, SnapshotManifestError, validate_manifest

    if manifest_id == OCTOBER_MANIFEST_ID:
        raise MeasureError(EXIT_SNAPSHOT, "manifest", f"{manifest_id} is the October manifest")
    path = Path(registry) / f"{manifest_id}.json"
    try:
        manifest = SnapshotManifest.load(path)
    except SnapshotManifestError as exc:
        raise MeasureError(EXIT_SNAPSHOT, "manifest", str(exc)) from exc
    if manifest.manifest_id == OCTOBER_MANIFEST_ID:
        raise MeasureError(EXIT_SNAPSHOT, "manifest", f"{path.name} holds the October manifest")
    if manifest.state != "published":
        raise MeasureError(EXIT_SNAPSHOT, "manifest", f"{manifest_id} is {manifest.state or 'legacy'}, not published")
    if manifest.cutoff_date < MINIMUM_CUTOFF:
        raise MeasureError(
            EXIT_SNAPSHOT,
            "manifest",
            f"cutoff {manifest.cutoff_date.isoformat()} is before {MINIMUM_CUTOFF.isoformat()}",
        )
    try:
        return (validator or validate_manifest)(manifest)
    except SnapshotManifestError as exc:
        raise MeasureError(EXIT_SNAPSHOT, "manifest", f"failed validation: {exc}") from exc


def selector_inputs(manifest: Any) -> Any:
    """Frozen selector CSVs of the manifest, as ``benchmark_premium`` builds them."""

    import benchmark_premium
    from snapshot_manifest import sha256_path

    snapshot = {
        "sources": {},
        "selector_paths": {},
    }
    for role, key in (("stock", "stocks"), ("fii", "fiis")):
        path = manifest.selector_source_for(role).resolved_path()
        snapshot["sources"][role] = sha256_path(path)
        snapshot["selector_paths"][key] = str(path)
    return benchmark_premium.selector_inputs_from_manifest(manifest, snapshot)


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


def run_pipeline(
    baseline: Mapping[str, Any],
    manifest: Any,
    seed: int,
    engines: Mapping[str, Callable[..., Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    import multirun_convergence
    from app.adapters.premium_optimization import run_premium_optimization

    policy = multirun_convergence.resolve_policy(baseline["record"], manifest, seed)
    workspace = Path(tempfile.mkdtemp(prefix="protection-zone-turnover-"))
    try:
        output = run_premium_optimization(
            policy,
            manifest,
            workspace,
            baseline["capital_brl"],
            selector_inputs=selector_inputs(manifest),
            snapshot_loader=engines["snapshot_loader"],
            allocation_engine=engines["allocation_engine"],
            stock_engine=engines["stock_engine"],
            fii_engine=engines["fii_engine"],
            seed=int(seed),
            incumbents={
                "run_id": baseline["run_id"],
                "stocks": list(baseline["stocks"]),
                "fiis": list(baseline["fiis"]),
            },
        )
    except Exception as exc:  # noqa: BLE001 - any pipeline failure fails the measurement
        cause = exc.__cause__ or exc
        raise MeasureError(EXIT_RUN_FAILED, "pipeline", f"{exc} ({cause!r})") from exc
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
    return policy, output


def class_report(output: Mapping[str, Any], selector: str) -> dict[str, Any]:
    zone = output.get("provenance", {}).get("protection_zone", {})
    record = zone.get(selector) if isinstance(zone, Mapping) else None
    if not isinstance(record, Mapping) or record.get("status") not in APPLIED_STATUSES:
        status = record.get("status") if isinstance(record, Mapping) else None
        raise MeasureError(EXIT_RUN_FAILED, selector, f"the zone was not applied (status {status})")
    return {
        "status": record["status"],
        "K": record["K"],
        "k_offset": record["k_offset"],
        "delta": record["delta"],
        "n_assets": record["n_assets"],
        "incumbents": list(record["incumbents"]),
        "zone_tickers": sorted(str(item["ticker"]) for item in output.get(selector, [])),
        "consensus_tickers": list(record["consensus_tickers"]),
        "turnover": record["turnover"],
        "consensus_turnover": record["consensus_turnover"],
        "exits": record["exits"],
        "kept": list(record["kept"]),
        "replaced": [dict(item) for item in record["replaced"]],
        "entered": list(record["entered"]),
        "departures": dict(record["departures"]),
        "buffer_fallback": record["buffer_fallback"],
    }


def build_report(
    baseline: Mapping[str, Any],
    manifest: Any,
    seed: int,
    policy: Mapping[str, Any],
    output: Mapping[str, Any],
    code: Mapping[str, Any],
) -> dict[str, Any]:
    import benchmark_premium

    return {
        "schema": SCHEMA,
        "issue": 46,
        "profile_id": baseline["profile_id"],
        "seed": int(seed),
        "policy_version": policy["provenance"]["policy_version"],
        "policy_sha256": benchmark_premium._sha256_json(policy),
        "protocol_sha256": baseline["protocol_sha256"],
        "october": {
            "run_id": baseline["run_id"],
            "seed": baseline["seed"],
            "manifest_id": OCTOBER_MANIFEST_ID,
        },
        "november": {
            "manifest_id": manifest.manifest_id,
            "manifest_sha256": manifest.manifest_sha256,
            "cutoff_date": manifest.cutoff_date.isoformat(),
        },
        "code": {"commit": code.get("commit"), "dirty": code.get("dirty")},
        "classes": {selector: class_report(output, selector) for selector in SELECTORS},
        "limitations": list(LIMITATIONS),
    }


def _tickers(values: Sequence[str]) -> str:
    return ", ".join(values) if values else "—"


def render_markdown(report: Mapping[str, Any]) -> str:
    november = report["november"]
    lines = [
        "# Turnover mês a mês com e sem zona de proteção",
        "",
        "Gerado por `scripts/protection_zone_turnover.py measure` (issue #46, fase 2 da #9). Não edite à mão.",
        "",
        f"- Perfil: `{report['profile_id']}`; seed {report['seed']}; política `{report['policy_version']}`.",
        f"- Outubro: run `{report['october']['run_id']}` (manifesto `{report['october']['manifest_id']}`).",
        f"- Novembro: manifesto `{november['manifest_id']}` (corte {november['cutoff_date']}, "
        f"sha256 `{november['manifest_sha256']}`).",
        f"- Código: commit `{report['code']['commit']}`" + (" (árvore com mudanças)" if report["code"]["dirty"] else "") + ".",
        "",
        "## Turnover (entradas contra a carteira de outubro)",
        "",
        "| Classe | K | δ | Sem zona | Com zona | Saídas com zona | Fallback HHI |",
        "| ------ | - | - | -------- | -------- | --------------- | ------------ |",
    ]
    for selector, item in report["classes"].items():
        lines.append(
            f"| {selector} | {item['K']} | {item['delta']} | {item['consensus_turnover']} | "
            f"{item['turnover']} | {item['exits']} | {item['buffer_fallback'] or 'não'} |"
        )
    for selector, item in report["classes"].items():
        lines += [
            "",
            f"## {selector}",
            "",
            f"- Mantidos: {_tickers(item['kept'])}",
            f"- Entraram: {_tickers(item['entered'])}",
            "- Trocas: "
            + (", ".join(f"{swap['out']} → {swap['in']}" for swap in item["replaced"]) or "—"),
            "- Saídas: "
            + (", ".join(f"{ticker} ({reason})" for ticker, reason in item["departures"].items()) or "—"),
            f"- Consenso puro: {_tickers(item['consensus_tickers'])}",
        ]
    lines += ["", "## Limitações", ""]
    lines += [f"- {text}" for text in report["limitations"]]
    return "\n".join(lines) + "\n"


def measure(
    *,
    registry: Path,
    manifest_id: str,
    seed: int,
    out_dir: Path,
    protocol_path: Path,
    engines: Mapping[str, Callable[..., Any]] | None = None,
    manifest_validator: Callable[[Any], Any] | None = None,
    code: Mapping[str, Any] | None = None,
) -> dict[Path, str]:
    import benchmark_premium

    baseline = load_baseline(protocol_path)
    manifest = load_november_manifest(registry, manifest_id, manifest_validator)
    policy, output = run_pipeline(
        baseline, manifest, seed, dict(engines or benchmark_premium.default_engines())
    )
    report = build_report(
        baseline, manifest, seed, policy, output, code or benchmark_premium.git_info()
    )
    out_dir = Path(out_dir)
    return {
        out_dir / JSON_NAME: analysis.dumps(report),
        out_dir / MD_NAME: render_markdown(report),
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    engines: Mapping[str, Callable[..., Any]] | None = None,
    manifest_validator: Callable[[Any], Any] | None = None,
    code: Mapping[str, Any] | None = None,
) -> int:
    parser = argparse.ArgumentParser(prog=PREFIX, description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("measure", help="run the November measurement and write the report")
    command.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    command.add_argument("--manifest-id", required=True)
    command.add_argument("--seed", type=int, default=DEFAULT_SEED)
    command.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    command.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return EXIT_USAGE if exc.code else EXIT_OK

    try:
        files = measure(
            registry=args.registry,
            manifest_id=args.manifest_id,
            seed=args.seed,
            out_dir=args.out_dir,
            protocol_path=args.protocol,
            engines=engines,
            manifest_validator=manifest_validator,
            code=code,
        )
    except MeasureError as exc:
        _error(exc.where, exc.reason)
        return exc.code
    args.out_dir.mkdir(parents=True, exist_ok=True)
    analysis.write_files_atomically(files)
    for path in files:
        print(f"{PREFIX}: wrote {path.name}", flush=True)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
