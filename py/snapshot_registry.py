"""Offline lifecycle CLI for immutable snapshot manifests.

Drives ``build``/``validate``/``publish``/``retire`` over a filesystem
registry of manifests (see ``docs/snapshot_registry.md``).  The CLI is a
single-operator tool: it only touches the local filesystem, never performs
remote access, never overwrites a published artifact, never deletes files on
``retire``, and never substitutes a fallback series.

Exit codes: 0 success; 2 usage/argument error; 3 CLI target missing on disk or
refused overwrite (plus ``build`` source inputs missing); 4 the artifact under
audit does not hold up (contract divergence); 5 invalid lifecycle transition.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

from allocation_config import ASSET_CLASSES
from allocation_data import (
    SnapshotError,
    build_equal_weight_sleeve,
    levels_to_returns,
    multiply_levels,
    read_levels_csv,
)
from snapshot_manifest import (
    ALLOCATION_FILES,
    MANIFEST_SCHEMA_VERSION,
    SnapshotManifest,
    SnapshotManifestError,
    file_sha256,
    manifest_canonical_sha256,
    sha256_path,
    validate_manifest,
)

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_TARGET_MISSING = 3
EXIT_CONTRACT_DIVERGENCE = 4
EXIT_INVALID_TRANSITION = 5

MANIFEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
RESERVED_MANIFEST_IDS = frozenset({"metadata"})

# Fixed provider literals per source role (AD-004/AD-006): fetch metadata is
# never copied into a manifest, so provenance stays registry-independent.
PROVIDER_ALLOCATION = "snapshot-build:allocation"
PROVIDER_SELECTOR = "snapshot-build:selector"

# Fixed disk serializer for every manifest write (AD-004).
DISK_JSON = dict(sort_keys=True, indent=2, ensure_ascii=False)

# Lifecycle envelope keys a rewrite may change (``_assert_envelope_only``).
ENVELOPE_KEYS = ("state", "state_changed_at", "retire_reason")


class TargetExistsError(FileExistsError):
    """Raised when an immutable target would be overwritten."""


class InvalidTransitionError(RuntimeError):
    """Raised when a lifecycle transition is not allowed (exit 5)."""


class AlreadyPublishedError(RuntimeError):
    """Raised when a published manifest is targeted again (exit 3)."""


def serialize_manifest(payload: Mapping[str, object]) -> str:
    """Serialize a manifest payload with the fixed disk form."""

    return json.dumps(dict(payload), **DISK_JSON) + "\n"


def write_manifest_atomic(
    path: Path,
    payload: Mapping[str, object],
    *,
    overwrite: bool = False,
) -> None:
    """Write a manifest file atomically (temp file in the target's own
    directory plus ``os.replace``) so no partial manifest is observable.

    Without ``overwrite`` an existing target is refused before anything is
    written; the target path itself is never opened exclusively - the
    temporary file carries the exclusivity burden.
    """

    path = Path(path)
    if not overwrite and path.exists():
        raise TargetExistsError(f"manifest already exists: {path}")
    text = serialize_manifest(payload)
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    try:
        with tmp:
            tmp.write(text)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp.name, path)
    except BaseException:
        Path(tmp.name).unlink(missing_ok=True)
        raise


def _assert_envelope_only(
    baseline: Mapping[str, object],
    result: Mapping[str, object],
    path: Path,
) -> None:
    """Enforce the rewrite contract: only envelope keys may change.

    ``manifest_sha256`` is self-referential: a transition may backfill it when
    absent, but never change or drop a declared value.
    """

    allowed = set(ENVELOPE_KEYS) | {"manifest_sha256"}
    for key in set(baseline) | set(result):
        if key in allowed:
            continue
        if key not in baseline or key not in result or baseline[key] != result[key]:
            raise SnapshotManifestError(
                f"envelope rewrite must not change content key {key!r}: {path}"
            )
    if (
        baseline.get("manifest_sha256") is not None
        and result.get("manifest_sha256") != baseline.get("manifest_sha256")
    ):
        raise SnapshotManifestError(
            f"envelope rewrite must not change manifest_sha256: {path}"
        )


def rewrite_envelope(
    path: Path,
    mutate: Callable[[dict], Optional[Mapping[str, object]]],
) -> dict:
    """Rewrite only the lifecycle envelope of a manifest file.

    The raw JSON payload is loaded so unknown keys survive verbatim; the
    mutate callable may touch ``ENVELOPE_KEYS`` only (enforced - any content
    key that appears, disappears, or changes raises ``SnapshotManifestError``
    before anything is written); the result is serialized with the fixed disk
    form and atomically replaced.
    """

    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SnapshotManifestError(f"manifest must be an object: {path}")
    baseline = copy.deepcopy(raw)
    updated = mutate(raw)
    if updated is None:
        updated = raw
    result = dict(updated)
    _assert_envelope_only(baseline, result, path)
    write_manifest_atomic(path, result, overwrite=True)
    return result


def transition_state(
    payload: Mapping[str, object],
    action: str,
    *,
    at: str,
    reason: Optional[str] = None,
) -> dict:
    """Return a new payload with the lifecycle envelope updated.

    Pure function enforcing the state machine: ``validate`` upgrades
    ``draft`` (or a legacy manifest without ``state``) to ``validated`` and is
    a no-op on ``validated``; ``publish`` accepts ``validated`` only and
    refuses an already-published artifact; ``retire`` accepts
    ``published``/``validated`` only.  A missing ``manifest_sha256`` is
    backfilled on every real transition.
    """

    if not isinstance(payload, Mapping):
        raise SnapshotManifestError("manifest payload must be an object")
    current = payload.get("state") or "draft"
    updated = dict(payload)
    if action == "validate":
        if current in ("published", "retired"):
            raise InvalidTransitionError(
                f"cannot validate a {current} manifest"
            )
        if current == "validated":
            return updated
        updated["state"] = "validated"
    elif action == "publish":
        if current == "published":
            raise AlreadyPublishedError("published manifest is never overwritten")
        if current != "validated":
            raise InvalidTransitionError(
                f"cannot publish a {current} manifest"
            )
        updated["state"] = "published"
    elif action == "retire":
        if current not in ("published", "validated"):
            raise InvalidTransitionError(f"cannot retire a {current} manifest")
        updated["state"] = "retired"
        if reason is not None:
            updated["retire_reason"] = reason
    else:
        raise SnapshotManifestError(f"unknown lifecycle action: {action}")
    updated["state_changed_at"] = at
    if not updated.get("manifest_sha256"):
        updated["manifest_sha256"] = manifest_canonical_sha256(payload)
    return updated


def _single_level_series(levels: Mapping[str, dict], label: str) -> dict:
    if "value" in levels:
        return dict(levels["value"])
    if len(levels) == 1:
        return dict(next(iter(levels.values())))
    raise SnapshotError(f"expected a single series for {label}; found {sorted(levels)}")


def premium_level_series(snapshot_dir: Path) -> dict:
    """Build the five class level series exactly as the Premium bundle
    consumes them: multi-ticker sleeves rebuilt per ticker basket,
    ``international_equity``/``crypto`` multiplied by ``ptax``."""

    def single(key: str) -> dict:
        return _single_level_series(
            read_levels_csv(snapshot_dir / ALLOCATION_FILES[key]), key
        )

    ptax = single("ptax")
    return {
        "brazilian_stocks": build_equal_weight_sleeve(
            read_levels_csv(snapshot_dir / ALLOCATION_FILES["brazilian_stocks"])
        ),
        "fiis": build_equal_weight_sleeve(
            read_levels_csv(snapshot_dir / ALLOCATION_FILES["fiis"])
        ),
        "international_equity": multiply_levels(
            single("international_equity"), ptax
        ),
        "fixed_income": single("fixed_income"),
        "crypto": multiply_levels(single("crypto"), ptax),
    }


def build_manifest(
    *,
    registry_root: Path,
    manifest_id: str,
    created_at: str,
    snapshot_dir: Path,
    stock_universe: Path,
    fii_universe: Path,
) -> dict:
    """Compose the ``draft`` manifest payload for one local snapshot.

    ``common_dates`` is the sorted intersection of the five class series
    exactly as ``load_premium_snapshot_bundle`` consumes them (via
    ``py/allocation_data.py`` - the intersection is never reimplemented);
    ``cutoff_date`` is the last of them.  Emits exactly the
    ``SnapshotManifest.as_dict()`` projection with fixed provider/metadata
    literals and registry-root-relative POSIX paths.
    """

    registry_root = registry_root.resolve()
    files = {name: ALLOCATION_FILES[name] for name in sorted(ALLOCATION_FILES)}
    rows = levels_to_returns(premium_level_series(snapshot_dir), ASSET_CLASSES)
    common_dates = tuple(row.date for row in rows)
    if not common_dates:
        raise SnapshotError("class series have no common dates")

    def rel(path: Path) -> str:
        return path.resolve().relative_to(registry_root).as_posix()

    payload = {
        "manifest_id": manifest_id,
        "manifest_version": MANIFEST_SCHEMA_VERSION,
        "created_at": created_at,
        "cutoff_date": common_dates[-1].isoformat(),
        "common_dates": [item.isoformat() for item in common_dates],
        "supported_classes": list(ASSET_CLASSES),
        "sources": {
            "allocation": {
                "path": rel(snapshot_dir),
                "sha256": sha256_path(snapshot_dir),
                "provider": PROVIDER_ALLOCATION,
                "metadata": {"files": files},
            },
            "stock": {
                "path": rel(stock_universe),
                "sha256": file_sha256(stock_universe),
                "provider": PROVIDER_SELECTOR,
                "metadata": {},
            },
            "fii": {
                "path": rel(fii_universe),
                "sha256": file_sha256(fii_universe),
                "provider": PROVIDER_SELECTOR,
                "metadata": {},
            },
        },
    }
    manifest = SnapshotManifest.from_dict(payload, base_dir=registry_root)
    result = manifest.as_dict()
    result["state"] = "draft"
    result["state_changed_at"] = created_at
    result["manifest_sha256"] = manifest_canonical_sha256(manifest)
    return result


def _contains_path(container: Path, path: Path) -> bool:
    """True when ``container`` equals or contains ``path`` (resolved forms)."""

    try:
        path.relative_to(container)
    except ValueError:
        return False
    return True


def _valid_iso_datetime(value: str) -> bool:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_manifest(args: argparse.Namespace) -> tuple:
    """Shared CLI preamble for manifest-consuming commands.

    Returns ``(exit_code, manifest_path, manifest)``.  Missing registry dir or
    manifest file is exit 3; an unloadable manifest (including an unknown
    ``state``) is exit 4; a ``--manifest-id`` that differs from the loaded
    ``manifest_id`` is exit 2.
    """

    registry_root = Path(args.registry)
    if not registry_root.is_dir():
        print(f"registry directory not found: {registry_root}", file=sys.stderr)
        return EXIT_TARGET_MISSING, None, None
    manifest_path = registry_root / f"{args.manifest_id}.json"
    if not manifest_path.is_file():
        print(f"manifest file not found: {manifest_path}", file=sys.stderr)
        return EXIT_TARGET_MISSING, None, None
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = SnapshotManifest.from_dict(
            raw, base_dir=registry_root, manifest_path=manifest_path
        )
    except (OSError, ValueError, SnapshotManifestError) as exc:
        print(f"manifest cannot be loaded: {exc}", file=sys.stderr)
        return EXIT_CONTRACT_DIVERGENCE, None, None
    if manifest.manifest_id != args.manifest_id:
        print(
            f"--manifest-id {args.manifest_id!r} differs from loaded "
            f"manifest_id {manifest.manifest_id!r}",
            file=sys.stderr,
        )
        return EXIT_USAGE, None, None
    return EXIT_OK, manifest_path, manifest


def class_series_map(manifest: SnapshotManifest) -> dict:
    """Return the deterministic class-to-series audit map (SNAP-09).

    One entry per supported class plus ``ptax`` and the explicit ``stock``/
    ``fii`` selector sources - never non-class files.  Each entry carries the
    registry-root-relative POSIX ``path`` and the ``file_sha256`` of the series
    FILE itself (never the ``allocation`` directory digest); derived series add
    ``derived_from: "allocation"`` for provenance (AD-006).
    """

    if manifest.manifest_path is not None:
        root = Path(manifest.manifest_path).parent
    elif manifest.sources:
        root = Path(next(iter(manifest.sources.values())).base_dir or Path.cwd())
    else:
        root = Path.cwd()
    root = root.resolve()

    result: dict = {}
    for key in (*ASSET_CLASSES, "ptax", "stock", "fii"):
        if key in ("stock", "fii"):
            source = manifest.selector_source_for(key)
        else:
            source = manifest.source_for(key)
        resolved = Path(source.resolved_path())
        try:
            relative = resolved.resolve().relative_to(root).as_posix()
        except ValueError:
            raise SnapshotManifestError(
                f"series path escapes the registry root: {resolved}"
            )
        entry = {
            "path": relative,
            "sha256": file_sha256(resolved),
            "provider": source.provider,
        }
        if source.metadata.get("derived_from") == "allocation":
            entry["derived_from"] = "allocation"
        result[key] = entry
    return result


def _cmd_validate(args: argparse.Namespace) -> int:
    code, manifest_path, manifest = _resolve_manifest(args)
    if code != EXIT_OK:
        return code
    if not args.check and manifest.state in ("published", "retired"):
        print(f"manifest is {manifest.state}: frozen, refusing to validate", file=sys.stderr)
        return EXIT_INVALID_TRANSITION
    try:
        validate_manifest(manifest)
        failure = None
    except (SnapshotManifestError, OSError) as exc:
        failure = exc
    if args.as_json:
        try:
            print(json.dumps(class_series_map(manifest), **DISK_JSON))
        except (SnapshotManifestError, OSError) as exc:
            if failure is None:
                failure = exc
    if failure is not None:
        print(f"validation failed: {failure}", file=sys.stderr)
        return EXIT_CONTRACT_DIVERGENCE
    # With --json the map owns stdout; status chatter moves to stderr.
    status = sys.stderr if args.as_json else sys.stdout
    if args.check:
        print("check: manifest contract holds", file=status)
        return EXIT_OK
    if manifest.state == "validated":
        print("manifest already validated; nothing to write", file=status)
        return EXIT_OK
    rewrite_envelope(
        manifest_path,
        lambda payload: transition_state(payload, "validate", at=_now_iso()),
    )
    print(f"validated {manifest_path}", file=status)
    return EXIT_OK


def _cmd_publish(args: argparse.Namespace) -> int:
    code, manifest_path, manifest = _resolve_manifest(args)
    if code != EXIT_OK:
        return code
    try:
        rewrite_envelope(
            manifest_path,
            lambda payload: transition_state(payload, "publish", at=_now_iso()),
        )
    except AlreadyPublishedError:
        print("published manifest is never overwritten", file=sys.stderr)
        return EXIT_TARGET_MISSING
    except InvalidTransitionError as exc:
        print(f"invalid transition: {exc}", file=sys.stderr)
        return EXIT_INVALID_TRANSITION
    print(f"published {manifest_path}")
    return EXIT_OK


def _cmd_retire(args: argparse.Namespace) -> int:
    code, manifest_path, manifest = _resolve_manifest(args)
    if code != EXIT_OK:
        return code
    try:
        rewrite_envelope(
            manifest_path,
            lambda payload: transition_state(
                payload, "retire", at=_now_iso(), reason=args.reason
            ),
        )
    except InvalidTransitionError as exc:
        print(f"invalid transition: {exc}", file=sys.stderr)
        return EXIT_INVALID_TRANSITION
    print(f"retired {manifest_path}")
    return EXIT_OK


def _cmd_build(args: argparse.Namespace) -> int:
    registry_root = Path(args.registry)
    if not registry_root.is_dir():
        print(f"registry directory not found: {registry_root}", file=sys.stderr)
        return EXIT_TARGET_MISSING
    if not _valid_iso_datetime(args.created_at):
        print("--created-at must be an ISO datetime", file=sys.stderr)
        return EXIT_USAGE

    snapshot_dir = Path(args.snapshot_dir)
    stock_universe = Path(args.stock_universe)
    fii_universe = Path(args.fii_universe)
    target = registry_root / f"{args.manifest_id}.json"
    for label, path, kind in (
        ("--snapshot-dir", snapshot_dir, "dir"),
        ("--stock-universe", stock_universe, "file"),
        ("--fii-universe", fii_universe, "file"),
    ):
        resolved = path.resolve()
        if kind == "dir" and _contains_path(resolved, target.resolve()):
            # The manifest must never land inside the hashed source tree:
            # sha256_path(snapshot_dir) runs before the write, so the manifest
            # would invalidate the very digest it records.
            print(
                f"{label} must not contain the destination manifest path: "
                f"{path} contains {target}; place the snapshot in "
                f"<registry>/data/<snapshot_name>/",
                file=sys.stderr,
            )
            return EXIT_USAGE
        try:
            resolved.relative_to(registry_root.resolve())
        except ValueError:
            print(
                f"{label} must resolve inside the registry root: {path}",
                file=sys.stderr,
            )
            return EXIT_USAGE
        exists = path.is_dir() if kind == "dir" else path.is_file()
        if not exists:
            print(f"{label} not found on disk: {path}", file=sys.stderr)
            return EXIT_TARGET_MISSING

    missing = [
        filename
        for filename in sorted(set(ALLOCATION_FILES.values()))
        if not (snapshot_dir / filename).is_file()
    ]
    if missing:
        print(
            f"snapshot inputs missing on disk: {missing} under {snapshot_dir}",
            file=sys.stderr,
        )
        return EXIT_TARGET_MISSING

    if target.exists():
        print(f"manifest already exists: {target}", file=sys.stderr)
        return EXIT_TARGET_MISSING

    try:
        payload = build_manifest(
            registry_root=registry_root,
            manifest_id=args.manifest_id,
            created_at=args.created_at,
            snapshot_dir=snapshot_dir,
            stock_universe=stock_universe,
            fii_universe=fii_universe,
        )
    except (SnapshotError, SnapshotManifestError) as exc:
        print(f"build failed: {exc}", file=sys.stderr)
        return EXIT_CONTRACT_DIVERGENCE

    try:
        write_manifest_atomic(target, payload, overwrite=False)
    except TargetExistsError:
        print(f"manifest already exists: {target}", file=sys.stderr)
        return EXIT_TARGET_MISSING
    print(f"wrote {target}")
    return EXIT_OK


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="snapshot_registry",
        description="Offline lifecycle CLI for immutable snapshot manifests.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build a draft manifest from local inputs")
    build.add_argument("--registry", required=True, help="registry root directory")
    build.add_argument("--manifest-id", required=True, help="manifest identifier")
    build.add_argument("--created-at", required=True, help="explicit ISO datetime")
    build.add_argument("--snapshot-dir", required=True, help="fetch output directory")
    build.add_argument("--stock-universe", required=True, help="stock selector CSV")
    build.add_argument("--fii-universe", required=True, help="FII selector CSV")
    build.set_defaults(handler=_cmd_build)

    validate = subparsers.add_parser(
        "validate", help="verify a manifest and upgrade draft to validated"
    )
    validate.add_argument("--registry", required=True, help="registry root directory")
    validate.add_argument("--manifest-id", required=True, help="manifest identifier")
    validate.add_argument(
        "--check",
        action="store_true",
        help="read-only audit in any lifecycle state (no writes)",
    )
    validate.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="print the class-to-series audit map",
    )
    validate.set_defaults(handler=_cmd_validate)

    publish = subparsers.add_parser(
        "publish", help="freeze a validated manifest as published"
    )
    publish.add_argument("--registry", required=True, help="registry root directory")
    publish.add_argument("--manifest-id", required=True, help="manifest identifier")
    publish.set_defaults(handler=_cmd_publish)

    retire = subparsers.add_parser(
        "retire", help="mark a validated or published manifest as retired"
    )
    retire.add_argument("--registry", required=True, help="registry root directory")
    retire.add_argument("--manifest-id", required=True, help="manifest identifier")
    retire.add_argument("--reason", help="optional retirement reason")
    retire.set_defaults(handler=_cmd_retire)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    if (
        not MANIFEST_ID_PATTERN.fullmatch(args.manifest_id)
        or args.manifest_id in RESERVED_MANIFEST_IDS
    ):
        print(f"invalid --manifest-id: {args.manifest_id!r}", file=sys.stderr)
        return EXIT_USAGE
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
