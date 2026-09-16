"""Immutable, local snapshot manifests used by Premium research runs.

Premium execution accepts only a validated manifest.  The registry is
deliberately filesystem-only: it never downloads data and it never falls back
to an older artifact after a compatible manifest fails validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping, Optional, Sequence, Tuple

from allocation_config import ASSET_CLASSES


MANIFEST_SCHEMA_VERSION = "1"
MANIFEST_FILE_GLOB = "*.json"

# These names are the local snapshot contract.  A manifest may override them
# with ``allocation_source.metadata.files`` or one source per class.
ALLOCATION_FILES = {
    "brazilian_stocks": "caio_stocks.csv",
    "fiis": "caio_fiis.csv",
    "international_equity": "sp500_total_return_usd.csv",
    "fixed_income": "di.csv",
    "crypto": "btc_usd.csv",
    "ptax": "ptax.csv",
}


class SnapshotManifestError(ValueError):
    """Raised when a snapshot manifest cannot be trusted for a run."""


# Public aliases make the failure boundary easy for API adapters to catch.
ManifestError = SnapshotManifestError
ManifestValidationError = SnapshotManifestError


def sha256_path(path: Path) -> str:
    """Return a stable SHA-256 for a file or a directory tree."""

    if not path.exists():
        raise SnapshotManifestError(f"snapshot source not found: {path}")
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if not path.is_dir():
        raise SnapshotManifestError(f"snapshot source is not a file or directory: {path}")

    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = child.relative_to(path).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        with child.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    """File-specific spelling retained for callers that hash individual CSVs."""

    if not path.exists() or not path.is_file():
        raise SnapshotManifestError(f"snapshot file not found: {path}")
    return sha256_path(path)


def _as_date(raw: object, label: str) -> date:
    if isinstance(raw, date) and not isinstance(raw, datetime):
        return raw
    try:
        return date.fromisoformat(str(raw)[:10])
    except (TypeError, ValueError) as exc:
        raise SnapshotManifestError(f"{label} must be an ISO date") from exc


def _as_iso_datetime(raw: object, label: str) -> str:
    value = str(raw or "").strip()
    if not value:
        raise SnapshotManifestError(f"{label} is required")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SnapshotManifestError(f"{label} must be an ISO datetime") from exc
    return value


def _normalise_hash(raw: object, label: str) -> str:
    value = str(raw or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise SnapshotManifestError(f"{label} must be a SHA-256 hex digest")
    return value


@dataclass(frozen=True)
class SnapshotSource:
    """One immutable source reference recorded by a manifest."""

    path: str
    sha256: str
    provider: str
    metadata: Mapping[str, object] = field(default_factory=dict)
    base_dir: Optional[Path] = field(default=None, compare=False, repr=False)

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
        *,
        base_dir: Optional[Path] = None,
        label: str = "source",
    ) -> "SnapshotSource":
        if not isinstance(value, Mapping):
            raise SnapshotManifestError(f"{label} must be an object")
        path = str(value.get("path") or value.get("source_path") or "").strip()
        if not path:
            raise SnapshotManifestError(f"{label}.path is required")
        raw_hash = value.get("sha256", value.get("hash"))
        sha256 = _normalise_hash(raw_hash, f"{label}.sha256")
        provider = str(value.get("provider") or value.get("source") or "").strip()
        if not provider:
            raise SnapshotManifestError(f"{label}.provider is required")
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise SnapshotManifestError(f"{label}.metadata must be an object")
        return cls(path, sha256, provider, dict(metadata), base_dir)

    def resolved_path(self, base_dir: Optional[Path] = None) -> Path:
        """Resolve a manifest-relative source path without changing its identity."""

        path = Path(self.path).expanduser()
        if path.is_absolute():
            return path
        root = base_dir or self.base_dir
        return (root / path if root is not None else path).resolve()

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "provider": self.provider,
            "metadata": dict(self.metadata),
        }


def _source_value(
    raw: object,
    *,
    base_dir: Optional[Path],
    label: str,
) -> SnapshotSource:
    if not isinstance(raw, Mapping):
        raise SnapshotManifestError(f"{label} must be an object")
    return SnapshotSource.from_dict(raw, base_dir=base_dir, label=label)


@dataclass(frozen=True)
class SnapshotManifest:
    """Validated manifest metadata and source registry for one run."""

    manifest_id: str
    manifest_version: str
    created_at: str
    cutoff_date: date
    common_dates: Tuple[date, ...]
    supported_classes: Tuple[str, ...]
    sources: Mapping[str, SnapshotSource]
    manifest_path: Optional[Path] = field(default=None, compare=False, repr=False)

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
        *,
        base_dir: Optional[Path] = None,
        manifest_path: Optional[Path] = None,
    ) -> "SnapshotManifest":
        if not isinstance(value, Mapping):
            raise SnapshotManifestError("manifest must be an object")
        manifest_id = str(value.get("manifest_id") or value.get("id") or "").strip()
        if not manifest_id:
            raise SnapshotManifestError("manifest_id is required")
        raw_version = value.get("manifest_version", value.get("version"))
        manifest_version = str(raw_version or "").strip().removeprefix("v")
        if not manifest_version:
            raise SnapshotManifestError("manifest_version is required")
        created_at = _as_iso_datetime(value.get("created_at"), "created_at")
        cutoff_date = _as_date(value.get("cutoff_date"), "cutoff_date")

        raw_dates = value.get("common_dates", ())
        if not isinstance(raw_dates, (list, tuple)):
            raise SnapshotManifestError("common_dates must be a list")
        common_dates = tuple(_as_date(item, "common_dates item") for item in raw_dates)

        raw_classes = value.get("supported_classes", ())
        if not isinstance(raw_classes, (list, tuple, set)):
            raise SnapshotManifestError("supported_classes must be a list")
        supported_classes = tuple(str(item).strip() for item in raw_classes if str(item).strip())

        sources: dict[str, SnapshotSource] = {}
        raw_sources = value.get("sources", {})
        if raw_sources is not None:
            if not isinstance(raw_sources, Mapping):
                raise SnapshotManifestError("sources must be an object")
            for key, raw_source in raw_sources.items():
                # A class source map may use an alias accepted by source_for().
                sources[str(key).strip()] = _source_value(
                    raw_source,
                    base_dir=base_dir,
                    label=f"sources.{key}",
                )

        for key in ("allocation_source", "stock_source", "fii_source"):
            raw_source = value.get(key)
            if raw_source is not None:
                if isinstance(raw_source, Mapping) and "path" not in raw_source:
                    # Keep a class-to-file source map inside the flat registry.
                    for child_key, child_source in raw_source.items():
                        sources[str(child_key).strip()] = _source_value(
                            child_source,
                            base_dir=base_dir,
                            label=f"{key}.{child_key}",
                        )
                else:
                    sources[key.removesuffix("_source")] = _source_value(
                        raw_source,
                        base_dir=base_dir,
                        label=key,
                    )

        return cls(
            manifest_id=manifest_id,
            manifest_version=manifest_version,
            created_at=created_at,
            cutoff_date=cutoff_date,
            common_dates=common_dates,
            supported_classes=supported_classes,
            sources=sources,
            manifest_path=manifest_path,
        )

    @classmethod
    def load(cls, path: Path) -> "SnapshotManifest":
        path = Path(path)
        if not path.exists() or not path.is_file():
            raise SnapshotManifestError(f"manifest file not found: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SnapshotManifestError(f"invalid manifest JSON: {path}") from exc
        return cls.from_dict(payload, base_dir=path.parent, manifest_path=path)

    def source_for(self, key: str) -> SnapshotSource:
        """Return a named source, including aliases and allocation file derivation."""

        aliases = {
            "stocks": "brazilian_stocks",
            "stock": "brazilian_stocks",
            "fii": "fiis",
            "international": "international_equity",
            "di": "fixed_income",
            "btc": "crypto",
        }
        canonical = aliases.get(key, key)
        for candidate in (canonical, key):
            source = self.sources.get(candidate)
            if source is not None:
                return source

        allocation = self.sources.get("allocation")
        filename = ALLOCATION_FILES.get(canonical)
        if allocation is not None and filename:
            files = allocation.metadata.get("files", {})
            filename = str(files.get(canonical, filename)) if isinstance(files, Mapping) else filename
            allocation_path = allocation.resolved_path()
            if allocation_path.is_dir():
                # The directory hash is authoritative for derived files.
                return SnapshotSource(
                    path=str((allocation_path / filename).resolve()),
                    sha256=allocation.sha256,
                    provider=allocation.provider,
                    metadata={"derived_from": "allocation", "filename": filename},
                    base_dir=allocation_path,
                )
        raise SnapshotManifestError(f"manifest has no source for {key}")

    def as_dict(self) -> dict[str, object]:
        sources = {
            key: source.as_dict()
            for key, source in sorted(self.sources.items())
        }
        result: dict[str, object] = {
            "manifest_id": self.manifest_id,
            "manifest_version": self.manifest_version,
            "created_at": self.created_at,
            "cutoff_date": self.cutoff_date.isoformat(),
            "common_dates": [item.isoformat() for item in self.common_dates],
            "supported_classes": list(self.supported_classes),
            "sources": sources,
        }
        for key in ("allocation", "stock", "fii"):
            if key in sources:
                result[f"{key}_source"] = sources[key]
        return result

    to_dict = as_dict


def _coerce_manifest(manifest: SnapshotManifest | Mapping[str, object] | Path) -> SnapshotManifest:
    if isinstance(manifest, SnapshotManifest):
        return manifest
    if isinstance(manifest, (str, Path)):
        return SnapshotManifest.load(Path(manifest))
    return SnapshotManifest.from_dict(manifest)


def _validate_dates(manifest: SnapshotManifest) -> None:
    if not manifest.common_dates:
        raise SnapshotManifestError("manifest common_dates cannot be empty")
    if tuple(sorted(set(manifest.common_dates))) != manifest.common_dates:
        raise SnapshotManifestError("manifest common_dates must be sorted and unique")
    if manifest.common_dates[-1] != manifest.cutoff_date:
        raise SnapshotManifestError(
            "manifest cutoff_date must equal the last declared common date"
        )

    for key, source in manifest.sources.items():
        metadata = source.metadata
        declared_dates = metadata.get("common_dates")
        if declared_dates is not None:
            if not isinstance(declared_dates, (list, tuple)):
                raise SnapshotManifestError(f"sources.{key}.common_dates must be a list")
            parsed = tuple(_as_date(item, f"sources.{key}.common_dates item") for item in declared_dates)
            if parsed != manifest.common_dates:
                raise SnapshotManifestError(
                    f"source {key} common dates do not match manifest"
                )
        for field_name in ("start_date", "end_date", "cutoff_date"):
            if field_name in metadata:
                source_date = _as_date(metadata[field_name], f"sources.{key}.{field_name}")
                if field_name == "start_date" and source_date > manifest.common_dates[0]:
                    raise SnapshotManifestError(f"source {key} starts after common dates")
                if field_name in {"end_date", "cutoff_date"} and source_date < manifest.cutoff_date:
                    raise SnapshotManifestError(f"source {key} ends before manifest cutoff")


def validate_manifest(
    manifest: SnapshotManifest | Mapping[str, object] | Path,
    *,
    required_classes: Sequence[str] = ASSET_CLASSES,
    verify_hashes: bool = True,
) -> SnapshotManifest:
    """Validate schema, source availability, hashes, and date compatibility.

    Returns the normalized manifest so an API adapter can retain its immutable
    ID and source metadata.  No source is fetched or substituted.
    """

    checked = _coerce_manifest(manifest)
    if checked.manifest_version != MANIFEST_SCHEMA_VERSION:
        raise SnapshotManifestError(
            f"unsupported manifest version: {checked.manifest_version}"
        )
    if not set(required_classes).issubset(set(checked.supported_classes)):
        missing = sorted(set(required_classes) - set(checked.supported_classes))
        raise SnapshotManifestError(f"manifest does not support classes: {missing}")
    _validate_dates(checked)

    required_source_keys = tuple(required_classes) + ("ptax",)
    for key in required_source_keys:
        source = checked.source_for(key)
        source_path = source.resolved_path()
        if not source_path.exists():
            raise SnapshotManifestError(f"source {key} not found: {source_path}")
        if verify_hashes:
            actual = sha256_path(source_path)
            # Derived files inherit a directory hash and are checked by the
            # allocation directory itself, not against the directory digest.
            if source.metadata.get("derived_from") == "allocation":
                allocation = checked.sources.get("allocation")
                if allocation is None:
                    raise SnapshotManifestError("derived source has no allocation source")
                allocation_path = allocation.resolved_path()
                allocation_actual = sha256_path(allocation_path)
                if allocation_actual != allocation.sha256:
                    raise SnapshotManifestError(
                        f"SHA-256 mismatch for source allocation: {allocation_path}"
                    )
            elif actual != source.sha256:
                raise SnapshotManifestError(
                    f"SHA-256 mismatch for source {key}: {source_path}"
                )
    # Allocation files cannot double as the security-selection universes.
    for key in ("stock", "fii"):
        source = checked.sources.get(key)
        if source is None:
            raise SnapshotManifestError(
                f"manifest has no explicit {key} selector source"
            )
        source_path = source.resolved_path()
        if not source_path.exists():
            raise SnapshotManifestError(f"source {key} not found: {source_path}")
        if verify_hashes and sha256_path(source_path) != source.sha256:
            raise SnapshotManifestError(
                f"SHA-256 mismatch for source {key}: {source_path}"
            )
    return checked


class SnapshotRegistry:
    """Filesystem registry selecting the latest compatible market cutoff."""

    def __init__(self, registry_dir: Path):
        self.registry_dir = Path(registry_dir)

    def manifests(self) -> Tuple[SnapshotManifest, ...]:
        if not self.registry_dir.exists() or not self.registry_dir.is_dir():
            raise SnapshotManifestError(f"manifest registry not found: {self.registry_dir}")
        manifests = []
        for path in sorted(self.registry_dir.glob(MANIFEST_FILE_GLOB)):
            if path.name == "metadata.json":
                continue
            manifests.append(SnapshotManifest.load(path))
        return tuple(manifests)

    def latest_compatible(
        self,
        *,
        required_classes: Sequence[str] = ASSET_CLASSES,
        max_cutoff_date: Optional[date] = None,
    ) -> SnapshotManifest:
        candidates = [
            manifest
            for manifest in self.manifests()
            if set(required_classes).issubset(set(manifest.supported_classes))
            and (max_cutoff_date is None or manifest.cutoff_date <= max_cutoff_date)
        ]
        candidates.sort(
            key=lambda item: (item.cutoff_date, item.created_at, item.manifest_id),
            reverse=True,
        )
        if not candidates:
            raise SnapshotManifestError("no compatible snapshot manifest found")
        # Do not silently fall back when the newest market snapshot changed.
        return validate_manifest(candidates[0], required_classes=required_classes)


ManifestRegistry = SnapshotRegistry


def latest_compatible_manifest(
    registry_dir: Path,
    *,
    required_classes: Sequence[str] = ASSET_CLASSES,
    max_cutoff_date: Optional[date] = None,
) -> SnapshotManifest:
    """Select by declared market cutoff, never by ``date.today()``."""

    return SnapshotRegistry(registry_dir).latest_compatible(
        required_classes=required_classes,
        max_cutoff_date=max_cutoff_date,
    )


load_manifest = SnapshotManifest.load
