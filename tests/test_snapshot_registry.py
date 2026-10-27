import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))
sys.path.insert(0, str(Path(__file__).parent))

from snapshot_manifest import (  # noqa: E402
    SnapshotManifest,
    SnapshotManifestError,
    SnapshotRegistry,
    latest_compatible_manifest,
    manifest_canonical_sha256,
    validate_manifest,
)
from test_snapshot_manifest import build_manifest_fixture  # noqa: E402


DRAFT_ENVELOPE = {
    "state": "draft",
    "state_changed_at": "2020-01-04T12:00:00+00:00",
    "manifest_sha256": "a" * 64,
}

RETIRED_ENVELOPE = {
    "state": "retired",
    "state_changed_at": "2020-01-06T09:30:00+00:00",
    "retire_reason": "superseded by newer snapshot",
    "manifest_sha256": "b" * 64,
}


class LifecycleEnvelopeTests(unittest.TestCase):
    def test_lifecycle_fields_round_trip_and_absent_fields_are_omitted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = build_manifest_fixture(root)

            legacy = SnapshotManifest.from_dict(payload)
            self.assertIsNone(legacy.state)
            self.assertIsNone(legacy.state_changed_at)
            self.assertIsNone(legacy.retire_reason)
            self.assertIsNone(legacy.manifest_sha256)
            for key in ("state", "state_changed_at", "retire_reason", "manifest_sha256"):
                self.assertNotIn(key, legacy.as_dict())

            enriched = SnapshotManifest.from_dict(dict(payload, **RETIRED_ENVELOPE))
            self.assertEqual(enriched.state, "retired")
            self.assertEqual(enriched.state_changed_at, "2020-01-06T09:30:00+00:00")
            self.assertEqual(enriched.retire_reason, "superseded by newer snapshot")
            self.assertEqual(enriched.manifest_sha256, "b" * 64)
            emitted = enriched.as_dict()
            for key, value in RETIRED_ENVELOPE.items():
                self.assertEqual(emitted[key], value)

    def test_unknown_state_string_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = build_manifest_fixture(root)
            payload["state"] = "frozen"

            with self.assertRaisesRegex(SnapshotManifestError, "unknown manifest state"):
                SnapshotManifest.from_dict(payload)

    def test_legacy_manifest_without_state_loads_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = build_manifest_fixture(root)
            manifest_path = root / "legacy.json"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            loaded = SnapshotManifest.load(manifest_path)

            self.assertIsNone(loaded.state)
            self.assertEqual(loaded.as_dict(), SnapshotManifest.from_dict(payload).as_dict())

    def test_from_dict_as_dict_round_trip_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = build_manifest_fixture(root)
            variants = {
                "legacy": {},
                "legacy-with-unknown-key": {"operator_note": "kept out of the projection"},
                "draft": DRAFT_ENVELOPE,
                "retired": RETIRED_ENVELOPE,
            }
            for label, extra in variants.items():
                with self.subTest(label=label):
                    manifest = SnapshotManifest.from_dict(dict(base, **extra))
                    self.assertEqual(
                        SnapshotManifest.from_dict(manifest.as_dict()).as_dict(),
                        manifest.as_dict(),
                    )


class CanonicalManifestHashTests(unittest.TestCase):
    def test_manifest_hash_survives_transitions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = build_manifest_fixture(root)

            baseline = manifest_canonical_sha256(payload)
            reordered = dict(reversed(list(payload.items())))
            self.assertEqual(manifest_canonical_sha256(reordered), baseline)
            for envelope in (DRAFT_ENVELOPE, RETIRED_ENVELOPE):
                self.assertEqual(
                    manifest_canonical_sha256(dict(payload, **envelope)),
                    baseline,
                )

    def test_manifest_hash_survives_premium_rehydration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = build_manifest_fixture(root)
            payload["manifest_sha256"] = manifest_canonical_sha256(payload)

            # Persist exactly as api/app/services/premium_recommendation.py
            # persists as_dict(), then rehydrate as premium_executor does.
            persisted = json.loads(json.dumps(SnapshotManifest.from_dict(payload).as_dict()))
            persisted["manifest_path"] = str(root / "manifest.json")
            rehydrated = SnapshotManifest.from_dict(persisted, base_dir=root)

            validated = validate_manifest(rehydrated)
            self.assertEqual(validated.manifest_sha256, payload["manifest_sha256"])
            self.assertEqual(manifest_canonical_sha256(persisted), payload["manifest_sha256"])

    def test_validate_manifest_accepts_matching_hash_and_rejects_divergence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = build_manifest_fixture(root)
            payload["manifest_sha256"] = manifest_canonical_sha256(payload)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            self.assertEqual(
                validate_manifest(manifest_path).manifest_sha256,
                payload["manifest_sha256"],
            )

            payload["manifest_sha256"] = "c" * 64
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                SnapshotManifestError, "manifest_sha256 does not match"
            ):
                validate_manifest(manifest_path)

    def test_legacy_manifest_without_manifest_sha256_still_validates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = build_manifest_fixture(root)
            self.assertNotIn("manifest_sha256", payload)
            legacy_path = root / "legacy.json"
            legacy_path.write_text(json.dumps(payload), encoding="utf-8")

            validated = validate_manifest(legacy_path)

            self.assertIsNone(validated.manifest_sha256)


def _manifest_variant(
    base: dict,
    manifest_id: str,
    cutoff_date: str,
    *,
    state=None,
    created_at: str = "2020-01-04T12:00:00+00:00",
) -> dict:
    common_dates = [item for item in base["common_dates"] if item <= cutoff_date]
    payload = dict(
        base,
        manifest_id=manifest_id,
        created_at=created_at,
        common_dates=common_dates,
        cutoff_date=cutoff_date,
    )
    if state is not None:
        payload["state"] = state
        payload["state_changed_at"] = created_at
    return payload


def _write_manifest(root: Path, payload: dict) -> Path:
    path = root / f"{payload['manifest_id']}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class SelectionStateTests(unittest.TestCase):
    def test_selection_excludes_draft_and_retired(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = build_manifest_fixture(root)
            _write_manifest(root, _manifest_variant(base, "published-old", "2020-01-02", state="published"))
            _write_manifest(root, _manifest_variant(base, "draft-new", "2020-01-03", state="draft"))
            _write_manifest(root, _manifest_variant(base, "retired-new", "2020-01-03", state="retired"))

            selected = latest_compatible_manifest(root)

            self.assertEqual(selected.manifest_id, "published-old")
            self.assertEqual(
                SnapshotRegistry(root).latest_compatible().manifest_id,
                "published-old",
            )

    def test_selection_admits_validated_only_with_include_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = build_manifest_fixture(root)
            _write_manifest(root, _manifest_variant(base, "published-old", "2020-01-02", state="published"))
            _write_manifest(root, _manifest_variant(base, "validated-new", "2020-01-03", state="validated"))

            self.assertEqual(latest_compatible_manifest(root).manifest_id, "published-old")
            self.assertEqual(
                latest_compatible_manifest(root, include_validated=True).manifest_id,
                "validated-new",
            )
            self.assertEqual(
                SnapshotRegistry(root)
                .latest_compatible(include_validated=True)
                .manifest_id,
                "validated-new",
            )

    def test_legacy_only_registry_selection_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = build_manifest_fixture(root)
            for manifest_id, cutoff_date in (("older", "2020-01-02"), ("newer", "2020-01-03")):
                payload = _manifest_variant(base, manifest_id, cutoff_date)
                self.assertNotIn("state", payload)
                _write_manifest(root, payload)

            selected = latest_compatible_manifest(root)

            self.assertEqual(selected.manifest_id, "newer")
            self.assertEqual(selected.cutoff_date.isoformat(), "2020-01-03")

    def test_selection_without_qualifying_candidate_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = build_manifest_fixture(root)
            _write_manifest(root, _manifest_variant(base, "draft-only", "2020-01-03", state="draft"))
            _write_manifest(root, _manifest_variant(base, "retired-only", "2020-01-03", state="retired"))

            with self.assertRaisesRegex(
                SnapshotManifestError, "no compatible snapshot manifest found"
            ):
                latest_compatible_manifest(root)
            with self.assertRaisesRegex(
                SnapshotManifestError, "no compatible snapshot manifest found"
            ):
                latest_compatible_manifest(root, include_validated=True)


if __name__ == "__main__":
    unittest.main()
