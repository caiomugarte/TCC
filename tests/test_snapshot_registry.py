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


if __name__ == "__main__":
    unittest.main()
