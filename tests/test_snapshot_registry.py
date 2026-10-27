import contextlib
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))
sys.path.insert(0, str(Path(__file__).parent))

import snapshot_registry  # noqa: E402
from allocation_config import ASSET_CLASSES  # noqa: E402
from allocation_data import load_premium_snapshot_bundle  # noqa: E402
from snapshot_manifest import (  # noqa: E402
    SnapshotManifest,
    SnapshotManifestError,
    SnapshotRegistry,
    latest_compatible_manifest,
    manifest_canonical_sha256,
    validate_manifest,
)
from snapshot_registry import (  # noqa: E402
    AlreadyPublishedError,
    InvalidTransitionError,
    rewrite_envelope,
    transition_state,
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


PLAIN_DATES = ["2020-01-01", "2020-01-02", "2020-01-03"]
CREATED_AT = "2020-01-04T12:00:00+00:00"


def write_levels_file(path: Path, columns: list, dates: list) -> None:
    lines = ["date," + ",".join(columns)]
    for index, current_date in enumerate(dates, start=1):
        values = [str(100.0 + index + column_index) for column_index in range(len(columns))]
        lines.append(",".join([current_date, *values]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_fetch_inputs(root: Path, **date_overrides) -> tuple:
    """Lay out a fetch-style snapshot (``data/<name>/``) inside a registry."""

    snapshot_dir = root / "data" / "snapshot-2020"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    dates = {
        name: date_overrides.get(f"{name}_dates", PLAIN_DATES)
        for name in ("stocks", "fiis", "sp500", "di", "btc", "ptax")
    }
    write_levels_file(snapshot_dir / "caio_stocks.csv", ["AAA3", "BBB3"], dates["stocks"])
    write_levels_file(snapshot_dir / "caio_fiis.csv", ["AAA11", "BBB11"], dates["fiis"])
    write_levels_file(snapshot_dir / "sp500_total_return_usd.csv", ["value"], dates["sp500"])
    write_levels_file(snapshot_dir / "di.csv", ["value"], dates["di"])
    write_levels_file(snapshot_dir / "btc_usd.csv", ["value"], dates["btc"])
    write_levels_file(snapshot_dir / "ptax.csv", ["value"], dates["ptax"])
    stock_universe = root / "stocks.csv"
    stock_universe.write_text("TICKER\nAAA3\n", encoding="utf-8")
    fii_universe = root / "fiis.csv"
    fii_universe.write_text("TICKER\nAAA11\n", encoding="utf-8")
    return snapshot_dir, stock_universe, fii_universe


def run_cli(*argv) -> tuple:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = snapshot_registry.main(list(argv))
    return code, out.getvalue(), err.getvalue()


def run_build(root: Path, manifest_id: str = "snap-2020-01", **date_overrides) -> tuple:
    snapshot_dir, stock_universe, fii_universe = build_fetch_inputs(
        root, **date_overrides
    )
    return run_cli(
        "build",
        "--registry",
        str(root),
        "--manifest-id",
        manifest_id,
        "--created-at",
        CREATED_AT,
        "--snapshot-dir",
        str(snapshot_dir),
        "--stock-universe",
        str(stock_universe),
        "--fii-universe",
        str(fii_universe),
    )


class BuildCommandTests(unittest.TestCase):
    def test_build_writes_valid_draft_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code, _, err = run_build(root)

            self.assertEqual(code, 0, err)
            target = root / "snap-2020-01.json"
            raw = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(raw["state"], "draft")
            self.assertEqual(raw["state_changed_at"], CREATED_AT)
            self.assertEqual(raw["supported_classes"], list(ASSET_CLASSES))
            manifest = SnapshotManifest.load(target)
            validated = validate_manifest(manifest)
            self.assertEqual(validated.manifest_sha256, raw["manifest_sha256"])
            self.assertEqual(raw["manifest_sha256"], manifest_canonical_sha256(manifest))

    def test_build_is_byte_reproducible_across_registries(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            first, second = base / "registry-one", base / "registry-two"
            for registry in (first, second):
                registry.mkdir()
                code, _, err = run_build(registry)
                self.assertEqual(code, 0, err)

            self.assertEqual(
                (first / "snap-2020-01.json").read_bytes(),
                (second / "snap-2020-01.json").read_bytes(),
            )

    def test_build_common_dates_match_premium_bundle_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code, _, err = run_build(
                root,
                stocks_dates=["2020-01-01", "2020-01-02", "2020-01-04", "2020-01-05"],
                fiis_dates=["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"],
                sp500_dates=["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"],
                di_dates=["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"],
                ptax_dates=["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
                btc_dates=[
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-04",
                    "2020-01-05",
                    "2020-01-06",
                ],
            )
            self.assertEqual(code, 0, err)
            raw = json.loads((root / "snap-2020-01.json").read_text(encoding="utf-8"))

            bundle = load_premium_snapshot_bundle(
                SnapshotManifest.load(root / "snap-2020-01.json")
            )

            row_dates = tuple(row.date.isoformat() for row in bundle.rows)
            self.assertEqual(row_dates, tuple(raw["common_dates"]))
            self.assertEqual(
                raw["common_dates"], ["2020-01-01", "2020-01-02", "2020-01-04"]
            )
            self.assertEqual(raw["cutoff_date"], "2020-01-04")

    def test_build_empty_intersection_exits_4(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code, _, err = run_build(
                root,
                ptax_dates=["2021-01-01", "2021-01-02"],
            )

            self.assertEqual(code, 4)
            self.assertFalse((root / "snap-2020-01.json").exists())

    def test_build_requires_explicit_created_at(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot_dir, stock_universe, fii_universe = build_fetch_inputs(root)
            common = [
                "build",
                "--registry",
                str(root),
                "--manifest-id",
                "snap-2020-01",
                "--snapshot-dir",
                str(snapshot_dir),
                "--stock-universe",
                str(stock_universe),
                "--fii-universe",
                str(fii_universe),
            ]

            code, _, _ = run_cli(*common)
            self.assertEqual(code, 2)

            code, _, _ = run_cli(*common, "--created-at", "not-a-datetime")
            self.assertEqual(code, 2)
            self.assertFalse((root / "snap-2020-01.json").exists())

    def test_manifest_id_rejects_traversal_and_reserved_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for manifest_id in ("../../x", "metadata", ".hidden"):
                with self.subTest(manifest_id=manifest_id):
                    code, _, err = run_build(root, manifest_id=manifest_id)
                    self.assertEqual(code, 2, err)
            self.assertEqual(sorted(root.glob("*.json")), [])

    def test_build_input_escaping_registry_root_exits_2(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "registry"
            root.mkdir()
            outside = base / "outside"
            outside.mkdir()
            write_levels_file(outside / "stocks.csv", ["TICKER"], PLAIN_DATES)

            code, _, err = run_cli(
                "build",
                "--registry",
                str(root),
                "--manifest-id",
                "snap-2020-01",
                "--created-at",
                CREATED_AT,
                "--snapshot-dir",
                str(outside),
                "--stock-universe",
                str(outside / "stocks.csv"),
                "--fii-universe",
                str(outside / "stocks.csv"),
            )

            self.assertEqual(code, 2, err)
            self.assertEqual(sorted(root.glob("*.json")), [])

    def test_build_existing_target_or_missing_source_exits_3(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code, _, err = run_build(root)
            self.assertEqual(code, 0, err)
            target = root / "snap-2020-01.json"
            original = target.read_bytes()

            code, _, err = run_build(root)
            self.assertEqual(code, 3)
            self.assertEqual(target.read_bytes(), original)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot_dir, stock_universe, fii_universe = build_fetch_inputs(root)
            (snapshot_dir / "caio_fiis.csv").unlink()
            (snapshot_dir / "ifix.csv").write_text(
                "date,value\n2020-01-01,100\n", encoding="utf-8"
            )

            code, _, err = run_cli(
                "build",
                "--registry",
                str(root),
                "--manifest-id",
                "snap-2020-01",
                "--created-at",
                CREATED_AT,
                "--snapshot-dir",
                str(snapshot_dir),
                "--stock-universe",
                str(stock_universe),
                "--fii-universe",
                str(fii_universe),
            )

            self.assertEqual(code, 3)
            self.assertIn("caio_fiis.csv", err)
            self.assertFalse((root / "snap-2020-01.json").exists())

    def test_failed_replace_keeps_original_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = build_manifest_fixture(root)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            original = manifest_path.read_bytes()

            with mock.patch("os.replace", side_effect=OSError("replace failed")):
                with self.assertRaises(OSError):
                    rewrite_envelope(
                        manifest_path,
                        lambda raw: {**raw, "state": "validated"},
                    )

            self.assertEqual(manifest_path.read_bytes(), original)
            leftovers = [item for item in root.iterdir() if item.name.endswith(".tmp")]
            self.assertEqual(leftovers, [])

    def test_build_source_metadata_is_fixed_and_absolute_path_free(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code, _, err = run_build(root)
            self.assertEqual(code, 0, err)
            text = (root / "snap-2020-01.json").read_text(encoding="utf-8")
            raw = json.loads(text)

            allocation = raw["sources"]["allocation"]
            self.assertEqual(allocation["provider"], "snapshot-build:allocation")
            self.assertEqual(
                allocation["metadata"],
                {
                    "files": {
                        "brazilian_stocks": "caio_stocks.csv",
                        "crypto": "btc_usd.csv",
                        "fiis": "caio_fiis.csv",
                        "fixed_income": "di.csv",
                        "international_equity": "sp500_total_return_usd.csv",
                        "ptax": "ptax.csv",
                    }
                },
            )
            self.assertEqual(allocation["path"], "data/snapshot-2020")
            for role in ("stock", "fii"):
                self.assertEqual(
                    raw["sources"][role]["provider"], "snapshot-build:selector"
                )
                self.assertEqual(raw["sources"][role]["metadata"], {})
            self.assertNotIn(str(root), text)
            self.assertNotIn("portfolio_path", text)
            self.assertNotIn("portfolio_artifacts", text)

    def test_cli_source_has_no_network_imports(self):
        source = (PROJECT_ROOT / "py" / "snapshot_registry.py").read_text(
            encoding="utf-8"
        )
        self.assertIsNone(
            re.search(r"urllib|socket|requests|http", source, re.IGNORECASE)
        )

    def test_registry_root_layout_keeps_loader_glob_intact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code, _, err = run_build(root)
            self.assertEqual(code, 0, err)

            self.assertEqual(
                sorted(path.name for path in root.glob("*.json")),
                ["snap-2020-01.json"],
            )
            self.assertTrue((root / "data" / "snapshot-2020").is_dir())
            loaded = SnapshotRegistry(root).manifests()
            self.assertEqual(
                [manifest.manifest_id for manifest in loaded], ["snap-2020-01"]
            )


class LifecycleHelperTests(unittest.TestCase):
    def test_transition_state_enforces_the_state_machine(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = build_manifest_fixture(root)
            at = "2020-01-05T00:00:00+00:00"

            validated = transition_state(dict(base), "validate", at=at)
            self.assertEqual(validated["state"], "validated")
            self.assertEqual(validated["state_changed_at"], at)
            self.assertTrue(validated["manifest_sha256"])
            drafted = transition_state(
                dict(base, **DRAFT_ENVELOPE), "validate", at=at
            )
            self.assertEqual(drafted["state"], "validated")

            noop = transition_state(dict(validated), "validate", at="2020-02-02T00:00:00+00:00")
            self.assertEqual(noop, validated)

            published = transition_state(dict(validated), "publish", at=at)
            self.assertEqual(published["state"], "published")
            with self.assertRaises(AlreadyPublishedError):
                transition_state(dict(published), "publish", at=at)

            retired = transition_state(
                dict(published), "retire", at=at, reason="superseded"
            )
            self.assertEqual(retired["state"], "retired")
            self.assertEqual(retired["retire_reason"], "superseded")

            for action, payload in (
                ("publish", dict(base)),
                ("publish", dict(RETIRED_ENVELOPE, **base)),
                ("retire", dict(base)),
                ("retire", dict(RETIRED_ENVELOPE, **base)),
                ("validate", dict(published)),
                ("validate", dict(RETIRED_ENVELOPE, **base)),
            ):
                with self.subTest(action=action, state=payload.get("state")):
                    with self.assertRaises(InvalidTransitionError):
                        transition_state(payload, action, at=at)

    def test_rewrite_envelope_preserves_unknown_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = build_manifest_fixture(root)
            payload["operator_note"] = "kept verbatim"
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            updated = rewrite_envelope(
                manifest_path,
                lambda raw: transition_state(
                    raw, "validate", at="2020-01-06T00:00:00+00:00"
                ),
            )

            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(raw["operator_note"], "kept verbatim")
            self.assertEqual(raw["state"], "validated")
            self.assertEqual(raw["state_changed_at"], "2020-01-06T00:00:00+00:00")
            for key in ("manifest_id", "created_at", "cutoff_date", "common_dates"):
                self.assertEqual(raw[key], payload[key])
            self.assertEqual(updated["operator_note"], "kept verbatim")


def run_validate(root: Path, manifest_id: str = "snap-2020-01", *flags) -> tuple:
    return run_cli(
        "validate", "--registry", str(root), "--manifest-id", manifest_id, *flags
    )


def load_raw(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_raw(path: Path, raw: dict) -> None:
    path.write_text(json.dumps(raw), encoding="utf-8")


def rehash_manifest(raw: dict) -> dict:
    raw["manifest_sha256"] = manifest_canonical_sha256(raw)
    return raw


def set_manifest_state(path: Path, state: str) -> None:
    raw = load_raw(path)
    raw["state"] = state
    raw["state_changed_at"] = "2020-01-05T00:00:00+00:00"
    dump_raw(path, raw)


class ValidateCommandTests(unittest.TestCase):
    def test_validate_rejects_divergent_source_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code, _, err = run_build(root)
            self.assertEqual(code, 0, err)
            target = root / "snap-2020-01.json"
            original = target.read_bytes()
            (root / "stocks.csv").write_text("TICKER\nCHANGED\n", encoding="utf-8")

            code, _, err = run_validate(root)
            self.assertEqual(code, 4)
            self.assertEqual(target.read_bytes(), original)

            code, _, err = run_validate(root, "snap-2020-01", "--check")
            self.assertEqual(code, 4)
            self.assertEqual(target.read_bytes(), original)

    def test_validate_rejects_cutoff_date_divergence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code, _, err = run_build(root)
            self.assertEqual(code, 0, err)
            target = root / "snap-2020-01.json"
            raw = load_raw(target)
            raw["cutoff_date"] = "2020-01-02"
            rehash_manifest(raw)
            dump_raw(target, raw)

            code, _, err = run_validate(root)

            self.assertEqual(code, 4)
            self.assertEqual(load_raw(target)["cutoff_date"], "2020-01-02")

    def test_validate_requires_caio_fiis_without_ifix_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code, _, err = run_build(root)
            self.assertEqual(code, 0, err)
            target = root / "snap-2020-01.json"
            snapshot_dir = root / "data" / "snapshot-2020"
            (snapshot_dir / "ifix.csv").write_text(
                "date,value\n2020-01-01,100\n", encoding="utf-8"
            )
            raw = load_raw(target)
            raw["supported_classes"] = [
                name for name in raw["supported_classes"] if name != "fiis"
            ]
            rehash_manifest(raw)
            dump_raw(target, raw)

            code, _, err = run_validate(root)
            self.assertEqual(code, 4)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code, _, err = run_build(root)
            self.assertEqual(code, 0, err)
            snapshot_dir = root / "data" / "snapshot-2020"
            (snapshot_dir / "caio_fiis.csv").unlink()
            (snapshot_dir / "ifix.csv").write_text(
                "date,value\n2020-01-01,100\n", encoding="utf-8"
            )

            code, _, err = run_validate(root)

            self.assertEqual(code, 4)

    def test_validate_missing_source_file_exits_4(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code, _, err = run_build(root)
            self.assertEqual(code, 0, err)
            target = root / "snap-2020-01.json"
            (root / "data" / "snapshot-2020" / "caio_fiis.csv").unlink()
            original = target.read_bytes()

            for flags in ((), ("--check",)):
                with self.subTest(flags=flags):
                    code, _, err = run_validate(root, "snap-2020-01", *flags)
                    self.assertEqual(code, 4)
                    self.assertEqual(target.read_bytes(), original)

    def test_check_runs_full_validation_not_just_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code, _, err = run_build(root)
            self.assertEqual(code, 0, err)
            target = root / "snap-2020-01.json"
            raw = load_raw(target)
            raw["supported_classes"] = [
                name for name in raw["supported_classes"] if name != "fiis"
            ]
            rehash_manifest(raw)
            dump_raw(target, raw)

            self.assertEqual(run_validate(root)[0], 4)
            self.assertEqual(run_validate(root, "snap-2020-01", "--check")[0], 4)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code, _, err = run_build(root)
            self.assertEqual(code, 0, err)
            target = root / "snap-2020-01.json"
            raw = load_raw(target)
            raw["cutoff_date"] = "2020-01-02"
            rehash_manifest(raw)
            dump_raw(target, raw)

            self.assertEqual(run_validate(root)[0], 4)
            self.assertEqual(run_validate(root, "snap-2020-01", "--check")[0], 4)

    def test_validate_success_sets_validated_and_preserves_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code, _, err = run_build(root)
            self.assertEqual(code, 0, err)
            target = root / "snap-2020-01.json"
            raw = load_raw(target)
            raw["operator_note"] = "kept verbatim"
            dump_raw(target, raw)
            declared_hash = raw["manifest_sha256"]

            code, _, err = run_validate(root)

            self.assertEqual(code, 0, err)
            after = load_raw(target)
            self.assertEqual(after["state"], "validated")
            self.assertEqual(after["manifest_sha256"], declared_hash)
            self.assertEqual(after["operator_note"], "kept verbatim")

    def test_validate_on_validated_is_idempotent_with_no_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code, _, err = run_build(root)
            self.assertEqual(code, 0, err)
            self.assertEqual(run_validate(root)[0], 0)
            target = root / "snap-2020-01.json"
            frozen = target.read_bytes()

            code, _, err = run_validate(root)

            self.assertEqual(code, 0, err)
            self.assertEqual(target.read_bytes(), frozen)

    def test_verify_published_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for state in ("draft", "validated", "published", "retired"):
                with self.subTest(state=state):
                    root = base / state
                    root.mkdir()
                    code, _, err = run_build(root)
                    self.assertEqual(code, 0, err)
                    target = root / "snap-2020-01.json"
                    set_manifest_state(target, state)

                    code, _, err = run_validate(root, "snap-2020-01", "--check")
                    self.assertEqual(code, 0, err)

                    raw = load_raw(target)
                    raw["created_at"] = "2020-02-01T00:00:00+00:00"
                    dump_raw(target, raw)
                    tampered = target.read_bytes()

                    code, _, err = run_validate(root, "snap-2020-01", "--check")
                    self.assertEqual(code, 4)
                    self.assertEqual(target.read_bytes(), tampered)

    def test_validate_on_published_or_retired_exits_5_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for state in ("published", "retired"):
                with self.subTest(state=state):
                    root = base / state
                    root.mkdir()
                    code, _, err = run_build(root)
                    self.assertEqual(code, 0, err)
                    target = root / "snap-2020-01.json"
                    set_manifest_state(target, state)
                    raw = load_raw(target)
                    raw["created_at"] = "2020-02-01T00:00:00+00:00"
                    dump_raw(target, raw)
                    frozen = target.read_bytes()

                    code, _, err = run_validate(root)

                    self.assertEqual(code, 5)
                    self.assertEqual(target.read_bytes(), frozen)

    def test_validate_backfills_manifest_sha256_on_legacy_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code, _, err = run_build(root)
            self.assertEqual(code, 0, err)
            target = root / "snap-2020-01.json"
            raw = load_raw(target)
            for key in ("state", "state_changed_at", "manifest_sha256"):
                raw.pop(key, None)
            raw["operator_note"] = "kept verbatim"
            dump_raw(target, raw)

            code, _, err = run_validate(root)

            self.assertEqual(code, 0, err)
            after = load_raw(target)
            self.assertEqual(after["state"], "validated")
            self.assertEqual(after["manifest_sha256"], manifest_canonical_sha256(after))
            self.assertEqual(after["operator_note"], "kept verbatim")

    def test_manifest_id_mismatch_with_loaded_manifest_exits_2(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code, _, err = run_build(root)
            self.assertEqual(code, 0, err)
            (root / "snap-2020-01.json").rename(root / "renamed.json")

            code, _, err = run_validate(root, "renamed")

            self.assertEqual(code, 2)
            self.assertIn("differs from loaded manifest_id", err)


def run_publish(root: Path, manifest_id: str = "snap-2020-01") -> tuple:
    return run_cli("publish", "--registry", str(root), "--manifest-id", manifest_id)


class PublishCommandTests(unittest.TestCase):
    def test_publish_transitions_validated_to_published(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code, _, err = run_build(root)
            self.assertEqual(code, 0, err)
            self.assertEqual(run_validate(root)[0], 0)
            target = root / "snap-2020-01.json"
            before = load_raw(target)

            code, _, err = run_publish(root)

            self.assertEqual(code, 0, err)
            after = load_raw(target)
            self.assertEqual(after["state"], "published")
            self.assertEqual(after["manifest_sha256"], before["manifest_sha256"])
            before_content = {
                key: value
                for key, value in before.items()
                if key not in ("state", "state_changed_at")
            }
            after_content = {
                key: value
                for key, value in after.items()
                if key not in ("state", "state_changed_at")
            }
            self.assertEqual(after_content, before_content)

    def test_publish_never_overwrites_published_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code, _, err = run_build(root)
            self.assertEqual(code, 0, err)
            self.assertEqual(run_validate(root)[0], 0)
            self.assertEqual(run_publish(root)[0], 0)
            target = root / "snap-2020-01.json"
            frozen = target.read_bytes()

            code, _, err = run_publish(root)

            self.assertEqual(code, 3)
            self.assertEqual(target.read_bytes(), frozen)

    def test_publish_refuses_draft_and_retired(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for state in ("draft", "retired"):
                with self.subTest(state=state):
                    root = base / state
                    root.mkdir()
                    code, _, err = run_build(root)
                    self.assertEqual(code, 0, err)
                    target = root / "snap-2020-01.json"
                    if state == "retired":
                        self.assertEqual(run_validate(root)[0], 0)
                        set_manifest_state(target, "retired")
                    frozen = target.read_bytes()

                    code, _, err = run_publish(root)

                    self.assertEqual(code, 5)
                    self.assertEqual(target.read_bytes(), frozen)


if __name__ == "__main__":
    unittest.main()
