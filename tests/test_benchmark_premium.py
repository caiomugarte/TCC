"""Offline tests for scripts/benchmark_premium.py (issue #8)."""

from contextlib import contextmanager
from datetime import date
import io
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import bench_measure as measure  # noqa: E402
import benchmark_premium as bench  # noqa: E402

from snapshot_manifest import SnapshotManifest, manifest_canonical_sha256, sha256_path  # noqa: E402


def _burn(seconds):
    deadline = time.process_time() + seconds
    value = 0
    while time.process_time() < deadline:
        value += 1
    return value


def _manifest_payload(registry, stock_hash, fii_hash, allocation_hash):
    payload = {
        "manifest_id": "bench-fixture",
        "manifest_version": "1",
        "created_at": "2026-10-29T00:00:00+00:00",
        "cutoff_date": "2026-10-28",
        "common_dates": ["2026-10-27", "2026-10-28"],
        "supported_classes": ["brazilian_stocks", "fiis", "international_equity", "fixed_income", "crypto"],
        "sources": {
            "allocation": {"path": "data/snap", "sha256": allocation_hash, "provider": "snapshot-build:allocation", "metadata": {}},
            "stock": {"path": "stocks.csv", "sha256": stock_hash, "provider": "snapshot-build:selector", "metadata": {}},
            "fii": {"path": "fiis.csv", "sha256": fii_hash, "provider": "snapshot-build:selector", "metadata": {}},
        },
    }
    payload["manifest_sha256"] = manifest_canonical_sha256(
        SnapshotManifest.from_dict(payload, base_dir=registry)
    )
    return payload


class _SnapshotFixture:
    """A registry holding one declared snapshot, plus its declaration file."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.registry = Path(self.temp.name) / "snapshots"
        (self.registry / "data" / "snap").mkdir(parents=True)
        (self.registry / "data" / "snap" / "ibovespa.csv").write_text("date,value\n2026-10-28,1\n")
        (self.registry / "stocks.csv").write_text("TICKER\nAAAA3\n")
        (self.registry / "fiis.csv").write_text("TICKER\nBBBB11\n")
        hashes = {
            "allocation": sha256_path(self.registry / "data" / "snap"),
            "stock": sha256_path(self.registry / "stocks.csv"),
            "fii": sha256_path(self.registry / "fiis.csv"),
        }
        payload = _manifest_payload(self.registry, hashes["stock"], hashes["fii"], hashes["allocation"])
        (self.registry / "bench-fixture.json").write_text(json.dumps(payload))
        self.declaration = {
            "manifest_id": "bench-fixture",
            "manifest_sha256": payload["manifest_sha256"],
            "sources": hashes,
        }
        self.declaration_path = Path(self.temp.name) / "declaration.json"
        self.declaration_path.write_text(json.dumps(self.declaration))

    def tearDown(self):
        self.temp.cleanup()

    def _main(self, *extra):
        return bench.main([
            "--declaration", str(self.declaration_path),
            "--registry", str(self.registry),
            *extra,
        ])


class SnapshotDeclarationTests(_SnapshotFixture, unittest.TestCase):
    def test_matching_snapshot_is_accepted(self):
        with patch("snapshot_manifest.validate_manifest", side_effect=lambda manifest: manifest):
            _manifest, report = bench.verify_snapshot(self.declaration, self.registry)
        self.assertEqual(report["sources"], self.declaration["sources"])
        self.assertEqual(report["selector_paths"]["stocks"], str((self.registry / "stocks.csv").resolve()))

    def test_changed_selector_csv_fails(self):
        (self.registry / "stocks.csv").write_text("TICKER\nAAAA3\nCCCC4\n")
        with self.assertRaisesRegex(bench.SnapshotDeclarationError, "SHA-256 mismatch for stock"):
            bench.verify_snapshot(self.declaration, self.registry)

    def test_changed_allocation_series_fails(self):
        (self.registry / "data" / "snap" / "ibovespa.csv").write_text("date,value\n2026-10-28,2\n")
        with self.assertRaisesRegex(bench.SnapshotDeclarationError, "SHA-256 mismatch for allocation"):
            bench.verify_snapshot(self.declaration, self.registry)

    def test_other_manifest_fails(self):
        declaration = dict(self.declaration, manifest_sha256="0" * 64)
        with self.assertRaisesRegex(bench.SnapshotDeclarationError, "manifest_sha256 mismatch"):
            bench.verify_snapshot(declaration, self.registry)

    def test_cli_exits_non_zero_before_running_on_mismatch(self):
        (self.registry / "fiis.csv").write_text("TICKER\nZZZZ11\n")
        with patch.object(bench, "run_profile") as run_profile:
            code = self._main("--output-dir", str(Path(self.temp.name) / "out"), "--isolation", "none")
        self.assertEqual(code, bench.EXIT_SNAPSHOT)
        run_profile.assert_not_called()

    def test_committed_declaration_is_well_formed(self):
        declaration = bench.load_declaration(bench.DEFAULT_DECLARATION)
        self.assertEqual(set(declaration["sources"]), {"allocation", "stock", "fii"})
        for digest in (declaration["manifest_sha256"], *declaration["sources"].values()):
            self.assertRegex(digest, r"^[0-9a-f]{64}$")


class RssSamplerTests(unittest.TestCase):
    def test_samples_at_one_second_and_matches_process_peak(self):
        current = bench.read_proc_status_kb("VmRSS")
        prior_peak = max(bench.ru_maxrss_kb(), bench.read_proc_status_kb("VmHWM") or 0)
        # Hold a block large enough that the new peak dominates any earlier one.
        block_kb = max(192 * 1024, prior_peak - current + 128 * 1024)
        sampler = bench.RssSampler(interval=1.0).start()
        try:
            block = bytearray(b"\x01") * (block_kb * 1024)
            time.sleep(3.2)
            del block
        finally:
            sampler.stop()
        samples = sampler.snapshot()
        # start + ~3 periodic + stop samples.
        self.assertGreaterEqual(len(samples), 4)
        periodic = [b - a for (a, _), (b, _) in zip(samples[:-2], samples[1:-1])]
        for gap in periodic:
            self.assertGreater(gap, 0.9)
            self.assertLess(gap, 1.5)
        check = bench.validate_rss(
            sampler.peak_kb(), bench.ru_maxrss_kb(), bench.read_proc_status_kb("VmHWM")
        )
        self.assertEqual(check["status"], "ok", check)
        self.assertGreaterEqual(sampler.peak_kb(), current + block_kb * 0.9)

    def test_validation_rejects_impossible_and_flags_undersampled(self):
        self.assertEqual(bench.validate_rss(100_000, 100_000, 100_000)["status"], "ok")
        self.assertEqual(bench.validate_rss(98_500, 98_000, 100_000)["status"], "ok")
        self.assertEqual(bench.validate_rss(110_000, 100_000, 100_000)["status"], "invalid")
        self.assertEqual(bench.validate_rss(100_000, 80_000, 100_000)["status"], "invalid")
        self.assertEqual(bench.validate_rss(50_000, 100_000, 100_000)["status"], "undersampled")
        self.assertEqual(bench.validate_rss(None, 100_000, 100_000)["status"], "invalid")

    def test_window_peak_uses_only_samples_inside_the_window(self):
        values = iter([10, 50, 20, 90])
        ticks = iter([1.0, 2.0, 3.0, 4.0])
        sampler = bench.RssSampler(reader=lambda: next(values), clock=lambda: next(ticks))
        for _ in range(4):
            sampler.sample_now()
        self.assertEqual(sampler.peak_kb(1.5, 3.5), 50)
        self.assertEqual(sampler.peak_kb(), 90)
        self.assertEqual(sampler.count(1.5, 3.5), 2)


def _report(wall_by_profile, **overrides):
    report = {
        "schema": bench.SCHEMA,
        "mode": "measured",
        "git": {"commit": "abc", "code_fingerprint": "code", "code_dirty": False},
        "snapshot": {"manifest_sha256": "m" * 64, "sources": {"stock": "s", "fii": "f", "allocation": "a"}},
        "seed": 42,
        "machine": {"fingerprint": "machine"},
        "profiles": {
            profile: {"policy_sha256": f"policy-{profile}", "stages": {"total": {"wall_s": wall}}}
            for profile, wall in wall_by_profile.items()
        },
    }
    report.update(overrides)
    return report


class RegressionGateTests(unittest.TestCase):
    def test_more_than_twenty_percent_slower_is_a_regression(self):
        comparison = bench.compare_to_baseline(
            _report({"moderado": 125.0, "arrojado": 100.0}),
            _report({"moderado": 100.0, "arrojado": 100.0}),
        )
        self.assertEqual(comparison["status"], "regression")
        self.assertEqual(bench.comparison_exit_code(comparison), bench.EXIT_REGRESSION)
        self.assertEqual([row["regression"] for row in comparison["profiles"]], [False, True])

    def test_within_twenty_percent_passes(self):
        for wall in (80.0, 100.0, 115.0, 120.0):
            comparison = bench.compare_to_baseline(_report({"moderado": wall}), _report({"moderado": 100.0}))
            self.assertEqual(bench.comparison_exit_code(comparison), bench.EXIT_OK, wall)

    def test_refuses_to_compare_different_conditions(self):
        baseline = _report({"moderado": 100.0})
        variants = {
            "seed": _report({"moderado": 100.0}, seed=7),
            "machine": _report({"moderado": 100.0}, machine={"fingerprint": "other"}),
            "snapshot": _report({"moderado": 100.0}, snapshot={"manifest_sha256": "x", "sources": {}}),
            "code": _report({"moderado": 100.0}, git={"code_fingerprint": "new", "code_dirty": False}),
            "dirty": _report({"moderado": 100.0}, git={"code_fingerprint": "code", "code_dirty": True}),
            "mode": _report({"moderado": 100.0}, mode="smoke"),
            "profile": _report({"arrojado": 100.0}),
        }
        for label, current in variants.items():
            comparison = bench.compare_to_baseline(current, baseline)
            self.assertEqual(comparison["status"], "not_comparable", label)
            self.assertTrue(comparison["problems"], label)
            self.assertEqual(bench.comparison_exit_code(comparison), bench.EXIT_NOT_COMPARABLE)

    def test_cli_compare_only_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "baseline.json"
            slow = Path(tmp) / "slow.json"
            same = Path(tmp) / "same.json"
            base.write_text(json.dumps(_report({"moderado": 100.0})))
            slow.write_text(json.dumps(_report({"moderado": 121.0})))
            same.write_text(json.dumps(_report({"moderado": 101.0})))
            with patch("sys.stdout"):
                self.assertEqual(bench.main(["--compare-only", str(slow), "--baseline", str(base)]), bench.EXIT_REGRESSION)
                self.assertEqual(bench.main(["--compare-only", str(same), "--baseline", str(base)]), bench.EXIT_OK)


WEIGHTS = {
    "brazilian_stocks": 0.2,
    "fiis": 0.2,
    "international_equity": 0.2,
    "fixed_income": 0.2,
    "crypto": 0.2,
}


def _fake_engines(fii_weight=0.2):
    weights = dict(WEIGHTS, fiis=fii_weight, fixed_income=0.4 - fii_weight)

    def snapshot_loader(_manifest):
        time.sleep(0.05)
        return SimpleNamespace(rows=[], metadata={})

    def allocation(*_args, **_kwargs):
        _burn(0.05)
        return {"current_target": {"selected": {"profile_winner": {"weights": weights}}}}

    def stocks(**_kwargs):
        _burn(0.15)
        runs = [{"generations_run": 120, "converged_early": True}, {"generations_run": 400}]
        return {
            "n_runs": 2,
            "selected_tickers": ["AAAA3", "BBBB4"],
            "sleeve_weights": {"AAAA3": 0.5, "BBBB4": 0.5},
            "all_runs": runs,
        }

    def fiis(**_kwargs):
        time.sleep(0.1)
        for generations in (50, 70, 90):
            CAPTURED.append({"generations_run": generations})
        return {"n_runs": 3, "selected_tickers": ["CCCC11"], "sleeve_weights": {"CCCC11": 1.0}}

    return {
        "snapshot_loader": snapshot_loader,
        "allocation_engine": allocation,
        "stock_engine": stocks,
        "fii_engine": fiis,
    }


CAPTURED = []


@contextmanager
def _fake_capture():
    CAPTURED.clear()
    yield CAPTURED


class StageAccountingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.manifest = SimpleNamespace(
            manifest_id="bench-fixture",
            manifest_version="1",
            cutoff_date=date(2026, 10, 28),
            created_at="2026-10-29T00:00:00+00:00",
            sources={},
        )
        from app.services.status_invest_inputs import SourceSnapshot, StatusInvestInputs

        self.inputs = StatusInvestInputs(
            stocks=SourceSnapshot(Path("/tmp/stocks.csv"), "statusinvest", "t", "s"),
            fiis=SourceSnapshot(Path("/tmp/fiis.csv"), "statusinvest", "t", "f"),
        )

    def tearDown(self):
        self.temp.cleanup()

    def _run(self, profile="moderado", **engine_kwargs):
        policy = bench.build_policy(profile, self.manifest, 42)
        return bench.run_profile(
            profile,
            policy,
            self.manifest,
            self.inputs,
            Path(self.temp.name) / profile,
            seed=42,
            engines=_fake_engines(**engine_kwargs),
            fii_capture=_fake_capture,
            sampler=bench.RssSampler(interval=0.05).start(),
        )

    def test_composition_is_total_minus_engines(self):
        measured = self._run()
        stages = measured["stages"]
        self.assertEqual(
            set(stages), {"snapshot_load", "allocation", "stocks", "fiis", "composition", "total"}
        )
        for field in measure.TIME_FIELDS:
            engines = sum(stages[name][field] for name in bench.ENGINE_STAGES)
            self.assertAlmostEqual(stages["composition"][field], stages["total"][field] - engines, places=9)
        self.assertGreaterEqual(stages["composition"]["wall_s"], 0.0)
        self.assertGreaterEqual(stages["snapshot_load"]["wall_s"], 0.05)
        self.assertGreaterEqual(stages["stocks"]["cpu_seconds"], 0.1)
        self.assertGreaterEqual(stages["fiis"]["wall_s"], 0.1)
        self.assertLess(stages["fiis"]["cpu_seconds"], 0.08)
        for name in (*bench.ENGINE_STAGES, "total"):
            self.assertTrue(stages[name]["executed"])
            self.assertIsNotNone(stages[name]["rss_peak_sampled_kb"])

    def test_generations_and_run_count_only_for_selectors(self):
        measured = self._run()
        engines = measured["engines"]
        self.assertFalse(engines["allocation"]["ga"])
        self.assertNotIn("n_runs_effective", engines["allocation"])
        self.assertEqual(engines["stocks"]["n_runs_effective"], 2)
        self.assertEqual(engines["stocks"]["generations_total"], 520)
        self.assertEqual(engines["stocks"]["converged_early_runs"], 1)
        self.assertEqual(engines["stocks"]["run_count_configured"], 150)
        self.assertEqual(engines["fiis"]["n_runs_effective"], 3)
        self.assertEqual(engines["fiis"]["generations_total"], 210)
        self.assertEqual(engines["fiis"]["generations_max"], 90)
        self.assertEqual(measured["output"]["stocks"], ["AAAA3", "BBBB4"])

    def test_skipped_selector_counts_as_zero(self):
        measured = self._run(fii_weight=0.0)
        fiis = measured["stages"]["fiis"]
        self.assertFalse(fiis["executed"])
        self.assertEqual(fiis["wall_s"], 0.0)
        self.assertIsNone(measured["engines"]["fiis"].get("n_runs_effective"))

    def test_fii_capture_restores_the_pipeline(self):
        import fii_selection

        original = fii_selection._run_fii_execution
        with bench.capture_fii_runs() as captured:
            self.assertIsNot(fii_selection._run_fii_execution, original)
            self.assertEqual(captured, [])
        self.assertIs(fii_selection._run_fii_execution, original)


def _measured(profile, policy, wall_s, rss_status="ok"):
    """A ``run_profile`` result reduced to what ``main`` reads."""

    return {
        "profile": profile,
        "policy_sha256": bench._sha256_json(policy),
        "stages": {"total": {"wall_s": wall_s}},
        "rss": {"status": rss_status},
    }


CLEAN_GIT = {"commit": "abc", "dirty": False, "code_dirty": False, "code_trees": {}, "code_fingerprint": "code"}


class RunPathTests(_SnapshotFixture, unittest.TestCase):
    """``main`` without ``--compare-only``: exit codes after a (faked) run."""

    def setUp(self):
        super().setUp()
        self.patches = [
            patch("snapshot_manifest.validate_manifest", side_effect=lambda manifest: manifest),
            patch.object(bench, "git_info", return_value=dict(CLEAN_GIT)),
            patch.object(bench, "machine_info", return_value={"fingerprint": "machine"}),
            patch.object(bench, "render_markdown", return_value=""),
        ]
        for item in self.patches:
            item.start()
        self.addCleanup(lambda: [item.stop() for item in self.patches])
        self.out = Path(self.temp.name) / "out"

    def _run(self, *extra, wall_s=100.0, rss_status="ok", out=None, side_effect=None):
        def fake_run_profile(profile, policy, *_args, **_kwargs):
            if side_effect is not None:
                raise side_effect
            return _measured(profile, policy, wall_s, rss_status)

        with patch.object(bench, "run_profile", side_effect=fake_run_profile) as run_profile, \
                patch("sys.stderr"):
            code = self._main(
                "--profiles", "moderado",
                "--isolation", "none",
                "--output-dir", str(out or self.out),
                *extra,
            )
        return code, run_profile

    def test_pipeline_failure_reports_the_chained_cause(self):
        try:
            try:
                raise ValueError("root cause: empty stock universe")
            except ValueError as cause:
                raise RuntimeError("stock selector failed") from cause
        except RuntimeError as exc:
            error = exc
        stderr = io.StringIO()
        with patch.object(bench, "run_profile", side_effect=error), patch("sys.stderr", stderr):
            code = self._main("--profiles", "moderado", "--isolation", "none", "--output-dir", str(self.out))
        self.assertEqual(code, bench.EXIT_RUN_FAILED)
        self.assertIn("stock selector failed", stderr.getvalue())
        self.assertIn("root cause: empty stock universe", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
