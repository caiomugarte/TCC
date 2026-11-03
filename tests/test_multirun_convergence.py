"""Tests for scripts/multirun_convergence.py (issue #9).

The fixture registry is built from the committed ``data/allocation`` series and the
``data/raw/status_invest_*.csv`` exports with ``snapshot_registry.build_manifest``;
the fixture GA is population 20, 5 generations, 12 runs (``--smoke``).
"""

from contextlib import contextmanager, redirect_stdout
import copy
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import convergence_analysis as ca  # noqa: E402
import multirun_convergence as mc  # noqa: E402

PROTOCOL_PATH = PROJECT_ROOT / "docs" / "validation" / "multirun-convergence-protocol.json"
PROFILE = "premium-2026-11-02"
SEEDS = [42, 43, 44]
RUN_COUNT = 12

FIXTURE = {}


def build_registry(root):
    from snapshot_registry import build_manifest

    registry = Path(root) / "registry"
    (registry / "data").mkdir(parents=True)
    shutil.copytree(
        PROJECT_ROOT / "data" / "allocation",
        registry / "data" / "snap",
        ignore=shutil.ignore_patterns("caio_stocks.csv", "README.md"),
    )
    shutil.copy(PROJECT_ROOT / "data" / "raw" / "status_invest_fundamentals.csv", registry / "stocks.csv")
    shutil.copy(PROJECT_ROOT / "data" / "raw" / "status_invest_fii.csv", registry / "fiis.csv")
    payload = build_manifest(
        registry_root=registry,
        manifest_id="fixture",
        created_at="2026-10-29T00:00:00+00:00",
        snapshot_dir=registry / "data" / "snap",
        stock_universe=registry / "stocks.csv",
        fii_universe=registry / "fiis.csv",
    )
    (registry / "fixture.json").write_text(json.dumps(payload), encoding="utf-8")
    declaration = {
        "manifest_id": "fixture",
        "manifest_sha256": payload["manifest_sha256"],
        "sources": {role: payload["sources"][role]["sha256"] for role in ("allocation", "stock", "fii")},
    }
    declaration_path = Path(root) / "declaration.json"
    declaration_path.write_text(json.dumps(declaration), encoding="utf-8")
    return registry, declaration_path


def fixture_protocol(declaration_path, **changes):
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol["snapshot_declaration"] = str(declaration_path)
    protocol["seeds"] = list(SEEDS)
    protocol["profiles"] = {PROFILE: protocol["profiles"][PROFILE]}
    protocol["production_run"]["seed"] = SEEDS[0]
    protocol["ga_override"]["run_count"] = RUN_COUNT
    protocol["prefixes"] = [4, 8, 12]
    protocol["curve_step"] = 4
    protocol["production_stop_rule"]["min_runs"] = 4
    protocol["decision"]["run_count"]["candidates"] = [4, 8, 12]
    protocol["decision"]["target"]["prefix"] = 12
    protocol.update(changes)
    return protocol


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")
    return Path(path)


def cli(*args, timeout=600):
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "multirun_convergence.py"), *map(str, args)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def run_main(*args):
    stderr = io.StringIO()
    with patch("sys.stderr", stderr), redirect_stdout(io.StringIO()):
        code = mc.main([str(a) for a in args])
    return code, stderr.getvalue()


def setUpModule():
    temp = tempfile.TemporaryDirectory()
    root = Path(temp.name)
    registry, declaration = build_registry(root)
    protocol_path = write_json(root / "protocol.json", fixture_protocol(declaration))
    FIXTURE.update(temp=temp, root=root, registry=registry, declaration=declaration, protocol=protocol_path)
    for label in ("a", "b"):
        lots = root / f"lots-{label}"
        completed = cli("run", "--protocol", protocol_path, "--registry", registry, "--lots-dir", lots, "--jobs", 3, "--smoke", "--poll-interval", 0.2)
        FIXTURE[f"run_{label}"] = completed
        FIXTURE[f"lots_{label}"] = lots
        report = cli("report", "--protocol", protocol_path, "--lots-dir", lots, "--out-dir", root / f"out-{label}")
        FIXTURE[f"report_{label}"] = report
        FIXTURE[f"out_{label}"] = root / f"out-{label}"


def tearDownModule():
    FIXTURE["temp"].cleanup()


def load_lot(label, seed, profile=PROFILE):
    return json.loads(ca.lot_path(FIXTURE[f"lots_{label}"], profile, seed).read_text(encoding="utf-8"))


def child_args(protocol_path, lots_dir, context_path, seed=SEEDS[0], profile=PROFILE):
    return [
        "_child", "--protocol", protocol_path, "--registry", FIXTURE["registry"], "--lots-dir", lots_dir,
        "--profile", profile, "--seed", seed, "--jobs", 1, "--context", context_path, "--smoke",
    ]


def write_context(root, protocol_path, **changes):
    import benchmark_premium

    _protocol, sha = ca.load_protocol(protocol_path)
    _manifest, snapshot = benchmark_premium.verify_snapshot(
        benchmark_premium.load_declaration(FIXTURE["declaration"]), FIXTURE["registry"]
    )
    context = mc.build_context(protocol_path, sha, snapshot, "smoke")
    context.update(changes)
    return write_json(Path(root) / "context.json", context), context


class FakeProcess:
    def __init__(self, polls, returncode=0, on_finish=None):
        self.polls = polls
        self.returncode = returncode
        self.on_finish = on_finish

    def poll(self):
        if self.polls > 0:
            self.polls -= 1
            return None
        if self.on_finish:
            self.on_finish()
            self.on_finish = None
        return self.returncode


# --------------------------------------------------------------------------
# S1
# --------------------------------------------------------------------------


class ProtocolValidationTests(unittest.TestCase):
    def test_invalid_protocol_exits_2(self):
        def edit(path, value):
            def apply(protocol):
                target = protocol
                for key in path[:-1]:
                    target = target[key]
                if value is KeyError:
                    del target[path[-1]]
                else:
                    target[path[-1]] = value
            return apply

        cases = [
            (edit(["prefixes"], [30, 60, 100]), "prefixes"),
            (edit(["seeds"], [42, 42]), "seeds"),
            (edit(["seeds"], [42, 43]), "seeds"),
            (edit(["ga_override", "adaptive_mode"], True), "adaptive_mode"),
            (edit(["curve_step"], 7), "curve_step"),
            (edit(["decision", "zone", "quantile"], 0), "quantile"),
            (edit(["decision", "zone", "quantile"], 1.5), "quantile"),
            (edit(["decision", "run_count", "candidates"], [30, 45, 150]), "candidates"),
            (edit(["decision", "target", "prefix"], 120), "prefix"),
            (edit(["decision", "run_count", "leave_one_seed_out"], "min"), "leave_one_seed_out"),
            (edit(["decision", "zone", "quantile_method"], "linear"), "quantile_method"),
            (edit(["decision", "zone", "delta_arithmetic"], "float"), "delta_arithmetic"),
            (edit(["production_run"], KeyError), "production_run"),
        ]
        base = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for mutate, field in cases:
                protocol = copy.deepcopy(base)
                mutate(protocol)
                path = write_json(tmp / "protocol.json", protocol)
                lots, out = tmp / "lots", tmp / "out"
                with patch.object(mc, "_spawn_child") as spawn:
                    code, err = run_main("run", "--protocol", path, "--registry", FIXTURE["registry"], "--lots-dir", lots, "--smoke")
                spawn.assert_not_called()
                self.assertEqual(code, mc.EXIT_USAGE, field)
                self.assertIn(field, err, field)
                self.assertEqual(len(err.strip().splitlines()), 1, err)
                self.assertTrue(err.startswith("multirun_convergence: "), err)
                self.assertFalse(lots.exists(), field)
                code, err = run_main("report", "--protocol", path, "--lots-dir", FIXTURE["lots_a"], "--out-dir", out)
                self.assertEqual(code, mc.EXIT_USAGE, field)
                self.assertIn(field, err, field)
                self.assertFalse(out.exists(), field)
            ca.validate_protocol(base)
            fine = copy.deepcopy(base)
            fine["decision"]["zone"]["quantile"] = 1
            ca.validate_protocol(fine)


class PreregistrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name) / "repo"
        for rel in ("api/app/x.py", "py/x.py", "scripts/multirun_convergence.py", "scripts/convergence_analysis.py",
                    "scripts/benchmark_premium.py", "scripts/bench_measure.py"):
            path = self.repo / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("x = 1\n")
        self.git("init", "-q")
        self.git("add", ".")
        self.git("commit", "-q", "-m", "init")
        self.protocol = write_json(self.repo / "protocol.json", fixture_protocol(FIXTURE["declaration"]))
        self.patch = patch.object(mc, "REPO_ROOT", self.repo)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.addCleanup(self.temp.cleanup)

    def git(self, *args):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=self.repo, check=True, capture_output=True)

    def run_measured(self, *extra):
        lots = Path(self.temp.name) / "lots"
        spawned = []

        def factory(args, context_path):
            context = json.loads(Path(context_path).read_text(encoding="utf-8"))

            def spawn(task):
                spawned.append((task, context))
                return FakeProcess(0)

            return spawn

        with patch.object(mc, "_spawn_child", factory):
            stderr = io.StringIO()
            with patch("sys.stderr", stderr):
                args = mc.build_parser().parse_args(
                    ["run", "--protocol", str(self.protocol), "--registry", str(FIXTURE["registry"]),
                     "--lots-dir", str(lots), "--poll-interval", "0", *extra]
                )
                code = mc.command_run(args, spawn_factory=factory)
        return code, stderr.getvalue(), spawned, lots

    def test_untracked_protocol_exits_6(self):
        code, err, spawned, lots = self.run_measured()
        self.assertEqual(code, mc.EXIT_PREREGISTRATION, err)
        self.assertIn("not committed", err)
        self.assertEqual(spawned, [])
        self.assertFalse(lots.exists())

    def test_edited_protocol_exits_6(self):
        self.git("add", "protocol.json")
        self.git("commit", "-q", "-m", "protocol")
        protocol = json.loads(self.protocol.read_text())
        protocol["decision"]["run_count"]["epsilon_swaps"] = 0.4
        write_json(self.protocol, protocol)
        code, err, spawned, lots = self.run_measured()
        self.assertEqual(code, mc.EXIT_PREREGISTRATION, err)
        self.assertIn("protocol.json", err)
        self.assertEqual(spawned, [])
        self.assertFalse(lots.exists())

    def test_modified_code_path_exits_6(self):
        self.git("add", "protocol.json")
        self.git("commit", "-q", "-m", "protocol")
        (self.repo / "py" / "x.py").write_text("x = 2\n")
        code, err, spawned, _ = self.run_measured()
        self.assertEqual(code, mc.EXIT_PREREGISTRATION, err)
        self.assertIn("py/x.py", err)
        self.assertEqual(spawned, [])

    def test_untracked_file_outside_paths_accepted(self):
        self.git("add", "protocol.json")
        self.git("commit", "-q", "-m", "protocol")
        (self.repo / "scripts" / "notes.py").write_text("y = 1\n")
        self.assertEqual(mc.preregistration_problems(self.protocol), [])
        code, err, spawned, _ = self.run_measured()
        self.assertEqual(code, mc.EXIT_OK, err)
        self.assertEqual([task for task, _ in spawned], [(PROFILE, s) for s in SEEDS])
        self.assertEqual(spawned[0][1]["mode"], "measured")
        self.assertEqual(len(spawned[0][1]["protocol_commit"]), 40)

    def test_smoke_skips_check(self):
        code, err, spawned, _ = self.run_measured("--smoke")
        self.assertEqual(code, mc.EXIT_OK, err)
        self.assertEqual(len(spawned), 3)
        self.assertEqual(spawned[0][1]["mode"], "smoke")
        for label in ("a", "b"):
            for seed in SEEDS:
                self.assertEqual(load_lot(label, seed)["mode"], "smoke")


class CodePathTests(unittest.TestCase):
    """C43: a modification under each of the six CODE_PATHS stops a measured run with exit 6."""

    setUp = PreregistrationTests.setUp
    git = PreregistrationTests.git
    run_measured = PreregistrationTests.run_measured

    def test_each_code_path_exits_6(self):
        self.git("add", "protocol.json")
        self.git("commit", "-q", "-m", "protocol")
        files = ("api/app/x.py", "py/x.py", "scripts/multirun_convergence.py", "scripts/convergence_analysis.py",
                 "scripts/benchmark_premium.py", "scripts/bench_measure.py")
        self.assertEqual(sorted(mc.CODE_PATHS), sorted(f if f.startswith("scripts/") else f.rsplit("/", 1)[0] for f in files))
        for rel in files:
            path = self.repo / rel
            path.write_text("x = 2\n")
            code, err, spawned, lots = self.run_measured()
            self.assertEqual(code, mc.EXIT_PREREGISTRATION, f"{rel}: {err}")
            self.assertIn(rel, err)
            self.assertEqual(spawned, [], rel)
            self.assertFalse(lots.exists(), rel)
            path.write_text("x = 1\n")
            self.assertEqual(mc.preregistration_problems(self.protocol), [], rel)


class ProvenanceTests(unittest.TestCase):
    def test_protocol_commit_of_committed_protocol(self):
        def git(*args):
            return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()

        rel = "docs/validation/multirun-convergence-protocol.json"
        provenance = mc.protocol_provenance(PROTOCOL_PATH)
        self.assertEqual(provenance["protocol_commit"], git("log", "-1", "--format=%H", "--", rel))
        self.assertEqual(provenance["protocol_committed_at"], git("log", "-1", "--format=%cI", "--", rel))
        self.assertRegex(provenance["protocol_commit"], r"^[0-9a-f]{40}$")
        snapshot = {"manifest_id": "m", "manifest_sha256": "s", "sources": {"stock": "1"}}
        context = mc.build_context(PROTOCOL_PATH, "sha", snapshot, "measured")
        self.assertEqual(context["protocol_commit"], provenance["protocol_commit"])
        self.assertEqual(context["protocol_committed_at"], provenance["protocol_committed_at"])
        # The child copies both into the lot and the report copies them from the lots.
        for label in ("a", "b"):
            lot = load_lot(label, SEEDS[0])
            report = json.loads((FIXTURE[f"out_{label}"] / "multirun-convergence.json").read_text(encoding="utf-8"))
            self.assertEqual(
                (report["protocol_commit"], report["protocol_committed_at"]),
                (lot["protocol_commit"], lot["protocol_committed_at"]),
            )


# --------------------------------------------------------------------------
# S2 - the fixture run
# --------------------------------------------------------------------------


class FixtureRunTests(unittest.TestCase):
    def test_lot_shape_and_runs(self):
        for label in ("a", "b"):
            self.assertEqual(FIXTURE[f"run_{label}"].returncode, 0, FIXTURE[f"run_{label}"].stderr[-2000:])
            self.assertEqual(FIXTURE[f"report_{label}"].returncode, 0, FIXTURE[f"report_{label}"].stderr[-2000:])
        for seed in SEEDS:
            lot = load_lot("a", seed)
            self.assertEqual(
                set(lot),
                {"schema", "mode", "protocol_id", "protocol_sha256", "protocol_commit", "protocol_committed_at",
                 "profile_id", "seed", "policy_sha256", "snapshot", "code", "selectors"},
            )
            self.assertEqual(set(lot["snapshot"]), {"manifest_id", "manifest_sha256", "sources"})
            self.assertEqual(set(lot["code"]), {"commit", "code_fingerprint", "code_trees", "experiment_fingerprint", "experiment_trees"})
            self.assertEqual(lot["seed"], seed)
            for selector in ca.SELECTORS:
                data = lot["selectors"][selector]
                self.assertEqual(
                    set(data), {"n_assets", "selection_config", "run_count", "selected_tickers", "candidates", "runs"}
                )
                self.assertEqual([run["run_id"] for run in data["runs"]], list(range(RUN_COUNT)))
                self.assertEqual(data["run_count"], RUN_COUNT)
                self.assertEqual(len(data["selected_tickers"]), 13)
                self.assertFalse(data["selection_config"]["system_ga_config"]["adaptive_mode"])
                self.assertEqual(data["selection_config"]["system_ga_config"]["run_count"], RUN_COUNT)
                for run in data["runs"]:
                    self.assertEqual(set(run), {"run_id", "seed", "tickers", "fitness", "hhi", "generations_run"})
                    self.assertEqual(run["tickers"], sorted(run["tickers"]))
            text = ca.lot_path(FIXTURE["lots_a"], PROFILE, seed).read_text(encoding="utf-8")
            self.assertNotIn("/tmp", text)
            self.assertNotIn("started_at", text)

    def test_byte_identical_lots_and_reports(self):
        for seed in SEEDS:
            self.assertEqual(
                ca.lot_path(FIXTURE["lots_a"], PROFILE, seed).read_bytes(),
                ca.lot_path(FIXTURE["lots_b"], PROFILE, seed).read_bytes(),
                seed,
            )
        self.assertEqual(
            (FIXTURE["out_a"] / "multirun-convergence.json").read_bytes(),
            (FIXTURE["out_b"] / "multirun-convergence.json").read_bytes(),
        )

    def test_prefix_equals_fresh_execution(self):
        from fii_selection import run_fii_selection
        from pipelines.multi_run import run_multi_execution_profile

        lot = load_lot("a", SEEDS[0])
        for selector in ca.SELECTORS:
            data = lot["selectors"][selector]
            consensus = ca.SeedSeries("x", selector, data, {}).consensus(8)
            with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), patch("sys.stderr", io.StringIO()):
                if selector == "stocks":
                    fresh = run_multi_execution_profile(
                        profile=None, n_runs=8, parallel=False, adaptive_mode=False,
                        selection_config=data["selection_config"], workspace=Path(tmp),
                        random_seed=SEEDS[0], input_path=FIXTURE["registry"] / "stocks.csv",
                    )
                else:
                    fresh = run_fii_selection(
                        input_path=FIXTURE["registry"] / "fiis.csv", selection_config=data["selection_config"],
                        n_runs=8, parallel=False, adaptive_mode=False, workspace=Path(tmp), seed=SEEDS[0],
                    )
            self.assertEqual(fresh["n_runs"], 8)
            self.assertEqual(consensus, sorted(fresh["selected_tickers"]), selector)

    def test_workspaces_distinct_and_removed(self):
        paths = re.findall(r"workspace (\S+)", FIXTURE["run_a"].stderr)
        self.assertEqual(len(paths), 3)
        self.assertEqual(len(set(paths)), 3)
        for path in paths:
            self.assertFalse(Path(path).exists(), path)
        for seed in SEEDS:
            for selector in ca.SELECTORS:
                self.assertEqual(len(load_lot("a", seed)["selectors"][selector]["runs"]), RUN_COUNT)

    def test_cost_and_summary_files(self):
        lots = FIXTURE["lots_a"]
        summary = json.loads((lots / "run-summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["jobs"], 3)
        for key in ("experiment_wall_s", "first_started_at", "last_finished_at", "lots"):
            self.assertIn(key, summary)
        self.assertEqual([item["lot"] for item in summary["lots"]], [f"{PROFILE}:{s}" for s in SEEDS])
        self.assertEqual({item["status"] for item in summary["lots"]}, {"written"})
        for seed in SEEDS:
            cost = json.loads(ca.cost_path(lots, PROFILE, seed).read_text(encoding="utf-8"))
            for key in ("started_at", "finished_at", "wall_s", "cpu_s", "stages", "rss_peak_kb", "jobs",
                        "load_avg_start", "load_avg_end", "machine"):
                self.assertIn(key, cost)
            self.assertEqual(cost["jobs"], 3)
            for stage in ("stocks", "fiis", "total"):
                self.assertGreater(cost["stages"][stage]["wall_s"], 0)
                self.assertGreater(cost["stages"][stage]["cpu_s"], 0)
            self.assertIn("numpy", cost["machine"]["libraries"])
            self.assertGreater(cost["cpu_s"], cost["stages"]["stocks"]["cpu_s"])
            self.assertLessEqual(cost["wall_s"], summary["experiment_wall_s"])
            self.assertLessEqual(summary["first_started_at"], cost["started_at"])


class AdaptiveOffTests(unittest.TestCase):
    def test_stop_disabled_records_all_runs(self):
        import fii_selection
        from pipelines import multi_run
        import app.adapters.premium_optimization as adapter

        calls = {}
        outputs = []
        original_stock = multi_run.run_multi_execution_profile
        original_fii = fii_selection.run_fii_selection
        original_rpo = adapter.run_premium_optimization

        def stock(*args, **kwargs):
            calls["stocks"] = kwargs
            return original_stock(*args, **kwargs)

        def fii(*args, **kwargs):
            calls["fiis"] = kwargs
            return original_fii(*args, **kwargs)

        def rpo(*args, **kwargs):
            outputs.append(original_rpo(*args, **kwargs))
            return outputs[-1]

        with tempfile.TemporaryDirectory() as tmp:
            context_path, context = write_context(tmp, FIXTURE["protocol"], protocol_commit="f" * 40, protocol_committed_at="2026-11-03T00:00:00+00:00")
            lots = Path(tmp) / "lots"
            with patch.dict("app.services.premium_policy.SYSTEM_GA_CONFIG", {"min_runs": 1, "target_cv": 1.0, "target_jaccard": 0.0}), \
                    patch.object(multi_run, "run_multi_execution_profile", stock), \
                    patch.object(fii_selection, "run_fii_selection", fii), \
                    patch.object(adapter, "run_premium_optimization", rpo):
                code, err = run_main(*child_args(FIXTURE["protocol"], lots, context_path))
            self.assertEqual(code, 0, err)
            lot = json.loads(ca.lot_path(lots, PROFILE, SEEDS[0]).read_text(encoding="utf-8"))
        for selector in ca.SELECTORS:
            self.assertFalse(calls[selector]["adaptive_mode"], selector)
            self.assertEqual(calls[selector]["n_runs"], RUN_COUNT, selector)
            self.assertEqual(calls[selector]["min_runs"], 1, selector)
            self.assertEqual(len(lot["selectors"][selector]["runs"]), RUN_COUNT, selector)
            self.assertEqual(
                lot["selectors"][selector]["selected_tickers"], [item["ticker"] for item in outputs[0][selector]]
            )
        self.assertEqual(lot["protocol_commit"], "f" * 40)
        self.assertEqual(lot["protocol_committed_at"], "2026-11-03T00:00:00+00:00")
        self.assertEqual(lot["code"], context["code"])
        self.assertEqual(lot["snapshot"], context["snapshot"])


class PolicyTests(unittest.TestCase):
    def test_profiles_resolve_expected_n_assets(self):
        import benchmark_premium

        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        manifest, _ = benchmark_premium.verify_snapshot(
            benchmark_premium.load_declaration(FIXTURE["declaration"]), FIXTURE["registry"]
        )
        expected = {"premium-2026-11-02": (13, 13), "arrojado-anchor": (15, 15)}
        for profile_id, declared in protocol["profiles"].items():
            policy = mc.resolve_policy(declared["record"], manifest, 42)
            self.assertEqual((policy["stocks"]["n_assets"], policy["fiis"]["n_assets"]), expected[profile_id])
            self.assertEqual(policy["profile"]["score"], 1.0 if profile_id == "arrojado-anchor" else 0.582)
            mc.check_n_assets(policy, declared["expected_n_assets"])
            overridden = mc.apply_override(copy.deepcopy(policy), protocol["ga_override"], smoke=False)
            for selector in ca.SELECTORS:
                ga = overridden[selector]["system_ga_config"]
                self.assertEqual((ga["adaptive_mode"], ga["run_count"]), (False, 150))
                self.assertEqual(ga["population"], policy[selector]["system_ga_config"]["population"])
                self.assertEqual(ga["min_runs"], policy[selector]["system_ga_config"]["min_runs"])

    def test_n_assets_mismatch_exits_2(self):
        protocol = fixture_protocol(FIXTURE["declaration"])
        protocol["profiles"][PROFILE]["expected_n_assets"]["stocks"] = 12
        with tempfile.TemporaryDirectory() as tmp:
            path = write_json(Path(tmp) / "protocol.json", protocol)
            context_path, _ = write_context(tmp, path)
            lots = Path(tmp) / "lots"
            with patch("app.adapters.premium_optimization.run_premium_optimization") as rpo:
                code, err = run_main(*child_args(path, lots, context_path))
            rpo.assert_not_called()
            self.assertEqual(code, mc.EXIT_USAGE)
            self.assertIn("expected_n_assets", err)
            self.assertFalse(lots.exists())


class SnapshotTests(unittest.TestCase):
    def test_registry_mismatch_exits_3(self):
        def stock_csv(registry):
            with open(registry / "stocks.csv", "a", encoding="utf-8") as handle:
                handle.write("\n")

        def allocation(registry):
            with open(registry / "data" / "snap" / "di.csv", "a", encoding="utf-8") as handle:
                handle.write("\n")

        def manifest(registry):
            payload = json.loads((registry / "fixture.json").read_text(encoding="utf-8"))
            payload["created_at"] = "2026-10-30T00:00:00+00:00"
            (registry / "fixture.json").write_text(json.dumps(payload), encoding="utf-8")

        for label, mutate in (("stock", stock_csv), ("allocation", allocation), ("manifest", manifest)):
            with tempfile.TemporaryDirectory() as tmp:
                registry = Path(tmp) / "registry"
                shutil.copytree(FIXTURE["registry"], registry)
                mutate(registry)
                lots = Path(tmp) / "lots"
                with patch.object(mc, "_spawn_child") as spawn:
                    code, err = run_main("run", "--protocol", FIXTURE["protocol"], "--registry", registry, "--lots-dir", lots, "--smoke")
                spawn.assert_not_called()
                self.assertEqual(code, mc.EXIT_SNAPSHOT, f"{label}: {err}")
                self.assertIn("snapshot", err)
                self.assertFalse(lots.exists(), label)


class SchedulerTests(unittest.TestCase):
    def test_at_most_jobs_alive(self):
        alive = []
        peak = [0]
        started = []

        def spawn(task):
            started.append(task)
            alive.append(task)
            peak[0] = max(peak[0], len(alive))
            return FakeProcess(polls=task % 3 + 1, on_finish=lambda: alive.remove(task))

        finished = mc.schedule([1, 2, 3, 4, 5], 2, spawn, poll_interval=0)
        self.assertEqual(peak[0], 2)
        self.assertEqual(sorted(started), [1, 2, 3, 4, 5])
        self.assertEqual(sorted(item["task"] for item in finished), [1, 2, 3, 4, 5])
        self.assertTrue(all(item["returncode"] == 0 for item in finished))


class SkipTests(unittest.TestCase):
    def test_valid_lot_skipped_and_stale_lot_exit_4(self):
        with tempfile.TemporaryDirectory() as tmp:
            lots = Path(tmp) / "lots"
            shutil.copytree(FIXTURE["lots_a"], lots)
            summary = (lots / "run-summary.json").read_bytes()
            with patch.object(mc, "_spawn_child") as spawn:
                code, err = run_main("run", "--protocol", FIXTURE["protocol"], "--registry", FIXTURE["registry"], "--lots-dir", lots, "--smoke")
            self.assertEqual(code, mc.EXIT_OK, err)
            spawn.assert_not_called()
            self.assertEqual(err.count("skipped"), 3)
            self.assertEqual((lots / "run-summary.json").read_bytes(), summary)

            before = ca.lot_path(lots, PROFILE, SEEDS[0]).read_bytes()
            real = mc.experiment_info()
            changed = dict(real, experiment_fingerprint="0" * 64)
            with patch.object(mc, "_spawn_child") as spawn, patch.object(mc, "experiment_info", return_value=changed):
                code, err = run_main("run", "--protocol", FIXTURE["protocol"], "--registry", FIXTURE["registry"], "--lots-dir", lots, "--smoke")
            self.assertEqual(code, mc.EXIT_INTEGRITY, err)
            self.assertIn("experiment_fingerprint", err)
            spawn.assert_not_called()
            self.assertEqual(ca.lot_path(lots, PROFILE, SEEDS[0]).read_bytes(), before)


    def test_each_reuse_field_exit_4(self):
        """C46: each field check_existing_lot compares with HEAD stops a reuse with exit 4."""

        def mode(lot):
            lot["mode"] = "measured"

        def snapshot(lot):
            lot["snapshot"]["manifest_id"] = "other"

        def code_fingerprint(lot):
            lot["code"]["code_fingerprint"] = "0" * 64

        def experiment_fingerprint(lot):
            lot["code"]["experiment_fingerprint"] = "0" * 64

        for mutate in (mode, snapshot, code_fingerprint, experiment_fingerprint):
            field = mutate.__name__
            with tempfile.TemporaryDirectory() as tmp:
                lots = Path(tmp) / "lots"
                shutil.copytree(FIXTURE["lots_a"], lots)
                path = ca.lot_path(lots, PROFILE, SEEDS[0])
                lot = json.loads(path.read_text(encoding="utf-8"))
                mutate(lot)
                path.write_text(ca.dumps(lot), encoding="utf-8")
                before = path.read_bytes()
                with patch.object(mc, "_spawn_child") as spawn:
                    code, err = run_main("run", "--protocol", FIXTURE["protocol"], "--registry", FIXTURE["registry"], "--lots-dir", lots, "--smoke")
                self.assertEqual(code, mc.EXIT_INTEGRITY, f"{field}: {err}")
                self.assertIn(f"another {field} than HEAD", err)
                spawn.assert_not_called()
                self.assertEqual(path.read_bytes(), before, field)


class ChildFailureTests(unittest.TestCase):
    def test_failed_child_exit_1_others_written(self):
        protocol = fixture_protocol(FIXTURE["declaration"], seeds=[42, 43])
        protocol["decision"]["run_count"]["leave_one_seed_out"] = None
        bad = copy.deepcopy(protocol["profiles"][PROFILE])
        bad["expected_n_assets"]["stocks"] = 12
        protocol["profiles"]["bad"] = bad
        with tempfile.TemporaryDirectory() as tmp:
            path = write_json(Path(tmp) / "protocol.json", protocol)
            lots = Path(tmp) / "lots"
            completed = cli("run", "--protocol", path, "--registry", FIXTURE["registry"], "--lots-dir", lots, "--jobs", 4, "--smoke", "--poll-interval", 0.2)
            self.assertEqual(completed.returncode, mc.EXIT_CHILD_FAILED, completed.stderr[-2000:])
            for seed in (42, 43):
                self.assertTrue(ca.lot_path(lots, PROFILE, seed).is_file())
                self.assertTrue(ca.cost_path(lots, PROFILE, seed).is_file())
                self.assertFalse(ca.lot_path(lots, "bad", seed).exists())
                self.assertFalse(ca.cost_path(lots, "bad", seed).exists())
                self.assertIn(f"multirun_convergence: bad:{seed}: ", completed.stderr)
            summary = json.loads((lots / "run-summary.json").read_text(encoding="utf-8"))
            statuses = {item["lot"]: item["status"] for item in summary["lots"]}
            self.assertEqual(statuses, {f"{PROFILE}:42": "written", f"{PROFILE}:43": "written", "bad:42": "failed", "bad:43": "failed"})


class RunCountGuardTests(unittest.TestCase):
    def run_child(self, engines=None, capture=None):
        with tempfile.TemporaryDirectory() as tmp:
            context_path, _ = write_context(tmp, FIXTURE["protocol"])
            lots = Path(tmp) / "lots"
            args = mc.build_parser().parse_args([str(a) for a in child_args(FIXTURE["protocol"], lots, context_path)])
            stderr = io.StringIO()
            patches = [patch("sys.stderr", stderr)]
            if capture is not None:
                patches.append(patch.object(mc, "capture_selector_runs", capture))
            with redirect_stdout(io.StringIO()):
                for item in patches:
                    item.start()
                try:
                    code = mc.command_child(args, engines=engines)
                finally:
                    for item in reversed(patches):
                        item.stop()
            written = sorted(p.name for p in lots.rglob("*")) if lots.exists() else []
        return code, stderr.getvalue(), written

    def test_dropped_run_exits_1(self):
        original = mc.capture_selector_runs

        @contextmanager
        def dropping():
            with original() as captured:
                yield captured
            captured["stocks"][:] = [run for run in captured["stocks"] if run["run_id"] != 7]

        code, err, written = self.run_child(capture=dropping)
        self.assertEqual(code, mc.EXIT_CHILD_FAILED, err)
        self.assertIn("stocks", err.strip().splitlines()[-1])
        self.assertEqual(written, [])

    def engines(self, change):
        import benchmark_premium

        engines = benchmark_premium.default_engines()
        original = engines["stock_engine"]

        def stock(*args, **kwargs):
            return change(dict(original(*args, **kwargs)))

        engines["stock_engine"] = stock
        return engines

    def test_n_runs_mismatch_exits_1(self):
        code, err, written = self.run_child(engines=self.engines(lambda r: {**r, "n_runs": 11}))
        self.assertEqual(code, mc.EXIT_CHILD_FAILED, err)
        self.assertIn("n_runs 11", err)
        self.assertEqual(written, [])

    def test_missing_n_runs_exits_1(self):
        code, err, written = self.run_child(engines=self.engines(lambda r: {k: v for k, v in r.items() if k != "n_runs"}))
        self.assertEqual(code, mc.EXIT_CHILD_FAILED, err)
        self.assertIn("no n_runs", err)
        self.assertEqual(written, [])

    def test_check_selector_runs(self):
        runs = [{"run_id": i} for i in range(12)]
        mc.check_selector_runs("fiis", runs, 12, {"n_runs": 12})
        for bad_runs, result in ((runs[:11], {"n_runs": 11}), (runs[1:] + runs[:1], {"n_runs": 12}), (runs, {"n_runs": 11}), (runs, {})):
            with self.assertRaises(mc.ChildError) as caught:
                mc.check_selector_runs("fiis", bad_runs, 12, result)
            self.assertEqual((caught.exception.code, caught.exception.where), (1, "fiis"))


if __name__ == "__main__":
    unittest.main()
