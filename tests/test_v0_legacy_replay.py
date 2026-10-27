"""S3: replay and its status (plan.md S3, AC 12-23).

The real executor needs `numpy` and `pandas`, which this repository does not vendor, so the
`REPRODUCED`, `MISMATCH` and failure branches are exercised through substitute drivers written
into a temporary directory and passed with `--driver`: plain scripts, no third-party import,
each writing one `runs.json`. Every substitute asserts that it is running with the scratch root
as cwd and without a `reference/` directory, so these tests also prove the executor seam.
"""

import ast
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUNDLE = REPO / "baselines" / "v0-legacy"
REPLAY = BUNDLE / "tools" / "replay.py"
REFERENCE_DIR = BUNDLE / "reference"
METRICS = REFERENCE_DIR / "metrics_stability_caio.csv"
CONSENSUS = REFERENCE_DIR / "carteira_caio_consensus.json"
TOTAL_RUNS = 30

SCRATCH_GUARD = """\
import json
import os
import sys
from pathlib import Path

if not Path("code/ga.py").is_file():
    sys.stderr.write("cwd is not the scratch root\\n")
    sys.exit(4)
if Path("reference").exists():
    sys.stderr.write("scratch root exposes reference/\\n")
    sys.exit(5)
"""


def reference_rows():
    with open(METRICS, newline="", encoding="utf-8") as handle:
        return {int(row["run_id"]): row for row in csv.DictReader(handle)}


def runs_document(mutate=None, recomputed=0.0):
    document = {"driver": "substitute", "runs": []}
    for run_id, row in sorted(reference_rows().items()):
        document["runs"].append(
            {
                "run_id": run_id,
                "seed": int(row["seed"]),
                "tickers": ast.literal_eval(row["tickers"]),
                "fitness": float(row["fitness"]),
                "hhi": float(row["hhi"]),
                "recomputed_fitness": recomputed,
            }
        )
    if mutate is not None:
        mutate(document)
    return document


def write_script(directory, name, body):
    path = Path(directory) / name
    path.write_text(body, encoding="utf-8")
    return path


def runs_driver(directory, name="driver.py", mutate=None, recomputed=0.0, tail=""):
    body = (
        SCRATCH_GUARD
        + "\nPath('runs.json').write_text(\n    "
        + repr(json.dumps(runs_document(mutate, recomputed)))
        + ",\n    encoding='utf-8',\n)\n"
        + tail
    )
    return write_script(directory, name, body)


def run_replay(bundle, report_dir, driver=None, extra=(), env=None, timeout=180):
    report_path = Path(report_dir) / "report.json"
    command = [
        sys.executable,
        str(REPLAY),
        "--bundle",
        str(bundle),
        "--report",
        str(report_path),
    ]
    if driver is not None:
        command += ["--driver", str(driver)]
    command += list(extra)
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=env)
    report = json.loads(completed.stdout) if completed.stdout.strip() else None
    return completed, report, report_path


def copy_bundle(destination):
    import shutil

    shutil.copytree(BUNDLE, destination)
    return Path(destination)


def rewrite_reference(bundle, rewrite):
    """Rewrite the recorded caio metrics CSV and re-hash the copy so the manifest still verifies."""
    metrics = bundle / "reference" / "metrics_stability_caio.csv"
    rewrite(metrics)
    rebuilt = subprocess.run(
        [sys.executable, str(bundle / "tools" / "build_manifest.py"), "--bundle", str(bundle)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if rebuilt.returncode != 0:
        raise AssertionError(f"build_manifest.py failed: {rebuilt.stderr}")


def blocked_dependency_env(directory):
    blocker = Path(directory) / "block"
    blocker.mkdir()
    for name in ("numpy", "pandas"):
        (blocker / f"{name}.py").write_text(
            f"raise ImportError('blocked by the replay test: {name}')\n", encoding="utf-8"
        )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(blocker)
    return environment


class PreExecutionTests(unittest.TestCase):
    def test_manifest_drift_stops_before_the_driver_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            bundle = copy_bundle(directory / "bundle")
            target = bundle / "code" / "ga.py"
            target.write_bytes(target.read_bytes() + b"#")
            sentinel = directory / "spawned"
            driver = write_script(
                directory,
                "sentinel.py",
                "from pathlib import Path\n"
                f"Path({str(sentinel)!r}).write_text('ran', encoding='utf-8')\n",
            )
            completed, report, _ = run_replay(bundle, directory, driver=driver)
            self.assertFalse(sentinel.exists(), "the driver ran on a bundle that did not verify")

        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertEqual(report["status"], "UNVERIFIABLE")
        self.assertEqual(report["cause"], "hash_mismatch")
        self.assertFalse(report["bundle_stable"])
        self.assertEqual(report["runs"], [])

    def test_missing_dependencies_are_named_and_exit_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            environment = blocked_dependency_env(tmp)
            completed, report, _ = run_replay(BUNDLE, tmp, env=environment)

        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertEqual(report["status"], "UNVERIFIABLE")
        self.assertEqual(report["cause"], "dependency_unavailable")
        self.assertEqual(report["missing_modules"], ["numpy", "pandas"])

    def test_uncovered_dataset_is_reported_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            bundle = copy_bundle(directory / "bundle")
            protocol_path = bundle / "protocol.json"
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            protocol["comparison"]["reference_metrics"] = "reference/metrics_stability_unknown.csv"
            protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
            completed, report, _ = run_replay(bundle, directory)

        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertEqual(report["status"], "UNVERIFIABLE")
        self.assertEqual(report["cause"], "dataset_unavailable")


class ComparisonTests(unittest.TestCase):
    def test_matching_runs_report_reproduced(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            driver = runs_driver(directory)
            completed, report, _ = run_replay(BUNDLE, directory, driver=driver)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(report["status"], "REPRODUCED")
        self.assertIsNone(report["cause"])
        self.assertEqual(report["fields"], [])
        self.assertTrue(report["bundle_stable"])

    def test_one_differing_field_reports_mismatch(self):
        def bump(document):
            document["runs"][3]["fitness"] = document["runs"][3]["fitness"] + 0.5

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            driver = runs_driver(directory, mutate=bump)
            completed, report, _ = run_replay(BUNDLE, directory, driver=driver)

        self.assertEqual(completed.returncode, 1, completed.stderr)
        self.assertEqual(report["status"], "MISMATCH")
        self.assertEqual(len(report["fields"]), 1)
        difference = report["fields"][0]
        self.assertEqual(difference["run_id"], 3)
        self.assertEqual(difference["field"], "fitness")
        self.assertAlmostEqual(difference["recomputed"] - difference["recorded"], 0.5, places=9)

    def test_report_carries_the_protocol_values_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            driver = runs_driver(directory)
            _, report, _ = run_replay(BUNDLE, directory, driver=driver)

        used = report["protocol"]
        self.assertEqual(used["profile"], "caio")
        self.assertEqual(used["n_assets"], 10)
        self.assertEqual(used["lambda"], 0.49)
        self.assertEqual(used["generations"], 460)
        self.assertEqual(used["pop_size"], 276)
        self.assertEqual(used["n_runs"], 30)
        self.assertEqual(used["run_id_start"], 0)
        self.assertEqual(used["rtol"], 1e-09)
        self.assertEqual(used["atol"], 1e-09)
        self.assertEqual(
            [spec["name"] for spec in used["fields"]],
            ["seed", "fitness", "hhi", "tickers"],
        )

    def test_missing_protocol_value_is_reported_as_protocol_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            bundle = copy_bundle(directory / "bundle")
            protocol_path = bundle / "protocol.json"
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            del protocol["runs"]["n_runs"]
            protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
            completed, report, _ = run_replay(bundle, directory)

        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertEqual(report["status"], "UNVERIFIABLE")
        self.assertEqual(report["cause"], "protocol_mismatch")
        self.assertIn("runs.n_runs", report["detail"])

    def test_no_reference_ticker_is_hardcoded(self):
        tickers = sorted({record["TICKER"] for record in json.loads(CONSENSUS.read_text())})
        self.assertGreaterEqual(len(tickers), 10)
        sources = {
            "replay.py": (BUNDLE / "tools" / "replay.py").read_text(encoding="utf-8"),
            "protocol.json": (BUNDLE / "protocol.json").read_text(encoding="utf-8"),
        }
        for name, text in sources.items():
            for ticker in tickers:
                self.assertNotIn(ticker, text, f"{ticker} is hardcoded in {name}")

    def test_reference_tree_is_untouched_and_no_golden_output(self):
        def snapshot():
            return {
                path.relative_to(REFERENCE_DIR).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(REFERENCE_DIR.rglob("*"))
                if path.is_file()
            }

        before = snapshot()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            driver = runs_driver(directory)
            completed, report, _ = run_replay(BUNDLE, directory, driver=driver)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(snapshot(), before)
        self.assertFalse(report["golden_output_written"])
        self.assertEqual(len(before), 9)

    def test_reported_and_recomputed_fitness_stay_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            driver = runs_driver(directory, recomputed=-999.5)
            completed, report, _ = run_replay(BUNDLE, directory, driver=driver)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(report["status"], "REPRODUCED")
        self.assertEqual([row["run_id"] for row in report["runs"]], list(range(TOTAL_RUNS)))
        first = report["runs"][0]
        self.assertAlmostEqual(first["reported_fitness"], float(reference_rows()[0]["fitness"]), places=12)
        self.assertEqual(first["recomputed_fitness"], -999.5)


class ReferenceReadTests(unittest.TestCase):
    """The recorded CSV is untrusted input: a bad one is `dataset_unavailable`, never a traceback.

    The replay exits 1 only for `MISMATCH`, so an exception escaping the reference reader would
    report a disagreement that was never measured. Each case re-hashes the copy first, otherwise
    the manifest check stops the replay before the reader is reached.
    """

    def run_with_rewritten_reference(self, rewrite):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            bundle = copy_bundle(directory / "bundle")
            rewrite_reference(bundle, rewrite)
            driver = runs_driver(directory)
            return run_replay(bundle, directory, driver=driver)

    def assert_dataset_unavailable(self, completed, report, needle):
        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertIsNotNone(report, completed.stderr)
        self.assertEqual(report["status"], "UNVERIFIABLE")
        self.assertEqual(report["cause"], "dataset_unavailable")
        self.assertIn(needle, report["detail"])
        self.assertNotIn("Traceback", completed.stderr)
        self.assertTrue(completed.stderr.strip())

    def test_reference_without_run_id_column_is_dataset_unavailable(self):
        def drop_run_id(path):
            with open(path, newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            column = rows[0].index("run_id")
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                for row in rows:
                    writer.writerow([cell for index, cell in enumerate(row) if index != column])

        completed, report, _ = self.run_with_rewritten_reference(drop_run_id)
        self.assert_dataset_unavailable(completed, report, "run_id")

    def test_reference_with_a_non_numeric_run_id_is_dataset_unavailable(self):
        def spoil_run_id(path):
            with open(path, newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            rows[1][rows[0].index("run_id")] = "first"
            with open(path, "w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows(rows)

        completed, report, _ = self.run_with_rewritten_reference(spoil_run_id)
        self.assert_dataset_unavailable(completed, report, "not an integer")


class ExecutorFailureTests(unittest.TestCase):
    def test_nonzero_driver_exit_reports_runtime_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            driver = write_script(
                directory,
                "failing.py",
                "import sys\nsys.stderr.write('E' * 5000)\nsys.exit(3)\n",
            )
            completed, report, _ = run_replay(BUNDLE, directory, driver=driver)

        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertEqual(report["status"], "UNVERIFIABLE")
        self.assertEqual(report["cause"], "runtime_error")
        self.assertEqual(len(report["stderr_tail"]), 4096)
        self.assertTrue(report["stderr_tail"].endswith("E"))

    def test_budget_exceeded_terminates_the_executor(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            driver = write_script(directory, "slow.py", "import time\ntime.sleep(30)\n")
            started = time.monotonic()
            completed, report, _ = run_replay(
                BUNDLE,
                directory,
                driver=driver,
                extra=["--budget-seconds", "1"],
            )
            elapsed = time.monotonic() - started

        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertEqual(report["status"], "UNVERIFIABLE")
        self.assertEqual(report["cause"], "runtime_budget_exceeded")
        self.assertLess(elapsed, 20)

    def test_child_without_runs_json_reports_runtime_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            driver = write_script(directory, "silent.py", SCRATCH_GUARD + "\nsys.exit(0)\n")
            completed, report, _ = run_replay(BUNDLE, directory, driver=driver)

        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertEqual(report["status"], "UNVERIFIABLE")
        self.assertEqual(report["cause"], "runtime_error")
        self.assertIn("runs.json", report["detail"])

    def test_bundle_moved_during_the_run_exits_three(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            bundle = copy_bundle(directory / "bundle")
            environment = dict(os.environ)
            environment["REPLAY_TEST_MUTATE"] = str(bundle / "code" / "ga.py")
            driver = runs_driver(
                directory,
                tail=(
                    "\ntarget = os.environ.get('REPLAY_TEST_MUTATE')\n"
                    "if target:\n"
                    "    lens = Path(target)\n"
                    "    lens.write_bytes(lens.read_bytes() + b'x')\n"
                ),
            )
            completed, report, _ = run_replay(bundle, directory, driver=driver, env=environment)

        self.assertEqual(completed.returncode, 3, completed.stderr)
        self.assertEqual(report["status"], "UNVERIFIABLE")
        self.assertEqual(report["cause"], "hash_mismatch")
        self.assertFalse(report["bundle_stable"])


class ExecutorSeamTests(unittest.TestCase):
    def test_archived_driver_compiles_and_names_runs_json(self):
        source = (BUNDLE / "tools" / "_legacy_driver.py").read_text(encoding="utf-8")
        tree = ast.parse(source, filename="_legacy_driver.py")
        compile(tree, "_legacy_driver.py", "exec")
        output = [
            node.value.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "OUTPUT" for target in node.targets)
            and isinstance(node.value, ast.Constant)
        ]
        self.assertEqual(output, ["runs.json"])

    def test_substitute_driver_is_resolved_and_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            driver = runs_driver(directory)
            completed, report, _ = run_replay(BUNDLE, directory, driver=driver)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(report["driver"], str(Path(driver).resolve()))
        # The substitute exits 4/5 unless cwd is the scratch root and reference/ is absent.
        self.assertEqual(report["status"], "REPRODUCED")


if __name__ == "__main__":
    unittest.main()
