"""S4: CLI contract and production isolation (plan.md S4, AC 24-28).

The isolation scan is a decision table over where a production file could reach the frozen
bundle, so it is proven twice: once over the real tree (zero offenders) and once over a copy
where the offending line is injected, because a scan that cannot fail is not a scan.
"""

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUNDLE = REPO / "baselines" / "v0-legacy"
VERIFY = BUNDLE / "tools" / "verify_manifest.py"
REPLAY = BUNDLE / "tools" / "replay.py"
DEFAULT_DRIVER = BUNDLE / "tools" / "_legacy_driver.py"

BUNDLE_MODULES = {
    "ga",
    "ga_multiple_runs",
    "profiles",
    "build_manifest",
    "verify_manifest",
    "replay",
    "_legacy_driver",
}
FORBIDDEN_SEGMENT = "baselines/"
PACKAGE_MARKERS = ("__init__.py", "setup.py", "pyproject.toml")
SCANNED_GROUPS = ("py", "api")


def scan(root):
    """Offenders as `path:line`, for every production file that could reach the bundle."""
    root = Path(root)
    targets = []
    for group in SCANNED_GROUPS:
        targets.extend(sorted((root / group).rglob("*.py")))
    targets.extend(sorted(root.glob("*.py")))

    offenders = []
    for path in targets:
        relative = path.relative_to(root).as_posix()
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            for number, line in enumerate(source.splitlines(), start=1):
                if FORBIDDEN_SEGMENT in line:
                    offenders.append(f"{relative}:{number}")
            continue
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            for name in names:
                if name in BUNDLE_MODULES and not (path.parent / f"{name}.py").is_file():
                    offenders.append(f"{relative}:{node.lineno}")
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if FORBIDDEN_SEGMENT in node.value:
                    offenders.append(f"{relative}:{node.lineno}")
    return sorted(set(offenders))


def blocked_dependency_env(directory):
    blocker = Path(directory) / "block"
    blocker.mkdir()
    for name in ("numpy", "pandas"):
        (blocker / f"{name}.py").write_text(
            f"raise ImportError('blocked by the isolation test: {name}')\n", encoding="utf-8"
        )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(blocker)
    return environment


def run(command, cwd=REPO, env=None):
    return subprocess.run(command, cwd=str(cwd), capture_output=True, text=True, env=env)


class VerifierCliTests(unittest.TestCase):
    def test_verifier_defaults_to_the_repository_bundle(self):
        completed = run([sys.executable, str(VERIFY)])
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual(report["status"], "VERIFIED")
        self.assertEqual(report["bundle"], str(BUNDLE.resolve()))

    def test_replay_flags_and_defaults(self):
        help_text = run([sys.executable, str(REPLAY), "--help"]).stdout
        for expected in (
            "baselines/v0-legacy",
            "outputs/baseline_v0_legacy/replay_report.json",
            "1800",
            "tools/_legacy_driver.py",
        ):
            self.assertIn(expected, help_text, expected)

        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.json"
            environment = blocked_dependency_env(tmp)
            completed = run(
                [
                    sys.executable,
                    str(REPLAY),
                    "--report",
                    str(report_path),
                    "--driver",
                    str(DEFAULT_DRIVER),
                ],
                env=environment,
            )
            written = report_path.read_text(encoding="utf-8")

        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertEqual(json.loads(written), json.loads(completed.stdout))
        self.assertEqual(json.loads(written)["driver"], str(DEFAULT_DRIVER.resolve()))


class CliContractTests(unittest.TestCase):
    def assert_single_object(self, completed):
        self.assertTrue(completed.stdout.strip().startswith("{"))
        self.assertTrue(completed.stdout.strip().endswith("}"))
        payload = json.loads(completed.stdout)
        self.assertIsInstance(payload, dict)
        return payload

    def test_exit_codes_and_single_json_object_on_stdout(self):
        annotated = self.assert_single_object(run([sys.executable, str(VERIFY)]))
        self.assertEqual(annotated["status"], "VERIFIED")

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)

            drifted = directory / "drifted"
            shutil.copytree(BUNDLE, drifted)
            target = drifted / "code" / "profiles.py"
            target.write_bytes(target.read_bytes() + b"#")
            drift = run([sys.executable, str(VERIFY), "--bundle", str(drifted)])
            self.assertEqual(drift.returncode, 1)
            self.assertEqual(self.assert_single_object(drift)["status"], "DRIFT")
            self.assertTrue(drift.stderr.strip())

            broken = directory / "broken"
            shutil.copytree(BUNDLE, broken)
            (broken / "baseline.json").write_text('{"schema_version": 1,', encoding="utf-8")
            untrusted = run([sys.executable, str(VERIFY), "--bundle", str(broken)])
            self.assertEqual(untrusted.returncode, 3)
            self.assertEqual(self.assert_single_object(untrusted)["status"], "ERROR")
            self.assertTrue(untrusted.stderr.strip())

            environment = blocked_dependency_env(directory)
            report_path = directory / "report.json"
            replay = run(
                [
                    sys.executable,
                    str(REPLAY),
                    "--report",
                    str(report_path),
                    "--driver",
                    str(DEFAULT_DRIVER),
                ],
                env=environment,
            )
            self.assertEqual(replay.returncode, 2)
            self.assertEqual(
                self.assert_single_object(replay)["status"],
                "UNVERIFIABLE",
            )


class IsolationTests(unittest.TestCase):
    def test_no_production_file_reaches_the_bundle(self):
        self.assertEqual(scan(REPO), [])

    def test_isolation_scan_flags_an_injected_bundle_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "py").mkdir()
            (root / "api").mkdir()
            (root / "py" / "config.py").write_text(
                'import sys\nsys.path.insert(0, "baselines/v0-legacy/tools")\n',
                encoding="utf-8",
            )
            (root / "api" / "runner.py").write_text(
                "from verify_manifest import verify_bundle\n",
                encoding="utf-8",
            )
            offenders = scan(root)

        self.assertIn("py/config.py:2", offenders)
        self.assertIn("api/runner.py:1", offenders)

    def test_no_package_marker_under_baselines(self):
        found = [
            path.relative_to(REPO).as_posix()
            for path in BUNDLE.parent.rglob("*")
            if path.is_file() and path.name in PACKAGE_MARKERS
        ]
        self.assertEqual(found, [])


if __name__ == "__main__":
    unittest.main()
