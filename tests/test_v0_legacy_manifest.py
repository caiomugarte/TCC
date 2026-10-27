"""S1: the frozen v0-legacy bundle and its manifest.

Every assertion here restates a criterion from
`.specs/features/p0-01-v0-legacy-baseline/plan.md` (S1, AC 1-5) and settles it through the
bundle's own CLI, so a bundle that drifts fails the suite that claims it is frozen.
"""

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUNDLE = REPO / "baselines" / "v0-legacy"
VERIFY = BUNDLE / "tools" / "verify_manifest.py"
BUILD = BUNDLE / "tools" / "build_manifest.py"
BASELINE_COMMIT = "97ea3275293da4e1902fcfeb684c814f04f5368f"
TOTAL_BYTES = 632981
COVERED_SUBTREES = ("code/", "data/", "reference/")
EXCLUDED_PATHS = ("baseline.json", "protocol.json", "README.md", "tools/")
REQUIRED_FIELDS = ("path", "source_path", "sha256", "git_blob_sha1", "bytes", "mode", "role")
ROLES = ("executed", "comparison", "provenance_only")
SOURCE_PREFIX = {"code/": "", "data/": "data/", "reference/": "outputs/"}

MANIFEST = json.loads((BUNDLE / "baseline.json").read_text(encoding="utf-8"))


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True)


def run_verify(bundle):
    completed = subprocess.run(
        [sys.executable, str(VERIFY), "--bundle", str(bundle)],
        capture_output=True,
        text=True,
    )
    return completed


def bundle_copy(destination):
    shutil.copytree(BUNDLE, destination)
    return Path(destination)


class BundleContentTests(unittest.TestCase):
    def test_bundle_holds_seventeen_files_and_632981_bytes(self):
        files = MANIFEST["files"]
        self.assertEqual(len(files), 17)
        self.assertEqual(sum(entry["bytes"] for entry in files), TOTAL_BYTES)

        on_disk = sorted(
            path.relative_to(BUNDLE).as_posix()
            for subtree in COVERED_SUBTREES
            for path in (BUNDLE / subtree).rglob("*")
            if path.is_file()
        )
        self.assertEqual(on_disk, sorted(entry["path"] for entry in files))

    def test_every_file_matches_its_commit_blob(self):
        for entry in MANIFEST["files"]:
            with self.subTest(path=entry["path"]):
                blob = git("cat-file", "blob", f"{BASELINE_COMMIT}:{entry['source_path']}")
                self.assertEqual(
                    blob.returncode,
                    0,
                    f"git cannot read {BASELINE_COMMIT}:{entry['source_path']}",
                )
                data = blob.stdout
                self.assertEqual(len(data), entry["bytes"], entry["path"])
                self.assertEqual(hashlib.sha256(data).hexdigest(), entry["sha256"], entry["path"])
                self.assertEqual(
                    hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest(),
                    entry["git_blob_sha1"],
                    entry["path"],
                )
                listed = git("ls-tree", BASELINE_COMMIT, "--", entry["source_path"]).stdout.decode()
                self.assertEqual(entry["mode"], listed.split()[0], entry["path"])


class ManifestSchemaTests(unittest.TestCase):
    def test_every_file_records_the_seven_manifest_fields(self):
        for entry in MANIFEST["files"]:
            with self.subTest(path=entry["path"]):
                for field in REQUIRED_FIELDS:
                    self.assertIn(field, entry)
                self.assertEqual(len(entry["sha256"]), 64)
                self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
                self.assertRegex(entry["git_blob_sha1"], r"^[0-9a-f]{40}$")
                self.assertIsInstance(entry["bytes"], int)
                self.assertGreater(entry["bytes"], 0)
                self.assertIn(entry["mode"], ("100644", "100755"))
                self.assertIn(entry["role"], ROLES)
                prefix = next(sub for sub in COVERED_SUBTREES if entry["path"].startswith(sub))
                self.assertEqual(
                    entry["source_path"],
                    SOURCE_PREFIX[prefix] + entry["path"][len(prefix):],
                )

    def test_roles_and_the_provenance_only_parquet(self):
        roles = {entry["path"]: entry["role"] for entry in MANIFEST["files"]}
        self.assertEqual(
            roles["data/processed/eligibility_filters.parquet"],
            "provenance_only",
        )
        for path, role in roles.items():
            if path.startswith("code/"):
                self.assertEqual(role, "executed", path)
            elif path.startswith("reference/"):
                self.assertEqual(role, "comparison", path)
            else:
                expected = "executed" if path.endswith(".csv") else "provenance_only"
                self.assertEqual(role, expected, path)

    def test_coverage_covers_three_subtrees_and_names_four_exclusions(self):
        coverage = MANIFEST["coverage"]
        self.assertEqual(tuple(coverage["included_subtrees"]), COVERED_SUBTREES)
        excluded = {entry["path"]: entry["reason"] for entry in coverage["excluded"]}
        self.assertEqual(sorted(excluded), sorted(EXCLUDED_PATHS))
        for path, reason in excluded.items():
            self.assertGreater(len(reason.strip()), 10, path)


class BuilderTests(unittest.TestCase):
    def test_builder_reproduces_the_committed_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "baseline.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(BUILD),
                    "--bundle",
                    str(BUNDLE),
                    "--out",
                    str(out),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads(completed.stdout)
            self.assertEqual(summary["status"], "WROTE")
            self.assertEqual(summary["bytes"], TOTAL_BYTES)
            rebuilt = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(rebuilt, MANIFEST)


class VerifyCliTests(unittest.TestCase):
    def test_verified_bundle_exits_zero_with_seventeen_files(self):
        completed = run_verify(BUNDLE)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual(report["status"], "VERIFIED")
        self.assertEqual(report["files"], 17)
        self.assertEqual(report["findings"], [])

    def test_missing_file_is_reported_by_path_and_kind(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = bundle_copy(Path(tmp) / "bundle")
            (bundle / "code" / "ga.py").unlink()
            completed = run_verify(bundle)

        self.assertEqual(completed.returncode, 1)
        report = json.loads(completed.stdout)
        self.assertEqual(report["status"], "DRIFT")
        self.assertEqual(
            report["findings"],
            [{"path": "code/ga.py", "kind": "missing"}],
        )
        self.assertIn("code/ga.py", completed.stderr)

    def test_modified_file_is_reported_by_path_and_kind(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = bundle_copy(Path(tmp) / "bundle")
            target = bundle / "code" / "ga.py"
            target.write_bytes(target.read_bytes() + b"#")
            completed = run_verify(bundle)

        self.assertEqual(completed.returncode, 1)
        report = json.loads(completed.stdout)
        self.assertEqual(report["status"], "DRIFT")
        self.assertEqual(
            report["findings"],
            [{"path": "code/ga.py", "kind": "modified"}],
        )

    def test_unexpected_file_is_reported_by_path_and_kind(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = bundle_copy(Path(tmp) / "bundle")
            (bundle / "code" / "stray.py").write_text("# not in the manifest\n", encoding="utf-8")
            completed = run_verify(bundle)

        self.assertEqual(completed.returncode, 1)
        report = json.loads(completed.stdout)
        self.assertEqual(report["status"], "DRIFT")
        self.assertEqual(
            report["findings"],
            [{"path": "code/stray.py", "kind": "unexpected"}],
        )

    def test_mode_change_is_reported_by_path_and_kind(self):
        recorded = {entry["path"]: entry["mode"] for entry in MANIFEST["files"]}
        self.assertEqual(recorded["code/ga.py"], "100644")

        with tempfile.TemporaryDirectory() as tmp:
            bundle = bundle_copy(Path(tmp) / "bundle")
            target = bundle / "code" / "ga.py"
            target.chmod(0o755)
            self.assertEqual(target.read_bytes(), (BUNDLE / "code" / "ga.py").read_bytes())
            completed = run_verify(bundle)

        self.assertEqual(completed.returncode, 1)
        report = json.loads(completed.stdout)
        self.assertEqual(report["status"], "DRIFT")
        self.assertEqual(
            report["findings"],
            [{"path": "code/ga.py", "kind": "modified"}],
        )


class ManifestStructureTests(unittest.TestCase):
    """Every structural arm of `load_manifest`, each reached through the CLI on its own fixture.

    The plan decides "a file with two roles is a manifest error" (Observable) and AC 26 sends a
    manifest the CLI cannot trust to exit 3. Malformed JSON is covered by C33; each closed-schema
    arm below carries its own broken copy and its own message, so removing any guard reddens this
    test instead of silently widening what the verifier accepts.
    """

    def assert_rejected(self, edit, *fragments):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = bundle_copy(Path(tmp) / "bundle")
            manifest = json.loads((bundle / "baseline.json").read_text(encoding="utf-8"))
            edit(manifest)
            (bundle / "baseline.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            completed = run_verify(bundle)

        self.assertEqual(completed.returncode, 3, completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual(report["status"], "ERROR")
        self.assertTrue(completed.stderr.strip())
        for fragment in fragments:
            self.assertIn(fragment, completed.stderr)

    def test_structurally_broken_manifest_is_rejected_naming_the_defect(self):
        def first_entry(manifest):
            return manifest["files"][0]

        def relisted_entry(manifest, path):
            entry = dict(first_entry(manifest))
            entry["path"] = path
            return entry

        def drop_field(manifest):
            del first_entry(manifest)["sha256"]

        def bad_role(manifest):
            first_entry(manifest)["role"] = "executor"

        def duplicate(manifest):
            manifest["files"].append(dict(first_entry(manifest)))

        def outside_subtrees(manifest):
            manifest["files"].append(relisted_entry(manifest, "tools/extra.py"))

        def wrong_schema_version(manifest):
            manifest["schema_version"] = 2

        def coverage_not_an_object(manifest):
            manifest["coverage"] = None

        def coverage_narrows(manifest):
            manifest["coverage"]["included_subtrees"] = ["code/"]

        def empty_files(manifest):
            manifest["files"] = []

        def non_object_entry(manifest):
            manifest["files"].append(1)

        arms = {
            "schema_version": (wrong_schema_version, "schema_version is 2", "expected 1"),
            "coverage shape": (coverage_not_an_object, "has no coverage object"),
            "coverage subtrees": (coverage_narrows, "coverage.included_subtrees"),
            "empty files": (empty_files, "has no files array"),
            "non-object entry": (non_object_entry, "non-object entry"),
            "missing field": (drop_field, "code/ga.py is missing sha256"),
            "bad role": (bad_role, "code/ga.py has role 'executor'"),
            "duplicate path": (duplicate, "code/ga.py is listed twice"),
            "outside subtrees": (outside_subtrees, "tools/extra.py is outside the covered subtrees"),
        }
        for name, (edit, *fragments) in arms.items():
            with self.subTest(arm=name):
                self.assert_rejected(edit, *fragments)

    def test_manifest_that_is_not_a_json_object_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = bundle_copy(Path(tmp) / "bundle")
            (bundle / "baseline.json").write_text("[]", encoding="utf-8")
            completed = run_verify(bundle)

        self.assertEqual(completed.returncode, 3, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["status"], "ERROR")
        self.assertIn("is not a JSON object", completed.stderr)


if __name__ == "__main__":
    unittest.main()
