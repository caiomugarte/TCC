"""Write guard and atomic report writes of the rebuild (SAFE-01, RPT-01, RPT-05).

The guard tests never write into data/, snapshots/ or baselines/: they only
resolve paths. The atomic write tests use a temporary directory.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))

from pit_rebuild.errors import ArgumentError  # noqa: E402
from pit_rebuild.safety import REPO_ROOT, assert_writable_dir, write_files_atomically  # noqa: E402


class WriteGuardTests(unittest.TestCase):
    def test_each_forbidden_root_is_refused_with_exit_2(self):
        for name in ("data", "snapshots", "baselines"):
            with self.subTest(root=name):
                with self.assertRaises(ArgumentError) as ctx:
                    assert_writable_dir(REPO_ROOT / name, "--out-dir")
                self.assertEqual(ctx.exception.exit_code, 2)
                self.assertEqual(ctx.exception.field_name, "--out-dir")

    def test_nested_child_of_a_forbidden_root_is_refused(self):
        with self.assertRaises(ArgumentError) as ctx:
            assert_writable_dir(REPO_ROOT / "data" / "raw" / "run-1", "--cache-dir")

        self.assertEqual(ctx.exception.exit_code, 2)

    def test_symlink_pointing_into_data_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            link = Path(tmp) / "out"
            link.symlink_to(REPO_ROOT / "data")

            with self.assertRaises(ArgumentError) as ctx:
                assert_writable_dir(link / "reports", "--out-dir")

        self.assertEqual(ctx.exception.exit_code, 2)

    def test_docs_validation_and_a_temp_dir_pass(self):
        docs = REPO_ROOT / "docs" / "validation"
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(assert_writable_dir(Path(tmp), "--out-dir"), Path(tmp).resolve())
        self.assertEqual(assert_writable_dir(docs, "--out-dir"), docs.resolve())

    def test_sibling_sharing_a_forbidden_name_prefix_passes(self):
        sibling = REPO_ROOT / "data-notes"

        self.assertEqual(assert_writable_dir(sibling, "--out-dir"), sibling.resolve())


class AtomicWriteTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.json_path = self.dir / "report.json"
        self.md_path = self.dir / "report.md"

    def tearDown(self):
        self._tmp.cleanup()

    def names_in_dir(self):
        return sorted(path.name for path in self.dir.iterdir())

    def test_success_writes_both_files(self):
        write_files_atomically({self.json_path: b"{}\n", self.md_path: b"# report\n"})

        self.assertEqual(self.json_path.read_bytes(), b"{}\n")
        self.assertEqual(self.md_path.read_bytes(), b"# report\n")
        self.assertEqual(self.names_in_dir(), ["report.json", "report.md"])

    def test_failure_on_second_replace_restores_both_targets_and_leaves_no_temp(self):
        self.json_path.write_bytes(b"old json")
        self.md_path.write_bytes(b"old md")
        real_replace = os.replace
        calls = []

        def second_replace_fails(src, dst):
            calls.append(dst)
            if len(calls) == 2:
                raise OSError("injected failure on the second replace")
            return real_replace(src, dst)

        with mock.patch("os.replace", side_effect=second_replace_fails):
            with self.assertRaises(OSError):
                write_files_atomically({self.json_path: b"new json", self.md_path: b"new md"})

        self.assertEqual(self.json_path.read_bytes(), b"old json")
        self.assertEqual(self.md_path.read_bytes(), b"old md")
        self.assertEqual(self.names_in_dir(), ["report.json", "report.md"])

    def test_failure_on_first_replace_leaves_targets_and_no_temp(self):
        self.json_path.write_bytes(b"old json")
        real_replace = os.replace
        calls = []

        def first_replace_fails(src, dst):
            calls.append(dst)
            if len(calls) == 1:
                raise OSError("injected failure on the first replace")
            return real_replace(src, dst)

        with mock.patch("os.replace", side_effect=first_replace_fails):
            with self.assertRaises(OSError):
                write_files_atomically({self.json_path: b"new json", self.md_path: b"new md"})

        self.assertEqual(self.json_path.read_bytes(), b"old json")
        self.assertEqual(self.names_in_dir(), ["report.json"])


if __name__ == "__main__":
    unittest.main()
