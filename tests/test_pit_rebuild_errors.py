"""Exit codes and messages of the rebuild's error classes (CLI-01, PIT-02)."""

import re
import sys
import unittest
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))

import pit_rebuild  # noqa: E402
from pit_rebuild.errors import (  # noqa: E402
    ArgumentError,
    IntegrityError,
    MissingPathError,
    PITLeakError,
    SourceError,
)


class ErrorClassTests(unittest.TestCase):
    def test_each_class_carries_its_cli_exit_code_and_one_line_message(self):
        self.assertEqual(ArgumentError.exit_code, 2)
        self.assertEqual(MissingPathError.exit_code, 3)
        self.assertEqual(IntegrityError.exit_code, 4)
        self.assertEqual(SourceError.exit_code, 5)

        err = ArgumentError("--out-dir", "resolves inside data/")
        self.assertEqual(str(err), "--out-dir: resolves inside data/")
        self.assertEqual(err.field_name, "--out-dir")
        self.assertEqual(err.reason, "resolves inside data/")

    def test_pit_leak_is_an_integrity_error_naming_ticker_and_both_dates(self):
        err = PITLeakError("PETR4", date(2026, 8, 7), date(2026, 8, 10))

        self.assertIsInstance(err, IntegrityError)
        self.assertEqual(err.exit_code, 4)
        self.assertIn("PETR4", str(err))
        self.assertIn("2026-08-07", str(err))
        self.assertIn("2026-08-10", str(err))


class PackageDocstringTests(unittest.TestCase):
    def test_docstring_lists_exit_codes_0_to_5(self):
        codes = set(re.findall(r"^\s+([0-5])\s{2,}\S", pit_rebuild.__doc__, re.MULTILINE))

        self.assertEqual(codes, {"0", "1", "2", "3", "4", "5"})


if __name__ == "__main__":
    unittest.main()
