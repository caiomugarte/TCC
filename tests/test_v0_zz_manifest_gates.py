"""v0-corrected (P0-02) suite trailer: the manifest gates after the whole suite (VC-23, VC-28).

The tests derive from the spec's acceptance criteria, not from the implementation:

- VC-28: WHEN any generator or test produces or imports bundle artifacts, nothing is
  written under `code/`, `data/` or `reference/` of either bundle and no scratch root
  lands in the repository - after the full suite both manifest gates SHALL report zero
  unexpected files.
- VC-23: WHEN any tool or test imports bundle code, it SHALL set
  `sys.dont_write_bytecode = True` before the import - enforced here as a static scan
  with a negative fixture (a scan that cannot fail is not a scan).

This module name (`test_v0_zz_...`) sorts last under `unittest discover -p 'test_v0_*.py'`,
so it observes the tree the whole suite leaves behind. Standard library only (subprocess
gates + an AST scan); nothing here writes anywhere.
"""

import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUNDLES = (
    REPO / "baselines" / "v0-legacy",
    REPO / "baselines" / "v0-corrected",
)
SCRATCH_DIR_PREFIXES = ("tie-", "v0-corrected-gen-", "v0-replay-")
SCRATCH_DIR_NAMES = frozenset({"scratch"})


def is_sys_path_insert(func):
    """The call shape `sys.path.insert(...)` - the path-based bundle-import enabler."""
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "insert"
        and isinstance(func.value, ast.Attribute)
        and func.value.attr == "path"
        and isinstance(func.value.value, ast.Name)
        and func.value.value.id == "sys"
    )


def is_dont_write_bytecode_assign(node):
    """The statement shape `sys.dont_write_bytecode = ...`."""
    return (
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "dont_write_bytecode"
            and isinstance(target.value, ast.Name)
            and target.value.id == "sys"
            for target in node.targets
        )
    )


def bundle_import_offenders(paths, root=None):
    """Files whose bundle imports are not preceded by `sys.dont_write_bytecode = True`.

    A bundle import is (a) a call to `importlib.util.spec_from_file_location` - the
    in-process loader every bundle-code importer here uses - or (b) a call to
    `sys.path.insert` in a file that references the `baselines` tree (such an insert
    enables `import ga`-style bundle imports). Each such call site must be preceded, in
    file order, by a `sys.dont_write_bytecode` assignment (AD-013).

    A `sys.path.insert` in a file that never references the bundle tree is out of
    scope: it cannot import bundle code (e.g. `py/` test helpers), and the frozen
    `baselines/v0-legacy/tools/_legacy_driver.py` is a scratch-root runner - its
    imports resolve to the spawn cwd's scratch copies, never to the bundle.
    """
    offenders = []
    for path in paths:
        path = Path(path)
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        flag_lines, load_lines, insert_lines = [], [], []
        references_bundle = False
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and "baselines" in node.value
            ):
                references_bundle = True
            if is_dont_write_bytecode_assign(node):
                flag_lines.append(node.lineno)
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr == "spec_from_file_location":
                    load_lines.append(node.lineno)
                elif is_sys_path_insert(node.func):
                    insert_lines.append(node.lineno)
        first_flag = min(flag_lines) if flag_lines else None
        name = (
            path.relative_to(root).as_posix()
            if root is not None
            else path.relative_to(REPO).as_posix()
        )
        for line in load_lines:
            if first_flag is None or first_flag > line:
                offenders.append(f"{name}:{line}: spec_from_file_location without a preceding dont_write_bytecode")
        if references_bundle:
            for line in insert_lines:
                if first_flag is None or first_flag > line:
                    offenders.append(f"{name}:{line}: sys.path.insert over the bundle without a preceding dont_write_bytecode")
    return sorted(offenders)


def scan_targets():
    """Every source file that could import bundle code in-process: tests/ + bundles."""
    targets = sorted((REPO / "tests").rglob("*.py"))
    for bundle in BUNDLES:
        targets.extend(sorted(bundle.rglob("*.py")))
    return targets


class ManifestTrailerTests(unittest.TestCase):
    """VC-23/VC-28: the tree after the whole suite, and the import discipline behind it."""

    def test_both_manifest_gates_exit_zero_after_the_full_suite(self):
        for bundle in BUNDLES:
            tool = bundle / "tools" / "verify_manifest.py"
            completed = subprocess.run(
                [sys.executable, str(tool)],
                cwd=str(REPO),
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, f"{tool}: {completed.stderr}")
            report = json.loads(completed.stdout)
            self.assertEqual(report["status"], "VERIFIED", report)
            unexpected = [f for f in report["findings"] if f["kind"] == "unexpected"]
            self.assertEqual(unexpected, [], f"{bundle.name} has unexpected files")

    def test_no_unexpected_files_exist_under_either_bundle(self):
        # no __pycache__ / .pyc may exist anywhere under baselines/ (AD-013 outcome)
        bytecode = []
        for bundle in BUNDLES:
            for path in bundle.rglob("*"):
                if path.name == "__pycache__" and path.is_dir():
                    bytecode.append(path.relative_to(REPO).as_posix())
                elif path.suffix in (".pyc", ".pyo"):
                    bytecode.append(path.relative_to(REPO).as_posix())
        self.assertEqual(bytecode, [], "bytecode written under the frozen bundles")

        # no scratch root may land in the repository (the tools' scratch prefixes)
        scratch_dirs = []
        for dirpath, dirnames, _ in os.walk(REPO):
            if ".git" in dirnames:
                dirnames.remove(".git")
            for name in dirnames:
                if name in SCRATCH_DIR_NAMES or name.startswith(SCRATCH_DIR_PREFIXES):
                    scratch_dirs.append(
                        Path(dirpath, name).relative_to(REPO).as_posix()
                    )
        self.assertEqual(scratch_dirs, [], "scratch roots left inside the repository")

        # the suite never writes to outputs/ (AD-007: outputs/ is never modified)
        completed = subprocess.run(
            ["git", "status", "--porcelain", "--", "outputs/"],
            cwd=str(REPO),
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "", "outputs/ was written to")

    def test_every_bundle_import_sets_dont_write_bytecode_before_the_import(self):
        offenders = bundle_import_offenders(scan_targets())
        self.assertEqual(offenders, [], "bundle imports without bytecode suppression")

        # negative fixture: the same scan must flag an injected offender (D-f)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "offender.py").write_text(
                "import importlib.util\n"
                "spec = importlib.util.spec_from_file_location('ga', 'baselines/v0-legacy/code/ga.py')\n",
                encoding="utf-8",
            )
            (root / "path_offender.py").write_text(
                "import sys\n"
                'sys.path.insert(0, "baselines/v0-legacy/code")\n',
                encoding="utf-8",
            )
            (root / "clean.py").write_text(
                "import importlib.util\n"
                "import sys\n"
                "sys.dont_write_bytecode = True\n"
                "spec = importlib.util.spec_from_file_location('ga', 'baselines/v0-legacy/code/ga.py')\n",
                encoding="utf-8",
            )
            (root / "py_helper.py").write_text(
                "import sys\n"
                'sys.path.insert(0, "py")\n',  # no bundle reference: out of scope
                encoding="utf-8",
            )
            flagged = bundle_import_offenders(sorted(root.glob("*.py")), root=root)
            self.assertEqual(len(flagged), 2, flagged)
            self.assertTrue(any(f.startswith("offender.py:") for f in flagged), flagged)
            self.assertTrue(any(f.startswith("path_offender.py:") for f in flagged), flagged)
            self.assertFalse(any(f.startswith("clean.py:") for f in flagged), flagged)
            self.assertFalse(any(f.startswith("py_helper.py:") for f in flagged), flagged)


if __name__ == "__main__":
    unittest.main()
