#!/usr/bin/env python3
"""verify_manifest.py - prove that the v0-corrected bytes are the committed bytes.

Reads `baseline.json` from the bundle, re-hashes every file it lists, and walks the three
covered subtrees for files it does not list. Standard library only.

The ten-line hashing loop is deliberately duplicated instead of importing
`py/snapshot_manifest.py::sha256_path`: a verifier that imports the code it verifies inherits
that code's future edits, and this bundle's whole claim is that its bytes do not move.

Exit codes: 0 `VERIFIED`, 1 `DRIFT`, 3 the manifest itself cannot be trusted.
Exactly one JSON object goes to stdout; diagnostics go to stderr.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

SCHEMA_VERSION = 1
DEFAULT_BUNDLE = "baselines/v0-corrected"
INCLUDED_SUBTREES = ("code/", "data/", "reference/")
REQUIRED_FILE_FIELDS = ("path", "source_path", "sha256", "git_blob_sha1", "bytes", "mode", "role")
VALID_ROLES = ("executed", "comparison", "provenance_only")

EXIT_VERIFIED = 0
EXIT_DRIFT = 1
EXIT_UNTRUSTWORTHY = 3


class ManifestError(Exception):
    """The manifest cannot be read or violates its own closed schema."""


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def git_blob_sha1(data):
    """The hash git uses for blob objects: sha1('blob <len>\\0' + data)."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def file_mode(path):
    """Git's regular-file modes: 100755 when any execute bit is set, else 100644."""
    return "100755" if os.stat(path).st_mode & 0o111 else "100644"


def is_covered(relpath, subtrees=INCLUDED_SUBTREES):
    return any(relpath.startswith(sub) and relpath != sub for sub in subtrees)


def walk_covered(bundle, subtrees=INCLUDED_SUBTREES):
    """Every regular file under the covered subtrees, as bundle-relative posix paths.

    Stray files count too - a `__pycache__/*.pyc` written under `code/` is `unexpected`.
    """
    bundle = Path(bundle)
    found = []
    for sub in subtrees:
        root = bundle / sub
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                found.append(path.relative_to(bundle).as_posix())
    return found


def load_manifest(bundle):
    """Read and structurally validate baseline.json. Raises ManifestError."""
    path = Path(bundle) / "baseline.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestError(f"cannot read {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ManifestError("baseline.json is not a JSON object")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(
            f"schema_version is {manifest.get('schema_version')!r}, expected {SCHEMA_VERSION}"
        )
    coverage = manifest.get("coverage")
    if not isinstance(coverage, dict):
        raise ManifestError("baseline.json has no coverage object")
    if list(coverage.get("included_subtrees") or ()) != list(INCLUDED_SUBTREES):
        raise ManifestError(
            "coverage.included_subtrees is "
            f"{coverage.get('included_subtrees')!r}, expected {list(INCLUDED_SUBTREES)!r}"
        )
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ManifestError("baseline.json has no files array")
    seen = set()
    for entry in files:
        if not isinstance(entry, dict):
            raise ManifestError("files[] holds a non-object entry")
        missing = [field for field in REQUIRED_FILE_FIELDS if field not in entry]
        if missing:
            raise ManifestError(f"{entry.get('path', '<no path>')} is missing {', '.join(missing)}")
        if entry["role"] not in VALID_ROLES:
            raise ManifestError(f"{entry['path']} has role {entry['role']!r}")
        if entry["path"] in seen:
            raise ManifestError(f"{entry['path']} is listed twice")
        if not is_covered(entry["path"], INCLUDED_SUBTREES):
            raise ManifestError(f"{entry['path']} is outside the covered subtrees")
        seen.add(entry["path"])
    return manifest


def verify_bundle(bundle):
    """Return the verification report: status, file count, and one finding per broken path."""
    bundle = Path(bundle).resolve()
    report = {
        "schema_version": SCHEMA_VERSION,
        "bundle": str(bundle),
        "baseline_id": None,
        "status": "VERIFIED",
        "files": 0,
        "findings": [],
        "errors": [],
    }
    try:
        manifest = load_manifest(bundle)
    except ManifestError as exc:
        report["status"] = "ERROR"
        report["errors"].append(str(exc))
        return report

    report["baseline_id"] = manifest.get("baseline_id")
    listed = {entry["path"]: entry for entry in manifest["files"]}
    report["files"] = len(listed)

    findings = []
    for rel in sorted(listed):
        entry = listed[rel]
        path = bundle / rel
        if not path.is_file():
            findings.append({"path": rel, "kind": "missing"})
            continue
        try:
            data = path.read_bytes()
        except OSError as exc:
            findings.append({"path": rel, "kind": "modified", "detail": str(exc)})
            continue
        if (
            sha256_bytes(data) != entry["sha256"]
            or git_blob_sha1(data) != entry["git_blob_sha1"]
            or len(data) != entry["bytes"]
            or file_mode(path) != entry["mode"]
        ):
            findings.append({"path": rel, "kind": "modified"})

    for rel in walk_covered(bundle):
        if rel not in listed:
            findings.append({"path": rel, "kind": "unexpected"})

    findings.sort(key=lambda finding: (finding["path"], finding["kind"]))
    report["findings"] = findings
    report["status"] = "DRIFT" if findings else "VERIFIED"
    return report


def exit_code(report):
    if report["status"] == "VERIFIED":
        return EXIT_VERIFIED
    if report["status"] == "DRIFT":
        return EXIT_DRIFT
    return EXIT_UNTRUSTWORTHY


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="verify_manifest.py",
        description="Re-hash the covered v0-corrected subtrees against baseline.json.",
    )
    parser.add_argument(
        "--bundle",
        default=DEFAULT_BUNDLE,
        help="bundle directory to verify (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    report = verify_bundle(args.bundle)
    for finding in report["findings"]:
        print(f"verify_manifest: {finding['kind']}: {finding['path']}", file=sys.stderr)
    for error in report["errors"]:
        print(f"verify_manifest: {error}", file=sys.stderr)
    print(json.dumps(report, indent=2, sort_keys=True))
    return exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
