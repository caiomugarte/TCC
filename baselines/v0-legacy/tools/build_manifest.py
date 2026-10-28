#!/usr/bin/env python3
"""build_manifest.py - hash the frozen v0-legacy subtrees into baseline.json.

Build time only: `verify_manifest.py` and `replay.py` never import or run this file, so a bug
here cannot make a bundle verify itself. The hash definitions come from `verify_manifest.py`
on purpose - the manifest is only meaningful relative to the definition its verifier applies.

Each covered file keeps the repo-relative path it was archived from (`source_path`), so the
manifest itself carries the `git show 97ea327:<source_path>` that filled it.
"""

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import verify_manifest as vm  # noqa: E402  (sibling tool, resolved from this file's directory)

BASELINE_ID = "v0-legacy"
BASELINE_COMMIT = "97ea3275293da4e1902fcfeb684c814f04f5368f"
SCHEMA_VERSION = 1

# Door B1: the subtree fixes the mapping and the file keeps its basename.
SOURCE_PREFIX = {
    "code/": "",
    "data/": "data/",
    "reference/": "outputs/",
}

EXCLUDED = [
    {
        "path": "baseline.json",
        "reason": "the manifest cannot hash itself without a fixed point",
    },
    {
        "path": "protocol.json",
        "reason": (
            "authored protocol, not an archived byte; equality with code/ is checked by "
            "tests/test_v0_legacy_protocol.py, which must be able to read it edited"
        ),
    },
    {
        "path": "README.md",
        "reason": "authored reader's guide, not evidence of the archived execution",
    },
    {
        "path": "tools/",
        "reason": (
            "authored verifier and replay; hashing them would let a changed verifier still "
            "verify, because the manifest is what the verifier reads"
        ),
    },
]


def source_path_for(relpath):
    for prefix, source in SOURCE_PREFIX.items():
        if relpath.startswith(prefix):
            return source + relpath[len(prefix):]
    raise ValueError(f"{relpath} is outside the covered subtrees")


def role_for(relpath):
    if relpath.startswith("code/"):
        return "executed"
    if relpath.startswith("reference/"):
        return "comparison"
    if relpath.startswith("data/"):
        # The replay loads the cleaned CSVs; anything else under data/ is archived as the
        # universe behind them and is never read by a covered module.
        return "executed" if relpath.endswith(".csv") else "provenance_only"
    raise ValueError(f"{relpath} is outside the covered subtrees")


def build_manifest(bundle, commit=BASELINE_COMMIT):
    bundle = Path(bundle).resolve()
    files = []
    for relpath in vm.walk_covered(bundle):
        path = bundle / relpath
        data = path.read_bytes()
        files.append(
            {
                "path": relpath,
                "source_path": source_path_for(relpath),
                "sha256": vm.sha256_bytes(data),
                "git_blob_sha1": vm.git_blob_sha1(data),
                "bytes": len(data),
                "mode": vm.file_mode(path),
                "role": role_for(relpath),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "baseline_id": BASELINE_ID,
        "created_from": {"commit": commit},
        "coverage": {
            "included_subtrees": list(vm.INCLUDED_SUBTREES),
            "excluded": [dict(entry) for entry in EXCLUDED],
        },
        "files": files,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="build_manifest.py",
        description="Hash the covered subtrees and write baseline.json (build time only).",
    )
    parser.add_argument(
        "--bundle",
        default="baselines/v0-legacy",
        help="bundle directory to hash (default: %(default)s)",
    )
    parser.add_argument(
        "--commit",
        default=BASELINE_COMMIT,
        help="commit the archived bytes came from (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="manifest path (default: <bundle>/baseline.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the manifest without writing it",
    )
    args = parser.parse_args(argv)

    bundle = Path(args.bundle).resolve()
    manifest = build_manifest(bundle, args.commit)
    text = json.dumps(manifest, indent=2, sort_keys=False) + "\n"
    out = Path(args.out) if args.out else bundle / "baseline.json"
    if not args.dry_run:
        out.write_text(text, encoding="utf-8")
    summary = {
        "status": "DRY_RUN" if args.dry_run else "WROTE",
        "path": str(out),
        "files": len(manifest["files"]),
        "bytes": sum(entry["bytes"] for entry in manifest["files"]),
        "commit": manifest["created_from"]["commit"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
