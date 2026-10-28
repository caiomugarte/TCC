#!/usr/bin/env python3
"""build_manifest.py - hash the frozen v0-corrected subtrees into baseline.json.

Build time only: `verify_manifest.py` never imports or runs this file, so a bug here cannot
make a bundle verify itself. The hash definitions come from `verify_manifest.py` on purpose -
the manifest is only meaningful relative to the definition its verifier applies. The sibling
is loaded from this file's own directory under a unique module name, so it can never resolve
to another bundle's verifier.

Each covered file keeps a provenance string (`source_path`): byte-identical copies and the
canonical-transform hunk say which v0-legacy bytes they came from, generated files say how
they were produced.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # never write __pycache__ under the bundle (AD-013)


def _load_sibling(name):
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_v0_corrected_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vm = _load_sibling("verify_manifest")

BASELINE_ID = "v0-corrected"
PINNED_BASE_COMMIT = "181453f986cce7941addf9092e38c6e005e18844"
SCHEMA_VERSION = 1

# Provenance per covered subtree; code/ga.py is the one deliberate difference.
GENERATED_PREFIX = "generated:tools/generate_reference.py#pass-a"
LEGACY_PREFIX = "baselines/v0-legacy/"
CANONICAL_HUNK = "baselines/v0-legacy/code/ga.py#canonical-transform"

EXCLUDED = [
    {
        "path": "baseline.json",
        "reason": "the manifest cannot hash itself without a fixed point",
    },
    {
        "path": "protocol.json",
        "reason": (
            "authored derived protocol, not a copied or generated artifact byte; its "
            "deep-equality with the v0-legacy protocol is checked by "
            "tests/test_v0_corrected_bundle.py, which must be able to read it edited"
        ),
    },
    {
        "path": "README.md",
        "reason": (
            "authored reader's guide and methodological note, not evidence of a "
            "generated execution"
        ),
    },
    {
        "path": "tools/",
        "reason": (
            "authored verifier and generator; hashing them would let a changed verifier "
            "still verify, because the manifest is what the verifier reads"
        ),
    },
    {
        "path": "status.json",
        "reason": (
            "authored verdict map derived from the covered artifacts; it records their "
            "digests and must be able to change without pinning itself"
        ),
    },
    {
        "path": "environment.txt",
        "reason": (
            "authored declaration of the generating environment, not an artifact byte"
        ),
    },
    {
        "path": "requirements.txt",
        "reason": (
            "authored install pin for the declared environment, not an artifact byte"
        ),
    },
]


def source_path_for(relpath):
    if relpath.startswith("reference/"):
        return GENERATED_PREFIX
    if relpath == "code/ga.py":
        return CANONICAL_HUNK
    if relpath.startswith(("code/", "data/")):
        return LEGACY_PREFIX + relpath
    raise ValueError(f"{relpath} is outside the covered subtrees")


def role_for(relpath):
    if relpath.startswith("code/"):
        return "executed"
    if relpath.startswith("reference/"):
        return "comparison"
    if relpath.startswith("data/"):
        # The recomputation loads the cleaned CSVs; anything else under data/ is copied as
        # the universe behind them and is never read by a covered module.
        return "executed" if relpath.endswith(".csv") else "provenance_only"
    raise ValueError(f"{relpath} is outside the covered subtrees")


def build(bundle):
    """Hash every covered file: path, source_path, sha256, git_blob_sha1, bytes, mode, role."""
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
        "created_from": {"pinned_base_commit": PINNED_BASE_COMMIT},
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
        default=vm.DEFAULT_BUNDLE,
        help="bundle directory to hash (default: %(default)s)",
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
    out = Path(args.out) if args.out else bundle / "baseline.json"
    if not args.dry_run and out.exists():
        print(f"build_manifest: refusing to overwrite {out}", file=sys.stderr)
        return vm.EXIT_UNTRUSTWORTHY
    manifest = build(bundle)
    text = json.dumps(manifest, indent=2, sort_keys=False) + "\n"
    if not args.dry_run:
        out.write_text(text, encoding="utf-8")
    summary = {
        "status": "DRY_RUN" if args.dry_run else "WROTE",
        "path": str(out),
        "files": len(manifest["files"]),
        "bytes": sum(entry["bytes"] for entry in manifest["files"]),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return vm.EXIT_VERIFIED


if __name__ == "__main__":
    raise SystemExit(main())
