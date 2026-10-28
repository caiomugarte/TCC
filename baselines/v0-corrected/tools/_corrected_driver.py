#!/usr/bin/env python3
"""_corrected_driver.py - execute the corrected v0 run sequence and write runs.json.

The generator and the replay spawn this with a scratch root as cwd. Everything it
needs is there: `code/`, `data/processed/` and `protocol.json` copied in by the
spawner. It writes exactly one file, `runs.json`, in the cwd, never reads
`reference/`, and never writes outside the cwd.

The per-run sequence is the exact mirror of `_legacy_driver.run_all` - random.seed +
np.random.seed + `importlib.reload(ga)` + random.seed - so E1 compares identical RNG
call sequences and a (profile, seed) row here must equal the frozen writer's row for
the same pair. The bundle's own `code/ga.py` and `code/profiles.py` (the scratch copies
of the bundle) are loaded via importlib under their canonical names; `reload(ga)` per
run re-executes that file, exactly as the frozen `ga_multiple_runs` re-imports it to
reset the module seed.

`fitness` is what the corrected `run_ga` reports through `carteira.attrs["fitness"]`,
which the corrected record block takes from the generation it belongs to.
`recomputed_fitness` is the frozen `ga.fitness` applied to the chromosome actually
returned; VC-01 requires the two to be equal on every row.
"""

import sys

sys.dont_write_bytecode = True  # before any bundle import (AD-013)

import importlib
import importlib.util
import json
import random
from pathlib import Path

DRIVER = "corrected-ga-v0"
OUTPUT = "runs.json"


def _load_code_module(name, code_dir):
    """Import one file of the bundle's own code under its canonical module name.

    The code dir goes on sys.path too, so the per-run `importlib.reload(ga)` mirror
    re-resolves to that same bundle file.
    """
    code_dir = Path(code_dir)
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    path = code_dir / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_all(root, run_ids=None):
    root = Path(root)
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    profile = protocol["profile"]
    runs = protocol["runs"]
    ga_config = protocol["ga"]

    import numpy as np
    import pandas as pd  # noqa: F401  (imported by the frozen modules this driver calls)

    ga = _load_code_module("ga", root / "code")
    pf = _load_code_module("profiles", root / "code")

    frame = pf.load_profile_data(profile, root / "data" / "processed")
    ranked = pf.build_scores(frame, profile)

    if run_ids is None:
        run_ids = range(runs["run_id_start"], runs["run_id_start"] + runs["n_runs"])

    results = []
    for run_id in run_ids:
        seed_value = runs["seed_base"] + run_id
        random.seed(seed_value)
        np.random.seed(seed_value)
        # The frozen ga_multiple_runs re-imports ga to reset the module seed, then
        # re-seeds random. Mirroring that sequence is what makes this a replay rather
        # than a rewrite.
        importlib.reload(ga)
        random.seed(seed_value)

        carteira = ga.run_ga(ranked, profile)
        tickers = sorted(carteira["TICKER"].tolist())
        mask = ranked["TICKER"].isin(tickers).to_numpy()
        results.append(
            {
                "run_id": run_id,
                "seed": seed_value,
                "tickers": tickers,
                "fitness": float(carteira.attrs["fitness"]),
                "hhi": float(carteira.attrs["hhi"]),
                "recomputed_fitness": float(
                    ga.fitness(ranked, mask, ga_config["lambda"], ga_config["n_assets"])
                ),
            }
        )
    return {"driver": DRIVER, "profile": profile, "runs": results}


def main(argv=None):
    root = Path.cwd()
    payload = run_all(root)
    (root / OUTPUT).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"_corrected_driver: wrote {len(payload['runs'])} runs to {root / OUTPUT}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
