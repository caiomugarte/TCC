#!/usr/bin/env python3
"""_legacy_driver.py - execute the archived v0-legacy run sequence and write runs.json.

The replay spawns this with the scratch root as cwd. Everything it needs is there: `code/`,
`data/processed/`, and `protocol.json` copied in by `replay.py`. It writes exactly one file,
`runs.json`, in the cwd, and never reads `reference/` - the expected values are the replay's
side of the comparison.

`fitness` is what the archived `run_ga` reports through `carteira.attrs["fitness"]`, which
`ga.py:132-134` takes from the previous generation's scores. `recomputed_fitness` is the
archived `ga.fitness` applied to the chromosome actually returned. Both are reported; neither
is corrected here, because the defect belongs to P0-02, not to the archive.
"""

import importlib
import json
import random
import sys
from pathlib import Path

DRIVER = "legacy-ga-v0"
OUTPUT = "runs.json"


def run_all(root):
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    profile = protocol["profile"]
    runs = protocol["runs"]
    ga_config = protocol["ga"]

    sys.path.insert(0, str(root / "code"))
    import numpy as np
    import pandas as pd  # noqa: F401  (imported by the archived modules this driver calls)
    import ga
    import profiles as pf

    frame = pf.load_profile_data(profile, root / "data" / "processed")
    ranked = pf.build_scores(frame, profile)

    results = []
    for run_id in range(runs["run_id_start"], runs["run_id_start"] + runs["n_runs"]):
        seed_value = runs["seed_base"] + run_id
        random.seed(seed_value)
        np.random.seed(seed_value)
        # The archived ga_multiple_runs re-imports ga to reset the module seed, then re-seeds
        # random. Mirroring that sequence is what makes this a replay rather than a rewrite.
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


def main():
    root = Path.cwd()
    payload = run_all(root)
    (root / OUTPUT).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"_legacy_driver: wrote {len(payload['runs'])} runs to {root / OUTPUT}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
