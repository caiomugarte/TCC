"""Itens (a) e (b): R runs do GA de produção para cada seed base, como o run Premium faz
(``derive_run_seed(seed, run_id, "stock")``). Grava um JSON por seed com tickers, fitness e gerações de cada run;
o consenso com R runs é o consenso dos R primeiros, igual a rodar com ``run_count`` = R sem parada adaptativa.

Uso: run_seeds.py <runs> <processos> <seed> [<seed> ...]
"""
import json, sys, time
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import common as c

OUT = Path(__file__).parent / "results"


def one_seed(args):
    seed, runs = args
    from core.optimizer import derive_run_seed
    df, cfg, _, _ = c.load_inputs()
    out, t0 = [], time.perf_counter()
    for run_id in range(runs):
        ga = c.make_ga(cfg, derive_run_seed(seed, run_id, "stock"))
        p = ga.optimize(df)
        out.append({"run_id": run_id, "tickers": sorted(p["TICKER"]), "fitness": float(p.attrs["fitness"]),
                    "generations_run": int(p.attrs["generations_run"])})
    wall = time.perf_counter() - t0
    (OUT / f"runs_seed{seed}.json").write_text(json.dumps({"seed": seed, "runs": runs, "wall_s": wall, "results": out}))
    return seed, wall


if __name__ == "__main__":
    runs, procs, seeds = int(sys.argv[1]), int(sys.argv[2]), [int(s) for s in sys.argv[3:]]
    t0 = time.perf_counter()
    with Pool(procs) as pool:
        for seed, wall in pool.imap_unordered(one_seed, [(s, runs) for s in seeds]):
            print(f"seed {seed}: {wall:.0f} s", flush=True)
    print(f"total {time.perf_counter() - t0:.0f} s")
