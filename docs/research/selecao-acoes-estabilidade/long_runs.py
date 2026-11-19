"""Item (d): runs mais longos chegam a fitness maior? Mesmo GA de produção, mudando só população, gerações e
paciência do early stopping (fixa em 50 em produção: ``optimize_portfolio`` não repassa o parâmetro).
Seeds dos runs: ``derive_run_seed(1, run_id, "stock")``, as mesmas do seed base 1 de run_seeds.py.

Sondagem fora do escopo de (d): um 6º campo opcional troca a taxa de mutação (produção: 0,02).

Uso: long_runs.py <processos> <nome:pop:gerações:paciência:runs[:mutação]> [...]
"""
import json, sys, time
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import common as c

OUT = Path(__file__).parent / "results"


def one(args):
    name, pop, gens, pat, run_id, mut = args
    from core.optimizer import derive_run_seed
    df, cfg, _, _ = c.load_inputs()
    t = time.perf_counter()
    p = c.make_ga(cfg, derive_run_seed(1, run_id, "stock"), generations=gens, pop_size=pop, patience=pat,
                 mutation_rate=mut).optimize(df)
    return name, {"run_id": run_id, "tickers": sorted(p["TICKER"]), "fitness": float(p.attrs["fitness"]),
                  "generations_run": int(p.attrs["generations_run"]), "wall_s": time.perf_counter() - t}


if __name__ == "__main__":
    procs, specs = int(sys.argv[1]), sys.argv[2:]
    tasks, meta = [], {}
    for spec in specs:
        name, pop, gens, pat, runs, *rest = spec.split(":")
        mut = float(rest[0]) if rest else None
        meta[name] = {"pop_size": int(pop), "generations": int(gens), "patience": int(pat), "runs": int(runs),
                      "mutation_rate": mut, "results": []}
        tasks += [(name, int(pop), int(gens), int(pat), i, mut) for i in range(int(runs))]
    t0 = time.perf_counter()
    with Pool(procs) as pool:
        for name, r in pool.imap_unordered(one, tasks):
            meta[name]["results"].append(r)
    for name, m in meta.items():
        m["results"].sort(key=lambda r: r["run_id"])
        (OUT / f"long_{name}.json").write_text(json.dumps(m))
        f = [r["fitness"] for r in m["results"]]
        print(name, "max", round(max(f), 4), "med", round(sorted(f)[len(f) // 2], 4), flush=True)
    print(f"total {time.perf_counter() - t0:.0f} s")
