"""Análise dos itens (a), (b) e (c) sobre os JSONs de run_seeds.py e reference.py.
Uso: analyze.py <out.json>   (imprime as tabelas em Markdown)"""
import itertools, json, statistics as st, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import common as c

RES = Path(__file__).parent / "results"
df, cfg, _, _ = c.load_inputs()
score = dict(zip(df.TICKER, df.SCORE.astype(float)))
ref = json.loads((RES / "reference.json").read_text())
OPT = ref["optimum_dedup"]["fitness_ga"]
opt_set = set(ref["optimum_dedup"]["tickers"])
seeds = sorted(json.loads(p.read_text())["seed"] for p in RES.glob("runs_seed*.json"))
runs = {s: json.loads((RES / f"runs_seed{s}.json").read_text())["results"] for s in seeds}
n = 12


def jac(a, b):
    a, b = set(a), set(b); return len(a & b) / len(a | b)


def stats(xs):
    return {"mean": st.mean(xs), "sd": st.pstdev(xs), "min": min(xs), "max": max(xs)}


out = {"seeds": seeds, "optimum": OPT, "by_R": {}}
Rs = list(range(10, 301, 10))
cons = {}
for R in Rs:
    cs = {s: c.consensus_tickers(runs[s][:R], df, cfg) for s in seeds}
    cons[R] = cs
    fit = {s: c.fitness_of(df, cfg, cs[s]) for s in seeds}
    pairs = [jac(cs[a], cs[b]) for a, b in itertools.combinations(seeds, 2)]
    swaps = [n - len(set(cs[a]) & set(cs[b])) for a, b in itertools.combinations(seeds, 2)]
    # margem de frequência entre o 12º e o 13º do ranking de frequência (sem dedup), média entre seeds
    margins = []
    for s in seeds:
        cnt = Counter(t for r in runs[s][:R] for t in r["tickers"])
        f = sorted(cnt.values(), reverse=True)
        margins.append((f[n - 1] - f[n]) / R)
    out["by_R"][R] = {
        "consensus_fitness": stats(list(fit.values())), "fitness_by_seed": fit,
        "pair_jaccard": stats(pairs), "pair_swaps": stats(swaps),
        "jaccard_vs_optimum": stats([jac(cs[s], opt_set) for s in seeds]),
        "freq_margin_12_13": stats(margins),
        "distinct_consensus": len({tuple(v) for v in cs.values()}),
    }
    if R > 10:
        prev = cons[R - 10]
        out["by_R"][R]["changed_vs_R_minus_10"] = sum(set(prev[s]) != set(cs[s]) for s in seeds)

# (a) dentro de um consenso: os runs que o formam, R = 150
R = 150
within = {s: stats([r["fitness"] for r in runs[s][:R]]) for s in seeds}
best_run = {s: max(r["fitness"] for r in runs[s][:R]) for s in seeds}
run_jac = {s: st.mean(jac(a["tickers"], b["tickers"]) for a, b in itertools.combinations(runs[s][:R], 2)) for s in seeds}
cs150 = cons[150]
freq_across = Counter(t for s in seeds for t in cs150[s])
modal = sorted(sorted(freq_across, key=lambda t: (-freq_across[t], -score[t]))[:n])
out["a"] = {
    "within_run_fitness_by_seed": within,
    "within_sd_mean": st.mean(v["sd"] for v in within.values()),
    "within_range_mean": st.mean(v["max"] - v["min"] for v in within.values()),
    "best_run_by_seed": best_run, "run_to_run_jaccard_by_seed": run_jac,
    "generations": stats([r["generations_run"] for s in seeds for r in runs[s][:R]]),
    "ticker_in_n_consensus": {t: [freq_across[t], score[t], t in opt_set] for t in sorted(freq_across, key=lambda t: (-freq_across[t], -score[t]))},
    "modal_set": modal, "modal_fitness": c.fitness_of(df, cfg, modal),
    "consensus_by_seed": cs150,
}
all_fit = [r["fitness"] for s in seeds for r in runs[s]]
out["all_runs"] = {"n": len(all_fit), **stats(all_fit), "share_of_optimum_max": max(all_fit) / OPT,
                   "distinct_portfolios": len({tuple(r["tickers"]) for s in seeds for r in runs[s]}),
                   "runs_hitting_optimum": sum(abs(f - OPT) < 1e-9 for f in all_fit)}
Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=str))

print(f"Ótimo exato: {OPT:.4f}\n")
print("| R | fitness do consenso (média ± dp) | mín–máx | Jaccard entre seeds | trocas entre seeds | Jaccard vs ótimo | margem freq. 12º–13º | consensos distintos | seeds que mudaram vs R−10 |")
print("|---:|---|---|---:|---:|---:|---:|---:|---:|")
for R in Rs:
    b = out["by_R"][R]
    print(f"| {R} | {b['consensus_fitness']['mean']:.3f} ± {b['consensus_fitness']['sd']:.3f} | {b['consensus_fitness']['min']:.3f}–{b['consensus_fitness']['max']:.3f} | "
          f"{b['pair_jaccard']['mean']:.2f} | {b['pair_swaps']['mean']:.1f} | {b['jaccard_vs_optimum']['mean']:.2f} | {b['freq_margin_12_13']['mean']:.3f} | {b['distinct_consensus']} | {b.get('changed_vs_R_minus_10', '-')} |")
print("\n(a) R=150 por seed: fitness consenso | runs média ± dp (mín–máx) | melhor run | Jaccard run a run")
for s in seeds:
    w = within[s]
    print(f"| {s} | {out['by_R'][150]['fitness_by_seed'][s]:.3f} | {w['mean']:.3f} ± {w['sd']:.3f} ({w['min']:.3f}–{w['max']:.3f}) | {best_run[s]:.3f} | {run_jac[s]:.3f} |")
print("\nPapéis nos consensos (R=150): ticker, em quantos dos 10, score, no ótimo?")
for t, v in out["a"]["ticker_in_n_consensus"].items():
    print(f"  {t}: {v[0]} | {v[1]:.3f} | {'sim' if v[2] else 'não'}")
print("modal", modal, out["a"]["modal_fitness"])
print("todos os runs", out["all_runs"], "\ngerações", out["a"]["generations"])
