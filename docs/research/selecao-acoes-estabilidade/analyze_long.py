"""Tabela do item (d) sobre os long_*.json; confere também o determinismo (long_base contra os 20 primeiros runs
do seed 1 de run_seeds.py, que usam as mesmas seeds). Uso: analyze_long.py <out.json>"""
import itertools, json, statistics as st, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import common as c

RES = Path(__file__).parent / "results"
df, cfg, _, _ = c.load_inputs()
ref = json.loads((RES / "reference.json").read_text())
OPT, opt = ref["optimum_dedup"]["fitness_ga"], set(ref["optimum_dedup"]["tickers"])
jac = lambda a, b: len(set(a) & set(b)) / len(set(a) | set(b))
order = ["base", "pat200", "full400", "gen2000", "gen10000", "pop1000", "pop1000full", "pop3000full",
         "mut005", "mut005full", "mut001full"]
out = {}
print("| variante | pop | gerações | paciência | mutação | runs | fitness mediana (mín–máx) | % do ótimo (mediana) | gerações rodadas (média) | s/run (média) | Jaccard run a run | Jaccard vs ótimo | fitness do consenso |")
print("|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|")
for name in order:
    p = RES / f"long_{name}.json"
    if not p.exists():
        continue
    m = json.loads(p.read_text()); rs = m["results"]
    f = [r["fitness"] for r in rs]
    cons = c.consensus_tickers(rs, df, cfg)
    row = {"pop": m["pop_size"], "gens": m["generations"], "pat": m["patience"], "mut": m.get("mutation_rate") or 0.02,
           "runs": len(rs), "median": st.median(f), "min": min(f), "max": max(f), "pct": st.median(f) / OPT,
           "gens_run": st.mean(r["generations_run"] for r in rs), "wall": st.mean(r["wall_s"] for r in rs),
           "jac_runs": st.mean(jac(a["tickers"], b["tickers"]) for a, b in itertools.combinations(rs, 2)),
           "jac_opt": st.mean(jac(r["tickers"], opt) for r in rs), "cons_fit": c.fitness_of(df, cfg, cons),
           "hits_opt": sum(abs(x - OPT) < 1e-9 for x in f)}
    out[name] = row
    print(f"| {name} | {row['pop']} | {row['gens']} | {row['pat']} | {row['mut']} | {row['runs']} | {row['median']:.3f} ({row['min']:.3f}–{row['max']:.3f}) | "
          f"{100*row['pct']:.0f}% | {row['gens_run']:.0f} | {row['wall']:.1f} | {row['jac_runs']:.2f} | {row['jac_opt']:.2f} | {row['cons_fit']:.3f} |")
seed1 = json.loads((RES / "runs_seed1.json").read_text())["results"][:20]
base = json.loads((RES / "long_base.json").read_text())["results"]
out["determinism_base_vs_seed1"] = all(a["tickers"] == b["tickers"] and a["fitness"] == b["fitness"] for a, b in zip(seed1, base))
print("determinismo:", out["determinism_base_vs_seed1"], "| ótimo", OPT)
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
