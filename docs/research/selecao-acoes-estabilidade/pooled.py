"""Complemento de (a)/(b): consenso com os runs de várias seeds juntos (R = 600 a 3000), para ver para onde o
voto por frequência converge, e as trocas de cada consenso de R = 150 contra o ótimo exato.
Uso: pooled.py <out.json>"""
import json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import common as c

RES = Path(__file__).parent / "results"
df, cfg, _, _ = c.load_inputs()
score = dict(zip(df.TICKER, df.SCORE.astype(float)))
ref = json.loads((RES / "reference.json").read_text())
opt = set(ref["optimum_dedup"]["tickers"]); OPT = ref["optimum_dedup"]["fitness_ga"]
runs = {s: json.loads((RES / f"runs_seed{s}.json").read_text())["results"] for s in range(1, 11)}
out = {"pooled": {}, "swaps_150": {}}
for k in (2, 5, 10):
    pool = [r for s in range(1, k + 1) for r in runs[s]]
    cons = c.consensus_tickers(pool, df, cfg)
    cnt = Counter(t for r in pool for t in r["tickers"])
    top = sorted(cnt.items(), key=lambda kv: -kv[1])[:16]
    out["pooled"][len(pool)] = {"fitness": c.fitness_of(df, cfg, cons), "is_optimum": set(cons) == opt,
                                "out_vs_opt": sorted(opt - set(cons)), "in_vs_opt": sorted(set(cons) - opt),
                                "top16_freq": [(t, v / len(pool), round(score[t], 3)) for t, v in top]}
for s in range(1, 11):
    cons = set(c.consensus_tickers(runs[s][:150], df, cfg))
    out["swaps_150"][s] = {"fitness_gap": OPT - c.fitness_of(df, cfg, cons),
                           "missing": [(t, round(score[t], 3)) for t in sorted(opt - cons)],
                           "extra": [(t, round(score[t], 3)) for t in sorted(cons - opt)]}
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
for R, v in out["pooled"].items():
    print(R, round(v["fitness"], 4), v["is_optimum"], v["out_vs_opt"], v["in_vs_opt"])
    print("   ", [(t, round(f, 3), s) for t, f, s in v["top16_freq"]])
for s, v in out["swaps_150"].items():
    print(s, round(v["fitness_gap"], 3), "sai", v["missing"], "entra", v["extra"])
