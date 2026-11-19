"""Referências sem sorteio para o item (c) da pesquisa #100.

- top-N por score: os N maiores SCORE, pulando a segunda classe da mesma empresa como o consenso faz;
- ótimo exato do fitness do GA. O fitness é ``soma(SCORE) - λ·HHI·n`` com ``HHI = Σ (c_s/n)²``, isto é,
  ``Σ_s [soma dos c_s melhores do setor s - (λ/n)·c_s²]``: separável por setor. Para um número c_s de papéis
  num setor, os melhores são sempre os c_s de maior score, então uma programação dinâmica sobre (setor,
  quantos papéis já escolhidos) acha o máximo global. Vale porque o moderado não tem ``hhi_max`` nem
  ``max_per_segment`` (conferido em common.load_inputs) e porque não há SETOR nem SCORE vazio.

Uso: reference.py <out.json>
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import common as c


def base_company(ticker):
    for suffix in ("11", "3", "4", "5", "6"):
        if ticker.endswith(suffix):
            return ticker[: -len(suffix)]
    return ticker


def exact_optimum(df, n, lam, dedup=False):
    groups = {}
    for t, s, sc in zip(df["TICKER"], df["SETOR"], df["SCORE"]):
        groups.setdefault(s, []).append((float(sc), t))
    sectors = []
    for s, items in groups.items():
        items.sort(key=lambda x: (-x[0], x[1]))
        if dedup:
            seen, kept = set(), []
            for sc, t in items:
                if base_company(t) not in seen:
                    seen.add(base_company(t)); kept.append((sc, t))
            items = kept
        sectors.append((s, items))
    NEG = float("-inf")
    best = {0: (0.0, [])}  # escolhidos -> (valor, lista de (setor, c))
    for s, items in sectors:
        new = {}
        for k, (v, picks) in best.items():
            acc = 0.0
            for cnt in range(0, min(len(items), n - k) + 1):
                if cnt:
                    acc += items[cnt - 1][0]
                val = v + acc - lam / n * cnt * cnt
                if val > new.get(k + cnt, (NEG,))[0]:
                    new[k + cnt] = (val, picks + [(s, cnt)])
        best = new
    val, picks = best[n]
    tickers = []
    for s, cnt in picks:
        items = dict(sectors)[s]
        tickers += [t for _, t in items[:cnt]]
    return val, sorted(tickers), {s: cnt for s, cnt in picks if cnt}


def top_n(df, n, dedup=True):
    out, seen = [], set()
    for t in df.sort_values(["SCORE", "TICKER"], ascending=[False, True])["TICKER"]:
        if dedup and base_company(t) in seen:
            continue
        seen.add(base_company(t)); out.append(t)
        if len(out) == n:
            break
    return sorted(out)


if __name__ == "__main__":
    df, cfg, _, _ = c.load_inputs()
    g = c.ga_settings(cfg)
    n, lam = int(g["n_assets"]), float(g["lambda"])
    assert g.get("hhi_max") is None and cfg.get("max_per_segment") is None
    assert df["SETOR"].notna().all() and df["SCORE"].notna().all()
    res = {}
    for name, dedup in (("optimum", False), ("optimum_dedup", True)):
        val, tick, alloc = exact_optimum(df, n, lam, dedup)
        res[name] = {"fitness_dp": val, "fitness_ga": c.fitness_of(df, cfg, tick), "tickers": tick, "per_sector": alloc}
    for name, dedup in (("top_n", False), ("top_n_dedup", True)):
        tick = top_n(df, n, dedup)
        res[name] = {"fitness_ga": c.fitness_of(df, cfg, tick), "tickers": tick,
                     "score_sum": float(df[df.TICKER.isin(tick)].SCORE.sum())}
    res["universe"] = len(df)
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(k, v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items()})
