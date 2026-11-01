"""Compare allocation methods on IBOV/IFIX data: single window (today), rolling-window resampling, risk parity."""
import sys, json, math
from pathlib import Path
from datetime import date
from multiprocessing import Pool
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "api"), str(ROOT / "py")]
from allocation_data import read_levels_csv, _single_series, multiply_levels, levels_to_returns
from pipelines.asset_allocation import optimize_window, _sorted_rows, _add_years
from allocation_config import ALLOCATION_CONFIG, ASSET_CLASSES, normalize_allocation_config
from core.allocation import simulate_portfolio
import app.adapters.premium_optimization as po
from app.services import premium_policy

D = ROOT / "snapshots/data/snap-2026-10-29"
lv = lambda f: read_levels_csv(D / f)
ptax = _single_series(lv("ptax.csv"))
levels = {
    "brazilian_stocks": _single_series(lv("ibovespa.csv")),
    "fiis": _single_series(lv("ifix.csv")),
    "international_equity": multiply_levels(_single_series(lv("sp500_total_return_usd.csv")), ptax),
    "fixed_income": _single_series(lv("di.csv")),
    "crypto": multiply_levels(_single_series(lv("btc_usd.csv")), ptax),
}
ROWS = _sorted_rows(levels_to_returns(levels, ASSET_CLASSES))
LAST = ROWS[-1].date
SHORT = {"brazilian_stocks": "BR", "fiis": "FII", "international_equity": "INT", "fixed_income": "RF", "crypto": "BTC"}
RISKY = ("brazilian_stocks", "fiis", "international_equity", "crypto")

def window(end, years):
    start = _add_years(end, -years)
    return tuple(r for r in ROWS if start <= r.date <= end)

def policy(score):
    return po._json_value(premium_policy._allocation_policy(score, {"nenhuma"}))

def grid(args):
    score, end, years = args
    sec = policy(score); cfg, cons = po._allocation_config(sec)
    settings = normalize_allocation_config({**cfg, "class_constraints": cons}, base={**ALLOCATION_CONFIG["caio"], "minimum_class_weight": 0.0})
    pen = float(cfg["hhi_penalty"])
    settings["hhi_penalties"] = tuple(sorted(set(settings.get("hhi_penalties", ())) | {pen}))
    c = optimize_window(window(end, years), ASSET_CLASSES, settings).penalty_winners.get(pen)
    return (score, end.isoformat(), years), (None if c is None else list(c.weights))

def metrics(rows, w):
    path = simulate_portfolio(rows, dict(zip(ASSET_CLASSES, w)), ASSET_CLASSES, annual_rebalance=True)
    v = np.array(path.values); r = v[1:] / v[:-1] - 1
    yrs = (path.dates[-1] - path.dates[0]).days / 365.25
    return (v[-1] / v[0]) ** (1 / yrs) - 1, r.std() * math.sqrt(252), (v / np.maximum.accumulate(v) - 1).min()

def risk_parity(score, end, cov_years=10):
    """Equal risk contribution across risky classes, scaled with fixed income to the profile caps."""
    sec = policy(score)
    rows = window(end, cov_years)
    R = np.array([[r.returns[k] for k in RISKY] for r in rows[1:]])
    cov = np.cov(R.T) * 252
    w = np.ones(len(RISKY)) / len(RISKY)
    for _ in range(5000):  # multiplicative ERC iteration
        rc = w * (cov @ w); w = w * (rc.mean() / rc) ** 0.5; w /= w.sum()
    best = None
    for x in np.arange(0, 1.0001, 0.01):
        full = [0.0] * 5
        for k, wk in zip(RISKY, w): full[ASSET_CLASSES.index(k)] = x * wk
        full[ASSET_CLASSES.index("fixed_income")] = 1 - x
        _, vol, dd = metrics(rows, full)
        if vol <= sec["volatility_cap"] and -dd <= sec["drawdown_cap"]:
            best = full
    return best

def fmt(w): return "inviável" if w is None else " ".join(f"{SHORT[k]}={x:.0%}" for k, x in zip(ASSET_CLASSES, w))
def fmtm(m): return f"ret {m[0]:5.1%} vol {m[1]:5.1%} dd {m[2]:6.1%}"

if __name__ == "__main__":
    SCORES = [0.0, 0.25, 0.5, 0.58, 0.75, 1.0]
    ends = []
    e = LAST
    while _add_years(e, -3) >= ROWS[0].date:
        ends.append(e); m = e.month - 6
        e = date(e.year + (m <= 0) * -1, m + 12 if m <= 0 else m, min(e.day, 28))
    # walk-forward rebalance dates for score 0.58 (decide with past data only, hold one year)
    wf_dates = [date(y, 11, 1) for y in range(2019, 2026)]
    wf_ends = set()
    for t in wf_dates:
        x = t
        while _add_years(x, -3) >= ROWS[0].date:
            wf_ends.add(x); m = x.month - 6
            x = date(x.year + (m <= 0) * -1, m + 12 if m <= 0 else m, min(x.day, 28))
    jobs = {(s, e2, 3) for s in SCORES for e2 in ends} | {(0.58, x, 3) for x in wf_ends}
    print(f"{len(ends)} janelas por perfil; {len(jobs)} buscas", file=sys.stderr, flush=True)
    import pickle
    cache = Path(__file__).resolve().parent / "grid_cache.pkl"
    G = pickle.loads(cache.read_bytes()) if cache.exists() else {}
    todo = sorted(j for j in jobs if (j[0], j[1].isoformat(), j[2]) not in G)
    print(f"{len(todo)} buscas novas", file=sys.stderr, flush=True)
    with Pool(8) as pool:
        G.update(dict(pool.map(grid, todo, chunksize=1)))
    cache.write_bytes(pickle.dumps(G))
    def resampled(score, upto_ends):
        ws = [G[(score, x.isoformat(), 3)] for x in upto_ends if G[(score, x.isoformat(), 3)] is not None]
        return list(np.mean(ws, axis=0)) if ws else None
    out = {"carteiras": {}, "walk_forward": {}}
    full10 = window(LAST, 10); last3 = window(LAST, 3)
    print("\n== CARTEIRAS POR PERFIL (decididas com dados até 2026-10) ==")
    for s in SCORES:
        single = G[(s, LAST.isoformat(), 3)]; rs = resampled(s, ends); rp = risk_parity(s, LAST)
        print(f"\nscore {s:.2f}  (vol≤{policy(s)['volatility_cap']:.1%} dd≤{policy(s)['drawdown_cap']:.1%})")
        for name, w in (("hoje (1 janela 3a)", single), ("reamostragem", rs), ("paridade de risco", rp)):
            m10 = metrics(full10, w) if w else None; m3 = metrics(last3, w) if w else None
            print(f"  {name:20} {fmt(w):42} | 10a: {fmtm(m10) if m10 else '-'} | 3a: {fmtm(m3) if m3 else '-'}")
            out["carteiras"].setdefault(str(s), {})[name] = w
    print("\n== RISCO-PARIDADE: sensibilidade à janela da covariância (score 0.58) ==")
    for cy in (3, 5, 10):
        print(f"  cov {cy:>2}a: {fmt(risk_parity(0.58, LAST, cy))}")
    print("\n== TESTE FORA DA AMOSTRA, score 0.58: decide em t só com o passado, segura 1 ano ==")
    def ends_upto(t):
        xs = []; x = t
        while _add_years(x, -3) >= ROWS[0].date:
            xs.append(x); m = x.month - 6
            x = date(x.year + (m <= 0) * -1, m + 12 if m <= 0 else m, min(x.day, 28))
        return xs
    curves = {k: [1.0] for k in ("hoje (1 janela 3a)", "reamostragem", "paridade de risco")}
    daily = {k: [] for k in curves}
    for t in wf_dates:
        hold = tuple(r for r in ROWS if t <= r.date <= _add_years(t, 1))
        ws = {"hoje (1 janela 3a)": G[(0.58, t.isoformat(), 3)], "reamostragem": resampled(0.58, ends_upto(t)), "paridade de risco": risk_parity(0.58, t, min(10, (t - ROWS[0].date).days // 365))}
        line = []
        for k, w in ws.items():
            path = simulate_portfolio(hold, dict(zip(ASSET_CLASSES, w)), ASSET_CLASSES, annual_rebalance=False)
            v = np.array(path.values); daily[k].extend(list(v[1:] / v[:-1] - 1))
            line.append(f"{k.split()[0]}: {fmt(w)} -> {v[-1]/v[0]-1:+.1%}")
        print(f"  {t}  " + " | ".join(line))
    print("\n  Resultado acumulado 2019-10 → 2026-10:")
    for k, r in daily.items():
        r = np.array(r); v = np.cumprod(1 + r); yrs = 7
        print(f"  {k:20} ret {v[-1]**(1/yrs)-1:5.1%} a.a.  vol {r.std()*math.sqrt(252):5.1%}  dd {(v/np.maximum.accumulate(v)-1).min():6.1%}")
    json.dump(out, open(Path(__file__).resolve().parent / "compare.json", "w"), default=str, indent=1)
