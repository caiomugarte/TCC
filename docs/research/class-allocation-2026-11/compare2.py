"""Round 2: today, rolling-window resampling, 1/N+CDI, block bootstrap (grid), skfolio (MaxDD / CDaR / CDaR+bootstrap)."""
import sys, math, pickle, warnings
from pathlib import Path
from datetime import date, timedelta
from multiprocessing import Pool
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare as C
from compare import ROWS, LAST, ASSET_CLASSES, RISKY, policy, metrics, fmt, fmtm, window, _add_years
from core.allocation import DailyReturn, simulate_portfolio
from pipelines.asset_allocation import optimize_window
from allocation_config import ALLOCATION_CONFIG, normalize_allocation_config
import app.adapters.premium_optimization as po

SCORES = [0.0, 0.25, 0.5, 0.58, 0.75, 1.0]
WF = [date(y, 11, 1) for y in range(2019, 2026)]
N_GRID, N_SK, BLOCK, LEN = 40, 300, 42, 756   # samples, mean block ~2 months, 3y samples
CACHE = Path(__file__).resolve().parent / "grid_cache.pkl"
G = pickle.loads(CACHE.read_bytes())

def past(t, years=10):
    start = max(_add_years(t, -years), ROWS[0].date)
    return tuple(r for r in ROWS if start <= r.date <= t)

def matrix(rows):
    return np.array([[r.returns[k] for k in ASSET_CLASSES] for r in rows[1:]])

def bootstrap_idx(n, seed):
    """Stationary bootstrap (Politis-Romano): random starts, geometric block lengths, circular."""
    rng = np.random.default_rng(seed); out = []
    while len(out) < LEN:
        s = rng.integers(n); L = rng.geometric(1 / BLOCK)
        out.extend((s + np.arange(L)) % n)
    return np.array(out[:LEN])

def as_rows(R):
    days = np.busday_offset("2000-01-03", np.arange(len(R) + 1), roll="forward").astype(date)
    return tuple([DailyReturn(days[0], {k: 0.0 for k in ASSET_CLASSES})] + [DailyReturn(days[i + 1], dict(zip(ASSET_CLASSES, R[i]))) for i in range(len(R))])

def grid_rows(score, rows):
    sec = policy(score); cfg, cons = po._allocation_config(sec)
    st = normalize_allocation_config({**cfg, "class_constraints": cons}, base={**ALLOCATION_CONFIG["caio"], "minimum_class_weight": 0.0})
    pen = float(cfg["hhi_penalty"]); st["hhi_penalties"] = tuple(sorted(set(st.get("hhi_penalties", ())) | {pen}))
    c = optimize_window(rows, ASSET_CLASSES, st).penalty_winners.get(pen)
    return None if c is None else np.array(c.weights)

def grid_boot_job(args):
    score, t, i = args
    R = matrix(past(t)); R = R[bootstrap_idx(len(R), hash((t.toordinal(), i)) % 2**32)]
    return args, grid_rows(score, as_rows(R))

def sk_solve(R, score, kind):
    from skfolio.optimization import MeanRisk, ObjectiveFunction
    sec = policy(score)
    kw = dict(objective_function=ObjectiveFunction.MAXIMIZE_RETURN, min_weights=0.0, max_weights=1.0,
              max_standard_deviation=sec["volatility_cap"] / math.sqrt(252))
    if kind == "maxdd": kw["max_max_drawdown"] = sec["drawdown_cap"]
    else: kw.update(max_cdar=sec["drawdown_cap"], cdar_beta=0.95)
    try:
        m = MeanRisk(**kw).fit(R); w = np.clip(np.asarray(m.weights_), 0, None)
        return w / w.sum() if w.sum() > 0 else None
    except Exception:
        return None

def sk_boot(score, t):
    R = matrix(past(t)); ws = []
    for i in range(N_SK):
        w = sk_solve(R[bootstrap_idx(len(R), 10_000 + i)], score, "cdar")
        if w is not None: ws.append(w)
    return (np.mean(ws, axis=0), len(ws)) if ws else (None, 0)

def scale_to_caps(risky_w, score, t):
    rows = past(t); sec = policy(score); best = None
    for x in np.arange(0, 1.0001, 0.01):
        full = np.zeros(5)
        for k, wk in zip(RISKY, risky_w): full[ASSET_CLASSES.index(k)] = x * wk
        full[ASSET_CLASSES.index("fixed_income")] = 1 - x
        _, vol, dd = metrics(rows, full)
        if vol <= sec["volatility_cap"] and -dd <= sec["drawdown_cap"]: best = full
    return best

def rolling_ends(t):
    xs = []; x = t
    while _add_years(x, -3) >= ROWS[0].date:
        xs.append(x); m = x.month - 6
        x = date(x.year - (m <= 0), m + 12 if m <= 0 else m, min(x.day, 28))
    return xs

def weights(method, score, t):
    if method == "hoje": return np.array(G[(score, t.isoformat(), 3)])
    if method == "janelas":
        ws = [G[(score, x.isoformat(), 3)] for x in rolling_ends(t) if G.get((score, x.isoformat(), 3)) is not None]
        return np.mean(ws, axis=0)
    if method == "1/N+RF": return scale_to_caps(np.ones(4) / 4, score, t)
    if method == "boot+grid":
        ws = [BG[(score, t, i)] for i in range(N_GRID) if BG.get((score, t, i)) is not None]
        return np.mean(ws, axis=0) if ws else None
    if method == "sk MaxDD": return sk_solve(matrix(window(t, 3)), score, "maxdd")
    if method == "sk CDaR": return sk_solve(matrix(window(t, 3)), score, "cdar")
    if method == "sk CDaR+boot": return sk_boot(score, t)[0]

METHODS = ["hoje", "janelas", "1/N+RF", "boot+grid", "sk MaxDD", "sk CDaR", "sk CDaR+boot"]

if __name__ == "__main__":
    need = {(s, x, 3) for s in SCORES for x in rolling_ends(LAST)} | {(0.58, x, 3) for t in WF for x in rolling_ends(t)}
    todo = sorted(j for j in need if (j[0], j[1].isoformat(), 3) not in G)
    jobs = [(s, LAST, i) for s in SCORES for i in range(N_GRID)] + [(0.58, t, i) for t in WF for i in range(N_GRID)]
    print(f"grid janelas novas: {len(todo)}; grid bootstrap: {len(jobs)}", file=sys.stderr, flush=True)
    with Pool(10) as pool:
        G.update(dict(pool.map(C.grid, todo, chunksize=1)))
        BG = dict(pool.map(grid_boot_job, jobs, chunksize=2))
    CACHE.write_bytes(pickle.dumps(G))
    full10, last3 = window(LAST, 10), window(LAST, 3)
    print("== CARTEIRAS POR PERFIL (dados até 2026-10) | métricas no histórico de 10a ==")
    for s in SCORES:
        sec = policy(s); print(f"\nscore {s:.2f} (vol≤{sec['volatility_cap']:.1%} dd≤{sec['drawdown_cap']:.1%})")
        for mth in METHODS:
            w = weights(mth, s, LAST)
            print(f"  {mth:13} {fmt(None if w is None else list(w)):42} | 10a: {fmtm(metrics(full10, w)) if w is not None else '-'}", flush=True)
    print("\n== FORA DA AMOSTRA, score 0.58 (decide em 1/nov com o passado, segura 1 ano) ==")
    for mth in METHODS:
        daily, prev, turn, ws_line = [], None, [], []
        for t in WF:
            w = weights(mth, 0.58, t)
            if w is None: w = prev if prev is not None else np.eye(5)[ASSET_CLASSES.index("fixed_income")]
            hold = tuple(r for r in ROWS if t <= r.date <= _add_years(t, 1))
            v = np.array(simulate_portfolio(hold, dict(zip(ASSET_CLASSES, w)), ASSET_CLASSES, annual_rebalance=False).values)
            daily.extend(v[1:] / v[:-1] - 1)
            if prev is not None: turn.append(np.abs(w - prev).sum() / 2)
            prev = w; ws_line.append(f"RF{w[3]:.0%}")
        r = np.array(daily); v = np.cumprod(1 + r)
        print(f"  {mth:13} ret {v[-1]**(1/7)-1:5.1%} a.a. vol {r.std()*math.sqrt(252):5.1%} dd {(v/np.maximum.accumulate(v)-1).min():6.1%} giro médio/ano {np.mean(turn):5.1%}  [{' '.join(ws_line)}]", flush=True)
