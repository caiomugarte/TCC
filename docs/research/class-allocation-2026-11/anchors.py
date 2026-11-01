"""Research: allocation by profile score x training window x anchor set.
FIIs use IFIX as a 10-year proxy (research only; production uses the FII sleeve)."""
import sys, copy, json, itertools, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "api"), str(ROOT / "py")]
import allocation_config as ac
from allocation_data import read_levels_csv, _single_series, multiply_levels, levels_to_returns, build_equal_weight_sleeve
from pipelines.asset_allocation import optimize_window, _latest_window, _sorted_rows
from allocation_config import ALLOCATION_CONFIG, ASSET_CLASSES, normalize_allocation_config
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
print(f"rows {ROWS[0].date} -> {ROWS[-1].date} ({len(ROWS)})", file=sys.stderr)

ORIG = copy.deepcopy(ac.ALLOCATION_PROFILE_ANCHORS)
def set_anchors(caps):
    for name, (vol, dd) in caps.items():
        ac.ALLOCATION_PROFILE_ANCHORS[name]["volatility_cap"] = vol
        ac.ALLOCATION_PROFILE_ANCHORS[name]["drawdown_cap"] = dd

def allocate(score, years):
    sec = po._json_value(premium_policy._allocation_policy(score, {"nenhuma"}))
    cfg, cons = po._allocation_config(sec)
    payload = {**cfg, "class_constraints": cons}
    settings = normalize_allocation_config(payload, base={**ALLOCATION_CONFIG["caio"], "minimum_class_weight": 0.0})
    pen = float(cfg["hhi_penalty"])
    settings["hhi_penalties"] = tuple(sorted(set(settings.get("hhi_penalties", ())) | {pen}))
    win = optimize_window(_latest_window(ROWS, years), ASSET_CLASSES, settings)
    c = win.penalty_winners.get(pen)
    if c is None:
        return None, sec
    return dict(zip(ASSET_CLASSES, c.weights)), sec

SETS = json.loads(sys.argv[1])
SCORES = json.loads(sys.argv[2])
YEARS = json.loads(sys.argv[3])
short = {"brazilian_stocks": "BR", "fiis": "FII", "international_equity": "INT", "fixed_income": "RF", "crypto": "BTC"}
for set_name, caps in SETS.items():
    set_anchors({k: tuple(v) for k, v in caps.items()})
    for years in YEARS:
        for score in SCORES:
            t = time.time()
            w, sec = allocate(score, years)
            txt = "inviável" if w is None else " ".join(f"{short[k]}={v:.0%}" for k, v in w.items())
            print(f"{set_name:8} janela={years:>2}a score={score:.2f} vol≤{sec['volatility_cap']:.1%} dd≤{sec['drawdown_cap']:.1%} -> {txt} ({time.time()-t:.0f}s)", flush=True)
