"""Score the Status Invest and the CVM/B3 stock CSVs with the run's stock policy and explain each swap (#80).

Usage: python scripts/compare_stock_selector_sources.py <repo> <si_csv> <cvm_csv> <policy_json> <old_tickers,> <new_tickers,> <out_json>

Variants (each scored with the same policy, no GA):
  si        the Status Invest CSV
  cvm       the CVM/B3 CSV
  cvm_si_rp the CVM/B3 CSV with ROIC and PEG RATIO taken from Status Invest (formula test)
"""
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

repo = Path(sys.argv[1])
sys.path[:0] = [str(repo / "py")]
from core.preprocessing import apply_robustness_filter, load_raw_data, preprocess_profile  # noqa: E402
from core.scoring import build_scores  # noqa: E402
from pipelines.single_run import _explicit_stock_config  # noqa: E402

si_csv, cvm_csv, policy_path = sys.argv[2:5]
old = sys.argv[5].split(",")
new = sys.argv[6].split(",")
policy = json.loads(Path(policy_path).read_text())
config = _explicit_stock_config(policy["stocks"])


def score(raw: pd.DataFrame) -> tuple[pd.DataFrame, set, set]:
    with redirect_stdout(io.StringIO()):
        clean = preprocess_profile(raw.copy(), config=config)
        eligible = set(clean["TICKER"])
        robust = apply_robustness_filter(clean)
        ranked = build_scores(robust, factor_weights=config["factor_weights"]).reset_index(drop=True)
    ranked["RANK"] = range(1, len(ranked) + 1)
    return ranked, eligible, set(robust["TICKER"])


si_raw = load_raw_data(Path(si_csv))
cvm_raw = load_raw_data(Path(cvm_csv))
hybrid = cvm_raw.copy()
si_idx = si_raw.set_index("TICKER")
for column in ("ROIC", "PEG RATIO"):
    hybrid[column] = [si_idx[column].get(t, v) if t in si_idx.index else v for t, v in zip(hybrid["TICKER"], hybrid[column])]

variants = {"si": si_raw, "cvm": cvm_raw, "cvm_si_rp": hybrid}
scored = {name: score(raw) for name, raw in variants.items()}
groups = [c for c in scored["cvm"][0].columns if c.startswith("AVG_")]
report = {"universe": {name: len(raw) for name, raw in variants.items()},
          "eligible": {name: len(s[1]) for name, s in scored.items()},
          "robust": {name: len(s[2]) for name, s in scored.items()},
          "ranked": {name: len(s[0]) for name, s in scored.items()},
          "tickers": {}}
cols = ["P/L", "P/VP", "ROE", "ROA", "ROIC", "PEG RATIO", "DY", "MARG. LIQUIDA", "EV/EBIT", "CAGR LUCROS 5 ANOS",
        "CAGR RECEITAS 5 ANOS", "LIQUIDEZ MEDIA DIARIA", "VALOR DE MERCADO", "SETOR"]
for ticker in sorted(set(old) ^ set(new) | set(old) & set(new)):
    item = {"in_old": ticker in old, "in_new": ticker in new}
    for name, raw in variants.items():
        ranked, eligible, robust = scored[name]
        row = ranked[ranked["TICKER"] == ticker]
        entry = {"in_csv": bool((raw["TICKER"] == ticker).any()), "eligible": ticker in eligible, "robust": ticker in robust}
        if not row.empty:
            r = row.iloc[0]
            entry |= {"rank": int(r["RANK"]), "score": round(float(r["SCORE"]), 4)}
            entry |= {g: (None if pd.isna(r[g]) else round(float(r[g]), 3)) for g in groups if g in r}
        if name != "cvm_si_rp":
            src = raw[raw["TICKER"] == ticker]
            if not src.empty:
                entry["values"] = {c: (None if c not in src or pd.isna(src.iloc[0][c]) else src.iloc[0][c]) for c in cols + (["SUBSETOR"] if "SUBSETOR" in src else [])}
        item[name] = entry
    report["tickers"][ticker] = item
Path(sys.argv[7]).write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str))
print(json.dumps({k: v for k, v in report.items() if k != "tickers"}))
