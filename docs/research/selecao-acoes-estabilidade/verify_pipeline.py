"""Confere que common.py reproduz o pipeline de produção: 3 runs do run_multi_execution_profile (seed 42)
contra 3 runs do GA montado aqui, com as mesmas seeds derivadas. Uso: verify_pipeline.py <out.json>"""
import contextlib, io, json, sys, tempfile, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import common as c
from core.optimizer import derive_run_seed
from pipelines.multi_run import run_multi_execution_profile

df, cfg, policy, manifest = c.load_inputs()
src = manifest.selector_source_for("stock").path
src = Path(src) if Path(src).is_absolute() else c.REGISTRY / src
t = time.perf_counter()
with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
    prod = run_multi_execution_profile(profile=None, n_runs=3, parallel=False, adaptive_mode=False,
                                       selection_config=cfg, workspace=Path(tmp), random_seed=42, input_path=src)
t_prod = time.perf_counter() - t
mine = []
for run_id in range(3):
    ga = c.make_ga(cfg, derive_run_seed(42, run_id, "stock"))
    p = ga.optimize(df)
    mine.append({"tickers": sorted(p["TICKER"]), "fitness": float(p.attrs["fitness"])})
prod_runs = [{"tickers": r["tickers"], "fitness": r["fitness"]} for r in prod["all_runs"]]
out = {"prod_keys": sorted(prod.keys()), "prod_runs": prod_runs, "mine": mine, "equal": prod_runs == mine, "t_prod_s": t_prod}
Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=str))
print("equal:", out["equal"], "keys:", out["prod_keys"])
