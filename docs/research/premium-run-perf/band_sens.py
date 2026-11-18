"""Sensibilidade do consenso: seed contra passo de score (pesquisa #90). GA vetorizado, config de producao.
Uso: band_sens.py <repo> <generic_profile> <score> <seed> <out.json>"""
import sys, json, tempfile, runpy
from pathlib import Path
repo = Path(sys.argv[1]); prof, score, seed, out = sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
sys.argv = [sys.argv[0], str(repo), prof, "150", "vec", out + ".e2e.json", "adaptive"]
sys.path[:0] = [str(repo / "scripts"), str(repo / "api"), str(repo / "py")]
from app.services import profile as prof_mod
prof_mod.ANCHOR_SCORES[prof] = score
import benchmark_premium as bp
_orig_run_profile = bp.run_profile
captured = {}
def run_profile(*a, **k):
    k["seed"] = seed
    r = _orig_run_profile(*a, **k); captured["r"] = r; return r
bp.run_profile = run_profile
runpy.run_path(str(Path(__file__).with_name("e2e.py")), run_name="__main__")
r = captured["r"]
Path(out).write_text(json.dumps({"profile": prof, "score": score, "seed": seed, "output": r.get("output"), "engines": r.get("engines")}, indent=1, default=str))
