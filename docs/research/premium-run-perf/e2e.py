"""Pipeline Premium inteiro com o GA corrigido por monkeypatch (pesquisa #90).
Uso: e2e.py <repo> <profile> <runs> <variant: orig|noattrs|vec> <out.json> [adaptive]"""
import sys, time, tempfile, json, re, inspect, textwrap
from pathlib import Path
import numpy as np
repo = Path(sys.argv[1]); profile, runs, variant, out = sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
adaptive = len(sys.argv) > 6
sys.path[:0] = [str(repo / "scripts")]
import benchmark_premium as bp
from core import optimizer as opt
orig = opt.GeneticAlgorithm.optimize
def fitness_batch(self, population):
    P = np.asarray(population, dtype=bool); out = np.full(len(P), -np.inf)
    ok = P.sum(1) == self.n_assets
    idx = np.nonzero(P[ok])[1].reshape(-1, self.n_assets)
    total = self._s[idx].sum(axis=1)
    cnt = np.zeros((len(idx), self._k), dtype=np.int64)
    np.add.at(cnt, (np.repeat(np.arange(len(idx)), self.n_assets), self._c[idx].ravel()), 1)
    n = self.n_assets; srt = -np.sort(-cnt, axis=1); hhi = np.empty(len(idx))
    for r in range(len(idx)):
        row = srt[r][srt[r] > 0]; hhi[r] = float(((row / n) ** 2).sum())
    fit = total - self.lambda_hhi * hhi * n; bad = np.zeros(len(idx), dtype=bool)
    if self.max_per_segment is not None: bad |= cnt.max(1) > self.max_per_segment
    if self.hhi_max is not None: bad |= hhi > self.hhi_max + 1e-12
    fit[bad] = -np.inf; out[ok] = fit; return out
src = textwrap.dedent(inspect.getsource(orig))
src = re.sub(r"scores = np\.array\(\[\s*self\.fitness\(df_ranked, chrom\)\s*for chrom in population\s*\]\)", "scores = self.fitness_batch(population)", src)
ns = {**opt.__dict__}; exec(src.replace("def optimize(self", "def _vec(self"), ns)
ga_time = [0.0]
def patched(self, df):
    t = time.perf_counter()
    if variant == "vec":
        self._s = df["SCORE"].to_numpy(dtype=float); c, u = df["SETOR"].factorize(); self._c, self._k = c, len(u)
        self.fitness_batch = fitness_batch.__get__(self); r = ns["_vec"](self, df)
    elif variant == "noattrs":
        d = df.copy(); d.attrs = {}; r = orig(self, d)
    else:
        r = orig(self, df)
    ga_time[0] += time.perf_counter() - t; return r
opt.GeneticAlgorithm.optimize = patched
args = bp.build_parser().parse_args(["--registry", "/home/caio/Documents/projetos/TCC/snapshots", "--profiles", profile])
manifest, snapshot = bp.verify_snapshot(bp.load_declaration(args.declaration), args.registry)
policy = bp.build_policy(profile, manifest, 42)
if not adaptive:
    for s in ("stocks", "fiis"):
        policy[s]["system_ga_config"].update({"run_count": runs, "adaptive_mode": False, "min_runs": 1})
with tempfile.TemporaryDirectory() as tmp:
    t0 = time.perf_counter()
    res = bp.run_profile(profile, policy, manifest, bp.selector_inputs_from_manifest(manifest, snapshot), Path(tmp), seed=42)
    wall = time.perf_counter() - t0
summary = {"variant": variant, "profile": profile, "runs": runs, "adaptive": adaptive, "wall": wall, "ga_time": ga_time[0],
           "stages": {k: v.get("wall_s") for k, v in res["stages"].items()}, "engines": res.get("engines"),
           "stocks": [c.get("ticker") for c in res.get("result", {}).get("stocks", [])] if isinstance(res.get("result"), dict) else None}
Path(out).write_text(json.dumps(summary, indent=1, default=str))
print("E2E", variant, profile, runs, round(wall, 1), "ga", round(ga_time[0], 1), {k: round(v, 2) for k, v in summary["stages"].items()})
