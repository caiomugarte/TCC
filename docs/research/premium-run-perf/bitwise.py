"""Prototipo (pesquisa #90): fitness vetorizado do GA contra o original.
Captura os df_ranked reais (acoes e FIIs) de um run smoke e roda o GA de producao
(pop 300, 400 ger.) com o fitness original e com o vetorizado, mesmas seeds.
Uso: vec_proto.py <repo> <profile> <n_seeds>"""
import sys, time, tempfile, json, pickle
from pathlib import Path
import numpy as np
repo = Path(sys.argv[1]); profile = sys.argv[2]; n_seeds = int(sys.argv[3])
sys.path[:0] = [str(repo / "scripts")]
import benchmark_premium as bp
args = bp.build_parser().parse_args(["--registry", "/home/caio/Documents/projetos/TCC/snapshots", "--profiles", profile])
manifest, snapshot = bp.verify_snapshot(bp.load_declaration(args.declaration), args.registry)
from core import optimizer as opt
captured = []
orig_optimize = opt.GeneticAlgorithm.optimize
def spy(self, df):
    captured.append((df.copy(), dict(n_assets=self.n_assets, lambda_hhi=self.lambda_hhi, hhi_max=self.hhi_max, max_per_segment=self.max_per_segment)))
    return orig_optimize(self, df)
opt.GeneticAlgorithm.optimize = spy
policy = bp.build_policy(profile, manifest, 42, smoke=True)
with tempfile.TemporaryDirectory() as tmp:
    bp.run_profile(profile, policy, manifest, bp.selector_inputs_from_manifest(manifest, snapshot), Path(tmp), seed=42)
opt.GeneticAlgorithm.optimize = orig_optimize

class VecGA(opt.GeneticAlgorithm):
    """Mesmo GA; so troca a avaliacao da populacao por numpy em lote."""
    def _prep(self, df):
        self._s = df["SCORE"].to_numpy(dtype=float)
        codes, uniq = (df["SETOR"].factorize() if "SETOR" in df.columns else (None, None))
        self._c = codes; self._k = len(uniq)
    def fitness_batch(self, population):
        P = np.asarray(population, dtype=bool)
        out = np.full(len(P), -np.inf)
        ok = P.sum(1) == self.n_assets
        idx = np.nonzero(P[ok])[1].reshape(-1, self.n_assets)  # indices em ordem crescente, como iloc
        total = self._s[idx].sum(axis=1)
        cnt = np.zeros((len(idx), self._k), dtype=np.int64)
        np.add.at(cnt, (np.repeat(np.arange(len(idx)), self.n_assets), self._c[idx].ravel()), 1)
        n = self.n_assets
        # HHI como pandas: value_counts(normalize) ordena por contagem desc; somamos so os nao-nulos nessa ordem
        srt = -np.sort(-cnt, axis=1)
        hhi = np.empty(len(idx))
        for r in range(len(idx)):
            row = srt[r][srt[r] > 0]
            hhi[r] = float(((row / n) ** 2).sum())
        fit = total - self.lambda_hhi * hhi * n
        bad = np.zeros(len(idx), dtype=bool)
        if self.max_per_segment is not None:
            bad |= cnt.max(1) > self.max_per_segment
        if self.hhi_max is not None:
            bad |= hhi > self.hhi_max + 1e-12
        fit[bad] = -np.inf
        out[ok] = fit
        return out

def run(cls, df, params, seed):
    ga = cls(generations=400, pop_size=300, random_seed=seed, **params)
    if cls is VecGA:
        ga._prep(df)
        ga.fitness = None
        # injeta: optimize chama self.fitness por cromossomo; trocamos o loop via monkeypatch local
    t = time.perf_counter()
    if cls is VecGA:
        orig_array = np.array
        res = _optimize_vec(ga, df)
    else:
        res = ga.optimize(df)
    return res, time.perf_counter() - t

import inspect, textwrap
src = textwrap.dedent(inspect.getsource(opt.GeneticAlgorithm.optimize))
import re
src = re.sub(r"scores = np\.array\(\[\s*self\.fitness\(df_ranked, chrom\)\s*for chrom in population\s*\]\)", "scores = self.fitness_batch(population)", src)
assert "fitness_batch" in src
ns = {**opt.__dict__}
exec(src.replace("def optimize(self", "def _optimize_vec(self"), ns)
_optimize_vec = ns["_optimize_vec"]


rng = np.random.default_rng(0)
for label, (df, params) in zip(("stocks", "fiis"), captured):
    for variant in ("base", "cap", "hhimax"):
        pr = dict(params)
        if variant == "cap": pr["max_per_segment"] = 2
        if variant == "hhimax": pr["hhi_max"] = 0.25
        ga = VecGA(generations=1, pop_size=2, random_seed=1, **pr); ga._prep(df)
        m = len(df); n = pr["n_assets"]; pop = []
        for _ in range(20000):
            c = np.zeros(m, dtype=int)
            k = n if rng.random() < 0.9 else int(rng.integers(1, m))
            c[rng.choice(m, size=k, replace=False)] = 1
            pop.append(c)
        a = np.array([ga.fitness(df, c) for c in pop]); b = ga.fitness_batch(pop)
        same = np.array_equal(a, b) and np.array_equal(np.isinf(a), np.isinf(b))
        print("BITWISE", label, variant, "n=20000", "infeasible", int(np.isinf(a).sum()), "identical", same, "maxdiff", float(np.nanmax(np.abs(np.where(np.isfinite(a), a - b, 0)))), flush=True)
