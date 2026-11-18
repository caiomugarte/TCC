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

report = {}
for label, (df, params) in zip(("stocks", "fiis"), captured):
    rows = []
    for k in range(n_seeds):
        seed = opt.derive_run_seed(42, k, "stock" if label == "stocks" else "fii")
        a, ta = run(opt.GeneticAlgorithm, df, params, seed)
        b, tb = run(VecGA, df, params, seed)
        df0 = df.copy(); df0.attrs = {}
        c, tc = run(opt.GeneticAlgorithm, df0, params, seed)
        same_c = sorted(a["TICKER"]) == sorted(c["TICKER"]) and a.attrs["fitness"] == c.attrs["fitness"] and a.attrs["generations_run"] == c.attrs["generations_run"]
        same = (sorted(a["TICKER"]) == sorted(b["TICKER"]) and a.attrs["fitness"] == b.attrs["fitness"]
                and a.attrs["generations_run"] == b.attrs["generations_run"])
        rows.append(dict(seed=seed, t_orig=ta, t_vec=tb, gens=a.attrs["generations_run"], identical=same, t_noattrs=tc, identical_noattrs=same_c))
        print(label, rows[-1], flush=True)
    import pickle as _p
    report_attrs = {k: len(_p.dumps(v)) for k, v in df.attrs.items()}
    report[label] = dict(attrs_bytes=report_attrs, speedup_noattrs=sum(r["t_orig"] for r in rows) / sum(r["t_noattrs"] for r in rows), m=len(df), n_sectors=int(df["SETOR"].nunique()), params=params, runs=rows,
                         speedup=sum(r["t_orig"] for r in rows) / sum(r["t_vec"] for r in rows))
    print(label, "m", len(df), "speedup", report[label]["speedup"], flush=True)
Path(sys.argv[4] if len(sys.argv) > 4 else "vec_report.json").write_text(json.dumps(report, indent=1, default=str))
