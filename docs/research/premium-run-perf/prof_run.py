"""Profiling reduzido de um run Premium (pesquisa #90).
Uso: prof_run.py <repo> <profile> <runs> <out_prefix> [--cprofile]
Politica igual a de producao (pop 300, 400 geracoes), mas run_count=<runs> e adaptive off."""
import cProfile, pstats, sys, time, tempfile, json
from pathlib import Path
repo = Path(sys.argv[1]); profile = sys.argv[2]; runs = int(sys.argv[3]); out = sys.argv[4]
use_prof = "--cprofile" in sys.argv
sys.path[:0] = [str(repo / "scripts")]
import benchmark_premium as bp
args = bp.build_parser().parse_args(["--registry", "/home/caio/Documents/projetos/TCC/snapshots", "--profiles", profile])
manifest, snapshot = bp.verify_snapshot(bp.load_declaration(args.declaration), args.registry)
policy = bp.build_policy(profile, manifest, 42)
for s in ("stocks", "fiis"):
    policy[s]["system_ga_config"].update({"run_count": runs, "adaptive_mode": False, "min_runs": 1})
print(json.dumps({s: policy[s]["system_ga_config"] for s in ("stocks","fiis")}))
selector_inputs = bp.selector_inputs_from_manifest(manifest, snapshot)
prof = cProfile.Profile() if use_prof else None
with tempfile.TemporaryDirectory() as tmp:
    t0 = time.perf_counter()
    if prof: prof.enable()
    res = bp.run_profile(profile, policy, manifest, selector_inputs, Path(tmp), seed=42)
    if prof: prof.disable()
    t1 = time.perf_counter()
summary = {"wall_total": t1 - t0, "stages": {k: v.get("wall_s") for k, v in res["stages"].items()}, "engines": res.get("engines")}
Path(out + ".json").write_text(json.dumps(summary, indent=1, default=str))
print("WALL", t1 - t0, {k: v.get("wall_s") for k, v in res["stages"].items()})
if prof:
    prof.dump_stats(out + ".prof")
    with open(out + ".txt", "w") as fh:
        st = pstats.Stats(prof, stream=fh).sort_stats("tottime"); st.print_stats(40)
        st.sort_stats("cumulative"); st.print_stats(50)
