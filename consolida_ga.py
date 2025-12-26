import profiles as pf, ga, pandas as pd, json
from pathlib import Path

DATA_DIR = Path("../data/processed")
OUT_DIR  = Path("outputs"); OUT_DIR.mkdir(exist_ok=True)

for perfil in ("conservador", "moderado", "arrojado"):
    df    = pf.load_profile_data(perfil, DATA_DIR)
    rank  = pf.build_scores(df, perfil)
    cart  = ga.run_ga(rank, perfil)

    # salva JSON
    file_out = OUT_DIR / f"carteira_{perfil}_ga.json"
    cart.to_json(file_out, orient="records", indent=2, force_ascii=False)
    print(perfil, "→ salvo em", file_out, " | HHI =", round(cart.attrs['hhi'], 3))
