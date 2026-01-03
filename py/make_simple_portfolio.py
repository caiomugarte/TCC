import profiles as pf
import pandas as pd
from pathlib import Path

TOP_N = {"conservador": 10, "moderado": 12, "arrojado": 15}
DATA_DIR = Path("../data/processed")
OUT_DIR  = Path("outputs"); OUT_DIR.mkdir(exist_ok=True)

for perfil, n in TOP_N.items():
    df   = pf.load_profile_data(perfil, DATA_DIR)
    rank = pf.build_scores(df, perfil)
    if perfil in ("conservador", "moderado"):
        rank = rank[rank["DY"].notna()]

    carteira = rank.head(n)

    # relatório conciso
    dist = carteira["SETOR"].value_counts(normalize=True) * 100
    print(f"--- {perfil.upper()} ---")
    print(dist.round(1).to_string())
    outfile = OUT_DIR / f"carteira_{perfil}.json"

    carteira.to_json(
        outfile,
        orient="records",  # lista de objetos { … }
        force_ascii=False,  # mantém acentuação
        indent=2  # mais legível
    )

    print(f"{perfil.capitalize():12}: {len(carteira)} papéis salvos em {outfile}")
