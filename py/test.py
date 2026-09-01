import pandas as pd, profiles as pf, ga

df_mod = pf.load_profile_data("moderado")
rank   = pf.build_scores(df_mod, "moderado")
carteria_mod = ga.run_ga(rank, "moderado")

print(carteria_mod[["TICKER", "SCORE", "SETOR"]])
print("HHI:", carteria_mod.attrs["hhi"])
