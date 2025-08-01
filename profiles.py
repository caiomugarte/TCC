"""profiles.py
----------------
Calcula pontuações (scores) fundamentalistas para perfis de investidor
conservador, moderado e arrojado, usando apenas indicadores contábeis.

Regras gerais
-------------
* Os indicadores já devem estar padronizados (z-score intra-setor).
* Ausência de preço nas features é garantida no preprocessing.
* Colunas faltantes são ignoradas na média de cada grupo; se todas
  estiverem ausentes, o grupo recebe NaN e o ativo perde relevância.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import pandas as pd

# ------------------------- Configuração ------------------------- #
# Grupos de métricas e colunas correspondentes (nomes iguais aos CSVs)
GROUPS: Dict[str, List[str]] = {
    "liquidez": [
        "LIQ. CORRENTE",
        "DIVIDA LIQUIDA / EBIT",
        "DIV. LIQ. / PATRI.",
    ],
    "rent": [
        "ROE",
        "ROA",
        "ROIC",
        "MARG. LIQUIDA",
        "MARGEM EBIT",
    ],
    "value": [
        "P/L",
        "P/VP",
        "EV/EBIT",  # se EV/EBITDA existir, ele é incluído dinamicamente
        "PSR",
    ],
    "growth": [
        "CAGR RECEITAS 5 ANOS",
        "CAGR LUCROS 5 ANOS",
        "PEG RATIO",
    ],
    "div": [
        "DY",
    ],
}

# Pesos por perfil de investidor
PROFILE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "conservador": {
        "liquidez": 0.30,
        "rent": 0.25,
        "value": 0.15,
        "growth": 0.10,
        "div": 0.20,
    },
    "moderado": {
        "liquidez": 0.20,
        "rent": 0.25,
        "value": 0.25,
        "growth": 0.20,
        "div": 0.10,
    },
    "arrojado": {
        "liquidez": 0.10,
        "rent": 0.20,
        "value": 0.20,
        "growth": 0.40,
        "div": 0.10,
    },
}

# ------------------------- Funções públicas -------------------- #

def add_dynamic_columns(df: pd.DataFrame) -> None:
    """Inclui dinamicamente colunas que existam no DataFrame mas não
    estejam em GROUPS (ex.: "EV/EBITDA" aparece em alguns setores)."""
    if "EV/EBITDA" in df.columns and "EV/EBITDA" not in GROUPS["value"]:
        GROUPS["value"].append("EV/EBITDA")


def build_scores(df: pd.DataFrame, profile: str) -> pd.DataFrame:
    """Calcula o score final para um *DataFrame* de métricas padronizadas.

    Parameters
    ----------
    df : pd.DataFrame
        Dados fundamentalistas (já limpos e padronizados).
    profile : str
        Um dos perfis definidos em PROFILE_WEIGHTS.

    Returns
    -------
    pd.DataFrame
        Mesmo DataFrame com colunas AVG_* para cada grupo e a coluna
        SCORE, ordenado do maior para o menor SCORE.
    """
    if profile not in PROFILE_WEIGHTS:
        raise ValueError(f"Perfil desconhecido: {profile}")

    add_dynamic_columns(df)

    weights = PROFILE_WEIGHTS[profile]
    df = df.copy()

    # Calcula média de cada grupo (skipna=True ignora NaNs)
    for group, cols in GROUPS.items():
        existing = [c for c in cols if c in df.columns]
        if not existing:
            df[f"AVG_{group.upper()}"] = float("nan")
        else:
            df[f"AVG_{group.upper()}"] = df[existing].mean(axis=1, skipna=True)

    # Score final = soma ponderada das médias de grupo
    df["SCORE"] = sum(weights[g] * df[f"AVG_{g.upper()}"] for g in weights)

    # Ordena do maior para o menor (quanto maior, melhor)
    return df.sort_values("SCORE", ascending=False).reset_index(drop=True)


def load_profile_data(perfil: str, data_dir: Path | str = "data/processed") -> pd.DataFrame:
    """Helper para ler o CSV padronizado de um perfil."""
    path = Path(data_dir) / f"fundamentals_clean_{perfil}.csv"
    return pd.read_csv(path)

# ---------------------- CLI rápido (opcional) ------------------ #
if __name__ == "__main__":
    import argparse, textwrap

    parser = argparse.ArgumentParser(
        description="Gera ranking fundamentalista por perfil",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """Exemplo:
            python profiles.py moderado --top 20
            """,
        ),
    )
    parser.add_argument("perfil", choices=list(PROFILE_WEIGHTS.keys()))
    parser.add_argument("--top", type=int, default=15, help="Número de ações a exibir")
    args = parser.parse_args()

    df_in = load_profile_data(args.perfil)
    ranked = build_scores(df_in, args.perfil)
    print(ranked[["TICKER", "SCORE"]].head(args.top).to_string(index=False))
