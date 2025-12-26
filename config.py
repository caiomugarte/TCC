"""config.py
=============================================================================
Configurações centralizadas para todo o projeto TCC.

Este módulo centraliza:
- Perfis de investidor (conservador, moderado, arrojado, caio)
- Parâmetros do Algoritmo Genético
- Filtros de elegibilidade
- Métricas e pesos por perfil
- Paths de dados
- Lista do Ibovespa
=============================================================================
"""

from pathlib import Path
from typing import Dict, List

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CACHE_DIR = PROJECT_ROOT / ".cache"

# Cria diretórios se não existirem
for dir_path in [DATA_RAW, DATA_PROCESSED, OUTPUTS_DIR, CACHE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DADOS
# ============================================================================

RAW_DATA_FILE = DATA_RAW / "status_invest_fundamentals.csv"

# ============================================================================
# PERFIS DE INVESTIDOR
# ============================================================================

PROFILES = ["conservador", "moderado", "arrojado", "caio", "caio2"]

# Filtros de elegibilidade por perfil (valor de mercado e liquidez)
FILTERS: Dict[str, Dict[str, float]] = {
    "conservador": {"cap_min": 5_000_000_000, "liq_min": 2_000_000},
    "moderado":    {"cap_min": 1_000_000_000, "liq_min":   500_000},
    "arrojado":    {"cap_min":   200_000_000, "liq_min":    50_000},
    "caio":        {"cap_min": 1_319_241_560, "liq_min": 461_735},
    "caio2": {"cap_min": 3_000_000_000, "liq_min": 1_050_000}
}

# ============================================================================
# ALGORITMO GENÉTICO - PARÂMETROS POR PERFIL
# ============================================================================

GA_CONFIG: Dict[str, Dict[str, int | float]] = {
    "conservador": {
        "n_assets": 10,
        "lambda": 0.50,
        "generations": 300,
        "pop_size": 200
    },
    "moderado": {
        "n_assets": 12,
        "lambda": 0.25,
        "generations": 400,
        "pop_size": 250
    },
    "arrojado": {
        "n_assets": 15,
        "lambda": 0.10,
        "generations": 500,
        "pop_size": 300
    },
    "caio": {
        "n_assets": 10,
        "lambda": 0.48,
        "generations": 460,
        "pop_size": 276
    },
    "caio2": {
    "n_assets": 10,
    "lambda": 0.50,
    "generations": 460,
    "pop_size": 276
}
}

# Parâmetros genéricos do GA (aplicam-se a todos os perfis)
GA_CROSSOVER_RATE = 0.8
GA_MUTATION_RATE = 0.02

# ============================================================================
# SCORING - GRUPOS DE MÉTRICAS
# ============================================================================

METRIC_GROUPS: Dict[str, List[str]] = {
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
        "EV/EBIT",
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

# Pesos dos grupos por perfil de investidor
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
    "caio": {
        "liquidez": 0.22,
        "rent": 0.35,
        "value": 0.08,
        "growth": 0.05,
        "div": 0.30
    },
    "caio2": {
        "liquidez": 0.29,
        "rent": 0.32,
        "value": 0.07,
        "growth": 0.05,
        "div": 0.27
    }
}

# ============================================================================
# PREPROCESSING
# ============================================================================

# Métricas que devem ser invertidas (quanto menor, melhor)
INVERT_METRICS = [
    "P/L",
    "P/VP",
    "P/ATIVOS",
    "PSR",
    "DIVIDA LIQUIDA / EBIT",
]

# Colunas removidas no preprocessing
PRICE_COLS = ["PRECO"]
FILTER_COLS = ["VALOR DE MERCADO", "LIQUIDEZ MEDIA DIARIA"]

# ============================================================================
# ANÁLISE E REPORTING
# ============================================================================

# Métricas principais para comparação
METRIC_COLS = [
    "DY", "P/VP", "EV/EBIT",
    "ROE", "ROIC", "MARGEM EBIT", "MARG. LIQUIDA",
    "CAGR RECEITAS 5 ANOS", "CAGR LUCROS 5 ANOS", "PEG RATIO",
    "LIQ. CORRENTE", "DIVIDA LIQUIDA / EBIT", "DIV. LIQ. / PATRI.",
]

# ============================================================================
# IBOVESPA (BENCHMARK)
# ============================================================================

IBOV_TICKERS = {
    "ABEV3", "ALOS3", "ASAI3", "AURE3", "AZUL4", "AZZA3", "B3SA3", "BBAS3",
    "BBDC3", "BBDC4", "BBSE3", "BEEF3", "BPAC11", "BRAP4", "BRAV3", "BRFS3",
    "BRKM5", "CMIG4", "CMIN3", "COGN3", "CPFE3", "CPLE6", "CRFB3", "CSAN3",
    "CSNA3", "CVCB3", "CXSE3", "CYRE3", "DIRR3", "EGIE3", "ELET3", "ELET6",
    "EMBR3", "ENEV3", "ENGI11", "EQTL3", "FLRY3", "GGBR4", "GOAU4", "HAPV3",
    "HYPE3", "IGTI11", "IRBR3", "ISAE4", "ITSA4", "ITUB4", "JBSS3", "KLBN11",
    "LREN3", "MGLU3", "MOTV3", "MRFG3", "MRVE3", "MULT3", "NTCO3", "PCAR3",
    "PETR3", "PETR4", "PETZ3", "POMO4", "PRIO3", "PSSA3", "RADL3", "RAIL3",
    "RAIZ4", "RDOR3", "RECV3", "RENT3", "SANB11", "SBSP3", "SLCE3", "SMFT3",
    "SMTO3", "STBP3", "SUZB3", "TAEE11", "TIMS3", "TOTS3", "UGPA3", "USIM5",
    "VALE3", "VAMO3", "VBBR3", "VIVA3", "VIVT3", "WEGE3", "YDUQ3"
}

# ============================================================================
# MÚLTIPLAS EXECUÇÕES (ANÁLISE DE ROBUSTEZ)
# ============================================================================

N_RUNS = 60  # Número de execuções independentes do GA

# ============================================================================
# BACKTEST
# ============================================================================

BACKTEST_PERIODS = {
    "5anos": 5,
    "10anos": 10
}
