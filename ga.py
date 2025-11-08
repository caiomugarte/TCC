"""ga.py
--------------
Algoritmo Genético (GA) para selecionar carteiras equiponderadas
maximizando o SCORE fundamentalista e controlando a concentração
setorial via penalização de HHI.

Perfis, tamanhos e penalizações
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PERFIL_CONFIG = {
    "conservador": {"n_assets": 10, "lambda": 0.50, "generations": 300, "pop_size": 200},
    "moderado":    {"n_assets": 12, "lambda": 0.25, "generations": 400, "pop_size": 250},
    "arrojado":    {"n_assets": 15, "lambda": 0.10, "generations": 500, "pop_size": 300},
}

Uso rápido
~~~~~~~~~~
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from random import randint, random, sample, seed

# ----------------------------- Config ------------------------------ #
PERFIL_CONFIG: Dict[str, Dict[str, int | float]] = {
    "conservador": {"n_assets": 10, "lambda": 0.50, "generations": 300, "pop_size": 200},
    "moderado":    {"n_assets": 12, "lambda": 0.25, "generations": 400, "pop_size": 250},
    "arrojado":    {"n_assets": 15, "lambda": 0.10, "generations": 500, "pop_size": 300},
}
# ---------- parâmetros de GA ----------
NCROSS_RATE = 0.8   # probabilidade de crossover
MUT_RATE   = 0.02   # probabilidade de mutação por gene
# seed(42)            # reprodutibilidade
# --------------------------------------
# ------------------------- Funções auxiliares ---------------------- #

def hhi_sector(df_sel: pd.DataFrame) -> float:
    """Calcula o Herfindahl-Hirschman Index por setor (peso igual)."""
    weights = df_sel["SETOR"].value_counts(normalize=True)
    return (weights ** 2).sum()


def fitness(df: pd.DataFrame, mask: np.ndarray, lam: float, n: int) -> float:
    """Score total da carteira menos penalidade λ*HHI.
    Presume peso 1/n; `mask` length == len(df) e contém 0/1.
    """
    if mask.sum() != n:
        return -np.inf  # inviável se não tiver exatamente n ativos
    sel = df.iloc[mask.astype(bool)]
    hhi = hhi_sector(sel)
    total_score = sel["SCORE"].sum()
    return total_score - lam * hhi * n  # escala pela qtde de ativos


def crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if random() > NCROSS_RATE:
        return parent1.copy(), parent2.copy()
    point = randint(1, len(parent1) - 2)
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    return child1, child2


def mutate(chrom: np.ndarray, n: int):
    for i in range(len(chrom)):
        if random() < MUT_RATE:
            chrom[i] = 1 - chrom[i]
    # garanta exatamente n ativos
    while chrom.sum() > n:
        idx = sample(list(np.where(chrom == 1)[0]), 1)[0]
        chrom[idx] = 0
    while chrom.sum() < n:
        idx = sample(list(np.where(chrom == 0)[0]), 1)[0]
        chrom[idx] = 1

# ---------------------- Função principal GA ------------------------ #

def run_ga(df_ranked: pd.DataFrame, profile: str) -> pd.DataFrame:
    """Executa GA e retorna DataFrame da carteira ótima para o perfil."""
    cfg = PERFIL_CONFIG[profile]
    n, lam, gens, pop_size = (
        cfg["n_assets"],
        cfg["lambda"],
        cfg["generations"],
        cfg["pop_size"],
    )
    m = len(df_ranked)

    # população inicial – amostras aleatórias enviesadas pelos top scores
    elite_idx = list(range(int(m * 0.25)))  # top 25 %
    population = []
    for _ in range(pop_size):
        chrom = np.zeros(m, dtype=int)
        # força inclusão de alguns top picks
        chosen = sample(elite_idx, k=int(n * 0.4))  # 40 % dos n vêm da elite
        chrom[chosen] = 1
        # preenche restante aleatório
        remaining = [i for i in range(m) if chrom[i] == 0]
        chrom[sample(remaining, k=n - len(chosen))] = 1
        population.append(chrom)

    best_fitness = -np.inf
    best_chrom   = None

    for _ in range(gens):
        # avalia fitness
        scores = np.array([fitness(df_ranked, c, lam, n) for c in population])

        # seleção por roleta (fitness shift para ≥0)
        min_fit = scores.min()
        probs = (scores - min_fit + 1e-9)  # evita divisão por zero
        probs /= probs.sum()
        new_pop = []
        for _ in range(pop_size // 2):
            idx1, idx2 = np.random.choice(pop_size, p=probs, size=2, replace=False)
            p1, p2 = population[idx1], population[idx2]
            c1, c2 = crossover(p1, p2)
            mutate(c1, n)
            mutate(c2, n)
            new_pop.extend([c1, c2])
        population = new_pop

        # regista melhor
        gen_best_idx = int(scores.argmax())
        if scores[gen_best_idx] > best_fitness:
            best_fitness = scores[gen_best_idx]
            best_chrom = population[gen_best_idx].copy()

    if best_chrom is None:
        raise RuntimeError("GA não convergiu para nenhuma solução válida.")

    carteira = df_ranked.iloc[best_chrom.astype(bool)].copy()
    carteira = carteira.reset_index(drop=True)
    hhi = hhi_sector(carteira)
    carteira.attrs["fitness"] = best_fitness
    carteira.attrs["hhi"] = hhi
    return carteira
