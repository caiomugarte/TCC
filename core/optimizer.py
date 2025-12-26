"""core/optimizer.py
=============================================================================
Algoritmo Genético para otimização de carteiras.

Seleciona carteiras equiponderadas maximizando o score fundamentalista
enquanto controla a concentração setorial via penalização de HHI.
=============================================================================
"""

import sys
from pathlib import Path

# Adiciona o diretório parent ao path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from random import randint, random, sample

from config import GA_CONFIG, GA_CROSSOVER_RATE, GA_MUTATION_RATE
from core.metrics import hhi_sector


class GeneticAlgorithm:
    """
    Otimizador de carteiras usando Algoritmo Genético.

    Attributes
    ----------
    n_assets : int
        Número de ativos na carteira.
    lambda_hhi : float
        Penalização para concentração setorial (HHI).
    generations : int
        Número de gerações do GA.
    pop_size : int
        Tamanho da população.
    crossover_rate : float
        Taxa de crossover.
    mutation_rate : float
        Taxa de mutação por gene.
    """

    def __init__(
        self,
        n_assets: int,
        lambda_hhi: float,
        generations: int,
        pop_size: int,
        crossover_rate: float = GA_CROSSOVER_RATE,
        mutation_rate: float = GA_MUTATION_RATE,
        random_seed: Optional[int] = None
    ):
        """
        Inicializa o otimizador GA.

        Parameters
        ----------
        n_assets : int
            Número de ativos na carteira.
        lambda_hhi : float
            Penalização para HHI.
        generations : int
            Número de gerações.
        pop_size : int
            Tamanho da população.
        crossover_rate : float
            Taxa de crossover.
        mutation_rate : float
            Taxa de mutação.
        random_seed : int, optional
            Seed para reprodutibilidade.
        """
        self.n_assets = n_assets
        self.lambda_hhi = lambda_hhi
        self.generations = generations
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

        if random_seed is not None:
            np.random.seed(random_seed)

    def fitness(self, df: pd.DataFrame, mask: np.ndarray) -> float:
        """
        Calcula fitness de uma carteira.

        Fitness = Score Total - λ × HHI × n

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame com scores.
        mask : np.ndarray
            Máscara binária (0/1) indicando ativos selecionados.

        Returns
        -------
        float
            Valor do fitness.
        """
        if mask.sum() != self.n_assets:
            return -np.inf  # Inviável

        selected = df.iloc[mask.astype(bool)]
        hhi = hhi_sector(selected)
        total_score = selected["SCORE"].sum()

        return total_score - self.lambda_hhi * hhi * self.n_assets

    def crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Operador de crossover de um ponto.

        Parameters
        ----------
        parent1 : np.ndarray
            Primeiro pai.
        parent2 : np.ndarray
            Segundo pai.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Dois filhos gerados.
        """
        if random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        point = randint(1, len(parent1) - 2)
        child1 = np.concatenate((parent1[:point], parent2[point:]))
        child2 = np.concatenate((parent2[:point], parent1[point:]))

        return child1, child2

    def mutate(self, chromosome: np.ndarray) -> None:
        """
        Operador de mutação.

        Aplica mutação bit-flip e garante exatamente n_assets ativos.

        Parameters
        ----------
        chromosome : np.ndarray
            Cromossomo a ser mutado (modificado in-place).
        """
        # Mutação bit-flip
        for i in range(len(chromosome)):
            if random() < self.mutation_rate:
                chromosome[i] = 1 - chromosome[i]

        # Garante exatamente n_assets
        while chromosome.sum() > self.n_assets:
            idx = sample(list(np.where(chromosome == 1)[0]), 1)[0]
            chromosome[idx] = 0

        while chromosome.sum() < self.n_assets:
            idx = sample(list(np.where(chromosome == 0)[0]), 1)[0]
            chromosome[idx] = 1

    def initialize_population(self, m: int) -> list:
        """
        Cria população inicial enviesada pelos top scores.

        Parameters
        ----------
        m : int
            Número total de ativos disponíveis.

        Returns
        -------
        list
            Lista de cromossomos (população).
        """
        elite_idx = list(range(int(m * 0.25)))  # Top 25%
        population = []

        for _ in range(self.pop_size):
            chrom = np.zeros(m, dtype=int)

            # 40% dos ativos vêm da elite
            chosen = sample(elite_idx, k=int(self.n_assets * 0.4))
            chrom[chosen] = 1

            # Preenche restante aleatoriamente
            remaining = [i for i in range(m) if chrom[i] == 0]
            chrom[sample(remaining, k=self.n_assets - len(chosen))] = 1

            population.append(chrom)

        return population

    def optimize(self, df_ranked: pd.DataFrame) -> pd.DataFrame:
        """
        Executa o Algoritmo Genético.

        Parameters
        ----------
        df_ranked : pd.DataFrame
            DataFrame com scores ordenados.

        Returns
        -------
        pd.DataFrame
            Carteira ótima encontrada.
        """
        m = len(df_ranked)
        population = self.initialize_population(m)

        best_fitness = -np.inf
        best_chrom = None

        for generation in range(self.generations):
            # Avalia fitness
            scores = np.array([
                self.fitness(df_ranked, chrom)
                for chrom in population
            ])

            # Seleção por roleta
            min_fit = scores.min()
            probs = scores - min_fit + 1e-9
            probs /= probs.sum()

            new_pop = []
            for _ in range(self.pop_size // 2):
                idx1, idx2 = np.random.choice(
                    self.pop_size,
                    p=probs,
                    size=2,
                    replace=False
                )

                p1, p2 = population[idx1], population[idx2]
                c1, c2 = self.crossover(p1, p2)

                self.mutate(c1)
                self.mutate(c2)

                new_pop.extend([c1, c2])

            population = new_pop

            # Registra melhor
            gen_best_idx = int(scores.argmax())
            if scores[gen_best_idx] > best_fitness:
                best_fitness = scores[gen_best_idx]
                best_chrom = population[gen_best_idx].copy()

        if best_chrom is None:
            raise RuntimeError("GA não convergiu para nenhuma solução válida.")

        # Constrói carteira final
        portfolio = df_ranked.iloc[best_chrom.astype(bool)].copy()
        portfolio = portfolio.reset_index(drop=True)

        hhi = hhi_sector(portfolio)
        portfolio.attrs["fitness"] = best_fitness
        portfolio.attrs["hhi"] = hhi

        return portfolio


def optimize_portfolio(
    df_ranked: pd.DataFrame,
    profile: str,
    random_seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Função wrapper para otimizar carteira usando GA.

    Parameters
    ----------
    df_ranked : pd.DataFrame
        DataFrame com scores ordenados.
    profile : str
        Perfil do investidor.
    random_seed : int, optional
        Seed para reprodutibilidade.

    Returns
    -------
    pd.DataFrame
        Carteira otimizada.

    Examples
    --------
    >>> df_ranked = build_scores(df_clean, "conservador")
    >>> portfolio = optimize_portfolio(df_ranked, "conservador")
    >>> print(portfolio[["TICKER", "SCORE"]])
    """
    if profile not in GA_CONFIG:
        raise ValueError(
            f"Perfil desconhecido: {profile}. "
            f"Perfis disponíveis: {list(GA_CONFIG.keys())}"
        )

    cfg = GA_CONFIG[profile]

    ga = GeneticAlgorithm(
        n_assets=cfg["n_assets"],
        lambda_hhi=cfg["lambda"],
        generations=cfg["generations"],
        pop_size=cfg["pop_size"],
        random_seed=random_seed
    )

    return ga.optimize(df_ranked)
