"""core/optimizer.py
=============================================================================
Algoritmo Genético para otimização de carteiras.

Seleciona carteiras equiponderadas maximizando o score fundamentalista
enquanto controla a concentração setorial via penalização de HHI.
=============================================================================
"""

import hashlib
import math
import random as random_module
import sys
from pathlib import Path

# Adiciona o diretório parent ao path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from typing import Mapping, Tuple, Optional

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
        random_seed: Optional[int] = None,
        hhi_max: Optional[float] = None,
        early_stopping_patience: int = 50,
        early_stopping_min_delta: float = 1e-6
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
        early_stopping_patience : int, optional
            Número de gerações sem melhoria para parar (default: 50).
        early_stopping_min_delta : float, optional
            Melhoria mínima considerada significativa (default: 1e-6).
        """
        if n_assets <= 0:
            raise ValueError("n_assets must be positive")
        if generations <= 0:
            raise ValueError("generations must be positive")
        if pop_size < 2 or pop_size % 2:
            raise ValueError("pop_size must be an even integer >= 2")
        if not math.isfinite(float(lambda_hhi)) or lambda_hhi < 0:
            raise ValueError("lambda_hhi must be finite and non-negative")
        if hhi_max is not None and (
            not math.isfinite(float(hhi_max)) or not 0.0 <= float(hhi_max) <= 1.0
        ):
            raise ValueError("hhi_max must be between 0 and 1")
        self.n_assets = int(n_assets)
        self.lambda_hhi = float(lambda_hhi)
        self.generations = int(generations)
        self.pop_size = int(pop_size)
        self.crossover_rate = float(crossover_rate)
        self.mutation_rate = float(mutation_rate)
        self.random_seed = random_seed
        self.hhi_max = None if hhi_max is None else float(hhi_max)
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self._random = random_module.Random(random_seed)
        self._numpy = np.random.default_rng(random_seed)

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
        if self.hhi_max is not None and (
            not math.isfinite(float(hhi)) or hhi > self.hhi_max + 1e-12
        ):
            return -np.inf
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
        if len(parent1) < 3 or self._random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        point = self._random.randint(1, len(parent1) - 2)
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
            if self._random.random() < self.mutation_rate:
                chromosome[i] = 1 - chromosome[i]

        # Garante exatamente n_assets
        while chromosome.sum() > self.n_assets:
            idx = self._random.sample(list(np.where(chromosome == 1)[0]), 1)[0]
            chromosome[idx] = 0

        while chromosome.sum() < self.n_assets:
            idx = self._random.sample(list(np.where(chromosome == 0)[0]), 1)[0]
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
        if m < self.n_assets:
            raise ValueError(
                f"universe has {m} assets; {self.n_assets} required"
            )
        elite_idx = list(range(max(1, int(m * 0.25))))  # Top 25%
        population = []

        for _ in range(self.pop_size):
            chrom = np.zeros(m, dtype=int)

            # 40% dos ativos vêm da elite
            elite_count = min(int(self.n_assets * 0.4), len(elite_idx))
            chosen = self._random.sample(elite_idx, k=elite_count)
            chrom[chosen] = 1

            # Preenche restante aleatoriamente
            remaining = [i for i in range(m) if chrom[i] == 0]
            chrom[self._random.sample(remaining, k=self.n_assets - len(chosen))] = 1

            population.append(chrom)

        return population

    def optimize(self, df_ranked: pd.DataFrame) -> pd.DataFrame:
        """
        Executa o Algoritmo Genético com early stopping.

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

        # Early stopping
        generations_without_improvement = 0
        last_improvement_gen = 0

        for generation in range(self.generations):
            # Avalia fitness
            scores = np.array([
                self.fitness(df_ranked, chrom)
                for chrom in population
            ])
            gen_best_idx = int(scores.argmax())
            current_best = scores[gen_best_idx]
            gen_best_chrom = population[gen_best_idx].copy()

            # Seleção por roleta
            finite_scores = np.isfinite(scores)
            if not finite_scores.any():
                probs = np.full(self.pop_size, 1.0 / self.pop_size)
                total_probability = 1.0
            else:
                min_fit = scores[finite_scores].min()
                probs = np.zeros(self.pop_size)
                probs[finite_scores] = scores[finite_scores] - min_fit + 1e-9
                total_probability = probs.sum()
            if (
                not np.isfinite(total_probability)
                or total_probability <= 0
                or (probs > 0).sum() < 2
            ):
                probs = np.full(self.pop_size, 1.0 / self.pop_size)
            else:
                probs /= total_probability

            new_pop = []
            for _ in range(self.pop_size // 2):
                idx1, idx2 = self._numpy.choice(
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

            if current_best > best_fitness + self.early_stopping_min_delta:
                best_fitness = current_best
                best_chrom = gen_best_chrom
                last_improvement_gen = generation
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            # Early stopping: para se não houver melhoria por N gerações
            if generations_without_improvement >= self.early_stopping_patience:
                # Armazena geração de parada nos attrs
                self.stopped_at_generation = generation + 1
                break
        else:
            # Caso complete todas as gerações sem early stopping
            self.stopped_at_generation = self.generations

        if best_chrom is None:
            raise RuntimeError("GA não convergiu para nenhuma solução válida.")

        # Constrói carteira final
        portfolio = df_ranked.iloc[best_chrom.astype(bool)].copy()
        portfolio = portfolio.reset_index(drop=True)

        hhi = hhi_sector(portfolio)
        portfolio.attrs["fitness"] = best_fitness
        portfolio.attrs["hhi"] = hhi
        portfolio.attrs["generations_run"] = self.stopped_at_generation
        portfolio.attrs["converged_early"] = self.stopped_at_generation < self.generations
        portfolio.attrs["seed"] = self.random_seed

        return portfolio


def derive_run_seed(
    base_seed: int,
    run_id: int,
    namespace: str = "selector",
) -> int:
    """Derive a stable non-process-dependent seed for one selector run."""

    payload = f"{int(base_seed)}:{int(run_id)}:{namespace}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def _explicit_ga_config(
    profile: Optional[str],
    ga_config: Optional[Mapping[str, int | float]],
    config: Optional[Mapping[str, object]],
) -> dict[str, int | float]:
    if config is not None:
        nested = config.get("system_ga_config")
        settings = dict(nested) if isinstance(nested, Mapping) else dict(config)
        for key in (
            "n_assets", "lambda", "lambda_hhi", "generations", "pop_size",
            "population", "crossover_rate", "mutation_rate", "hhi_max",
        ):
            if key in config:
                settings[key] = config[key]
        filters = config.get("liquidity_and_size_filters") or config.get("filters")
        if isinstance(filters, Mapping) and "hhi_max" in filters:
            settings.setdefault("hhi_max", filters["hhi_max"])
    else:
        settings = dict(ga_config) if ga_config is not None else None
        if settings is None:
            if profile not in GA_CONFIG:
                raise ValueError(
                    f"profile or explicit GA config is required; profiles: {list(GA_CONFIG)}"
                )
            settings = dict(GA_CONFIG[profile])
    if "population" in settings and "pop_size" not in settings:
        settings["pop_size"] = settings["population"]
    if "lambda_hhi" in settings and "lambda" not in settings:
        settings["lambda"] = settings["lambda_hhi"]
    required = ("n_assets", "generations", "pop_size")
    missing = [key for key in required if key not in settings]
    if missing:
        raise ValueError(f"GA config missing fields: {missing}")
    return settings


def optimize_portfolio(
    df_ranked: pd.DataFrame,
    profile: Optional[str] = None,
    random_seed: Optional[int] = None,
    ga_config: Optional[Mapping[str, int | float]] = None,
    config: Optional[Mapping[str, object]] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Função wrapper para otimizar carteira usando GA.

    Parameters
    ----------
    df_ranked : pd.DataFrame
        DataFrame com scores ordenados.
    profile : str, optional
        Named profile retained for offline callers.
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
    if random_seed is not None and seed is not None and random_seed != seed:
        raise ValueError("random_seed and seed disagree")
    random_seed = seed if seed is not None else random_seed
    cfg = _explicit_ga_config(profile, ga_config, config)

    ga = GeneticAlgorithm(
        n_assets=cfg["n_assets"],
        lambda_hhi=cfg.get("lambda", cfg.get("lambda_hhi", 0.0)),
        generations=cfg["generations"],
        pop_size=cfg["pop_size"],
        crossover_rate=cfg.get("crossover_rate", GA_CROSSOVER_RATE),
        mutation_rate=cfg.get("mutation_rate", GA_MUTATION_RATE),
        random_seed=random_seed,
        hhi_max=cfg.get("hhi_max"),
    )

    return ga.optimize(df_ranked)
