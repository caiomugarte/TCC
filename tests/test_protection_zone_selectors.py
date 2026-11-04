"""Protection zone in the stock consensus (issue #46, PZ-14, PZ-17, PZ-18).

``_pre_change_build_consensus`` is ``build_consensus_portfolio`` at commit 76a6d70,
copied verbatim (spliced in from git). With no incumbents the new function must
return the same frame and attrs (PZ-17).
"""

from collections import Counter
import contextlib
import io
import math
from pathlib import Path
import sys
import tempfile
from typing import Dict, List, Mapping, Optional
import unittest
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))

from config import PROFILE_WEIGHTS  # noqa: E402
from core.metrics import hhi_sector  # noqa: E402
from fetch_status_invest import SOURCE_COLUMNS as STOCK_COLUMNS  # noqa: E402
from pipelines.multi_run import (  # noqa: E402
    GA_CONFIG,
    build_consensus_portfolio,
    run_multi_execution_profile,
)


# build_consensus_portfolio at 76a6d70 (py/pipelines/multi_run.py:187-295), copied verbatim.
def _pre_change_build_consensus(
    results: List[Dict],
    df_ranked: pd.DataFrame,
    profile: Optional[str] = None,
    selection_config: Optional[Mapping[str, object]] = None,
) -> pd.DataFrame:
    """
    Constrói carteira consenso baseada na frequência de aparição.

    Parameters
    ----------
    results : List[Dict]
        Resultados de todas as execuções.
    df_ranked : pd.DataFrame
        DataFrame com dados ranqueados.
    profile : str
        Perfil do investidor.

    Returns
    -------
    pd.DataFrame
        Carteira consenso.
    """
    if selection_config is not None:
        n_assets = int(
            selection_config.get(
                "n_assets",
                selection_config.get("system_ga_config", {}).get("n_assets"),
            )
        )
    else:
        if profile not in GA_CONFIG:
            raise ValueError(f"unknown stock profile: {profile}")
        n_assets = GA_CONFIG[profile]["n_assets"]

    # Conta frequência de cada ticker
    all_tickers = []
    for r in results:
        all_tickers.extend(r["tickers"])

    ticker_counts = Counter(all_tickers)

    # Seleciona os N mais frequentes com critério de desempate por SCORE
    # Cria lista de (ticker, frequência, score) para ordenação
    ticker_data = []
    for ticker, count in ticker_counts.items():
        score = df_ranked[df_ranked["TICKER"] == ticker]["SCORE"].iloc[0]
        ticker_data.append((ticker, count, score))

    # Ordena por: 1) frequência (desc), 2) score (desc), 3) ticker (alfabético)
    ticker_data.sort(key=lambda x: (-x[1], -x[2], x[0]))

    # Filtra duplicatas de mesma empresa (ex: POMO3 e POMO4)
    # Remove sufixos 3, 4, 5, 6, 11 para identificar empresa base
    def get_base_ticker(ticker):
        """Remove sufixo de classe de ação para identificar empresa base."""
        for suffix in ['11', '3', '4', '5', '6']:
            if ticker.endswith(suffix):
                return ticker[:-len(suffix)]
        return ticker

    # Seleciona ativos evitando duplicatas de mesma empresa
    consensus_tickers = []
    seen_companies = set()
    skipped_duplicates = []

    for ticker, freq, score in ticker_data:
        base_ticker = get_base_ticker(ticker)

        # Se ainda não temos ativo dessa empresa, adiciona
        if base_ticker not in seen_companies:
            consensus_tickers.append(ticker)
            seen_companies.add(base_ticker)

            # Para quando atingir o número desejado
            if len(consensus_tickers) == n_assets:
                break
        else:
            # Registra duplicata pulada
            existing_ticker = [t for t in consensus_tickers if get_base_ticker(t) == base_ticker][0]
            skipped_duplicates.append((ticker, existing_ticker, freq, score))

    # Informa sobre duplicatas removidas
    if skipped_duplicates:
        print(f"\n  ⚠️  Duplicatas de mesma empresa removidas da carteira consensual:")
        for skipped, kept, freq, score in skipped_duplicates:
            if skipped in [t for t, _, _ in ticker_data[:n_assets * 2]]:  # Só mostra se estava entre os candidatos
                print(f"      • {skipped} (freq: {freq/len(results):.1%}, score: {score:.3f}) "
                      f"→ já tem {kept} da mesma empresa")

    # Constrói DataFrame da carteira consenso
    consensus_portfolio = df_ranked[
        df_ranked["TICKER"].isin(consensus_tickers)
    ].copy()

    # Adiciona coluna de frequência
    consensus_portfolio["FREQUENCY"] = consensus_portfolio["TICKER"].map(
        lambda t: ticker_counts[t] / len(results)
    )

    # Calcula métricas
    hhi = hhi_sector(consensus_portfolio)
    avg_score = consensus_portfolio["SCORE"].mean()

    consensus_portfolio.attrs["hhi"] = hhi
    consensus_portfolio.attrs["avg_score"] = avg_score
    consensus_portfolio.attrs["method"] = "consensus_frequency"

    return consensus_portfolio.sort_values("FREQUENCY", ascending=False)


def _quiet(function, *args, **kwargs):
    """Call ``function`` with its duplicate-removal report silenced."""

    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def _ranked(rows):
    """df_ranked from (ticker, score, sector) rows."""

    return pd.DataFrame(
        {
            "TICKER": [row[0] for row in rows],
            "SCORE": [row[1] for row in rows],
            "SETOR": [row[2] for row in rows],
        }
    )


def _runs_from_counts(counts, runs):
    """``runs`` GA runs in which every ticker appears in exactly ``counts[ticker]`` of them."""

    return [{"tickers": [t for t, c in counts.items() if index < c]} for index in range(runs)]


REGRESSION_RESULTS = [
    {"tickers": ["AAAA3", "BBBB4", "CCCC3", "DDDD11"]},
    {"tickers": ["AAAA3", "BBBB4", "EEEE3", "FFFF11"]},
    {"tickers": ["AAAA4", "CCCC3", "EEEE3", "GGGG5"]},
    {"tickers": ["AAAA3", "CCCC3", "HHHH6", "DDDD11"]},
]
REGRESSION_RANKED = _ranked(
    [
        ("AAAA3", 0.8, "S1"), ("AAAA4", 0.7, "S1"), ("BBBB4", 0.6, "S2"), ("CCCC3", 0.5, "S3"),
        ("DDDD11", 0.4, "S4"), ("EEEE3", 0.3, "S2"), ("FFFF11", 0.2, "S5"), ("GGGG5", 0.1, "S3"),
        ("HHHH6", 0.05, "S6"),
    ]
)

# Ten runs: ALFA3 0.9, BETA3 0.8, GAMA3 0.7, DELT3 0.6, EPSI3 0.5, ZETA3 0.4.
ZONE_COUNTS = {"ALFA3": 9, "BETA3": 8, "GAMA3": 7, "DELT3": 6, "EPSI3": 5, "ZETA3": 4}
ZONE_RESULTS = _runs_from_counts(ZONE_COUNTS, 10)
ZONE_RANKED = _ranked(
    [
        ("ALFA3", 0.5, "S1"), ("BETA3", 0.9, "S2"), ("GAMA3", 0.4, "S3"),
        ("DELT3", 0.3, "S4"), ("EPSI3", 0.2, "S5"), ("ZETA3", 0.1, "S6"),
    ]
)
ZONE_CONFIG = {
    "n_assets": 3,
    "protection_zone": {"K": 6, "delta": 0.11, "k_offset": 3},
}


class ConsensusWithoutIncumbentsTests(unittest.TestCase):
    def test_omitted_none_and_empty_incumbents_return_the_pre_change_frame(self):  # PZ-17
        # The second config carries a zone and an HHI limit, as a Premium run does.
        configs = {
            "plain": {"n_assets": 3},
            "premium": {
                "n_assets": 3,
                "protection_zone": {"K": 6, "delta": 0.11, "k_offset": 3},
                "liquidity_and_size_filters": {"hhi_max": 0.2},
            },
        }
        variants = {"omitted": {}, "none": {"incumbents": None}, "empty": {"incumbents": []}}
        for config_label, config in configs.items():
            expected = _quiet(
                _pre_change_build_consensus, REGRESSION_RESULTS, REGRESSION_RANKED, None, config
            )
            for label, kwargs in variants.items():
                with self.subTest(config=config_label, incumbents=label):
                    actual = _quiet(
                        build_consensus_portfolio, REGRESSION_RESULTS, REGRESSION_RANKED, None, config,
                        **kwargs,
                    )
                    assert_frame_equal(actual, expected)
                    self.assertEqual(dict(actual.attrs), dict(expected.attrs))
                    self.assertNotIn("protection_zone", actual.attrs)


class StockZoneTests(unittest.TestCase):
    def test_zone_rows_are_the_zone_portfolio_with_its_record(self):  # PZ-18
        frame = _quiet(
            build_consensus_portfolio, ZONE_RESULTS, ZONE_RANKED, None, ZONE_CONFIG,
            incumbents=["DELT3", "ZETA3"],
        )
        self.assertEqual(frame["TICKER"].tolist(), ["ALFA3", "BETA3", "DELT3"])
        self.assertEqual(frame["FREQUENCY"].tolist(), [0.9, 0.8, 0.6])
        record = frame.attrs["protection_zone"]
        self.assertEqual(
            record,
            {
                "status": "applied",
                "K": 6,
                "k_offset": 3,
                "delta": 0.11,
                "n_assets": 3,
                "incumbents": ["DELT3", "ZETA3"],
                "kept": ["DELT3"],
                "replaced": [{"out": "ZETA3", "in": "BETA3"}],
                "entered": ["ALFA3", "BETA3"],
                "departures": {"ZETA3": "saiu: substituído por BETA3"},
                "turnover": 2,
                "exits": 1,
                "buffer_fallback": None,
                "consensus_tickers": ["ALFA3", "BETA3", "GAMA3"],
                "consensus_turnover": 3,
            },
        )
        self.assertEqual(frame.attrs["method"], "consensus_frequency")

    def test_zone_over_the_hhi_limit_falls_back_to_the_consensus(self):  # PZ-14
        # DELT3 shares ALFA3's sector: the zone HHI is 5/9, the consensus HHI is 1/3.
        ranked = ZONE_RANKED.copy()
        ranked.loc[ranked["TICKER"] == "DELT3", "SETOR"] = "S1"
        config = {**ZONE_CONFIG, "liquidity_and_size_filters": {"hhi_max": 0.4}}
        frame = _quiet(
            build_consensus_portfolio, ZONE_RESULTS, ranked, None, config, incumbents=["DELT3", "ZETA3"]
        )
        self.assertEqual(frame["TICKER"].tolist(), ["ALFA3", "BETA3", "GAMA3"])
        self.assertEqual(frame.attrs["protection_zone"]["status"], "fallback_hhi")
        self.assertEqual(frame.attrs["protection_zone"]["buffer_fallback"], "hhi")
        self.assertAlmostEqual(frame.attrs["hhi"], 1 / 3)

    def test_outranked_same_company_incumbent_leaves_and_its_outsider_enters(self):  # PZ-09, PZ-07
        # POMO3 is in more runs than POMO4, so POMO4 has no rank once the company is deduped.
        counts = {"POMO3": 9, "POMO4": 8, "ALFA3": 7, "BETA3": 6}
        ranked = _ranked(
            [("POMO3", 0.5, "S1"), ("POMO4", 0.9, "S1"), ("ALFA3", 0.4, "S2"), ("BETA3", 0.3, "S3")]
        )
        config = {"n_assets": 2, "protection_zone": {"K": 4, "delta": 0.11, "k_offset": 2}}
        frame = _quiet(
            build_consensus_portfolio, _runs_from_counts(counts, 10), ranked, None, config,
            incumbents=["POMO4", "BETA3"],
        )
        self.assertEqual(frame["TICKER"].tolist(), ["POMO3", "BETA3"])
        record = frame.attrs["protection_zone"]
        self.assertEqual(record["departures"], {"POMO4": "saiu: fora do top K"})
        self.assertEqual(record["entered"], ["POMO3"])

    def test_top_ranked_same_company_incumbent_is_kept_and_its_outsider_stays_out(self):  # PZ-09, PZ-03
        # POMO4 is in more runs than POMO3. The vacancy goes to ALFA3, never to POMO3: its company is held.
        counts = {"POMO4": 9, "POMO3": 8, "ALFA3": 7, "BETA3": 6}
        ranked = _ranked(
            [("POMO4", 0.5, "S1"), ("POMO3", 0.9, "S1"), ("ALFA3", 0.4, "S2"), ("BETA3", 0.3, "S3")]
        )
        config = {"n_assets": 3, "protection_zone": {"K": 4, "delta": 0.11, "k_offset": 1}}
        frame = _quiet(
            build_consensus_portfolio, _runs_from_counts(counts, 10), ranked, None, config,
            incumbents=["POMO4", "BETA3"],
        )
        self.assertEqual(frame["TICKER"].tolist(), ["POMO4", "ALFA3", "BETA3"])
        record = frame.attrs["protection_zone"]
        self.assertEqual(record["kept"], ["BETA3", "POMO4"])
        self.assertEqual(record["entered"], ["ALFA3"])
        self.assertEqual(record["departures"], {})

    def test_zone_within_the_hhi_limit_is_used_when_the_consensus_exceeds_it(self):  # PZ-16
        # Zone HHI is 5/9 (ALFA3 and BETA3 in S1, DELT3 in S4); the pure consensus is all S1, HHI 1.
        ranked = ZONE_RANKED.copy()
        ranked.loc[ranked["TICKER"].isin(["BETA3", "GAMA3"]), "SETOR"] = "S1"
        config = {**ZONE_CONFIG, "liquidity_and_size_filters": {"hhi_max": 0.6}}
        frame = _quiet(
            build_consensus_portfolio, ZONE_RESULTS, ranked, None, config, incumbents=["DELT3", "ZETA3"]
        )
        self.assertEqual(frame["TICKER"].tolist(), ["ALFA3", "BETA3", "DELT3"])
        self.assertEqual(frame.attrs["protection_zone"]["status"], "applied")
        self.assertIsNone(frame.attrs["protection_zone"]["buffer_fallback"])
        self.assertAlmostEqual(frame.attrs["hhi"], 5 / 9)

    def test_incumbent_missing_from_the_ranked_frame_is_ineligible(self):  # PZ-06, A-5
        frame = _quiet(
            build_consensus_portfolio, ZONE_RESULTS, ZONE_RANKED, None, ZONE_CONFIG,
            incumbents=["ZZZZ3", "BETA3"],
        )
        self.assertNotIn("ZZZZ3", frame["TICKER"].tolist())
        self.assertEqual(
            frame.attrs["protection_zone"]["departures"], {"ZZZZ3": "saiu: inelegível"}
        )

    def test_zone_with_k_zero_selects_what_production_selects_on_nan_scores(self):  # PZ-02, AD-025
        results = [{"tickers": ["AAAA3", "BBBB3", "CCCC3"]}] * 4
        ranked = _ranked([("AAAA3", 0.5, "S1"), ("BBBB3", math.nan, "S2"), ("CCCC3", 0.7, "S3")])
        production = _quiet(build_consensus_portfolio, results, ranked, None, {"n_assets": 2})
        config = {"n_assets": 2, "protection_zone": {"K": 0, "delta": 0.11, "k_offset": 0}}
        zone = _quiet(
            build_consensus_portfolio, results, ranked, None, config, incumbents=["CCCC3"]
        )
        self.assertEqual(sorted(zone["TICKER"]), sorted(production["TICKER"]))
        self.assertEqual(zone.attrs["protection_zone"]["status"], "applied")


def _stock_frame() -> pd.DataFrame:
    """Six stocks in one sector with every metric present (the fixture of test_missing_score_exclusion)."""

    rows = []
    for index, ticker in enumerate(("AAA3", "BBB3", "CCC3", "DDD3", "EEE3", "FFF3"), start=1):
        row = {column: float(index) + 0.1 * index * index for column in STOCK_COLUMNS}
        row.update(
            {
                "TICKER": ticker,
                "PRECO": 100.0,
                "DY": float(index),
                "VALOR DE MERCADO": 1000.0,
                "LIQUIDEZ MEDIA DIARIA": 1000.0,
                "SETOR": "Sector A",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=[*STOCK_COLUMNS, "SETOR"])


def _pipeline_config(with_zone: bool) -> dict:
    config = {
        "selection_preset": "fixture",
        "factor_weights": dict(PROFILE_WEIGHTS["caio_new"]),
        "liquidity_and_size_filters": {"cap_min": 100.0, "liq_min": 100.0},
        "system_ga_config": {"n_assets": 2, "lambda": 0.0, "generations": 3, "pop_size": 4},
    }
    if with_zone:
        config["protection_zone"] = {"K": 8, "delta": 0.11, "k_offset": 6}
    return config


class PipelineForwardsIncumbentsTests(unittest.TestCase):
    """Explicit mode with two GA runs on the small fixture (PZ-20)."""

    def _run(self, *, with_zone: bool = True, **kwargs) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "stocks.csv"
            _stock_frame().to_csv(input_path, index=False)
            with patch("builtins.input", side_effect=AssertionError("unexpected prompt")):
                return _quiet(
                    run_multi_execution_profile,
                    input_path=input_path,
                    config=_pipeline_config(with_zone),
                    n_runs=2,
                    parallel=False,
                    use_cache=False,
                    save_interval=1,
                    workspace=root / "workspace",
                    seed=7,
                    **kwargs,
                )

    def test_explicit_result_records_the_zone_for_the_incumbents(self):  # PZ-20
        result = self._run(incumbents=["aaa3", "FFF3"])
        record = result["protection_zone"]
        self.assertEqual(record["status"], "applied")
        self.assertEqual(record["incumbents"], ["AAA3", "FFF3"])
        self.assertEqual(record["K"], 8)
        self.assertEqual(sorted(record["kept"] + record["entered"]), result["selected_tickers"])

    def test_explicit_result_has_no_zone_without_incumbents(self):  # PZ-20
        cases = {"omitted": {}, "none": {"incumbents": None}, "empty": {"incumbents": []}}
        for label, kwargs in cases.items():
            with self.subTest(incumbents=label):
                self.assertNotIn("protection_zone", self._run(**kwargs))

    def test_explicit_result_has_no_zone_when_the_config_declares_none(self):  # PZ-20
        result = self._run(with_zone=False, incumbents=["AAA3"])
        self.assertNotIn("protection_zone", result)


if __name__ == "__main__":
    unittest.main()
