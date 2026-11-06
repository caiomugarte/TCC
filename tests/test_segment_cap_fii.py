"""FII selector wiring of the segment cap (issue #56, SEG-15..SEG-18, SEG-26..SEG-29, SEG-37, SEG-40, SEG-41)."""

from collections import Counter
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import pandas as pd
from pandas.testing import assert_frame_equal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))

from core.metrics import hhi_sector  # noqa: E402
from core.optimizer import GeneticAlgorithm  # noqa: E402
from core.protection_zone import SegmentCapInfeasibleError, select_with_fallback  # noqa: E402
from fetch_status_invest_fii import SOURCE_COLUMNS  # noqa: E402
from fii_selection import (  # noqa: E402
    FII_PROFILE_WEIGHTS,
    FiiSelectionError,
    _consensus,
    run_fii_selection,
)


def _fii_rows(rows) -> pd.DataFrame:
    """Raw FII CSV frame from (ticker, segment, quality) rows; a higher quality scores higher."""

    records = []
    for ticker, sector, quality in rows:
        records.append(
            {
                "TICKER": ticker,
                "PRECO": 100.0,
                "ULTIMO DIVIDENDO": 1.0,
                "DY": 4.0 + quality,
                "VALOR PATRIMONIAL COTA": 100.0,
                "P/VP": 1.3 - quality / 20.0,
                "LIQUIDEZ MEDIA DIARIA": (40.0 + 10.0 * quality) * 10000,
                "PERCENTUAL EM CAIXA": 0.5 + quality / 4.0,
                "CAGR DIVIDENDOS 3 ANOS": 4.0 + quality,
                "CAGR VALOR COTA 3 ANOS": 4.0 + quality,
                "PATRIMONIO": (40.0 + 10.0 * quality) * 1000000,
                "N COTISTAS": (40.0 + 10.0 * quality) * 10,
                "GESTAO": "",
                "N COTAS": (40.0 + 10.0 * quality) * 1000,
                "SETOR": sector,
            }
        )
    return pd.DataFrame(records, columns=list(SOURCE_COLUMNS) + ["SETOR"])


# Six "Papéis" funds hold the best qualities; five other segments follow.
DOMINANT_SEGMENT_ROWS = [
    ("PAP111", "Papéis", 10.0),
    ("PAP211", "Papéis", 9.5),
    ("PAP311", "Papéis", 9.0),
    ("PAP411", "Papéis", 8.5),
    ("PAP511", "Papéis", 8.0),
    ("PAP611", "Papéis", 7.5),
    ("LAJ111", "Lajes", 5.0),
    ("LOG111", "Logística", 4.5),
    ("SHO111", "Shoppings", 4.0),
    ("HIB111", "Híbrido", 3.5),
    ("IND111", "Indefinido", 3.0),
]


def _config(n_assets=5, **extra) -> dict:
    config = {
        "factor_weights": dict(FII_PROFILE_WEIGHTS["conservador"]),
        "liquidity_and_size_filters": {},
        "n_assets": n_assets,
        "lambda": 0.0,
        "system_ga_config": {"generations": 6, "pop_size": 8},
    }
    config.update(extra)
    return config


class _RecordingGeneticAlgorithm(GeneticAlgorithm):
    """Captures the ``max_per_segment`` each GA run is constructed with."""

    constructed_with: list = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        type(self).constructed_with.append(kwargs.get("max_per_segment"))


def _run(rows, config, **kwargs):
    with tempfile.TemporaryDirectory() as directory:
        input_path = Path(directory) / "fii.csv"
        _fii_rows(rows).to_csv(input_path, index=False)
        return run_fii_selection(input_path=input_path, selection_config=config, **kwargs)


class GaCapWiringTests(unittest.TestCase):
    def test_config_cap_reaches_every_ga_run(self):  # SEG-27
        _RecordingGeneticAlgorithm.constructed_with = []
        with mock.patch("fii_selection.GeneticAlgorithm", _RecordingGeneticAlgorithm):
            _run(DOMINANT_SEGMENT_ROWS, _config(max_per_segment=2), n_runs=3)
        self.assertEqual(_RecordingGeneticAlgorithm.constructed_with, [2, 2, 2])

    def test_config_without_cap_constructs_the_ga_with_none(self):  # SEG-27, SEG-32
        _RecordingGeneticAlgorithm.constructed_with = []
        with mock.patch("fii_selection.GeneticAlgorithm", _RecordingGeneticAlgorithm):
            _run(DOMINANT_SEGMENT_ROWS, _config(), n_runs=1)
        self.assertEqual(_RecordingGeneticAlgorithm.constructed_with, [None])


# _consensus at a61164a (py/fii_selection.py:555-597), copied verbatim as the SEG-18 oracle.
def _pre_change_consensus(
    portfolios,
    df_ranked,
    n_assets,
    *,
    incumbents=None,
    protection_zone=None,
    hhi_max=None,
):
    counts = Counter(
        ticker
        for portfolio in portfolios
        for ticker in portfolio["TICKER"].tolist()
    )
    scores = df_ranked.set_index("TICKER")["SCORE"]
    zone_record = None
    if incumbents and protection_zone is not None:
        # AD-022 zone (#46). FIIs take no base-company dedup.
        selection, zone_record = select_with_fallback(
            incumbents,
            {ticker: count / len(portfolios) for ticker, count in counts.items()},
            {ticker: float(scores[ticker]) for ticker in counts},
            n_assets,
            protection_zone,
            eligible=set(df_ranked["TICKER"]),
            dedup_base_company=False,
            hhi_of=lambda tickers: hhi_sector(df_ranked[df_ranked["TICKER"].isin(list(tickers))]),
            hhi_max=None if hhi_max is None else float(hhi_max),
        )
        selected = list(selection.tickers)
    else:
        selected = sorted(
            counts,
            key=lambda ticker: (-counts[ticker], -float(scores[ticker]), ticker),
        )[:n_assets]
    result = df_ranked[df_ranked["TICKER"].isin(selected)].copy()
    result["FREQUENCY"] = result["TICKER"].map(
        lambda ticker: counts[ticker] / len(portfolios)
    )
    result = result.sort_values(["FREQUENCY", "SCORE", "TICKER"], ascending=[False, False, True])
    if zone_record is not None:
        result.attrs["protection_zone"] = zone_record
    return result


SEGMENT_NAMES = {"P": "Papéis", "Q": "Lajes", "R": "Logística", "S": "Shoppings",
                 "T": "Híbrido", "U": "Recebíveis", "I": "Indefinido"}


def _ranked(tickers, scores=None) -> pd.DataFrame:
    """df_ranked over tickers whose leading letter names the segment."""

    scores = scores or {}
    return pd.DataFrame(
        {
            "TICKER": list(tickers),
            "SCORE": [scores.get(t, 1.0 - 0.01 * i) for i, t in enumerate(tickers)],
            "SETOR": [SEGMENT_NAMES[t[0]] for t in tickers],
        }
    )


def _portfolios_from_counts(counts) -> list[pd.DataFrame]:
    """GA runs in which each ticker appears ``counts[ticker]`` times."""

    runs = max(counts.values())
    return [
        pd.DataFrame({"TICKER": [t for t, c in counts.items() if c > run]}) for run in range(runs)
    ]


ZONE = {"K": 7, "delta": 0.11, "k_offset": 3}


class ConsensusSegmentCapTests(unittest.TestCase):
    def test_pure_consensus_skips_the_third_ticker_of_a_full_segment(self):  # SEG-15, SEG-16
        counts = {"P1": 5, "P2": 4, "P3": 3, "T1": 2, "Q1": 2}
        ranked = _ranked(counts, scores={"Q1": 0.9, "T1": 0.5})
        result = _consensus(_portfolios_from_counts(counts), ranked, 3, max_per_segment=2)
        self.assertEqual(result["TICKER"].tolist(), ["P1", "P2", "Q1"])

    def test_pure_consensus_caps_indefinido(self):  # SEG-17
        counts = {"I1": 5, "I2": 4, "I3": 3, "Q1": 2}
        result = _consensus(_portfolios_from_counts(counts), _ranked(counts), 3, max_per_segment=2)
        self.assertEqual(result["TICKER"].tolist(), ["I1", "I2", "Q1"])
        self.assertEqual((result["SETOR"] == "Indefinido").sum(), 2)

    def test_pure_consensus_without_cap_equals_pre_change(self):  # SEG-18, SEG-24
        counts = {"P1": 5, "P2": 4, "P3": 3, "T1": 2, "Q1": 2, "R1": 1}
        ranked = _ranked(counts, scores={"Q1": 0.9, "T1": 0.5})
        portfolios = _portfolios_from_counts(counts)
        cases = {
            "pure, omitted": ({}, {}),
            "pure, None": ({"max_per_segment": None}, {}),
            "zone, None": (
                {"max_per_segment": None, "incumbents": ["P3", "R1"], "protection_zone": ZONE},
                {"incumbents": ["P3", "R1"], "protection_zone": ZONE},
            ),
            "zone with hhi_max, None": (
                {"max_per_segment": None, "incumbents": ["P3"], "protection_zone": ZONE, "hhi_max": 0.9},
                {"incumbents": ["P3"], "protection_zone": ZONE, "hhi_max": 0.9},
            ),
        }
        for label, (kwargs, pre_kwargs) in cases.items():
            with self.subTest(case=label):
                actual = _consensus(portfolios, ranked, 3, **kwargs)
                expected = _pre_change_consensus(portfolios, ranked, 3, **pre_kwargs)
                assert_frame_equal(actual, expected)
                self.assertEqual(dict(actual.attrs), dict(expected.attrs))

    def test_pure_consensus_skips_a_pair_that_breaks_hhi_max(self):  # SEG-37
        counts = {"P1": 7, "P2": 6, "Q1": 5, "R1": 4, "S1": 3, "T1": 2, "U1": 2}
        ranked = _ranked(counts, scores={"T1": 0.9, "U1": 0.1})
        result = _consensus(
            _portfolios_from_counts(counts), ranked, 5, max_per_segment=2, hhi_max=0.25
        )
        self.assertEqual(result["TICKER"].tolist(), ["P1", "Q1", "R1", "S1", "T1"])

    def test_pure_consensus_with_cap_and_no_hhi_max_keeps_the_pair(self):  # SEG-41
        counts = {"P1": 7, "P2": 6, "Q1": 5, "R1": 4, "S1": 3, "T1": 2, "U1": 2}
        ranked = _ranked(counts, scores={"T1": 0.9, "U1": 0.1})
        result = _consensus(_portfolios_from_counts(counts), ranked, 5, max_per_segment=2)
        self.assertEqual(result["TICKER"].tolist(), ["P1", "P2", "Q1", "R1", "S1"])

    def test_infeasible_consensus_raises_fii_selection_error(self):  # SEG-40 (pure path)
        counts = {"P1": 5, "P2": 4, "Q1": 3, "Q2": 2, "R1": 1}
        with self.assertRaises(FiiSelectionError) as caught:
            _consensus(
                _portfolios_from_counts(counts), _ranked(counts), 4, max_per_segment=2, hhi_max=0.25
            )
        self.assertNotIsInstance(caught.exception, SegmentCapInfeasibleError)
        self.assertRegex(str(caught.exception), "max_per_segment")
        self.assertRegex(str(caught.exception), "hhi_max")

    def test_infeasible_zone_consensus_raises_fii_selection_error(self):  # SEG-40 (zone path, review S7)
        counts = {"P1": 5, "P2": 4, "Q1": 3, "Q2": 2, "R1": 1}
        with self.assertRaises(FiiSelectionError) as caught:
            _consensus(
                _portfolios_from_counts(counts), _ranked(counts), 4,
                incumbents=["P1"], protection_zone=ZONE, max_per_segment=2, hhi_max=0.25,
            )
        self.assertNotIsInstance(caught.exception, SegmentCapInfeasibleError)
        self.assertIsInstance(caught.exception.__cause__, SegmentCapInfeasibleError)
        self.assertRegex(str(caught.exception), "max_per_segment")
        self.assertRegex(str(caught.exception), "hhi_max")

    def test_zone_consensus_passes_the_segment_map(self):  # SEG-19 (selector level)
        counts = {"P1": 5, "P2": 4, "P3": 3, "Q1": 2, "R1": 1}
        result = _consensus(
            _portfolios_from_counts(counts), _ranked(counts), 4,
            incumbents=["P1", "P2", "P3"], protection_zone={"K": 5, "delta": 0.11, "k_offset": 1},
            max_per_segment=2,
        )
        record = result.attrs["protection_zone"]
        self.assertEqual(record["departures"], {"P3": "saiu: teto do segmento"})
        self.assertEqual(sorted(result["TICKER"]), ["P1", "P2", "Q1", "R1"])


if __name__ == "__main__":
    unittest.main()
