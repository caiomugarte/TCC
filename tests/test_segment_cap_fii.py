"""FII selector wiring of the segment cap (issue #56, SEG-15..SEG-18, SEG-26..SEG-29, SEG-37, SEG-40, SEG-41)."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))

from core.optimizer import GeneticAlgorithm  # noqa: E402
from fetch_status_invest_fii import SOURCE_COLUMNS  # noqa: E402
from fii_selection import FII_PROFILE_WEIGHTS, run_fii_selection  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
