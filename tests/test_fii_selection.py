from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))

from fetch_status_invest_fii import SOURCE_COLUMNS  # noqa: E402
from fii_selection import (  # noqa: E402
    FII_PROFILE_WEIGHTS,
    FiiSelectionError,
    apply_fii_eligibility,
    build_fii_scores,
    load_fii_data,
    preprocess_fii,
    run_fii_selection,
)
from run_fii import build_parser, execution_settings  # noqa: E402


def _fixture_frame():
    rows = []
    values = [
        ("AAA11", "Segment A", 0.80, 12.0, 10.0, 100.0, 3.0),
        ("BBB11", "Segment A", 1.00, 8.0, 8.0, 80.0, 2.0),
        ("CCC11", "Segment A", 1.20, 5.0, 6.0, 60.0, 1.0),
        ("DDD11", "Segment B", 0.85, 11.0, 9.0, 90.0, 2.5),
        ("EEE11", "Segment B", 1.05, 7.0, 7.0, 70.0, 1.5),
        ("FFF11", "Segment B", 1.25, 4.0, 5.0, 50.0, 0.5),
    ]
    for ticker, sector, p_vp, dy, dividend_cagr, liquidity, cash in values:
        rows.append(
            {
                "TICKER": ticker,
                "PRECO": 100.0,
                "ULTIMO DIVIDENDO": 1.0,
                "DY": dy,
                "VALOR PATRIMONIAL COTA": 100.0,
                "P/VP": p_vp,
                "LIQUIDEZ MEDIA DIARIA": liquidity * 10000,
                "PERCENTUAL EM CAIXA": cash,
                "CAGR DIVIDENDOS 3 ANOS": dividend_cagr,
                "CAGR VALOR COTA 3 ANOS": dividend_cagr,
                "PATRIMONIO": liquidity * 1000000,
                "N COTISTAS": liquidity * 10,
                "GESTAO": "",
                "N COTAS": liquidity * 1000,
                "SETOR": sector,
            }
        )
    return pd.DataFrame(rows, columns=list(SOURCE_COLUMNS) + ["SETOR"])


class FiiSelectionTests(unittest.TestCase):
    def test_profile_weights_are_independent_and_complete(self):
        for weights in FII_PROFILE_WEIGHTS.values():
            self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_preprocess_and_score_keep_real_segments(self):
        ranked = build_fii_scores(preprocess_fii(_fixture_frame()), "caio")

        self.assertEqual(set(ranked["SETOR"]), {"Segment A", "Segment B"})
        self.assertTrue(ranked["SCORE"].notna().all())
        self.assertEqual(ranked.iloc[0]["TICKER"], "AAA11")

    def test_run_writes_selected_assets_without_allocation_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            input_path = directory / "fii.csv"
            output_path = directory / "portfolio.json"
            processed_path = directory / "clean.csv"
            _fixture_frame().to_csv(input_path, index=False)

            result = run_fii_selection(
                "caio",
                input_path=input_path,
                output_path=output_path,
                processed_path=processed_path,
                n_runs=2,
                ga_config={
                    "n_assets": 2,
                    "lambda": 0.0,
                    "generations": 4,
                    "pop_size": 4,
                },
            )

            portfolio = result["portfolio"]
            self.assertEqual(len(portfolio), 2)
            self.assertNotIn("WEIGHT", portfolio.columns)
            self.assertTrue(output_path.exists())
            self.assertTrue(processed_path.exists())

    def test_run_supports_adaptive_consensus(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            input_path = directory / "fii.csv"
            _fixture_frame().to_csv(input_path, index=False)

            result = run_fii_selection(
                "caio",
                input_path=input_path,
                output_path=directory / "portfolio.json",
                processed_path=directory / "clean.csv",
                n_runs=4,
                adaptive_mode=True,
                min_runs=2,
                target_cv=999.0,
                target_jaccard=0.0,
                ga_config={
                    "n_assets": 2,
                    "lambda": 0.0,
                    "generations": 4,
                    "pop_size": 4,
                },
            )

            self.assertEqual(result["n_runs"], 2)
            self.assertIn("fitness_cv", result["stability"])

    def test_run_supports_parallel_consensus(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            input_path = directory / "fii.csv"
            _fixture_frame().to_csv(input_path, index=False)

            result = run_fii_selection(
                "caio",
                input_path=input_path,
                output_path=directory / "portfolio.json",
                processed_path=directory / "clean.csv",
                n_runs=2,
                parallel=True,
                ga_config={
                    "n_assets": 2,
                    "lambda": 0.0,
                    "generations": 2,
                    "pop_size": 4,
                },
            )

            self.assertEqual(result["n_runs"], 2)
            self.assertEqual(len(result["portfolio"]), 2)

    def test_cli_presets_match_stock_run_budgets(self):
        parser = build_parser()
        self.assertEqual(execution_settings(parser.parse_args(["--once"]))["n_runs"], 1)
        self.assertEqual(execution_settings(parser.parse_args(["--production"]))["n_runs"], 100)
        self.assertEqual(execution_settings(parser.parse_args(["--max-quality"]))["n_runs"], 150)

    def test_unlabelled_input_requires_explicit_segment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "segment.csv"
            _fixture_frame().drop(columns=["SETOR"]).to_csv(path, index=False)
            with self.assertRaisesRegex(FiiSelectionError, "must contain SETOR"):
                load_fii_data(path)


# Provenance: the rows below are the normative eligibility matrix frozen by
# .specs/features/fii-universe-feasibility/spec.md (Assumption 2) and
# docs/adr/0002-fii-universe-feasibility-policy.md, measured on the committed
# snapshot data/raw/status_invest_fii.csv (588 FIIs) with the production
# pipeline run_fii_selection uses (load_fii_data -> preprocess_fii ->
# build_fii_scores, py/fii_selection.py:566-572). "eligible" counts the
# post-eligibility rows; "ranked" counts the post-scoring rows the GA can
# actually pick from (scoring drops rows with unusable fundamentals).
NORMATIVE_MATRIX = (
    # (size_min, liq_min, eligible, ranked)
    (5.0e9, 2.0e6, 7, 7),
    (3.0e9, 2.0e6, 13, 12),
    (2.0e9, 2.0e6, 28, 26),
    (1.0e9, 2.0e6, 50, 47),
    (3.4e9, 1.4e6, 10, 9),
    (1.0e9, 5.0e5, 61, 57),
    (2.0e8, 5.0e4, 162, 137),
)


def _production_counts(filters):
    raw = load_fii_data()
    clean = preprocess_fii(raw, eligibility_filters=filters)
    ranked = build_fii_scores(
        clean, None, factor_weights=FII_PROFILE_WEIGHTS["conservador"]
    )
    return len(clean), len(ranked)


class FiiNormativeMatrixTests(unittest.TestCase):
    def _assert_counts(self, size_min, liq_min, eligible, ranked):
        self.assertEqual(
            _production_counts({"size_min": size_min, "liq_min": liq_min}),
            (eligible, ranked),
        )

    def test_matrix_row_5e9_2e6(self):
        self._assert_counts(5.0e9, 2.0e6, 7, 7)

    def test_matrix_row_3e9_2e6(self):
        self._assert_counts(3.0e9, 2.0e6, 13, 12)

    def test_matrix_row_2e9_2e6(self):
        self._assert_counts(2.0e9, 2.0e6, 28, 26)

    def test_matrix_row_1e9_2e6(self):
        self._assert_counts(1.0e9, 2.0e6, 50, 47)

    def test_matrix_row_34e9_14e6(self):
        self._assert_counts(3.4e9, 1.4e6, 10, 9)

    def test_matrix_row_1e9_5e5_moderado_anchor(self):
        self._assert_counts(1.0e9, 5.0e5, 61, 57)

    def test_matrix_row_2e8_5e4_arrojado_anchor(self):
        self._assert_counts(2.0e8, 5.0e4, 162, 137)

    def test_eligibility_only_counts_match_preprocess_pipeline(self):
        raw = load_fii_data()
        for size_min, liq_min, eligible, _ranked in NORMATIVE_MATRIX:
            with self.subTest(size_min=size_min, liq_min=liq_min):
                filters = {"size_min": size_min, "liq_min": liq_min}
                only_eligibility = apply_fii_eligibility(raw, filters=filters)
                preprocessed = preprocess_fii(raw, eligibility_filters=filters)
                self.assertEqual(len(only_eligibility), len(preprocessed))
                self.assertEqual(len(only_eligibility), eligible)

    def test_production_entry_point_reports_ranked_universe(self):
        result = run_fii_selection(
            selection_config={
                "factor_weights": dict(FII_PROFILE_WEIGHTS["conservador"]),
                "liquidity_and_size_filters": {
                    "size_min": 3_000_000_000,
                    "liq_min": 2_000_000,
                },
                "n_assets": 10,
                "lambda": 0.5,
                "system_ga_config": {},
            },
            n_runs=1,
            ga_config={"generations": 2, "pop_size": 8},
        )

        self.assertEqual(result["n_candidates"], 12)
        self.assertIsNone(result["output_path"])
        self.assertIsNone(result["processed_path"])
        self.assertIsNone(result["workspace"])


if __name__ == "__main__":
    unittest.main()
