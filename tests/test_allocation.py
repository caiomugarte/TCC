import sys
import unittest
from datetime import date, timedelta
import json
from pathlib import Path
import tempfile
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))

from core.allocation import (  # noqa: E402
    AllocationError,
    Candidate,
    DailyReturn,
    Metrics,
    bounded_simplex_grid,
    class_hhi,
    evaluate_candidates,
    non_dominated_frontier,
    penalty_sweep,
    portfolio_metrics,
    risk_contributions,
    select_frontier_points,
    simulate_portfolio,
    simplex_grid,
    validate_weights,
)
from allocation_data import (  # noqa: E402
    SnapshotError,
    load_portfolio_tickers,
    load_snapshot_bundle,
)
from allocation_config import (  # noqa: E402
    ALLOCATION_PROFILE_ANCHORS,
    ASSET_CLASSES,
)
from allocation_profiles import (  # noqa: E402
    AllocationProfile,
    AllocationProfileError,
    build_anchor_profiles,
    interpolate_profile,
)
from run_allocation import build_parser as build_allocation_parser  # noqa: E402
from run_allocation import resolve_suitability_score  # noqa: E402
from fetch_allocation_snapshot import (  # noqa: E402
    SnapshotFetchError,
    _is_permanent_ticker_failure,
    build_parser as build_snapshot_parser,
    fetch_snapshot,
)
from pipelines.asset_allocation import (  # noqa: E402
    evaluate_crypto_weight_scenarios,
    optimize_window,
    run_allocation,
    write_outputs,
)


class AllocationCoreTests(unittest.TestCase):
    def test_portfolio_ticker_loader_rejects_invalid_identity(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            valid_path = directory / "valid.json"
            valid_path.write_text(
                json.dumps([{"TICKER": " aaa "}, {"TICKER": "BBB", "SCORE": 0.9}]),
                encoding="utf-8",
            )
            self.assertEqual(load_portfolio_tickers(valid_path, "FII"), ("AAA", "BBB"))

            for index, payload in enumerate(
                (
                    {},
                    [],
                    [{"TICKER": "AAA"}, {"TICKER": "aaa"}],
                    [{"TICKER": " "}],
                    [{"SCORE": 1.0}],
                    ["AAA"],
                )
            ):
                path = directory / f"invalid-{index}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(SnapshotError):
                    load_portfolio_tickers(path, "FII")

    def test_snapshot_parser_exposes_fii_portfolio_and_optional_ifix(self):
        snapshot_args = build_snapshot_parser().parse_args(["--skip-ifix"])
        self.assertEqual(
            snapshot_args.fii_portfolio.name,
            "carteira_fii_caio_consensus.json",
        )
        self.assertTrue(snapshot_args.skip_ifix)

        allocation_args = build_allocation_parser().parse_args([])
        self.assertEqual(
            allocation_args.fii_portfolio.name,
            "carteira_fii_caio_consensus.json",
        )

    def test_ticker_no_data_is_permanent_but_rate_limit_is_not(self):
        self.assertTrue(
            _is_permanent_ticker_failure(
                SnapshotFetchError("could not fetch Yahoo: HTTP Error 404: Not Found")
            )
        )
        self.assertFalse(
            _is_permanent_ticker_failure(
                SnapshotFetchError("could not fetch Yahoo: HTTP Error 429: Too Many Requests")
            )
        )

    def test_snapshot_fetch_auto_skips_ticker_with_no_yahoo_history(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            portfolio_path = directory / "portfolio.json"
            portfolio_path.write_text(
                json.dumps([{"TICKER": "AAA"}, {"TICKER": "BBB"}]),
                encoding="utf-8",
            )
            levels = {
                date(2020, 1, 1): 100.0,
                date(2020, 1, 2): 101.0,
            }

            def fake_yahoo(symbol, *args, **kwargs):
                if symbol == "BBB.SA":
                    raise SnapshotFetchError(
                        "could not fetch Yahoo: HTTP Error 404: Not Found"
                    )
                return levels

            with patch("fetch_allocation_snapshot.fetch_yahoo_levels", side_effect=fake_yahoo):
                with patch("fetch_allocation_snapshot.fetch_ifix_levels", return_value=levels):
                    with patch("fetch_allocation_snapshot.fetch_bcb_cdi_levels", return_value=levels):
                        with patch("fetch_allocation_snapshot.fetch_ptax_levels", return_value=levels):
                            result = fetch_snapshot(
                                directory / "snapshot",
                                portfolio_path,
                                date(2020, 1, 1),
                                date(2020, 1, 2),
                            )

            self.assertEqual(result["tickers"], ("AAA",))
            self.assertEqual(result["skipped_tickers"], ["BBB"])
            self.assertIn("HTTP Error 404", result["skipped_ticker_reasons"]["BBB"])
            metadata = json.loads(
                (directory / "snapshot" / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["skipped_tickers"], ["BBB"])
            self.assertEqual(
                metadata["sources"]["caio_stocks"]["symbols"], ["AAA.SA"]
            )

    def test_snapshot_fetch_writes_selected_fii_levels_without_ifix(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            stock_path = directory / "stocks.json"
            stock_path.write_text(json.dumps([{"TICKER": "AAA"}]), encoding="utf-8")
            fii_path = directory / "fiis.json"
            fii_path.write_text(
                json.dumps([{"TICKER": "FII1", "SCORE": 0.9}, {"TICKER": "FII2"}]),
                encoding="utf-8",
            )
            levels = {date(2020, 1, 1): 100.0, date(2020, 1, 2): 101.0}

            def fake_yahoo(symbol, *args, **kwargs):
                self.assertTrue(kwargs["adjusted"] if symbol.endswith(".SA") else True)
                return levels

            with patch("fetch_allocation_snapshot.fetch_yahoo_levels", side_effect=fake_yahoo):
                with patch("fetch_allocation_snapshot.fetch_ifix_levels") as ifix:
                    with patch(
                        "fetch_allocation_snapshot.fetch_bcb_cdi_levels",
                        return_value=levels,
                    ):
                        with patch(
                            "fetch_allocation_snapshot.fetch_ptax_levels",
                            return_value=levels,
                        ):
                            result = fetch_snapshot(
                                directory / "snapshot",
                                stock_path,
                                date(2020, 1, 1),
                                date(2020, 1, 2),
                                fii_portfolio_path=fii_path,
                                include_ifix=False,
                            )

            snapshot = directory / "snapshot"
            metadata = json.loads((snapshot / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(result["fii_tickers"], ("FII1", "FII2"))
            self.assertEqual(
                metadata["sources"]["caio_fiis"]["portfolio_path"], str(fii_path)
            )
            self.assertEqual(
                metadata["sources"]["caio_fiis"]["symbols"], ["FII1.SA", "FII2.SA"]
            )
            self.assertTrue((snapshot / "caio_fiis.csv").exists())
            self.assertFalse((snapshot / "ifix.csv").exists())
            ifix.assert_not_called()

    def test_snapshot_fetch_skips_permanent_fii_failure_and_records_reason(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            stock_path = directory / "stocks.json"
            stock_path.write_text(json.dumps([{"TICKER": "AAA"}]), encoding="utf-8")
            fii_path = directory / "fiis.json"
            fii_path.write_text(
                json.dumps([{"TICKER": "FII1"}, {"TICKER": "FII2"}]),
                encoding="utf-8",
            )
            levels = {date(2020, 1, 1): 100.0, date(2020, 1, 2): 101.0}

            def fake_yahoo(symbol, *args, **kwargs):
                if symbol == "FII2.SA":
                    raise SnapshotFetchError("Yahoo returned no usable observations for FII2.SA")
                return levels

            with patch("fetch_allocation_snapshot.fetch_yahoo_levels", side_effect=fake_yahoo):
                with patch("fetch_allocation_snapshot.fetch_ifix_levels", return_value=levels):
                    with patch(
                        "fetch_allocation_snapshot.fetch_bcb_cdi_levels",
                        return_value=levels,
                    ):
                        with patch(
                            "fetch_allocation_snapshot.fetch_ptax_levels",
                            return_value=levels,
                        ):
                            result = fetch_snapshot(
                                directory / "snapshot",
                                stock_path,
                                date(2020, 1, 1),
                                date(2020, 1, 2),
                                fii_portfolio_path=fii_path,
                            )

            metadata = json.loads(
                (directory / "snapshot" / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(result["fii_tickers"], ("FII1",))
            self.assertEqual(result["fii_skipped_tickers"], ["FII2"])
            self.assertIn("no usable observations", result["fii_skipped_ticker_reasons"]["FII2"])
            self.assertEqual(metadata["fii_skipped_tickers"], ["FII2"])
            self.assertEqual(
                metadata["sources"]["caio_fiis"]["skipped_tickers"], ["FII2"]
            )

    def test_snapshot_fetch_aborts_transient_fii_failure_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            stock_path = directory / "stocks.json"
            stock_path.write_text(json.dumps([{"TICKER": "AAA"}]), encoding="utf-8")
            fii_path = directory / "fiis.json"
            fii_path.write_text(
                json.dumps([{"TICKER": "FII1"}, {"TICKER": "FII2"}]),
                encoding="utf-8",
            )
            levels = {date(2020, 1, 1): 100.0, date(2020, 1, 2): 101.0}

            def fake_yahoo(symbol, *args, **kwargs):
                if symbol == "FII2.SA":
                    raise SnapshotFetchError("Yahoo request timed out")
                return levels

            with patch("fetch_allocation_snapshot.fetch_yahoo_levels", side_effect=fake_yahoo):
                with self.assertRaises(SnapshotFetchError):
                    fetch_snapshot(
                        directory / "snapshot",
                        stock_path,
                        date(2020, 1, 1),
                        date(2020, 1, 2),
                        fii_portfolio_path=fii_path,
                        include_ifix=False,
                    )
            self.assertFalse((directory / "snapshot").exists())

    def test_snapshot_fetch_rejects_malformed_fii_before_network(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            stock_path = directory / "stocks.json"
            stock_path.write_text(json.dumps([{"TICKER": "AAA"}]), encoding="utf-8")
            fii_path = directory / "fiis.json"
            fii_path.write_text("{not-json", encoding="utf-8")

            with patch("fetch_allocation_snapshot.fetch_yahoo_levels") as yahoo:
                with self.assertRaises(SnapshotError):
                    fetch_snapshot(
                        directory / "snapshot",
                        stock_path,
                        date(2020, 1, 1),
                        date(2020, 1, 2),
                        fii_portfolio_path=fii_path,
                        include_ifix=False,
                    )
            yahoo.assert_not_called()

    def test_allocation_profile_interpolates_between_three_anchors(self):
        anchors = {
            "conservador": AllocationProfile(
                "conservador", 0.0, 0.10, 0.15, 0.20, 0.50, "fixture"
            ),
            "moderado": AllocationProfile(
                "moderado", 0.5, 0.20, 0.30, 0.40, 0.25, "fixture"
            ),
            "arrojado": AllocationProfile(
                "arrojado", 1.0, 0.30, 0.45, 0.60, 0.10, "fixture"
            ),
        }
        profile = interpolate_profile(0.25, anchors)
        self.assertAlmostEqual(profile.volatility_cap, 0.15)
        self.assertAlmostEqual(profile.drawdown_cap, 0.225)
        self.assertAlmostEqual(profile.crypto_risk_contribution_cap, 0.30)
        self.assertAlmostEqual(profile.hhi_penalty, 0.375)

    def test_allocation_profile_rejects_invalid_score_or_anchor_order(self):
        anchors = {
            "conservador": AllocationProfile(
                "conservador", 0.0, 0.10, 0.15, 0.20, 0.50, "fixture"
            ),
            "moderado": AllocationProfile(
                "moderado", 0.5, 0.20, 0.30, 0.40, 0.25, "fixture"
            ),
            "arrojado": AllocationProfile(
                "arrojado", 1.0, 0.30, 0.45, 0.60, 0.10, "fixture"
            ),
        }
        with self.assertRaises(AllocationProfileError):
            interpolate_profile(1.1, anchors)
        invalid = dict(anchors)
        invalid["moderado"] = AllocationProfile(
            "moderado", 0.4, 0.20, 0.30, 0.40, 0.25, "fixture"
        )
        with self.assertRaises(AllocationProfileError):
            interpolate_profile(0.5, invalid)

    def test_configured_caio_score_interpolates_profile_policy(self):
        profile = interpolate_profile(
            0.831,
            build_anchor_profiles(ALLOCATION_PROFILE_ANCHORS),
            name="caio_new",
        )
        self.assertAlmostEqual(profile.volatility_cap, 0.1831)
        self.assertAlmostEqual(profile.drawdown_cap, 0.3162)
        self.assertAlmostEqual(profile.crypto_risk_contribution_cap, 0.4662)
        self.assertAlmostEqual(profile.hhi_penalty, 0.1507)
        self.assertAlmostEqual(profile.risk_adjusted_weights["return"], 0.6324)
        self.assertAlmostEqual(profile.risk_adjusted_weights["volatility"], 0.1838)
        self.assertAlmostEqual(profile.risk_adjusted_weights["drawdown"], 0.1838)

    def test_caio_last_defaults_to_conservative_allocation_score(self):
        score = resolve_suitability_score("caio_last", None)
        profile = interpolate_profile(
            score,
            build_anchor_profiles(ALLOCATION_PROFILE_ANCHORS),
            name="caio_last",
        )
        self.assertEqual(score, 0.0)
        self.assertAlmostEqual(profile.volatility_cap, 0.10)
        self.assertAlmostEqual(profile.drawdown_cap, 0.15)
        self.assertAlmostEqual(profile.crypto_risk_contribution_cap, 0.25)
        self.assertEqual(resolve_suitability_score("caio_last", 0.831), 0.831)
        self.assertIsNone(resolve_suitability_score("caio_new", None))

    def test_run_allocation_records_personalized_profile_parameters(self):
        rows = tuple(
            DailyReturn(
                current_date,
                {name: 0.0 for name in ASSET_CLASSES},
            )
            for current_date in (
                date(2020, 1, 1),
                date(2025, 1, 1),
                date(2027, 1, 1),
                date(2030, 1, 1),
            )
        )
        profile = AllocationProfile(
            "caio", 0.25, 0.15, 0.20, 0.30, 0.35, "fixture calibration"
        )
        result = run_allocation(rows, allocation_profile=profile)
        self.assertEqual(result["allocation_profile"]["status"], "personalized")
        self.assertEqual(
            result["allocation_profile"]["parameters"]["name"],
            "caio",
        )
        self.assertEqual(result["config"]["volatility_cap"], 0.15)
        self.assertEqual(result["config"]["drawdown_cap"], 0.20)
        self.assertIn("profile_winner", result["current_target"]["selected"])

    def test_optimizer_enforces_minimum_weight_for_every_class(self):
        rows = tuple(
            DailyReturn(
                date(2020, 1, 1) + timedelta(days=index),
                {name: 0.0 for name in ASSET_CLASSES},
            )
            for index in range(4)
        )
        window = optimize_window(
            rows,
            ASSET_CLASSES,
            {
                "volatility_cap": 0.20,
                "drawdown_cap": 0.30,
                "minimum_class_weight": 0.05,
                "coarse_step": 0.05,
                "refinement_step": 0.01,
                "refinement_radius": 0.02,
                "hhi_penalties": (0.0,),
            },
        )
        self.assertTrue(
            all(
                weight >= 0.05 - 1e-12
                for candidate in window.candidates
                for weight in candidate.weights
            )
        )

    def test_weights_are_validated_and_hhi_is_class_based(self):
        classes = ("stocks", "fixed_income")
        self.assertEqual(validate_weights({"stocks": 0.0, "fixed_income": 1.0}, classes), (0.0, 1.0))
        self.assertEqual(class_hhi({"stocks": 0.5, "fixed_income": 0.5}, classes), 0.5)
        with self.assertRaises(AllocationError):
            validate_weights({"stocks": 0.8, "fixed_income": 0.3}, classes)

    def test_risk_contributions_measure_variance_not_weight(self):
        classes = ("stocks", "fixed_income")
        rows = [
            DailyReturn(
                date(2020, 1, 1) + timedelta(days=index),
                {"stocks": stock_return, "fixed_income": fixed_return},
            )
            for index, (stock_return, fixed_return) in enumerate(
                ((0.10, 0.02), (-0.10, -0.02), (0.10, 0.02), (-0.10, -0.02))
            )
        ]
        contributions = risk_contributions(
            rows,
            {"stocks": 0.5, "fixed_income": 0.5},
            classes,
        )
        self.assertAlmostEqual(contributions["stocks"], 5 / 6)
        self.assertAlmostEqual(contributions["fixed_income"], 1 / 6)
        self.assertAlmostEqual(sum(contributions.values()), 1.0)

    def test_risk_contribution_cap_filters_candidates_without_weight_cap(self):
        classes = ("stocks", "fixed_income")
        rows = [
            DailyReturn(
                date(2020, 1, 1) + timedelta(days=index),
                {"stocks": stock_return, "fixed_income": fixed_return},
            )
            for index, (stock_return, fixed_return) in enumerate(
                ((0.10, 0.02), (-0.10, -0.02), (0.10, 0.02), (-0.10, -0.02))
            )
        ]
        candidate = ((0.5, 0.5),)
        unrestricted = evaluate_candidates(rows, classes, candidate, 10.0, 1.0)
        capped = evaluate_candidates(
            rows,
            classes,
            candidate,
            10.0,
            1.0,
            risk_contribution_cap=0.25,
        )
        self.assertTrue(unrestricted[0].feasible)
        self.assertFalse(capped[0].feasible)

    def test_class_specific_risk_contribution_cap_filters_only_named_class(self):
        classes = ("stocks", "fixed_income")
        rows = [
            DailyReturn(
                date(2020, 1, 1) + timedelta(days=index),
                {"stocks": stock_return, "fixed_income": fixed_return},
            )
            for index, (stock_return, fixed_return) in enumerate(
                ((0.10, 0.02), (-0.10, -0.02), (0.10, 0.02), (-0.10, -0.02))
            )
        ]
        capped = evaluate_candidates(
            rows,
            classes,
            ((0.5, 0.5),),
            10.0,
            1.0,
            risk_contribution_caps={"stocks": 0.25},
        )
        self.assertFalse(capped[0].feasible)

    def test_crypto_weight_scenarios_redirect_difference_to_fixed_income(self):
        rows = [
            DailyReturn(
                date(2020, 1, 1) + timedelta(days=index),
                {name: 0.0 for name in ASSET_CLASSES},
            )
            for index in range(2)
        ]
        base_weights = {
            "brazilian_stocks": 0.2,
            "fiis": 0.1,
            "international_equity": 0.25,
            "fixed_income": 0.15,
            "crypto": 0.3,
        }
        scenarios = evaluate_crypto_weight_scenarios(
            rows,
            rows,
            rows,
            ASSET_CLASSES,
            base_weights,
            (0.1,),
            {"volatility_cap": 0.2, "drawdown_cap": 0.3},
        )
        weights = scenarios["crypto_10pct_to_fixed_income"]["current_training"]["weights"]
        self.assertEqual(weights["crypto"], 0.1)
        self.assertEqual(weights["fixed_income"], 0.35)

    def test_crypto_weight_scenarios_skip_weights_above_personalized_base(self):
        rows = [
            DailyReturn(
                date(2020, 1, 1) + timedelta(days=index),
                {name: 0.0 for name in ASSET_CLASSES},
            )
            for index in range(2)
        ]
        base_weights = {
            "brazilian_stocks": 0.2,
            "fiis": 0.15,
            "international_equity": 0.2,
            "fixed_income": 0.4,
            "crypto": 0.05,
        }
        scenarios = evaluate_crypto_weight_scenarios(
            rows,
            rows,
            rows,
            ASSET_CLASSES,
            base_weights,
            (0.1,),
            {"volatility_cap": 0.1, "drawdown_cap": 0.15},
        )
        self.assertEqual(scenarios, {})

    def test_annual_rebalance_waits_for_next_trading_day(self):
        classes = ("stocks", "fixed_income")
        rows = [
            DailyReturn(date(2020, 1, 1), {"stocks": 0.0, "fixed_income": 0.0}),
            DailyReturn(date(2021, 1, 1), {"stocks": 0.0, "fixed_income": 0.0}),
            DailyReturn(date(2021, 1, 4), {"stocks": 0.0, "fixed_income": 0.0}),
        ]
        path = simulate_portfolio(
            rows,
            {"stocks": 0.5, "fixed_income": 0.5},
            classes,
        )
        self.assertEqual(path.rebalance_dates, (date(2021, 1, 4),))

    def test_metrics_capture_a_drawdown(self):
        classes = ("stocks", "fixed_income")
        start = date(2020, 1, 1)
        rows = [
            DailyReturn(start + timedelta(days=index), {"stocks": value, "fixed_income": 0.0})
            for index, value in enumerate((0.0, 0.10, -0.20, 0.05))
        ]
        path = simulate_portfolio(
            rows,
            {"stocks": 1.0, "fixed_income": 0.0},
            classes,
            annual_rebalance=False,
        )
        metrics = portfolio_metrics(path)
        self.assertLess(metrics.max_drawdown, 0.0)
        self.assertGreater(metrics.annualized_volatility, 0.0)

    def test_simplex_grid_and_refinement_are_deterministic(self):
        classes = ("stocks", "fixed_income")
        coarse = simplex_grid(classes, 0.05)
        self.assertEqual(len(coarse), 21)
        refined = bounded_simplex_grid(
            [(0.5, 0.5)], classes, step=0.01, radius=0.05
        )
        self.assertIn((0.5, 0.5), refined)
        self.assertTrue(all(abs(sum(weights) - 1.0) < 1e-9 for weights in refined))

    def test_frontier_exposes_endpoints_and_knee(self):
        candidates = (
            Candidate((0.7, 0.3), Metrics(0.40, 0.10, 0.18, -0.25, 1.6), 0.58, True),
            Candidate((0.5, 0.5), Metrics(0.25, 0.08, 0.13, -0.15, 1.67), 0.50, True),
            Candidate((0.3, 0.7), Metrics(0.10, 0.05, 0.08, -0.10, 1.8), 0.58, True),
        )
        frontier = non_dominated_frontier(candidates)
        points = select_frontier_points(frontier)
        self.assertEqual(points["max_return"].weights, (0.7, 0.3))
        self.assertEqual(points["most_diversified"].weights, (0.5, 0.5))
        self.assertEqual(points["knee"].weights, (0.7, 0.3))

    def test_penalty_sweep_returns_deterministic_feasible_winners(self):
        candidates = (
            Candidate((1.0, 0.0), Metrics(0.30, 0.10, 0.15, -0.20, 1.5), 1.0, True),
            Candidate((0.5, 0.5), Metrics(0.20, 0.08, 0.10, -0.10, 1.0), 0.5, True),
        )
        winners = penalty_sweep(candidates, (0.0, 1.0))
        self.assertEqual(winners[0.0].weights, (1.0, 0.0))
        self.assertEqual(winners[1.0].weights, (0.5, 0.5))

    def test_risk_adjusted_preferences_can_prefer_lower_risk(self):
        candidates = (
            Candidate((1.0, 0.0), Metrics(0.30, 0.30, 0.30, -0.30, 1.0), 1.0, True),
            Candidate((0.5, 0.5), Metrics(0.25, 0.25, 0.10, -0.10, 2.5), 0.5, True),
        )
        winners = penalty_sweep(
            candidates,
            (0.0,),
            risk_adjusted_weights={
                "return": 0.20,
                "volatility": 0.40,
                "drawdown": 0.40,
            },
        )
        self.assertEqual(winners[0.0].weights, (0.5, 0.5))

    def test_snapshot_loader_aligns_and_converts_series_without_filling(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            (directory / "metadata.json").write_text(
                json.dumps({
                    "source": "fixture",
                    "retrieved_at": "2026-07-21T12:00:00-03:00",
                    "cutoff_date": "2020-01-03",
                    "skipped_tickers": ["BBB"],
                }),
                encoding="utf-8",
            )
            (directory / "caio_stocks.csv").write_text(
                "date,AAA,BBB\n"
                "2020-01-01,100,100\n"
                "2020-01-02,101,99\n"
                "2020-01-03,102,98\n",
                encoding="utf-8",
            )
            for filename, values in {
                "ifix.csv": ("100", "101", "102"),
                "sp500_total_return_usd.csv": ("100", "101", "102"),
                "di.csv": ("100", "100.1", "100.2"),
                "btc_usd.csv": ("100", "110", "120"),
                "ptax.csv": ("5", "5.1", "5.2"),
            }.items():
                (directory / filename).write_text(
                    "date,value\n"
                    f"2020-01-01,{values[0]}\n"
                    f"2020-01-02,{values[1]}\n"
                    f"2020-01-03,{values[2]}\n",
                    encoding="utf-8",
                )
            portfolio_path = directory / "portfolio.json"
            portfolio_path.write_text(
                json.dumps([{"TICKER": "AAA"}, {"TICKER": "BBB"}]),
                encoding="utf-8",
            )

            bundle = load_snapshot_bundle(directory, portfolio_path)

        self.assertEqual(bundle.start_date, date(2020, 1, 1))
        self.assertEqual(bundle.end_date, date(2020, 1, 3))
        self.assertEqual(len(bundle.rows), 3)
        self.assertAlmostEqual(bundle.rows[1].returns["brazilian_stocks"], 0.01)
        self.assertEqual(set(bundle.rows[-1].returns), {
            "brazilian_stocks", "fiis", "international_equity", "fixed_income", "crypto"
        })
        self.assertGreater(bundle.rows[-1].returns["international_equity"], 0.0)

    def test_snapshot_loader_uses_equal_weight_fii_sleeve_without_ifix(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            (directory / "metadata.json").write_text(
                json.dumps({
                    "source": "fixture",
                    "retrieved_at": "2026-07-21T12:00:00-03:00",
                    "cutoff_date": "2021-01-04",
                }),
                encoding="utf-8",
            )
            dates = ("2020-01-01", "2020-01-02", "2021-01-01", "2021-01-04")
            (directory / "caio_stocks.csv").write_text(
                "date,AAA\n" + "\n".join(
                    f"{current_date},{value}"
                    for current_date, value in zip(dates, (100, 101, 102, 103))
                ) + "\n",
                encoding="utf-8",
            )
            (directory / "caio_fiis.csv").write_text(
                "date,FII1,FII2\n"
                "2020-01-01,100,100\n"
                "2020-01-02,110,100\n"
                "2021-01-01,220,100\n"
                "2021-01-04,242,100\n",
                encoding="utf-8",
            )
            for filename, values in {
                "sp500_total_return_usd.csv": (100, 101, 102, 103),
                "di.csv": (100, 100.1, 100.2, 100.3),
                "btc_usd.csv": (100, 110, 120, 130),
                "ptax.csv": (5, 5.1, 5.2, 5.3),
            }.items():
                (directory / filename).write_text(
                    "date,value\n" + "\n".join(
                        f"{current_date},{value}"
                        for current_date, value in zip(dates, values)
                    ) + "\n",
                    encoding="utf-8",
                )
            stock_path = directory / "stocks.json"
            stock_path.write_text(json.dumps([{"TICKER": "AAA"}]), encoding="utf-8")
            fii_path = directory / "fiis.json"
            fii_path.write_text(
                json.dumps([{"TICKER": "FII1"}, {"TICKER": "FII2"}]),
                encoding="utf-8",
            )

            bundle = load_snapshot_bundle(
                directory,
                stock_path,
                fii_portfolio_path=fii_path,
            )

            self.assertEqual(
                [round(row.returns["fiis"], 8) for row in bundle.rows],
                [0.0, 0.05, 0.52380952, 0.05],
            )
            (directory / "ifix.csv").write_text(
                "date,value\n2020-01-01,1\n2020-01-02,10000\n2021-01-01,2\n2021-01-04,20000\n",
                encoding="utf-8",
            )
            changed_ifix_bundle = load_snapshot_bundle(
                directory,
                stock_path,
                fii_portfolio_path=fii_path,
            )
            self.assertEqual(
                [row.returns["fiis"] for row in changed_ifix_bundle.rows],
                [row.returns["fiis"] for row in bundle.rows],
            )

    def test_output_writer_uses_nested_walk_forward_selection(self):
        classes = ["stocks", "fixed_income"]
        candidate = {
            "weights": {"stocks": 0.5, "fixed_income": 0.5},
            "metrics": {
                "total_return": 0.1,
                "annualized_return": 0.1,
                "annualized_volatility": 0.1,
                "max_drawdown": -0.1,
                "calmar": 1.0,
                "hhi": 0.5,
                "feasible": True,
            },
        }
        result = {
            "classes": classes,
            "current_target": {"frontier": [candidate]},
            "walk_forward": {
                "primary": [{
                    "train_start": "2020-01-01",
                    "train_end": "2021-01-01",
                    "test_start": "2021-01-02",
                    "test_end": "2022-01-01",
                    "training": {"selected": {"knee": candidate}},
                }],
            },
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = write_outputs(result, Path(temporary_directory), "fixture")
            self.assertTrue(all(path.exists() for path in paths))


if __name__ == "__main__":
    unittest.main()
