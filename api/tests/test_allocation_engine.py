from datetime import date, timedelta
import math
import unittest

# The adapter owns the `py/` import seam, so it must be imported first.
from app.adapters.allocation_engine import (
    AllocationAdapterError,
    BasicRecommendationInput,
    generate_basic_recommendation,
)
from allocation_data import SnapshotBundle  # noqa: E402
from core.allocation import DailyReturn  # noqa: E402
from pipelines.equal_weight_risky import METHOD_ID  # noqa: E402


ENGINE_WEIGHTS = {
    "brazilian_stocks": 0.2,
    "fiis": 0.1,
    "international_equity": 0.25,
    "fixed_income": 0.15,
    "crypto": 0.3,
}


def fake_bundle() -> SnapshotBundle:
    return SnapshotBundle(
        rows=(),
        metadata={"cutoff_date": "2026-07-21", "snapshot_id": "fixture-v1"},
        start_date=date(2020, 1, 1),
        end_date=date(2026, 7, 21),
    )


def fake_result():
    return {
        "status": "ok",
        "method": METHOD_ID,
        "weights": ENGINE_WEIGHTS,
        "provenance": {"method": METHOD_ID, "risky_share": 0.85},
    }


def synthetic_rows(days: int = 3 * 365) -> tuple[DailyReturn, ...]:
    """Deterministic daily returns with a riskier tail than fixed income."""

    start = date(2020, 1, 1)
    rows = []
    for index in range(days):
        wave = math.sin(index / 7.0)
        rows.append(
            DailyReturn(
                start + timedelta(days=index),
                {
                    "brazilian_stocks": 0.0006 + 0.024 * wave,
                    "fiis": 0.0004 + 0.012 * math.sin(index / 5.0),
                    "international_equity": 0.0006 + 0.020 * math.cos(index / 6.0),
                    "fixed_income": 0.0004,
                    "crypto": 0.002 + 0.070 * math.sin(index / 3.0),
                },
            )
        )
    return tuple(rows)


def synthetic_bundle() -> SnapshotBundle:
    rows = synthetic_rows()
    return SnapshotBundle(
        rows=rows,
        metadata={"cutoff_date": rows[-1].date.isoformat(), "snapshot_id": "synthetic-v1"},
        start_date=rows[0].date,
        end_date=rows[-1].date,
    )


class AllocationEngineAdapterTests(unittest.TestCase):
    def test_normalizes_engine_result_and_amounts(self):
        recommendation = generate_basic_recommendation(
            BasicRecommendationInput("moderado", 100_000),
            load_bundle=lambda _snapshot_dir, _portfolio_path: fake_bundle(),
            run_analysis=lambda *_args, **_kwargs: fake_result(),
        )

        self.assertEqual(recommendation["plan"], "basic")
        self.assertEqual(recommendation["model_version"], "allocation-v1")
        self.assertEqual(recommendation["snapshot_cutoff"], "2026-07-21")
        self.assertEqual(recommendation["allocation_method"]["method"], METHOD_ID)
        self.assertEqual(recommendation["snapshot_id"], "fixture-v1")
        self.assertEqual(
            {item["key"] for item in recommendation["classes"]},
            {"brazilian_stocks", "fiis", "international", "fixed_income", "crypto"},
        )
        self.assertEqual(
            sum(item["target_amount_brl"] for item in recommendation["classes"]),
            100_000,
        )

    def test_passes_anchor_caps_without_class_constraints(self):
        calls = []

        def runner(rows, **kwargs):
            calls.append(kwargs)
            return fake_result()

        generate_basic_recommendation(
            BasicRecommendationInput("arrojado", 100_000),
            load_bundle=lambda _snapshot_dir, _portfolio_path: fake_bundle(),
            run_analysis=runner,
        )

        (kwargs,) = calls
        self.assertNotIn("class_constraints", kwargs)
        self.assertEqual(kwargs["allocation_config"]["score"], 1.0)
        self.assertIn("volatility_cap", kwargs["allocation_config"])
        self.assertIn("drawdown_cap", kwargs["allocation_config"])
        self.assertEqual(kwargs["metadata"]["snapshot_id"], "fixture-v1")

    def test_rejects_missing_weights(self):
        with self.assertRaisesRegex(AllocationAdapterError, "no class weights"):
            generate_basic_recommendation(
                BasicRecommendationInput("moderado", 100_000),
                load_bundle=lambda _snapshot_dir, _portfolio_path: fake_bundle(),
                run_analysis=lambda *_args, **_kwargs: {"provenance": {}},
            )

    def test_rejects_missing_provenance(self):
        result = fake_result()
        del result["provenance"]
        with self.assertRaisesRegex(AllocationAdapterError, "provenance"):
            generate_basic_recommendation(
                BasicRecommendationInput("moderado", 100_000),
                load_bundle=lambda _snapshot_dir, _portfolio_path: fake_bundle(),
                run_analysis=lambda *_args, **_kwargs: result,
            )

    def test_real_method_orders_profiles_by_risky_share(self):
        capital = 123_456.78
        risky_shares = []
        for profile in ("conservador", "moderado", "arrojado"):
            with self.subTest(profile=profile):
                recommendation = generate_basic_recommendation(
                    BasicRecommendationInput(profile, capital),
                    load_bundle=lambda _snapshot_dir, _portfolio_path: synthetic_bundle(),
                )
                weights = {
                    item["key"]: item["target_weight"]
                    for item in recommendation["classes"]
                }
                self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)
                self.assertAlmostEqual(
                    sum(item["target_amount_brl"] for item in recommendation["classes"]),
                    capital,
                    places=2,
                )
                method = recommendation["allocation_method"]
                self.assertEqual(method["method"], METHOD_ID)
                self.assertTrue(method["risk"]["within_caps"])
                risky_shares.append(1.0 - weights["fixed_income"])

        self.assertGreater(risky_shares[0], 0.0)
        self.assertLess(risky_shares[-1], 1.0)
        self.assertLess(risky_shares[0], risky_shares[1])
        self.assertLess(risky_shares[1], risky_shares[2])

    def test_rejects_invalid_capital_before_loading_snapshot(self):
        with self.assertRaisesRegex(AllocationAdapterError, "positive"):
            generate_basic_recommendation(
                BasicRecommendationInput("moderado", 0),
                load_bundle=lambda *_args: self.fail("snapshot must not load"),
                run_analysis=lambda *_args, **_kwargs: fake_result(),
            )

    def test_real_snapshot_reads_index_series_and_orders_profiles(self):
        risky_shares = []
        for profile in ("conservador", "moderado", "arrojado"):
            with self.subTest(profile=profile):
                recommendation = generate_basic_recommendation(
                    BasicRecommendationInput(profile, 100_000),
                )
                weights = {
                    item["key"]: item["target_weight"]
                    for item in recommendation["classes"]
                }
                self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)
                self.assertEqual(
                    sum(item["target_amount_brl"] for item in recommendation["classes"]),
                    100_000,
                )
                method = recommendation["allocation_method"]
                self.assertEqual(method["method"], METHOD_ID)
                self.assertEqual(
                    method["class_series"]["brazilian_stocks"]["file"], "ibovespa.csv"
                )
                self.assertEqual(method["class_series"]["fiis"]["file"], "ifix.csv")
                risky_shares.append(1.0 - weights["fixed_income"])

        self.assertLess(risky_shares[0], risky_shares[1])
        self.assertLess(risky_shares[1], risky_shares[2])


if __name__ == "__main__":
    unittest.main()
