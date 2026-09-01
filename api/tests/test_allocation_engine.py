from datetime import date
from pathlib import Path
import unittest

from allocation_data import SnapshotBundle
from app.adapters.allocation_engine import (
    AllocationAdapterError,
    BasicRecommendationInput,
    generate_basic_recommendation,
)


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
        "current_target": {
            "selected": {
                "profile_winner": {
                    "weights": ENGINE_WEIGHTS,
                },
            },
        },
    }


class AllocationEngineAdapterTests(unittest.TestCase):
    def test_normalizes_engine_result_and_amounts(self):
        recommendation = generate_basic_recommendation(
            BasicRecommendationInput("moderado", 100_000),
            load_bundle=lambda _snapshot_dir, _portfolio_path: fake_bundle(),
            run_analysis=lambda *_args, **_kwargs: fake_result(),
        )

        self.assertEqual(recommendation["plan"], "basic")
        self.assertEqual(recommendation["snapshot_id"], "fixture-v1")
        self.assertEqual(
            {item["key"] for item in recommendation["classes"]},
            {"brazilian_stocks", "fiis", "international", "fixed_income", "crypto"},
        )
        self.assertEqual(
            sum(item["target_amount_brl"] for item in recommendation["classes"]),
            100_000,
        )

    def test_rejects_missing_profile_winner(self):
        with self.assertRaisesRegex(AllocationAdapterError, "Basic profile winner"):
            generate_basic_recommendation(
                BasicRecommendationInput("moderado", 100_000),
                load_bundle=lambda _snapshot_dir, _portfolio_path: fake_bundle(),
                run_analysis=lambda *_args, **_kwargs: {"current_target": {"selected": {}}},
            )

    def test_rejects_invalid_capital_before_loading_snapshot(self):
        with self.assertRaisesRegex(AllocationAdapterError, "positive"):
            generate_basic_recommendation(
                BasicRecommendationInput("moderado", 0),
                load_bundle=lambda *_args: self.fail("snapshot must not load"),
                run_analysis=lambda *_args, **_kwargs: fake_result(),
            )


if __name__ == "__main__":
    unittest.main()
