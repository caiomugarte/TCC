from __future__ import annotations

from datetime import datetime, timezone
import unittest

from pydantic import ValidationError

from app.schemas.recommendation import (
    AllocationClass,
    DroppedConstituent,
    RecommendationConstituent,
    RecommendationResponse,
)


def legacy_constituent() -> dict:
    return {
        "ticker": "KNRI11",
        "sleeve_weight": 0.1,
        "portfolio_weight": 0.013,
        "target_amount_brl": 100.0,
        "reasons": ["mantido"],
    }


def legacy_class() -> dict:
    return {
        "key": "fiis",
        "label": "FIIs",
        "target_weight": 0.13,
        "target_amount_brl": 1300.0,
        "metrics": {},
    }


def dropped_entry() -> dict:
    return {
        "ticker": "KNRI11",
        "sleeve_weight": 0.1,
        "portfolio_weight": 0.013,
        "target_amount_brl": 100.0,
        "price_brl": 147.89,
        "reasons": ["não coube no capital"],
    }


def response_payload(**overrides) -> dict:
    payload = {
        "id": "run-1",
        "account_id": "account-1",
        "profile_version": 1,
        "plan": "premium",
        "model_version": "premium-policy-v4",
        "snapshot_id": "snapshot-1",
        "snapshot_cutoff": "2026-11-06",
        "classes": [legacy_class()],
        "assumptions": [],
        "risks": [],
        "created_at": datetime(2026, 11, 9, tzinfo=timezone.utc),
        "fiis": [legacy_constituent()],
    }
    payload.update(overrides)
    return payload


class RecommendationSchemaTests(unittest.TestCase):
    def test_legacy_entries_validate_with_null_new_fields(self):
        response = RecommendationResponse(**response_payload())

        constituent = response.fiis[0]
        self.assertIsNone(constituent.price_brl)
        self.assertIsNone(constituent.quantity)
        self.assertIsNone(constituent.amount_brl)
        self.assertIsNone(constituent.effective_portfolio_weight)
        allocation_class = response.classes[0]
        self.assertIsNone(allocation_class.cash_residual_brl)
        self.assertIsNone(allocation_class.over_target_brl)
        self.assertEqual(response.stocks_dropped, [])
        self.assertEqual(response.fiis_dropped, [])

    def test_new_fields_serialise_camel_case(self):
        constituent = {
            **legacy_constituent(),
            "price_brl": 147.89,
            "quantity": 1,
            "amount_brl": 147.89,
            "effective_portfolio_weight": 0.014789,
        }
        allocation_class = {**legacy_class(), "cash_residual_brl": 4.04, "over_target_brl": 0.0}
        response = RecommendationResponse(
            **response_payload(
                classes=[allocation_class],
                fiis=[constituent],
                stocks_dropped=[dropped_entry()],
                fiis_dropped=[dropped_entry()],
            )
        )

        dumped = response.model_dump(by_alias=True)

        fii = dumped["fiis"][0]
        self.assertEqual(fii["priceBrl"], 147.89)
        self.assertEqual(fii["quantity"], 1)
        self.assertEqual(fii["amountBrl"], 147.89)
        self.assertEqual(fii["effectivePortfolioWeight"], 0.014789)
        self.assertEqual(dumped["classes"][0]["cashResidualBrl"], 4.04)
        self.assertEqual(dumped["classes"][0]["overTargetBrl"], 0.0)
        expected_dropped = {
            "ticker": "KNRI11",
            "sleeveWeight": 0.1,
            "portfolioWeight": 0.013,
            "targetAmountBrl": 100.0,
            "priceBrl": 147.89,
            "reasons": ["não coube no capital"],
        }
        self.assertEqual(dumped["stocksDropped"], [expected_dropped])
        self.assertEqual(dumped["fiisDropped"], [expected_dropped])

    def test_each_bound_rejects_its_own_input(self):
        constituent_cases = {
            "quantity": 0,
            "price_brl": 0,
            "amount_brl": -0.01,
            "effective_portfolio_weight": -0.01,
        }
        for field, value in constituent_cases.items():
            with self.subTest(model="constituent", field=field):
                with self.assertRaises(ValidationError):
                    RecommendationConstituent(**{**legacy_constituent(), field: value})
        for field in ("cash_residual_brl", "over_target_brl"):
            with self.subTest(model="class", field=field):
                with self.assertRaises(ValidationError):
                    AllocationClass(**{**legacy_class(), field: -0.01})
        with self.subTest(model="dropped", field="extra key"):
            with self.assertRaises(ValidationError):
                DroppedConstituent(**{**dropped_entry(), "quantity": 0})

        forced = RecommendationConstituent(
            **{**legacy_constituent(), "effective_portfolio_weight": 1.5}
        )
        self.assertEqual(forced.effective_portfolio_weight, 1.5)


if __name__ == "__main__":
    unittest.main()
