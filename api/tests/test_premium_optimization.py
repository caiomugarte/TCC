import unittest
from pathlib import Path
from types import SimpleNamespace

from app.adapters.premium_optimization import (
    PremiumOptimizationError,
    _rounded_amounts,
    run_premium_optimization,
)


class Source:
    def __init__(self, path: str):
        self.path = Path(path)

    def resolved_path(self) -> Path:
        return self.path


class Manifest:
    manifest_id = "manifest-fixture"
    manifest_path = None

    def source_for(self, key: str) -> Source:
        return Source(f"/tmp/{key}.csv")


def policy(weights=None, selector_hhi_max=None):
    return {
        "allocation": {
            "volatility_cap": 0.2,
            "drawdown_cap": 0.3,
            "crypto_risk_contribution_cap": 0.5,
            "hhi_penalty": 0.1,
            "risk_adjusted_weights": {"return": 0.7},
            "class_constraints": {
                "minimum_class_weight": 0.0,
                "minimum_weights": {},
                "maximum_weights": {},
                "hhi_max": None,
            },
        },
        "stocks": {
            "selection_preset": "fixture",
            "n_assets": 2,
            "factor_weights": {"liquidez": 1.0},
            "liquidity_and_size_filters": (
                {} if selector_hhi_max is None else {"hhi_max": selector_hhi_max}
            ),
            "lambda_hhi": 0.2,
            "system_ga_config": {"population": 4, "generations": 2, "run_count": 1},
        },
        "fiis": {
            "selection_preset": "fixture",
            "n_assets": 2,
            "factor_weights": {"liquidity": 1.0},
            "liquidity_and_size_filters": (
                {} if selector_hhi_max is None else {"hhi_max": selector_hhi_max}
            ),
            "lambda_hhi": 0.2,
            "system_ga_config": {"population": 4, "generations": 2, "run_count": 1},
        },
        "provenance": {
            "policy_version": "premium-policy-v1",
            "model_versions": {"allocation": "allocation-fixture"},
            "source_snapshot_ids": ["manifest-fixture"],
            "source_snapshot_hashes": {},
            "cutoff_date": "2026-07-21",
            "random_seed": 19,
        },
    }


class PremiumOptimizationTests(unittest.TestCase):
    def test_composes_sleeves_and_keeps_engine_sections_separate(self):
        calls = {}

        def allocation(rows, *, metadata, allocation_config, class_constraints):
            calls["allocation"] = (allocation_config, class_constraints)
            return {
                "current_target": {
                    "selected": {
                        "profile_winner": {
                            "weights": {
                                "brazilian_stocks": 0.4,
                                "fiis": 0.2,
                                "international_equity": 0.1,
                                "fixed_income": 0.2,
                                "crypto": 0.1,
                            },
                            "metrics": {"annualized_return": 0.12},
                        }
                    }
                }
            }

        def stocks(**kwargs):
            calls["stocks"] = kwargs["config"]
            return {"selected_tickers": ["AAA3", "BBB3"], "sleeve_weights": {"AAA3": 0.6, "BBB3": 0.4}}

        def fiis(**kwargs):
            calls["fiis"] = kwargs["selection_config"]
            return {"selected_tickers": ["AAA11", "BBB11"], "sleeve_weights": {"AAA11": 0.5, "BBB11": 0.5}}

        result = run_premium_optimization(
            policy(),
            Manifest(),
            Path("/tmp/prumo-test-run"),
            100_000,
            snapshot_loader=lambda _manifest: SimpleNamespace(rows=(), metadata={}),
            allocation_engine=allocation,
            stock_engine=stocks,
            fii_engine=fiis,
        )

        self.assertEqual(sum(item["target_weight"] for item in result["classes"]), 1.0)
        self.assertEqual(sum(item["target_amount_brl"] for item in result["classes"]), 100_000)
        self.assertAlmostEqual(sum(item["portfolio_weight"] for item in result["stocks"]), 0.4)
        self.assertEqual(
            sum(item["target_amount_brl"] for item in result["stocks"]),
            40_000,
        )
        self.assertAlmostEqual(sum(item["portfolio_weight"] for item in result["fiis"]), 0.2)
        self.assertNotIn("stocks", calls["allocation"])
        self.assertNotIn("fiis", calls["allocation"])
        self.assertNotIn("allocation", calls["stocks"])
        self.assertNotIn("fiis", calls["stocks"])
        self.assertNotIn("allocation", calls["fiis"])

    def test_invalid_engine_output_never_returns_partial_result(self):
        with self.assertRaises(PremiumOptimizationError):
            run_premium_optimization(
                policy(),
                Manifest(),
                Path("/tmp/prumo-test-run"),
                100_000,
                snapshot_loader=lambda _manifest: SimpleNamespace(rows=(), metadata={}),
                allocation_engine=lambda *_args, **_kwargs: {
                    "current_target": {
                        "selected": {
                            "profile_winner": {
                                "weights": {
                                    "brazilian_stocks": 0.8,
                                    "fiis": 0.8,
                                    "international_equity": 0,
                                    "fixed_income": 0,
                                    "crypto": 0,
                                }
                            }
                        }
                    }
                },
            )

    def test_selector_hhi_is_a_hard_constraint(self):
        def allocation(*_args, **_kwargs):
            return {
                "current_target": {
                    "selected": {
                        "profile_winner": {
                            "weights": {
                                "brazilian_stocks": 0.4,
                                "fiis": 0.2,
                                "international_equity": 0.1,
                                "fixed_income": 0.2,
                                "crypto": 0.1,
                            }
                        }
                    }
                }
            }

        with self.assertRaises(PremiumOptimizationError) as error:
            run_premium_optimization(
                policy(selector_hhi_max=0.25),
                Manifest(),
                Path("/tmp/prumo-test-run"),
                100_000,
                snapshot_loader=lambda _manifest: SimpleNamespace(rows=(), metadata={}),
                allocation_engine=allocation,
                stock_engine=lambda **_kwargs: {
                    "selected_tickers": ["AAA3", "BBB3"],
                    "sleeve_weights": {"AAA3": 0.5, "BBB3": 0.5},
                    "metrics": {"hhi": 1.0},
                },
            )
        self.assertEqual(error.exception.code, "infeasible_constraints")

    def test_amount_rounding_never_creates_negative_targets(self):
        amounts = _rounded_amounts({"zero": 0.0, "small": 0.5, "other": 0.5}, 0.01)

        self.assertEqual(sum(amounts.values()), 0.01)
        self.assertTrue(all(value >= 0 for value in amounts.values()))


if __name__ == "__main__":
    unittest.main()
