import unittest
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from app.adapters.premium_optimization import (
    PremiumOptimizationError,
    _default_fii_engine,
    _default_stock_engine,
    _rounded_amounts,
    run_premium_optimization,
)
from app.services.premium_policy import resolve_recommendation_policy
from app.services.status_invest_inputs import SourceSnapshot, StatusInvestInputs


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


def selector_inputs():
    return StatusInvestInputs(
        stocks=SourceSnapshot(
            Path("/tmp/run/status-invest/stocks.csv"),
            "statusinvest",
            "2026-07-21T00:00:00Z",
            "stock-hash",
        ),
        fiis=SourceSnapshot(
            Path("/tmp/run/status-invest/fiis.csv"),
            "statusinvest",
            "2026-07-21T00:00:00Z",
            "fii-hash",
        ),
    )


def resolved_policy(plan: str):
    profile = SimpleNamespace(
        version=3,
        suitability_score=0.8,
        raw_score=0.8,
        dimensions={
            "apetite": 0.8,
            "capacidade": 0.8,
            "liquidez": 0.8,
            "conhecimento": 0.8,
        },
        generic_profile="moderado",
        restrictions_json=["nenhuma"],
        answers={"restricoes": ["nenhuma"]},
        rules_json=[],
        warnings_json=[],
        schema_version=1,
    )
    return resolve_recommendation_policy(
        profile,
        plan=plan,
        source_snapshot_ids=["history-v1"],
        source_snapshot_hashes={"history-v1": "history-hash"},
        cutoff_date="2026-07-21",
        random_seed=19,
    )


def allocation_result():
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


def selector_result(tickers):
    return {
        "selected_tickers": tickers,
        "sleeve_weights": {ticker: 1 / len(tickers) for ticker in tickers},
    }


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
            selector_inputs=selector_inputs(),
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
                selector_inputs=selector_inputs(),
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
                selector_inputs=selector_inputs(),
                snapshot_loader=lambda _manifest: SimpleNamespace(rows=(), metadata={}),
                allocation_engine=allocation,
                stock_engine=lambda **_kwargs: {
                    "selected_tickers": ["AAA3", "BBB3"],
                    "sleeve_weights": {"AAA3": 0.5, "BBB3": 0.5},
                    "metrics": {"hhi": 1.0},
                },
            )
        self.assertEqual(error.exception.code, "infeasible_constraints")

    def test_basic_and_premium_policies_share_combined_result_contract(self):
        results = {}
        policies = {}
        for plan in ("basic", "premium"):
            policies[plan] = resolved_policy(plan)
            results[plan] = run_premium_optimization(
                policies[plan],
                Manifest(),
                Path(f"/tmp/prumo-{plan}-run"),
                100_000,
                selector_inputs=selector_inputs(),
                snapshot_loader=lambda _manifest: SimpleNamespace(rows=(), metadata={}),
                allocation_engine=lambda *_args, **_kwargs: allocation_result(),
                stock_engine=lambda **_kwargs: selector_result(["AAA3", "BBB3"]),
                fii_engine=lambda **_kwargs: selector_result(["AAA11", "BBB11"]),
            )

        self.assertNotEqual(
            policies["basic"].stocks.factor_weights,
            policies["premium"].stocks.factor_weights,
        )
        expected_keys = {
            "policy", "classes", "stocks", "fiis", "assumptions", "risks",
            "provenance", "diagnostics",
        }
        for result in results.values():
            self.assertEqual(set(result), expected_keys)
            self.assertEqual(sum(item["target_amount_brl"] for item in result["classes"]), 100_000)
            self.assertEqual(sum(item["target_amount_brl"] for item in result["stocks"]), 40_000)
            self.assertEqual(sum(item["target_amount_brl"] for item in result["fiis"]), 20_000)

    def test_selectors_use_only_per_run_status_invest_paths(self):
        sources = selector_inputs()
        paths = {}

        def stocks(**kwargs):
            paths["stocks"] = kwargs["input_path"]
            return selector_result(["AAA3"])

        def fiis(**kwargs):
            paths["fiis"] = kwargs["input_path"]
            return selector_result(["AAA11"])

        run_premium_optimization(
            policy(),
            Manifest(),
            Path("/tmp/prumo-test-run"),
            100_000,
            selector_inputs=sources,
            snapshot_loader=lambda _manifest: SimpleNamespace(rows=(), metadata={}),
            allocation_engine=lambda *_args, **_kwargs: allocation_result(),
            stock_engine=stocks,
            fii_engine=fiis,
        )

        self.assertEqual(paths, {"stocks": sources.stocks.path, "fiis": sources.fiis.path})

    def test_allocation_history_still_uses_manifest_reference_inputs(self):
        manifest = Manifest()
        history_rows = [{"date": "2026-07-21", "close": 100.0}]
        history_metadata = {"source": "allocation-history"}
        seen = {}

        def load_history(value):
            seen["manifest"] = value
            return SimpleNamespace(rows=history_rows, metadata=history_metadata)

        def allocate(rows, *, metadata, allocation_config, class_constraints):
            seen["rows"] = rows
            seen["metadata"] = metadata
            return allocation_result()

        run_premium_optimization(
            policy(),
            manifest,
            Path("/tmp/prumo-test-run"),
            100_000,
            selector_inputs=selector_inputs(),
            snapshot_loader=load_history,
            allocation_engine=allocate,
            stock_engine=lambda **_kwargs: selector_result(["AAA3"]),
            fii_engine=lambda **_kwargs: selector_result(["AAA11"]),
        )

        self.assertIs(seen["manifest"], manifest)
        self.assertIs(seen["rows"], history_rows)
        self.assertIs(seen["metadata"], history_metadata)

    def test_provenance_separates_selector_sources_from_allocation_history(self):
        sources = selector_inputs()
        result = run_premium_optimization(
            policy(),
            Manifest(),
            Path("/tmp/prumo-test-run"),
            100_000,
            selector_inputs=sources,
            snapshot_loader=lambda _manifest: SimpleNamespace(rows=(), metadata={}),
            allocation_engine=lambda *_args, **_kwargs: allocation_result(),
            stock_engine=lambda **_kwargs: selector_result(["AAA3"]),
            fii_engine=lambda **_kwargs: selector_result(["AAA11"]),
        )

        provenance = result["provenance"]
        self.assertEqual(provenance["manifest_id"], "manifest-fixture")
        self.assertEqual(
            provenance["allocation_history"]["source_snapshot_ids"],
            ["manifest-fixture"],
        )
        self.assertEqual(
            provenance["allocation_history"]["source_snapshot_hashes"], {}
        )
        self.assertEqual(provenance["selector_sources"], sources.provenance())
        self.assertNotIn("manifest_path", provenance)
        self.assertNotIn("workspace", provenance)
        self.assertNotIn("/tmp/", repr(provenance))

    def test_premium_selector_ga_controls_pass_through_unchanged(self):
        policy_value = resolved_policy("premium").to_dict()
        stock_call = {}
        fii_call = {}
        pipelines = ModuleType("pipelines")
        pipelines.__path__ = []
        multi_run = ModuleType("pipelines.multi_run")
        multi_run.run_multi_execution_profile = lambda **kwargs: stock_call.update(kwargs)
        fii_selection = ModuleType("fii_selection")
        fii_selection.run_fii_selection = lambda *args, **kwargs: fii_call.update(kwargs)

        with patch.dict(
            sys.modules,
            {
                "pipelines": pipelines,
                "pipelines.multi_run": multi_run,
                "fii_selection": fii_selection,
            },
        ):
            _default_stock_engine(
                config=policy_value["stocks"],
                workspace=Path("/tmp/prumo-test-run/stocks"),
                seed=19,
                input_path=Path("/tmp/run/status-invest/stocks.csv"),
            )
            _default_fii_engine(
                selection_config=policy_value["fiis"],
                n_runs=policy_value["fiis"]["system_ga_config"]["run_count"],
                parallel=False,
                workspace=Path("/tmp/prumo-test-run/fiis"),
                seed=19,
                input_path=Path("/tmp/run/status-invest/fiis.csv"),
            )

        for call, section in (
            (stock_call, policy_value["stocks"]),
            (fii_call, policy_value["fiis"]),
        ):
            controls = section["system_ga_config"]
            self.assertEqual(call["selection_config"]["system_ga_config"], controls)
            self.assertEqual(call["n_runs"], controls["run_count"])
            for key in ("adaptive_mode", "min_runs", "target_cv", "target_jaccard"):
                self.assertEqual(call[key], controls[key])

    def test_missing_validated_selector_inputs_fail_before_selection(self):
        selected = []
        with self.assertRaises(PremiumOptimizationError) as error:
            run_premium_optimization(
                policy(),
                Manifest(),
                Path("/tmp/prumo-test-run"),
                100_000,
                selector_inputs=None,
                stock_engine=lambda **_kwargs: selected.append("stocks"),
            )

        self.assertEqual(error.exception.code, "snapshot_unavailable")
        self.assertEqual(selected, [])

    def test_amount_rounding_never_creates_negative_targets(self):
        amounts = _rounded_amounts({"zero": 0.0, "small": 0.5, "other": 0.5}, 0.01)

        self.assertEqual(sum(amounts.values()), 0.01)
        self.assertTrue(all(value >= 0 for value in amounts.values()))


if __name__ == "__main__":
    unittest.main()
