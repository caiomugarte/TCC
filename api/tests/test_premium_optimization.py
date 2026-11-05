import random
import unittest
from datetime import date, timedelta
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
from core.allocation import DailyReturn
from core.protection_zone import select_with_fallback
from fii_selection import FiiSelectionError, FiiUniverseInfeasibleError


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


def resolved_policy_profile(restrictions=("nenhuma",)):
    return SimpleNamespace(
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
        restrictions_json=list(restrictions),
        answers={"restricoes": list(restrictions)},
        rules_json=[],
        warnings_json=[],
        schema_version=1,
    )


def resolved_policy(plan: str, restrictions=("nenhuma",)):
    return resolve_recommendation_policy(
        resolved_policy_profile(restrictions),
        plan=plan,
        source_snapshot_ids=["history-v1"],
        source_snapshot_hashes={"history-v1": "history-hash"},
        cutoff_date="2026-07-21",
        random_seed=19,
    )


FIXTURE_WEIGHTS = {
    "brazilian_stocks": 0.4,
    "fiis": 0.2,
    "international_equity": 0.1,
    "fixed_income": 0.2,
    "crypto": 0.1,
}


def allocation_result(weights=None):
    """Result shape of ``pipelines.equal_weight_risky.run_equal_weight_risky``."""

    return {
        "status": "ok",
        "method": "equal-weight-risky-v1",
        "weights": dict(FIXTURE_WEIGHTS if weights is None else weights),
        "metrics": {"annualized_return": 0.12},
        "provenance": {"method": "equal-weight-risky-v1", "risky_share": 0.8},
    }


def synthetic_rows(years=11, seed=3):
    rng = random.Random(seed)
    end = date(2026, 10, 28)
    day = end - timedelta(days=int(years * 365.25))
    rows = []
    while day <= end:
        if day.weekday() < 5:
            returns = {
                name: rng.gauss(0.0005, 0.03)
                for name in ("brazilian_stocks", "fiis", "international_equity", "crypto")
            }
            returns["fixed_income"] = 0.0003 + rng.gauss(0.0, 0.0001)
            rows.append(DailyReturn(day, returns))
        day += timedelta(days=1)
    return tuple(rows)


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
            return allocation_result()

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
                allocation_engine=lambda *_args, **_kwargs: allocation_result(
                    {
                        "brazilian_stocks": 0.8,
                        "fiis": 0.8,
                        "international_equity": 0,
                        "fixed_income": 0,
                        "crypto": 0,
                    }
                ),
            )

    def test_selector_hhi_is_a_hard_constraint(self):
        def allocation(*_args, **_kwargs):
            return allocation_result()

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

    def test_default_allocation_engine_is_equal_weight_risky(self):
        rows = synthetic_rows()
        resolved = resolved_policy("premium")
        # The grid search must not be reached by the default engine.
        with patch.dict(sys.modules, {"pipelines.asset_allocation": None}):
            result = run_premium_optimization(
                resolved,
                Manifest(),
                Path("/tmp/prumo-test-run"),
                100_000,
                selector_inputs=selector_inputs(),
                snapshot_loader=lambda _manifest: SimpleNamespace(rows=rows, metadata={}),
                stock_engine=lambda **_kwargs: selector_result(["AAA3", "BBB3"]),
                fii_engine=lambda **_kwargs: selector_result(["AAA11", "BBB11"]),
            )

        weights = {item["key"]: item["target_weight"] for item in result["classes"]}
        method = result["provenance"]["allocation_method"]
        share = method["risky_share"]
        self.assertEqual(method["method"], "equal-weight-risky-v1")
        self.assertGreater(share, 0.0)
        for key in ("brazilian_stocks", "fiis", "international", "crypto"):
            self.assertAlmostEqual(weights[key], share / 4, places=6)
        self.assertAlmostEqual(weights["fixed_income"], 1 - share, places=6)
        self.assertEqual(sum(item["target_amount_brl"] for item in result["classes"]), 100_000)
        self.assertEqual(
            method["risk"]["volatility_cap"], resolved.allocation.volatility_cap
        )
        self.assertEqual(method["risk"]["drawdown_cap"], resolved.allocation.drawdown_cap)
        self.assertTrue(method["risk"]["within_caps"])
        self.assertEqual(method["window"]["end_date"], "2026-10-28")
        self.assertEqual(method["window"]["requested_years"], 10)
        self.assertEqual(set(method["class_series"]), {
            "brazilian_stocks", "fiis", "international_equity", "fixed_income", "crypto",
        })

    def test_default_engine_drops_avoided_classes(self):
        resolved = resolved_policy(
            "premium", ("evitar_cripto", "evitar_exterior", "limitar_concentracao")
        )
        rows = synthetic_rows()
        result = run_premium_optimization(
            resolved,
            Manifest(),
            Path("/tmp/prumo-test-run"),
            100_000,
            selector_inputs=selector_inputs(),
            snapshot_loader=lambda _manifest: SimpleNamespace(rows=rows, metadata={}),
            # Selector HHI stays enforced under "limitar_concentracao".
            stock_engine=lambda **_kwargs: {
                **selector_result(["AAA3", "BBB3"]), "metrics": {"hhi": 0.2},
            },
            fii_engine=lambda **_kwargs: {
                **selector_result(["AAA11", "BBB11"]), "metrics": {"hhi": 0.2},
            },
        )

        weights = {item["key"]: item["target_weight"] for item in result["classes"]}
        method = result["provenance"]["allocation_method"]
        self.assertEqual(weights["crypto"], 0.0)
        self.assertEqual(weights["international"], 0.0)
        self.assertAlmostEqual(weights["brazilian_stocks"], weights["fiis"], places=6)
        self.assertEqual(method["excluded_classes"], ["international_equity", "crypto"])

    def test_class_hhi_does_not_reject_the_allocation(self):
        concentrated_policy = policy()
        concentrated_policy["allocation"]["class_constraints"]["hhi_max"] = 0.25
        weights = {
            "brazilian_stocks": 0.1,
            "fiis": 0.1,
            "international_equity": 0.1,
            "fixed_income": 0.6,
            "crypto": 0.1,
        }

        result = run_premium_optimization(
            concentrated_policy,
            Manifest(),
            Path("/tmp/prumo-test-run"),
            100_000,
            selector_inputs=selector_inputs(),
            snapshot_loader=lambda _manifest: SimpleNamespace(rows=(), metadata={}),
            allocation_engine=lambda *_args, **_kwargs: allocation_result(weights),
            stock_engine=lambda **_kwargs: selector_result(["AAA3"]),
            fii_engine=lambda **_kwargs: selector_result(["AAA11"]),
        )

        self.assertEqual(
            {item["key"]: item["target_weight"] for item in result["classes"]}["fixed_income"],
            0.6,
        )

    def test_allocation_method_provenance_is_recorded(self):
        result = run_premium_optimization(
            policy(),
            Manifest(),
            Path("/tmp/prumo-test-run"),
            100_000,
            selector_inputs=selector_inputs(),
            snapshot_loader=lambda _manifest: SimpleNamespace(rows=(), metadata={}),
            allocation_engine=lambda *_args, **_kwargs: allocation_result(),
            stock_engine=lambda **_kwargs: selector_result(["AAA3"]),
            fii_engine=lambda **_kwargs: selector_result(["AAA11"]),
        )

        self.assertEqual(
            result["provenance"]["allocation_method"],
            {"method": "equal-weight-risky-v1", "risky_share": 0.8},
        )

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

    def test_fii_universe_shortage_maps_to_infeasible_universe_with_details(self):
        shortage = FiiUniverseInfeasibleError(
            "conservador", 7, 7, 10, 5_000_000_000.0, 2_000_000.0
        )

        with self.assertRaises(PremiumOptimizationError) as error:
            run_premium_optimization(
                policy(),
                Manifest(),
                Path("/tmp/prumo-test-run"),
                100_000,
                selector_inputs=selector_inputs(),
                snapshot_loader=lambda _manifest: SimpleNamespace(rows=(), metadata={}),
                allocation_engine=lambda *_args, **_kwargs: allocation_result(),
                stock_engine=lambda **_kwargs: selector_result(["AAA3"]),
                fii_engine=lambda **_kwargs: (_ for _ in ()).throw(shortage),
            )

        failure = error.exception
        self.assertEqual(failure.code, "infeasible_universe")
        self.assertEqual(
            dict(failure.details),
            {
                "profile": "conservador",
                "eligible_count": 7,
                "universe_count": 7,
                "required_count": 10,
                "size_min": 5_000_000_000.0,
                "liq_min": 2_000_000.0,
            },
        )

    def test_fii_universe_shortage_never_maps_to_constraint_or_engine_codes(self):
        with self.assertRaises(PremiumOptimizationError) as error:
            run_premium_optimization(
                policy(selector_hhi_max=0.25),
                Manifest(),
                Path("/tmp/prumo-test-run"),
                100_000,
                selector_inputs=selector_inputs(),
                snapshot_loader=lambda _manifest: SimpleNamespace(rows=(), metadata={}),
                allocation_engine=lambda *_args, **_kwargs: allocation_result(),
                stock_engine=lambda **_kwargs: {
                    "selected_tickers": ["AAA3"],
                    "sleeve_weights": {"AAA3": 1.0},
                    "metrics": {"hhi": 0.2},
                },
                fii_engine=lambda **_kwargs: (_ for _ in ()).throw(
                    FiiUniverseInfeasibleError(
                        "conservador", 13, 12, 13, 3_000_000_000.0, 2_000_000.0
                    )
                ),
            )

        self.assertEqual(error.exception.code, "infeasible_universe")

    def test_hhi_breach_text_path_keeps_infeasible_constraints_classification(self):
        with self.assertRaises(PremiumOptimizationError) as error:
            run_premium_optimization(
                policy(selector_hhi_max=0.25),
                Manifest(),
                Path("/tmp/prumo-test-run"),
                100_000,
                selector_inputs=selector_inputs(),
                snapshot_loader=lambda _manifest: SimpleNamespace(rows=(), metadata={}),
                allocation_engine=lambda *_args, **_kwargs: allocation_result(),
                stock_engine=lambda **_kwargs: {
                    "selected_tickers": ["AAA3"],
                    "sleeve_weights": {"AAA3": 1.0},
                    "metrics": {"hhi": 0.2},
                },
                fii_engine=lambda **_kwargs: (_ for _ in ()).throw(
                    FiiSelectionError("FII selection violates the configured HHI limit")
                ),
            )

        self.assertEqual(error.exception.code, "infeasible_constraints")

    def test_other_fii_selection_errors_still_classify_as_engine_error(self):
        with self.assertRaises(PremiumOptimizationError) as error:
            run_premium_optimization(
                policy(),
                Manifest(),
                Path("/tmp/prumo-test-run"),
                100_000,
                selector_inputs=selector_inputs(),
                snapshot_loader=lambda _manifest: SimpleNamespace(rows=(), metadata={}),
                allocation_engine=lambda *_args, **_kwargs: allocation_result(),
                stock_engine=lambda **_kwargs: selector_result(["AAA3"]),
                fii_engine=lambda **_kwargs: (_ for _ in ()).throw(
                    FiiSelectionError("fixture boom")
                ),
            )

        failure = error.exception
        self.assertEqual(failure.code, "engine_error")
        self.assertIsNone(failure.details)


STOCK_ZONE = {"k_offset": 6, "K": 8, "delta": 0.11}
FII_ZONE = {"k_offset": 3, "K": 5, "delta": 0.11}


def zone_policy(selector_hhi_max=None):
    value = policy(selector_hhi_max=selector_hhi_max)
    value["stocks"]["protection_zone"] = dict(STOCK_ZONE)
    value["fiis"]["protection_zone"] = dict(FII_ZONE)
    value["provenance"]["policy_version"] = "premium-policy-v3"
    return value


def run_with(policy_value, *, stock_engine, fii_engine, weights=None, **kwargs):
    return run_premium_optimization(
        policy_value,
        Manifest(),
        Path("/tmp/prumo-test-run"),
        100_000,
        selector_inputs=selector_inputs(),
        snapshot_loader=lambda _manifest: SimpleNamespace(rows=(), metadata={}),
        allocation_engine=lambda *_args, **_kwargs: allocation_result(weights),
        stock_engine=stock_engine,
        fii_engine=fii_engine,
        **kwargs,
    )


def recording_engine(calls, key, tickers, extra=None):
    def engine(**kwargs):
        calls[key] = kwargs
        result = selector_result(tickers)
        if kwargs.get("incumbents"):
            # A real selector always returns its class record with incumbents (PZ-20).
            result["protection_zone"] = {
                "status": "applied",
                "incumbents": list(kwargs["incumbents"]),
                "kept": [],
                "entered": [],
                "replaced": [],
            }
        result.update(extra or {})
        return result

    return engine


class ProtectionZoneForwardingTests(unittest.TestCase):
    """PZ-30, PZ-16: zone config and incumbents reach the selectors."""

    def test_incumbents_reach_only_classes_with_a_non_empty_list(self):
        cases = {
            "stocks only": ({"run_id": "r1", "stocks": ["AAA3"], "fiis": []}, ["AAA3"], None),
            "fiis only": ({"run_id": "r1", "stocks": None, "fiis": ["AAA11"]}, None, ["AAA11"]),
            "both": ({"run_id": "r1", "stocks": ["AAA3"], "fiis": ["AAA11"]}, ["AAA3"], ["AAA11"]),
            "none passed": (None, None, None),
        }
        for label, (incumbents, stock_expected, fii_expected) in cases.items():
            with self.subTest(case=label):
                calls = {}
                kwargs = {} if incumbents is None else {"incumbents": incumbents}
                run_with(
                    zone_policy(),
                    stock_engine=recording_engine(calls, "stocks", ["AAA3", "BBB3"]),
                    fii_engine=recording_engine(calls, "fiis", ["AAA11", "BBB11"]),
                    **kwargs,
                )
                for key, expected in (("stocks", stock_expected), ("fiis", fii_expected)):
                    if expected is None:
                        self.assertNotIn("incumbents", calls[key])
                    else:
                        self.assertEqual(calls[key]["incumbents"], expected)
                self.assertEqual(calls["stocks"]["config"]["protection_zone"], STOCK_ZONE)
                self.assertEqual(calls["fiis"]["selection_config"]["protection_zone"], FII_ZONE)

    def test_resolved_premium_policy_zone_is_copied_into_both_selector_configs(self):
        resolved = resolved_policy("premium")
        calls = {}
        run_with(
            resolved,
            stock_engine=recording_engine(calls, "stocks", ["AAA3", "BBB3"]),
            fii_engine=recording_engine(calls, "fiis", ["AAA11", "BBB11"]),
            incumbents={"run_id": "r1", "stocks": ["AAA3"], "fiis": ["AAA11"]},
        )
        self.assertEqual(
            calls["stocks"]["config"]["protection_zone"],
            {"k_offset": 6, "K": resolved.stocks.n_assets + 6, "delta": 0.11},
        )
        self.assertEqual(
            calls["fiis"]["selection_config"]["protection_zone"],
            {"k_offset": 3, "K": resolved.fiis.n_assets + 3, "delta": 0.11},
        )
        self.assertEqual(calls["stocks"]["incumbents"], ["AAA3"])
        self.assertEqual(calls["fiis"]["incumbents"], ["AAA11"])

    def test_policy_without_zone_never_forwards_incumbents(self):
        for label, policy_value in (("basic", resolved_policy("basic")), ("legacy", policy())):
            with self.subTest(policy=label):
                calls = {}
                run_with(
                    policy_value,
                    stock_engine=recording_engine(calls, "stocks", ["AAA3", "BBB3"]),
                    fii_engine=recording_engine(calls, "fiis", ["AAA11", "BBB11"]),
                    incumbents={"run_id": "r1", "stocks": ["AAA3"], "fiis": ["AAA11"]},
                )
                self.assertNotIn("incumbents", calls["stocks"])
                self.assertNotIn("incumbents", calls["fiis"])
                self.assertNotIn("protection_zone", calls["stocks"]["config"])
                self.assertNotIn("protection_zone", calls["fiis"]["selection_config"])

    def test_default_stock_engine_forwards_incumbents(self):
        policy_value = resolved_policy("premium").to_dict()
        calls = []
        pipelines = ModuleType("pipelines")
        pipelines.__path__ = []
        multi_run = ModuleType("pipelines.multi_run")
        multi_run.run_multi_execution_profile = lambda **kwargs: calls.append(kwargs)
        with patch.dict(sys.modules, {"pipelines": pipelines, "pipelines.multi_run": multi_run}):
            for incumbents in (["AAA3", "BBB3"], None):
                kwargs = {} if incumbents is None else {"incumbents": incumbents}
                _default_stock_engine(
                    config=policy_value["stocks"],
                    workspace=Path("/tmp/prumo-test-run/stocks"),
                    seed=19,
                    input_path=Path("/tmp/run/status-invest/stocks.csv"),
                    **kwargs,
                )
        self.assertEqual(calls[0]["incumbents"], ["AAA3", "BBB3"])
        self.assertIsNone(calls[1]["incumbents"])
        self.assertEqual(
            calls[0]["selection_config"]["protection_zone"],
            policy_value["stocks"]["protection_zone"],
        )

    def test_result_policy_omits_the_zone_for_a_basic_dataclass_policy(self):
        results = {}
        for plan in ("basic", "premium"):
            results[plan] = run_with(
                resolved_policy(plan),
                stock_engine=lambda **_kwargs: selector_result(["AAA3", "BBB3"]),
                fii_engine=lambda **_kwargs: selector_result(["AAA11", "BBB11"]),
            )
        basic_policy = results["basic"]["policy"]
        self.assertNotIn("protection_zone", basic_policy["stocks"])
        self.assertNotIn("protection_zone", basic_policy["fiis"])
        self.assertEqual(basic_policy, resolved_policy("basic").to_dict())
        premium_policy = results["premium"]["policy"]
        self.assertEqual(premium_policy["stocks"]["protection_zone"]["k_offset"], 6)
        self.assertEqual(premium_policy["fiis"]["protection_zone"]["k_offset"], 3)

    def test_stock_zone_fallback_that_still_breaks_hhi_is_infeasible(self):
        fallback_record = {"status": "fallback_hhi", "buffer_fallback": "hhi"}
        with self.assertRaises(PremiumOptimizationError) as error:
            run_with(
                zone_policy(selector_hhi_max=0.25),
                stock_engine=recording_engine(
                    {}, "stocks", ["AAA3", "BBB3"],
                    {"metrics": {"hhi": 0.5}, "protection_zone": fallback_record},
                ),
                fii_engine=lambda **_kwargs: selector_result(["AAA11", "BBB11"]),
                incumbents={"run_id": "r1", "stocks": ["AAA3"], "fiis": []},
            )
        self.assertEqual(error.exception.code, "infeasible_constraints")

    def test_stock_zone_portfolio_within_the_limit_is_used(self):
        calls = {}
        result = run_with(
            zone_policy(selector_hhi_max=0.25),
            stock_engine=recording_engine(
                calls, "stocks", ["AAA3", "BBB3"], {"metrics": {"hhi": 0.25}}
            ),
            fii_engine=recording_engine(
                calls, "fiis", ["AAA11", "BBB11"], {"metrics": {"hhi": 0.2}}
            ),
            incumbents={"run_id": "r1", "stocks": ["AAA3"], "fiis": []},
        )
        self.assertEqual([item["ticker"] for item in result["stocks"]], ["AAA3", "BBB3"])
        self.assertEqual(calls["stocks"]["incumbents"], ["AAA3"])


def zone_engine(incumbents, frequencies, zone, *, hhi_of=None, hhi_max=None, calls=None, key=None):
    """Fake selector whose zone output comes from the real core function."""

    def engine(**kwargs):
        if calls is not None:
            calls[key] = kwargs
        selection, record = select_with_fallback(
            kwargs.get("incumbents", incumbents),
            frequencies,
            {ticker: 0.5 for ticker in frequencies},
            2,
            zone,
            eligible=set(frequencies),
            dedup_base_company=False,
            hhi_of=hhi_of or (lambda _tickers: 0.0),
            hhi_max=hhi_max,
        )
        result = selector_result(list(selection.tickers))
        if kwargs.get("incumbents"):
            result["protection_zone"] = record
        return result

    return engine


STOCK_FREQUENCIES = {"DDD3": 0.9, "AAA3": 0.8, "CCC3": 0.5, "EEE3": 0.1}
FII_FREQUENCIES = {"BBB11": 0.9, "AAA11": 0.6, "CCC11": 0.2}


def reasons_by_ticker(constituents):
    return {item["ticker"]: item["reasons"] for item in constituents}


class ProtectionZoneProvenanceTests(unittest.TestCase):
    """PZ-26, PZ-28, PZ-29, PZ-31: the run says what it kept and why."""

    def run_zone(self, incumbents, *, weights=None, stock_engine=None, fii_engine=None, policy_value=None):
        kwargs = {} if incumbents is None else {"incumbents": incumbents}
        return run_with(
            zone_policy() if policy_value is None else policy_value,
            stock_engine=stock_engine or zone_engine([], STOCK_FREQUENCIES, STOCK_ZONE),
            fii_engine=fii_engine or zone_engine([], FII_FREQUENCIES, FII_ZONE),
            weights=weights,
            **kwargs,
        )

    def test_applied_records_the_core_record_and_appends_reasons(self):
        result = self.run_zone(
            {"run_id": "run-1", "stocks": ["AAA3", "CCC3"], "fiis": ["AAA11"]}
        )
        zone = result["provenance"]["protection_zone"]
        self.assertEqual(zone["incumbent_run_id"], "run-1")
        stocks = zone["stocks"]
        self.assertEqual(stocks["status"], "applied")
        self.assertEqual(
            (stocks["K"], stocks["k_offset"], stocks["delta"], stocks["n_assets"]), (8, 6, 0.11, 2)
        )
        self.assertEqual(stocks["incumbents"], ["AAA3", "CCC3"])
        self.assertEqual(stocks["kept"], ["AAA3"])
        self.assertEqual(stocks["replaced"], [{"out": "CCC3", "in": "DDD3"}])
        self.assertEqual(stocks["entered"], ["DDD3"])
        self.assertEqual(stocks["departures"], {"CCC3": "saiu: substituído por DDD3"})
        self.assertEqual((stocks["turnover"], stocks["exits"]), (1, 1))
        self.assertIsNone(stocks["buffer_fallback"])
        self.assertEqual(stocks["consensus_tickers"], ["AAA3", "DDD3"])
        self.assertEqual(stocks["consensus_turnover"], 1)
        self.assertEqual(
            reasons_by_ticker(result["stocks"]),
            {"DDD3": ["substituiu CCC3"], "AAA3": ["mantido"]},
        )
        fiis = zone["fiis"]
        self.assertEqual(fiis["status"], "applied")
        self.assertEqual((fiis["K"], fiis["k_offset"]), (5, 3))
        self.assertEqual(fiis["kept"], ["AAA11"])
        self.assertEqual(fiis["entered"], ["BBB11"])
        self.assertEqual(
            reasons_by_ticker(result["fiis"]), {"BBB11": ["novo"], "AAA11": ["mantido"]}
        )

    def test_hhi_fallback_is_recorded_with_reasons_against_the_consensus(self):
        frequencies = {"DDD3": 0.9, "AAA3": 0.8, "CCC3": 0.75}
        result = self.run_zone(
            {"run_id": "run-1", "stocks": ["CCC3"], "fiis": ["AAA11"]},
            stock_engine=zone_engine(
                [], frequencies, STOCK_ZONE,
                hhi_of=lambda tickers: 1.0 if "CCC3" in tickers else 0.1,
                hhi_max=0.25,
            ),
        )
        stocks = result["provenance"]["protection_zone"]["stocks"]
        self.assertEqual(stocks["status"], "fallback_hhi")
        self.assertEqual(stocks["buffer_fallback"], "hhi")
        self.assertEqual(stocks["departures"], {"CCC3": "saiu: fora do top K"})
        self.assertEqual(
            reasons_by_ticker(result["stocks"]), {"DDD3": ["novo"], "AAA3": ["novo"]}
        )

    def test_class_with_an_empty_incumbent_list_records_no_incumbents(self):
        result = self.run_zone({"run_id": "run-1", "stocks": ["AAA3"], "fiis": []})
        fiis = result["provenance"]["protection_zone"]["fiis"]
        self.assertEqual(
            fiis,
            {"status": "no_incumbents", "K": 5, "k_offset": 3, "delta": 0.11, "n_assets": 2},
        )
        self.assertEqual(reasons_by_ticker(result["fiis"]), {"BBB11": [], "AAA11": []})

    def test_first_run_records_no_incumbents_and_keeps_reasons_unchanged(self):
        result = self.run_zone(None)
        zone = result["provenance"]["protection_zone"]
        self.assertIsNone(zone["incumbent_run_id"])
        self.assertEqual(
            zone["stocks"],
            {"status": "no_incumbents", "K": 8, "k_offset": 6, "delta": 0.11, "n_assets": 2},
        )
        self.assertEqual(zone["fiis"]["status"], "no_incumbents")
        for constituents in (result["stocks"], result["fiis"]):
            self.assertTrue(all(item["reasons"] == [] for item in constituents))

    def test_unreadable_class_records_incumbents_unreadable_and_completes(self):
        calls = {}
        result = self.run_zone(
            {"run_id": "run-1", "stocks": None, "fiis": ["AAA11"]},
            stock_engine=zone_engine([], STOCK_FREQUENCIES, STOCK_ZONE, calls=calls, key="stocks"),
        )
        zone = result["provenance"]["protection_zone"]
        self.assertEqual(zone["incumbent_run_id"], "run-1")
        self.assertEqual(
            zone["stocks"],
            {"status": "incumbents_unreadable", "K": 8, "k_offset": 6, "delta": 0.11, "n_assets": 2},
        )
        self.assertNotIn("incumbents", calls["stocks"])
        self.assertEqual([item["ticker"] for item in result["stocks"]], ["DDD3", "AAA3"])
        self.assertTrue(all(item["reasons"] == [] for item in result["stocks"]))
        self.assertEqual(zone["fiis"]["status"], "applied")

    def test_class_with_zero_weight_records_class_not_selected(self):
        weights = dict(FIXTURE_WEIGHTS, brazilian_stocks=0.0, fixed_income=0.6)
        result = self.run_zone(
            {"run_id": "run-1", "stocks": ["AAA3"], "fiis": ["AAA11"]},
            weights=weights,
            stock_engine=lambda **_kwargs: self.fail("stock selector must not run"),
        )
        zone = result["provenance"]["protection_zone"]
        self.assertEqual(
            zone["stocks"],
            {"status": "class_not_selected", "K": 8, "k_offset": 6, "delta": 0.11, "n_assets": 2},
        )
        self.assertNotIn("turnover", zone["stocks"])
        self.assertEqual(result["stocks"], [])

    def test_policy_without_zone_has_no_protection_zone_provenance(self):
        for label, policy_value in (("basic", resolved_policy("basic")), ("legacy", policy())):
            with self.subTest(policy=label):
                result = self.run_zone(
                    {"run_id": "run-1", "stocks": ["AAA3"], "fiis": ["AAA11"]},
                    policy_value=policy_value,
                )
                self.assertNotIn("protection_zone", result["provenance"])
                self.assertTrue(all(item["reasons"] == [] for item in result["stocks"]))

    def test_selector_ignoring_incumbents_fails_instead_of_recording_a_wrong_status(self):
        with self.assertRaises(PremiumOptimizationError) as error:
            self.run_zone(
                {"run_id": "run-1", "stocks": ["AAA3"], "fiis": []},
                stock_engine=lambda **_kwargs: selector_result(["AAA3", "BBB3"]),
            )
        self.assertEqual(error.exception.code, "engine_error")


if __name__ == "__main__":
    unittest.main()
