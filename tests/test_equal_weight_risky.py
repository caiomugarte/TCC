import random
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))

from allocation_config import ALLOCATION_PROFILE_ANCHORS, ASSET_CLASSES  # noqa: E402
from allocation_profiles import build_anchor_profiles, interpolate_profile  # noqa: E402
from core.allocation import (  # noqa: E402
    AllocationError,
    DailyReturn,
    is_feasible,
    portfolio_metrics,
    simulate_portfolio,
)
from pipelines.equal_weight_risky import (  # noqa: E402
    METHOD_ID,
    RISKY_CLASSES,
    run_equal_weight_risky,
)


def synthetic_rows(years: float, seed: int = 7, end: date = date(2026, 10, 28)):
    """Weekday rows: noisy risky classes and a near-riskless fixed income."""

    rng = random.Random(seed)
    start = end - timedelta(days=int(years * 365.25))
    rows = []
    day = start
    while day <= end:
        if day.weekday() < 5:
            returns = {name: rng.gauss(0.0005, 0.03) for name in RISKY_CLASSES}
            returns["fixed_income"] = 0.0003 + rng.gauss(0.0, 0.0001)
            rows.append(DailyReturn(day, returns))
        day += timedelta(days=1)
    return tuple(rows)


def caps(score: float):
    profile = interpolate_profile(score, build_anchor_profiles(ALLOCATION_PROFILE_ANCHORS))
    return {
        "volatility_cap": float(profile.volatility_cap),
        "drawdown_cap": float(profile.drawdown_cap),
    }


LOOSE = {"volatility_cap": 10.0, "drawdown_cap": 1.0}
NO_CONSTRAINTS = {"minimum_weights": {}, "maximum_weights": {}, "hhi_max": None}


class EqualWeightRiskyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = synthetic_rows(12)

    def run_method(self, config, constraints=NO_CONSTRAINTS, rows=None, metadata=None):
        return run_equal_weight_risky(
            self.rows if rows is None else rows,
            metadata=metadata,
            allocation_config=config,
            class_constraints=constraints,
        )

    def test_higher_scores_get_a_larger_risky_share_split_equally(self):
        shares = []
        for score in (0.0, 0.58, 1.0):
            result = self.run_method(caps(score))
            weights = result["weights"]
            share = result["provenance"]["risky_share"]
            shares.append(share)

            self.assertEqual(set(weights), set(ASSET_CLASSES))
            self.assertAlmostEqual(sum(weights.values()), 1.0, places=12)
            self.assertAlmostEqual(weights["fixed_income"], 1.0 - share, places=12)
            for name in RISKY_CLASSES:
                self.assertAlmostEqual(weights[name], share / 4, places=12)
            self.assertAlmostEqual(share * 100, round(share * 100), places=9)
            self.assertTrue(result["provenance"]["risk"]["within_caps"])
        self.assertTrue(0 < shares[0] < shares[1] < shares[2] < 1, shares)

    def test_risky_share_is_the_largest_step_within_both_caps(self):
        config = caps(0.58)
        provenance = self.run_method(config)["provenance"]
        steps = round(provenance["risky_share"] * 100)
        risk = provenance["risk"]
        self.assertLessEqual(risk["annualized_volatility"], config["volatility_cap"])
        self.assertLessEqual(risk["max_drawdown"], config["drawdown_cap"])

        window = tuple(
            row for row in self.rows if row.date.isoformat() >= provenance["window"]["start_date"]
        )
        for larger in range(steps + 1, 101):
            share = larger / 100
            weights = {name: share / 4 for name in RISKY_CLASSES}
            weights["fixed_income"] = 1 - share
            metrics = portfolio_metrics(
                simulate_portfolio(window, weights, ASSET_CLASSES, annual_rebalance=True)
            )
            self.assertFalse(
                is_feasible(metrics, config["volatility_cap"], config["drawdown_cap"]),
                larger,
            )

    def test_avoided_classes_leave_the_split(self):
        constraints = {
            "minimum_weights": {},
            "maximum_weights": {"crypto": 0.0, "international_equity": 0.0},
            "hhi_max": None,
        }
        result = self.run_method(caps(0.58), constraints)
        weights = result["weights"]
        provenance = result["provenance"]

        self.assertEqual(weights["crypto"], 0.0)
        self.assertEqual(weights["international_equity"], 0.0)
        self.assertGreater(weights["brazilian_stocks"], 0.0)
        self.assertEqual(weights["brazilian_stocks"], weights["fiis"])
        self.assertAlmostEqual(
            weights["brazilian_stocks"] * 2, provenance["risky_share"], places=12
        )
        self.assertEqual(provenance["excluded_classes"], ["international_equity", "crypto"])
        self.assertEqual(provenance["risky_classes"], ["brazilian_stocks", "fiis"])

    def test_income_priority_keeps_fixed_income_at_least_forty_percent(self):
        constraints = {
            "minimum_weights": {"fixed_income": 0.40},
            "maximum_weights": {},
            "hhi_max": None,
        }
        result = self.run_method(LOOSE, constraints)

        self.assertAlmostEqual(result["weights"]["fixed_income"], 0.40, places=12)
        self.assertEqual(result["provenance"]["risky_share"], 0.60)
        self.assertEqual(result["provenance"]["risky_share_ceiling"], 0.60)
        self.assertEqual(result["provenance"]["fixed_income_minimum"], 0.40)

    def test_class_hhi_limit_is_not_applied(self):
        constraints = {"minimum_weights": {}, "maximum_weights": {}, "hhi_max": 0.25}
        result = self.run_method(caps(0.0), constraints)
        hhi = sum(value * value for value in result["weights"].values())
        self.assertGreater(hhi, 0.25)

    def test_no_feasible_share_is_all_fixed_income_not_an_error(self):
        # Fixed income alone has some noise, so a zero cap fits nothing.
        result = self.run_method({"volatility_cap": 0.0, "drawdown_cap": 0.0})

        self.assertEqual(result["weights"]["fixed_income"], 1.0)
        self.assertTrue(all(result["weights"][name] == 0.0 for name in RISKY_CLASSES))
        self.assertEqual(result["provenance"]["risky_share"], 0.0)
        self.assertFalse(result["provenance"]["risk"]["within_caps"])
        self.assertEqual(result["status"], "ok")
        self.assertNotIn("diagnostics", result)

    def test_all_risky_classes_excluded_is_all_fixed_income(self):
        constraints = {
            "minimum_weights": {},
            "maximum_weights": {name: 0.0 for name in RISKY_CLASSES},
            "hhi_max": None,
        }
        result = self.run_method(LOOSE, constraints)
        self.assertEqual(result["weights"]["fixed_income"], 1.0)
        self.assertEqual(result["provenance"]["risky_classes"], [])

    def test_window_is_the_last_ten_years(self):
        result = self.run_method(caps(0.58))
        window = result["provenance"]["window"]

        self.assertEqual(window["end_date"], "2026-10-28")
        self.assertGreaterEqual(window["start_date"], "2016-10-28")
        self.assertLess(window["start_date"], "2016-11-02")
        self.assertFalse(window["full_history"])
        self.assertEqual(window["requested_years"], 10)
        self.assertAlmostEqual(window["years_used"], 10.0, delta=0.02)
        self.assertLess(window["observations"], len(self.rows))

    def test_old_history_outside_the_window_does_not_change_the_result(self):
        crash = tuple(
            DailyReturn(
                row.date,
                {name: (-0.5 if name in RISKY_CLASSES else 0.0) for name in ASSET_CLASSES},
            )
            for row in self.rows[:20]
        )
        with_crash = crash + self.rows[20:]
        self.assertEqual(
            self.run_method(caps(0.58), rows=with_crash)["weights"],
            self.run_method(caps(0.58))["weights"],
        )

    def test_short_history_uses_everything_and_records_the_real_window(self):
        rows = synthetic_rows(5)
        window = self.run_method(caps(0.58), rows=rows)["provenance"]["window"]

        self.assertTrue(window["full_history"])
        self.assertEqual(window["start_date"], rows[0].date.isoformat())
        self.assertEqual(window["observations"], len(rows))
        self.assertAlmostEqual(window["years_used"], 5.0, delta=0.02)

    def test_provenance_is_complete(self):
        metadata = {
            "sources": {
                "allocation": {
                    "path": "/srv/snapshots/data/snap-x",
                    "sha256": "alloc-hash",
                    "provider": "prumo-snapshot",
                    "metadata": {"files": {"fixed_income": "cdi.csv"}},
                },
                "crypto": {
                    "path": "/srv/snapshots/data/btc.csv",
                    "sha256": "btc-hash",
                    "provider": "yahoo",
                    "metadata": {},
                },
            }
        }
        config = caps(0.58)
        result = self.run_method(config, metadata=metadata)
        provenance = result["provenance"]

        self.assertEqual(result["method"], METHOD_ID)
        self.assertEqual(provenance["method"], "equal-weight-risky-v1")
        self.assertEqual(
            set(provenance),
            {
                "method", "risky_share", "risky_share_step", "risky_share_ceiling",
                "risky_classes", "excluded_classes", "fixed_income_minimum",
                "window", "risk", "class_series",
            },
        )
        self.assertEqual(provenance["risk"]["volatility_cap"], config["volatility_cap"])
        self.assertEqual(provenance["risk"]["drawdown_cap"], config["drawdown_cap"])
        self.assertEqual(
            provenance["risk"]["annualized_volatility"],
            result["metrics"]["annualized_volatility"],
        )
        self.assertEqual(
            provenance["risk"]["max_drawdown"], abs(result["metrics"]["max_drawdown"])
        )
        series = provenance["class_series"]
        self.assertEqual(set(series), set(ASSET_CLASSES))
        self.assertEqual(series["brazilian_stocks"]["file"], "ibovespa.csv")
        self.assertEqual(series["fiis"]["file"], "ifix.csv")
        self.assertEqual(series["fixed_income"]["file"], "cdi.csv")
        self.assertEqual(series["brazilian_stocks"]["sha256"], "alloc-hash")
        self.assertEqual(series["crypto"]["source"], "crypto")
        self.assertEqual(series["crypto"]["file"], "btc.csv")
        self.assertEqual(series["crypto"]["fx"]["file"], "ptax.csv")
        self.assertEqual(series["international_equity"]["fx"]["file"], "ptax.csv")
        self.assertNotIn("/srv/", repr(provenance))

    def test_missing_caps_are_rejected(self):
        with self.assertRaises(AllocationError):
            self.run_method({"volatility_cap": 0.1})
        with self.assertRaises(AllocationError):
            self.run_method(caps(0.58), rows=())


if __name__ == "__main__":
    unittest.main()
