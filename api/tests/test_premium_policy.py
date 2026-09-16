import unittest
from decimal import Decimal

from app.db.models import ProfileRecord
from app.services.premium_policy import PremiumPolicyError, resolve_premium_policy


def profile(
    *,
    score: float = 0.5,
    raw_score: float = 0.73,
    generic_profile: str = "moderado",
    restrictions: list[str] | None = None,
    knowledge: float = 0.8,
) -> ProfileRecord:
    return ProfileRecord(
        id="profile-1",
        version=3,
        answers={"restricoes": restrictions or ["nenhuma"]},
        dimensions={
            "apetite": score,
            "capacidade": score,
            "liquidez": score,
            "conhecimento": knowledge,
        },
        suitability_score=Decimal(str(score)),
        raw_score=Decimal(str(raw_score)),
        rules_json=["fixture rule"],
        warnings_json=["fixture warning"],
        restrictions_json=restrictions or ["nenhuma"],
        schema_version=1,
        generic_profile=generic_profile,
    )


class PremiumPolicyTests(unittest.TestCase):
    def test_policy_has_stable_serialization_and_separate_sections(self) -> None:
        kwargs = {
            "source_snapshot_ids": ["allocation-v1", "stock-v1", "fii-v1"],
            "source_snapshot_hashes": {"allocation-v1": "abc"},
            "cutoff_date": "2026-07-21",
            "random_seed": 42,
        }
        resolved = resolve_premium_policy(profile(), **kwargs)

        self.assertEqual(resolved.to_json(), resolve_premium_policy(profile(), **kwargs).to_json())
        self.assertEqual(
            set(resolved.to_dict()),
            {"profile", "allocation", "stocks", "fiis", "provenance"},
        )
        self.assertEqual(resolved.profile.raw_score, 0.73)
        self.assertEqual(resolved.allocation.score, 0.5)
        self.assertEqual(resolved.stocks.system_ga_config, resolved.fiis.system_ga_config)
        self.assertNotEqual(resolved.allocation.risk_adjusted_weights, resolved.stocks.factor_weights)

    def test_score_and_dimensions_drive_selector_values(self) -> None:
        conservative = resolve_premium_policy(
            profile(score=0.1, raw_score=0.2, generic_profile="conservador", knowledge=0.1)
        )
        aggressive = resolve_premium_policy(
            profile(score=0.9, raw_score=0.95, generic_profile="arrojado", knowledge=0.9)
        )

        self.assertLess(conservative.allocation.volatility_cap, aggressive.allocation.volatility_cap)
        self.assertLess(conservative.stocks.n_assets, aggressive.stocks.n_assets)
        self.assertNotEqual(
            conservative.stocks.factor_weights,
            aggressive.stocks.factor_weights,
        )

    def test_each_persisted_dimension_affects_selector_resolution(self) -> None:
        baseline = resolve_premium_policy(profile())
        for dimension in ("apetite", "capacidade", "liquidez", "conhecimento"):
            changed = profile()
            changed.dimensions = {
                **changed.dimensions,
                dimension: 0.1,
            }
            resolved = resolve_premium_policy(changed)
            self.assertNotEqual(
                baseline.stocks.factor_weights,
                resolved.stocks.factor_weights,
                dimension,
            )

    def test_restrictions_intersect_without_relaxing_defaults(self) -> None:
        resolved = resolve_premium_policy(
            profile(
                restrictions=["evitar_exterior", "evitar_cripto", "priorizar_renda", "evitar_illiquidez"]
            )
        )

        constraints = resolved.allocation.class_constraints
        self.assertEqual(constraints["minimum_class_weight"], 0.0)
        self.assertEqual(constraints["minimum_weights"], {"fixed_income": 0.4})
        self.assertEqual(
            constraints["maximum_weights"],
            {"crypto": 0.0, "international_equity": 0.0},
        )
        self.assertTrue(resolved.stocks.liquidity_and_size_filters["enforce_liquidity_and_size"])
        self.assertTrue(resolved.fiis.liquidity_and_size_filters["enforce_liquidity_and_size"])

    def test_concentration_restriction_is_explicit_and_infeasible_intersection_fails(self) -> None:
        resolved = resolve_premium_policy(profile(restrictions=["limitar_concentracao"]))

        self.assertEqual(resolved.allocation.class_constraints["hhi_max"], 0.25)
        self.assertEqual(resolved.stocks.liquidity_and_size_filters["hhi_max"], 0.25)
        self.assertGreaterEqual(resolved.stocks.lambda_hhi, 0.25)

        with self.assertRaises(PremiumPolicyError):
            resolve_premium_policy(
                profile(restrictions=["limitar_concentracao", "evitar_cripto", "evitar_exterior"])
            )

    def test_none_cannot_coexist_with_another_restriction(self) -> None:
        with self.assertRaisesRegex(PremiumPolicyError, "cannot coexist"):
            resolve_premium_policy(profile(restrictions=["nenhuma", "evitar_cripto"]))

    def test_invalid_profile_input_fails_before_policy_is_built(self) -> None:
        with self.assertRaises(PremiumPolicyError):
            resolve_premium_policy(profile(score=1.2))


if __name__ == "__main__":
    unittest.main()
