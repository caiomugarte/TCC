import re
import unittest
from collections import Counter
from decimal import Decimal
from pathlib import Path

from app.db.models import ProfileRecord
from app.services.premium_policy import (
    BASIC_POLICY_VERSION,
    FII_RULES,
    POLICY_VERSION,
    STOCK_RULES,
    SYSTEM_GA_CONFIG,
    PremiumPolicyError,
    _selector_policy,
    resolve_basic_policy,
    resolve_premium_policy,
    resolve_recommendation_policy,
)
from fii_selection import (
    apply_fii_eligibility,
    build_fii_scores,
    load_fii_data,
    preprocess_fii,
)


SUITABILITY_PAGE = Path(__file__).resolve().parents[2] / "suitability.html"


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
    def test_system_ga_config_uses_max_quality_adaptive_defaults(self) -> None:
        self.assertEqual(SYSTEM_GA_CONFIG["population"], 300)
        self.assertEqual(SYSTEM_GA_CONFIG["generations"], 400)
        self.assertEqual(SYSTEM_GA_CONFIG["run_count"], 150)
        self.assertTrue(SYSTEM_GA_CONFIG["adaptive_mode"])
        self.assertEqual(SYSTEM_GA_CONFIG["min_runs"], 40)
        self.assertEqual(SYSTEM_GA_CONFIG["target_cv"], 0.02)
        self.assertEqual(SYSTEM_GA_CONFIG["target_jaccard"], 0.75)

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

    def test_basic_generic_profile_bands_map_to_discrete_stock_and_fii_rules(self) -> None:
        expected = {
            "conservador": (10, "conservador"),
            "moderado": (12, "moderado"),
            "arrojado": (15, "arrojado"),
        }
        for band, (asset_count, preset) in expected.items():
            with self.subTest(band=band):
                resolved = resolve_basic_policy(profile(generic_profile=band))
                self.assertEqual(resolved.profile.generic_profile, band)
                self.assertEqual(resolved.stocks.n_assets, asset_count)
                self.assertEqual(resolved.fiis.n_assets, asset_count)
                self.assertEqual(resolved.stocks.selection_preset, preset)
                self.assertEqual(resolved.fiis.selection_preset, preset)
                self.assertEqual(resolved.provenance.policy_version, BASIC_POLICY_VERSION)

    def test_basic_policy_uses_shared_ga_and_profile_restrictions(self) -> None:
        resolved = resolve_recommendation_policy(
            profile(generic_profile="conservador", restrictions=["evitar_illiquidez"]),
            plan="basic",
        )

        self.assertEqual(resolved.stocks.system_ga_config, SYSTEM_GA_CONFIG)
        self.assertEqual(resolved.fiis.system_ga_config, SYSTEM_GA_CONFIG)
        self.assertTrue(resolved.stocks.liquidity_and_size_filters["enforce_liquidity_and_size"])
        self.assertTrue(resolved.fiis.liquidity_and_size_filters["enforce_liquidity_and_size"])
        self.assertNotIn("caio", resolved.stocks.selection_preset)

    def test_recommendation_policy_maps_premium_profile_fields_and_restrictions(self) -> None:
        conservative = resolve_recommendation_policy(
            profile(score=0.1, generic_profile="conservador"), plan="premium"
        )
        aggressive = resolve_recommendation_policy(
            profile(score=0.9, generic_profile="arrojado"), plan="premium"
        )
        restricted = resolve_recommendation_policy(
            profile(restrictions=["evitar_illiquidez"]), plan="premium"
        )

        self.assertNotEqual(conservative.stocks.factor_weights, aggressive.stocks.factor_weights)
        self.assertTrue(restricted.stocks.liquidity_and_size_filters["enforce_liquidity_and_size"])
        self.assertEqual(restricted.stocks.system_ga_config, SYSTEM_GA_CONFIG)

    def test_same_basic_profile_resolves_identically(self) -> None:
        expected = resolve_basic_policy(profile(generic_profile="moderado")).to_json()

        self.assertEqual(
            resolve_basic_policy(profile(generic_profile="moderado")).to_json(),
            expected,
        )

    def test_plan_dispatch_keeps_existing_premium_mapping_unchanged(self) -> None:
        kwargs = {
            "source_snapshot_ids": ["allocation-v1"],
            "source_snapshot_hashes": {"allocation-v1": "abc"},
            "cutoff_date": "2026-07-21",
            "random_seed": 42,
        }

        self.assertEqual(
            resolve_recommendation_policy(profile(), plan="premium", **kwargs).to_json(),
            resolve_premium_policy(profile(), **kwargs).to_json(),
        )

    def test_conservador_fii_rules_pin_the_adr_0002_calibration(self) -> None:
        self.assertEqual(
            FII_RULES["conservador"]["filters"],
            {"size_min": 3_000_000_000, "liq_min": 2_000_000},
        )
        self.assertEqual(
            {band: rules["n_assets"] for band, rules in FII_RULES.items()},
            {"conservador": 10, "moderado": 12, "arrojado": 15},
        )
        self.assertEqual(
            FII_RULES["moderado"]["filters"],
            {"size_min": 1_000_000_000, "liq_min": 500_000},
        )
        self.assertEqual(
            FII_RULES["arrojado"]["filters"],
            {"size_min": 200_000_000, "liq_min": 50_000},
        )

    def test_stock_rules_are_untouched_by_the_fii_calibration(self) -> None:
        self.assertEqual(
            STOCK_RULES["conservador"]["filters"],
            {"cap_min": 5_000_000_000, "liq_min": 2_000_000},
        )
        self.assertEqual(
            STOCK_RULES["moderado"]["filters"],
            {"cap_min": 1_000_000_000, "liq_min": 500_000},
        )
        self.assertEqual(
            STOCK_RULES["arrojado"]["filters"],
            {"cap_min": 200_000_000, "liq_min": 50_000},
        )
        self.assertEqual(
            {band: rules["n_assets"] for band, rules in STOCK_RULES.items()},
            {"conservador": 10, "moderado": 12, "arrojado": 15},
        )

    def test_policy_versions_are_bumped_and_reject_the_v1_strings(self) -> None:
        self.assertEqual(POLICY_VERSION, "premium-policy-v2")
        self.assertEqual(BASIC_POLICY_VERSION, "basic-policy-v2")
        with self.assertRaisesRegex(PremiumPolicyError, "unsupported policy version"):
            resolve_premium_policy(profile(), rules_version="premium-policy-v1")
        with self.assertRaisesRegex(PremiumPolicyError, "unsupported policy version"):
            resolve_basic_policy(profile(), rules_version="basic-policy-v1")

    def test_persisted_provenance_reports_the_bumped_policy_versions(self) -> None:
        premium = resolve_premium_policy(profile())
        basic = resolve_recommendation_policy(profile(), plan="basic")

        self.assertEqual(premium.provenance.policy_version, "premium-policy-v2")
        self.assertEqual(basic.provenance.policy_version, "basic-policy-v2")


class PolicyDriftGuardTests(unittest.TestCase):
    """DFT-01: suitability.html mirrors STOCK_RULES; the flag key stays inert."""

    def test_suitability_anchor_filter_pairs_match_stock_rules(self) -> None:
        text = SUITABILITY_PAGE.read_text(encoding="utf-8")
        block = re.search(r"const BASE_PROFILES = \[(.*?)\n    \];", text, re.DOTALL)
        self.assertIsNotNone(block, "suitability.html BASE_PROFILES block not found")

        anchors = [
            (match.group("key"), int(match.group("cap_min")), int(match.group("liq_min")))
            for match in re.finditer(
                r'key:\s*"(?P<key>conservador|moderado|arrojado)"'
                r"[^{}]*?"
                r"filters:\s*\{\s*cap_min:\s*(?P<cap_min>\d+)\s*,\s*liq_min:\s*(?P<liq_min>\d+)\s*\}",
                block.group(1),
            )
        ]

        self.assertEqual(
            [key for key, _cap_min, _liq_min in anchors],
            ["conservador", "moderado", "arrojado"],
        )
        for key, cap_min, liq_min in anchors:
            with self.subTest(anchor=key):
                self.assertEqual(
                    {"cap_min": cap_min, "liq_min": liq_min},
                    STOCK_RULES[key]["filters"],
                )

    def test_enforce_liquidity_and_size_flag_never_changes_the_eligible_set(self) -> None:
        raw = load_fii_data()
        for filters in (
            {"size_min": 3_000_000_000, "liq_min": 2_000_000},
            {"size_min": 1_000_000_000, "liq_min": 500_000},
        ):
            baseline = apply_fii_eligibility(raw, filters=dict(filters))
            expected = baseline["TICKER"].tolist()
            self.assertTrue(expected)
            for flag in (True, False):
                with self.subTest(filters=filters, flag=flag):
                    flagged = apply_fii_eligibility(
                        raw,
                        filters={**filters, "enforce_liquidity_and_size": flag},
                    )
                    self.assertEqual(flagged["TICKER"].tolist(), expected)
                    self.assertEqual(len(flagged), len(baseline))


def _raw_snapshot():
    """The committed snapshot, loaded once (read-only; the pipeline copies)."""
    global _RAW_SNAPSHOT
    if _RAW_SNAPSHOT is None:
        _RAW_SNAPSHOT = load_fii_data()
    return _RAW_SNAPSHOT


_RAW_SNAPSHOT = None


def _min_sector_hhi_sq(sector_multiplicities, n_assets):
    """Exact count-based DP for the minimum achievable sector HHI numerator.

    An equal-weight portfolio of ``n_assets`` assets has sector HHI
    ``Σ (c_s/n)² = Σ c_s² / n²`` over its sector counts ``c_s``, so the minimum
    achievable sector HHI is ``min Σ c_s² / n²`` subject to ``Σ c_s = n_assets``
    and ``0 ≤ c_s ≤ m_s`` (``m_s`` = sector multiplicities of the ranked
    universe). The DP below computes that minimum exactly in integers; subset
    enumeration is deliberately never used (``C(57, 12)`` is infeasible).
    """
    best = {0: 0}
    for multiplicity in sector_multiplicities:
        updated = {}
        for taken, squared in best.items():
            for count in range(0, min(multiplicity, n_assets - taken) + 1):
                total = taken + count
                value = squared + count * count
                if total not in updated or value < updated[total]:
                    updated[total] = value
        best = updated
    return best[n_assets]


def _band_point(selector_score):
    """Resolve one band point through the real ``_selector_policy``/``_interpolate``
    path with ``limitar_concentracao`` and score it with the production pipeline
    (``load_fii_data`` → ``preprocess_fii`` → ``build_fii_scores``)."""
    policy = _selector_policy(FII_RULES, selector_score, {"limitar_concentracao"})
    clean = preprocess_fii(
        _raw_snapshot(),
        eligibility_filters=dict(policy.liquidity_and_size_filters),
    )
    ranked = build_fii_scores(clean, None, factor_weights=policy.factor_weights)
    multiplicities = sorted(Counter(ranked["SETOR"]).values(), reverse=True)
    if len(ranked) < policy.n_assets:
        min_hhi = float("inf")
    else:
        min_hhi = _min_sector_hhi_sq(multiplicities, policy.n_assets) / (
            policy.n_assets**2
        )
    return policy, len(ranked), min_hhi


class PremiumInterpolationBandTests(unittest.TestCase):
    """POL-03 band half: ADR-0002's band claim made executable (21 + 21 points)."""

    def _assert_segment_feasible(self, label, score_of):
        failures = []
        for step in range(21):
            fraction = step / 20
            score = score_of(fraction)
            policy, ranked_count, min_hhi = _band_point(score)
            with self.subTest(segment=label, fraction=f"{fraction:.2f}", score=score):
                self.assertGreaterEqual(ranked_count, policy.n_assets)
                self.assertLessEqual(min_hhi, 0.25 + 1e-12)
            if ranked_count < policy.n_assets or min_hhi > 0.25 + 1e-12:
                failures.append(
                    (round(score, 4), ranked_count, policy.n_assets, round(min_hhi, 4))
                )
        self.assertEqual(
            failures, [], f"{label}: expected 0/21 failing band points, got {failures}"
        )

    def test_min_sector_hhi_dp_matches_the_hand_computed_pins(self) -> None:
        conservador = _selector_policy(FII_RULES, 0.0, {"limitar_concentracao"})
        raw = _raw_snapshot()

        # 10-of-12 at the 3e9 anchor (sector multiplicities 5/3/2/1/1):
        # min Σc² = 3²+3²+2²+1²+1² = 24 -> 24/100 = 0.2400.
        anchor_ranked = build_fii_scores(
            preprocess_fii(
                raw,
                eligibility_filters=dict(conservador.liquidity_and_size_filters),
            ),
            None,
            factor_weights=conservador.factor_weights,
        )
        self.assertEqual(len(anchor_ranked), 12)
        anchor_sq = _min_sector_hhi_sq(Counter(anchor_ranked["SETOR"]).values(), 10)
        self.assertEqual(anchor_sq, 24)
        self.assertAlmostEqual(anchor_sq / 100, 0.2400, places=4)

        # The 7-asset set at the pre-calibration 5e9 anchor IS the portfolio
        # (multiplicities 3/2/1/1): min Σc² = 3²+2²+1²+1² = 15 -> 15/49 = 0.3061.
        full_ranked = build_fii_scores(
            preprocess_fii(
                raw,
                eligibility_filters={"size_min": 5_000_000_000, "liq_min": 2_000_000},
            ),
            None,
            factor_weights=conservador.factor_weights,
        )
        self.assertEqual(len(full_ranked), 7)
        full_sq = _min_sector_hhi_sq(Counter(full_ranked["SETOR"]).values(), 7)
        self.assertEqual(full_sq, 15)
        self.assertAlmostEqual(full_sq / 49, 0.3061, places=4)

    def test_conservador_to_moderado_band_is_feasible_at_all_21_points(self) -> None:
        self._assert_segment_feasible(
            "conservador->moderado", lambda fraction: 0.5 * fraction
        )
        # The coarse score grid 0.00…0.50 step 0.05 is a subset of the sweep
        # (segment fraction steps of 0.05 = score steps of 0.025).
        swept_scores = {round(0.5 * (step / 20), 6) for step in range(21)}
        coarse_scores = {round(0.05 * index, 2) for index in range(11)}
        self.assertTrue(coarse_scores <= swept_scores)

    def test_moderado_to_arrojado_band_is_feasible_at_all_21_points(self) -> None:
        self._assert_segment_feasible(
            "moderado->arrojado", lambda fraction: 0.5 + 0.5 * fraction
        )


if __name__ == "__main__":
    unittest.main()
