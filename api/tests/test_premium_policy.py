import json
import re
import unittest
from collections import Counter
from decimal import Decimal
from pathlib import Path
from unittest import mock

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
    GeneticAlgorithm,
    apply_fii_eligibility,
    build_fii_scores,
    load_fii_data,
    preprocess_fii,
    run_fii_selection,
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

    def test_concentration_restriction_applies_only_to_selectors(self) -> None:
        resolved = resolve_premium_policy(profile(restrictions=["limitar_concentracao"]))

        self.assertIsNone(resolved.allocation.class_constraints["hhi_max"])
        self.assertEqual(resolved.stocks.liquidity_and_size_filters["hhi_max"], 0.25)
        self.assertEqual(resolved.fiis.liquidity_and_size_filters["hhi_max"], 0.25)
        self.assertGreaterEqual(resolved.stocks.lambda_hhi, 0.25)
        self.assertGreaterEqual(resolved.fiis.lambda_hhi, 0.25)

    def test_concentration_with_avoided_classes_is_feasible(self) -> None:
        resolved = resolve_premium_policy(
            profile(restrictions=["limitar_concentracao", "evitar_cripto", "evitar_exterior"])
        )

        constraints = resolved.allocation.class_constraints
        self.assertIsNone(constraints["hhi_max"])
        self.assertEqual(
            constraints["maximum_weights"],
            {"crypto": 0.0, "international_equity": 0.0},
        )
        self.assertEqual(resolved.stocks.liquidity_and_size_filters["hhi_max"], 0.25)

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
            {"conservador": 4, "moderado": 5, "arrojado": 6},
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
        self.assertEqual(POLICY_VERSION, "premium-policy-v4")
        self.assertEqual(BASIC_POLICY_VERSION, "basic-policy-v2")
        with self.assertRaisesRegex(PremiumPolicyError, "unsupported policy version"):
            resolve_premium_policy(profile(), rules_version="premium-policy-v1")
        with self.assertRaisesRegex(PremiumPolicyError, "unsupported policy version"):
            resolve_basic_policy(profile(), rules_version="basic-policy-v1")

    def test_persisted_provenance_reports_the_bumped_policy_versions(self) -> None:
        premium = resolve_premium_policy(profile())
        basic = resolve_recommendation_policy(profile(), plan="basic")

        self.assertEqual(premium.provenance.policy_version, "premium-policy-v4")
        self.assertEqual(basic.provenance.policy_version, "basic-policy-v2")


# Basic to_json() for profile(generic_profile="moderado",
# restrictions=["limitar_concentracao"]), captured at 76a6d70 (before #46).
BASIC_GOLDEN_JSON = (
    '{"allocation":{"class_constraints":{"hhi_max":null,"maximum_weights":{},'
    '"minimum_class_weight":0.0,"minimum_weights":{}},"crypto_risk_contribution_cap":0.4,'
    '"drawdown_cap":0.25,"hhi_penalty":0.25,"risk_adjusted_weights":{"drawdown":0.25,'
    '"return":0.5,"volatility":0.25},"score":0.5,"volatility_cap":0.15},'
    '"fiis":{"factor_weights":{"dividend":0.1,"growth":0.2,"liquidity":0.2,"size_cash":0.25,'
    '"value":0.25},"lambda_hhi":0.25,"liquidity_and_size_filters":{"enforce_liquidity_and_size":false,'
    '"hhi_max":0.25,"liq_min":500000.0,"size_min":1000000000.0},"n_assets":12,'
    '"selection_preset":"moderado","system_ga_config":{"adaptive_mode":true,"crossover_rate":0.8,'
    '"generations":400,"min_runs":40,"mutation_rate":0.02,"population":300,"run_count":150,'
    '"target_cv":0.02,"target_jaccard":0.75}},'
    '"profile":{"applied_rules":["fixture rule"],"dimensions":{"apetite":0.5,"capacidade":0.5,'
    '"conhecimento":0.8,"liquidez":0.5},"generic_profile":"moderado","profile_revision":3,'
    '"raw_score":0.73,"restrictions":["limitar_concentracao"],"schema_version":1,"score":0.5,'
    '"warnings":["fixture warning"]},'
    '"provenance":{"cutoff_date":null,"model_versions":{"allocation":"allocation-v1",'
    '"fiis":"fii-selection-v1","stocks":"stock-selection-v1"},"policy_version":"basic-policy-v2",'
    '"random_seed":null,"source_snapshot_hashes":{},"source_snapshot_ids":[]},'
    '"stocks":{"factor_weights":{"div":0.1,"growth":0.2,"liquidez":0.2,"rent":0.25,"value":0.25},'
    '"lambda_hhi":0.25,"liquidity_and_size_filters":{"cap_min":1000000000.0,'
    '"enforce_liquidity_and_size":false,"hhi_max":0.25,"liq_min":500000.0},"n_assets":12,'
    '"selection_preset":"moderado","system_ga_config":{"adaptive_mode":true,"crossover_rate":0.8,'
    '"generations":400,"min_runs":40,"mutation_rate":0.02,"population":300,"run_count":150,'
    '"target_cv":0.02,"target_jaccard":0.75}}}'
)


class ProtectionZonePolicyTests(unittest.TestCase):
    """PZ-21..PZ-23: the zone lives in the Premium policy v3 only."""

    def test_premium_policy_v2_is_rejected(self) -> None:
        with self.assertRaisesRegex(PremiumPolicyError, "unsupported policy version"):
            resolve_premium_policy(profile(), rules_version="premium-policy-v2")
        with self.assertRaisesRegex(PremiumPolicyError, "unsupported policy version"):
            resolve_recommendation_policy(
                profile(), rules_version="premium-policy-v2", plan="premium"
            )

    def test_premium_sections_carry_the_issue_9_zone_values_for_every_anchor(self) -> None:
        anchors = {
            "conservador": (0.0, 10, 4),
            "moderado": (0.5, 12, 5),
            "arrojado": (1.0, 15, 6),
        }
        for band, (score, stock_n, fii_n) in anchors.items():
            with self.subTest(band=band):
                resolved = resolve_premium_policy(
                    profile(score=score, knowledge=score, generic_profile=band)
                )
                self.assertEqual(resolved.stocks.n_assets, stock_n)
                self.assertEqual(resolved.fiis.n_assets, fii_n)
                self.assertEqual(
                    resolved.stocks.protection_zone,
                    {"k_offset": 6, "K": stock_n + 6, "delta": 0.11},
                )
                self.assertEqual(
                    resolved.fiis.protection_zone,
                    {"k_offset": 3, "K": fii_n + 3, "delta": 0.11},
                )
                payload = resolved.to_dict()
                self.assertEqual(
                    payload["stocks"]["protection_zone"],
                    {"k_offset": 6, "K": stock_n + 6, "delta": 0.11},
                )
                self.assertEqual(
                    payload["fiis"]["protection_zone"],
                    {"k_offset": 3, "K": fii_n + 3, "delta": 0.11},
                )

    def test_zone_k_follows_an_interpolated_asset_count(self) -> None:
        resolved = resolve_premium_policy(profile(score=0.75, knowledge=0.75))
        self.assertEqual(resolved.stocks.protection_zone["K"], resolved.stocks.n_assets + 6)
        self.assertEqual(resolved.fiis.protection_zone["K"], resolved.fiis.n_assets + 3)

    def test_basic_policy_has_no_zone_and_matches_the_pre_change_json(self) -> None:
        resolved = resolve_basic_policy(
            profile(generic_profile="moderado", restrictions=["limitar_concentracao"])
        )
        self.assertIsNone(resolved.stocks.protection_zone)
        self.assertIsNone(resolved.fiis.protection_zone)
        payload = resolved.to_dict()
        self.assertNotIn("protection_zone", payload["stocks"])
        self.assertNotIn("protection_zone", payload["fiis"])
        self.assertNotIn("protection_zone", resolved.to_json())
        self.assertEqual(resolved.to_json(), BASIC_GOLDEN_JSON)


class SegmentCapPolicyTests(unittest.TestCase):
    """SEG-01..SEG-08 (issue #56): the Premium FII sleeve is capped per segment."""

    def test_premium_fiis_carry_max_per_segment_2_for_every_score(self) -> None:
        for score in (0.0, 0.25, 0.5, 0.7, 1.0):
            with self.subTest(score=score):
                resolved = resolve_premium_policy(profile(score=score, knowledge=score))
                self.assertEqual(resolved.fiis.max_per_segment, 2)
                self.assertEqual(resolved.to_dict()["fiis"]["max_per_segment"], 2)
                self.assertEqual(
                    json.loads(resolved.to_json())["fiis"]["max_per_segment"], 2
                )

    def test_stocks_have_no_max_per_segment(self) -> None:
        for score in (0.0, 0.5, 1.0):
            with self.subTest(score=score):
                resolved = resolve_premium_policy(profile(score=score, knowledge=score))
                self.assertIsNone(resolved.stocks.max_per_segment)
                self.assertNotIn("max_per_segment", resolved.to_dict()["stocks"])
                self.assertNotIn(
                    "max_per_segment", json.loads(resolved.to_json())["stocks"]
                )

    def test_fii_count_interpolates_half_to_even(self) -> None:
        # 4 + 1 * 0.5 = 4.5 -> 4 and 5 + 1 * 0.5 = 5.5 -> 6 (Python round, A-14).
        for score, n_assets in ((0.25, 4), (0.75, 6)):
            with self.subTest(score=score):
                resolved = resolve_premium_policy(profile(score=score, knowledge=score))
                self.assertEqual(resolved.fiis.n_assets, n_assets)

    def test_owner_profile_resolves_five_fiis_capped_at_two(self) -> None:
        resolved = resolve_premium_policy(profile(score=0.7, knowledge=0.7))
        self.assertEqual(resolved.provenance.policy_version, "premium-policy-v4")
        self.assertEqual(resolved.fiis.n_assets, 5)
        self.assertEqual(resolved.fiis.max_per_segment, 2)
        self.assertEqual(
            resolved.fiis.protection_zone, {"k_offset": 3, "K": 8, "delta": 0.11}
        )
        self.assertEqual(resolved.stocks.n_assets, 13)

    def test_premium_policy_v3_is_rejected(self) -> None:
        with self.assertRaisesRegex(PremiumPolicyError, "unsupported policy version"):
            resolve_premium_policy(profile(), rules_version="premium-policy-v3")
        with self.assertRaisesRegex(PremiumPolicyError, "unsupported policy version"):
            resolve_recommendation_policy(
                profile(), rules_version="premium-policy-v3", plan="premium"
            )

    def test_model_versions_record_fii_selection_v2(self) -> None:
        premium_versions = {
            "allocation": "allocation-v1",
            "stocks": "stock-selection-v1",
            "fiis": "fii-selection-v2",
        }
        basic_versions = {**premium_versions, "fiis": "fii-selection-v1"}
        premium = resolve_premium_policy(profile())
        self.assertEqual(dict(premium.provenance.model_versions), premium_versions)
        self.assertEqual(
            json.loads(premium.to_json())["provenance"]["model_versions"], premium_versions
        )
        self.assertEqual(
            dict(resolve_recommendation_policy(profile(), plan="premium").provenance.model_versions),
            premium_versions,
        )
        for basic in (
            resolve_basic_policy(profile()),
            resolve_recommendation_policy(profile(), plan="basic"),
        ):
            with self.subTest(policy_version=basic.provenance.policy_version):
                self.assertEqual(dict(basic.provenance.model_versions), basic_versions)
                self.assertEqual(
                    json.loads(basic.to_json())["provenance"]["model_versions"], basic_versions
                )


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


class _RecordingGeneticAlgorithm(GeneticAlgorithm):
    """Captures the GA constructor's ``hhi_max`` to prove the nested-filters
    wiring reaches the GA settings (``optimize_fii_portfolio`` seam)."""

    constructed_with: list = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        type(self).constructed_with.append(kwargs.get("hhi_max"))


class ConservadorAnchorExecutabilityTests(unittest.TestCase):
    """POL-03 anchor half: ADR-0002's central claim made executable — the
    conservador anchor runs under ``limitar_concentracao`` and the only wiring
    difference is the nested ``hhi_max``."""

    PINNED_SEED = 42
    GA_BUDGET = {"generations": 30, "pop_size": 20}

    def _anchor_fiis_policy(self, restrictions):
        # Conservador anchor through the real resolution path: selector score
        # 0.0 -> _selector_policy/_interpolate at f=0 (premium_policy.py:199-243).
        resolved = resolve_premium_policy(
            profile(
                score=0.0,
                raw_score=0.0,
                generic_profile="conservador",
                restrictions=restrictions,
                knowledge=0.0,
            )
        )
        return resolved.fiis

    def _run_real_selector(self, selector_policy, n_assets=None):
        config = {
            "selection_preset": "conservador",
            "factor_weights": dict(selector_policy.factor_weights),
            "liquidity_and_size_filters": dict(
                selector_policy.liquidity_and_size_filters
            ),
            "n_assets": selector_policy.n_assets if n_assets is None else n_assets,
            "lambda_hhi": selector_policy.lambda_hhi,
            "system_ga_config": dict(selector_policy.system_ga_config),
        }
        _RecordingGeneticAlgorithm.constructed_with = []
        with mock.patch("fii_selection.GeneticAlgorithm", _RecordingGeneticAlgorithm):
            result = run_fii_selection(
                selection_config=config,
                n_runs=1,
                ga_config=dict(self.GA_BUDGET),
                random_seed=self.PINNED_SEED,
            )
        return result

    def test_conservador_anchor_executes_with_the_hhi_cap_reaching_the_ga(self) -> None:
        restricted = self._anchor_fiis_policy(["limitar_concentracao"])
        self.assertEqual(restricted.liquidity_and_size_filters["hhi_max"], 0.25)

        result = self._run_real_selector(restricted)

        self.assertEqual(_RecordingGeneticAlgorithm.constructed_with, [0.25])
        self.assertLessEqual(result["hhi"], 0.25 + 1e-12)
        self.assertEqual(result["n_selected"], restricted.n_assets)

    def test_unrestricted_anchor_differs_only_by_hhi_max_and_runs_above_the_cap(self) -> None:
        restricted = self._anchor_fiis_policy(["limitar_concentracao"])
        unrestricted = self._anchor_fiis_policy(["nenhuma"])

        restricted_filters = dict(restricted.liquidity_and_size_filters)
        unrestricted_filters = dict(unrestricted.liquidity_and_size_filters)
        self.assertEqual(restricted_filters.pop("hhi_max"), 0.25)
        self.assertNotIn("hhi_max", unrestricted_filters)
        self.assertEqual(restricted_filters, unrestricted_filters)

        # Subject: the uncapped GA exceeds HHI 0.25 at the conservador anchor.
        # At the v4 count (4) it reaches 0.25 on its own, so the pre-#56 count
        # 10 keeps the subject (spec A-12).
        result = self._run_real_selector(unrestricted, n_assets=10)

        self.assertEqual(_RecordingGeneticAlgorithm.constructed_with, [None])
        self.assertGreater(result["hhi"], 0.25)


class _CapRecordingGeneticAlgorithm(GeneticAlgorithm):
    """Captures the GA constructor's ``max_per_segment`` (SEG-27 on real data)."""

    constructed_with: list = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        type(self).constructed_with.append(kwargs.get("max_per_segment"))


A6_DEPARTURE_REASONS = {
    "saiu: inelegível",
    "saiu: teto do segmento",
    "saiu: vagas reduzidas",
    "saiu: fora do top K",
}


class V4FiiExecutabilityTests(unittest.TestCase):
    """SEG-05, SEG-27, SEG-42, SEG-43: the v4 FII sections run on the committed
    ``data/raw/status_invest_fii.csv`` within the segment cap."""

    ANCHOR_SEED = 42
    ANCHOR_GA_BUDGET = {"generations": 30, "pop_size": 20}
    MANY_RUNS = 8
    MANY_RUNS_GA_BUDGET = {"generations": 40, "pop_size": 30}
    SEEDS = (0, 1, 2, 3)

    @staticmethod
    def _v4_fiis(score, restrictions=None):
        return resolve_premium_policy(
            profile(score=score, knowledge=score, restrictions=restrictions)
        ).fiis

    @staticmethod
    def _config(fiis):
        # Same keys the adapter hands the FII engine for a v4 section.
        return {
            "selection_preset": fiis.selection_preset,
            "factor_weights": dict(fiis.factor_weights),
            "liquidity_and_size_filters": dict(fiis.liquidity_and_size_filters),
            "n_assets": fiis.n_assets,
            "lambda_hhi": fiis.lambda_hhi,
            "system_ga_config": dict(fiis.system_ga_config),
            "protection_zone": dict(fiis.protection_zone),
            "max_per_segment": fiis.max_per_segment,
        }

    def _assert_capped_sleeve(self, result, n_assets, hhi_max=None):
        self.assertEqual(result["n_selected"], n_assets)
        self.assertEqual(len(result["selected_tickers"]), n_assets)
        segments = Counter(result["portfolio"]["SETOR"].astype(str))
        self.assertLessEqual(max(segments.values()), 2, segments)
        if hhi_max is not None:
            self.assertLessEqual(result["hhi"], hhi_max + 1e-12)

    def test_real_data_anchors_fill_n_assets_within_the_cap(self) -> None:
        for score, n_assets in ((0.0, 4), (0.5, 5), (0.7, 5), (1.0, 6)):
            with self.subTest(score=score):
                fiis = self._v4_fiis(score)
                self.assertEqual((fiis.n_assets, fiis.max_per_segment), (n_assets, 2))
                _CapRecordingGeneticAlgorithm.constructed_with = []
                with mock.patch("fii_selection.GeneticAlgorithm", _CapRecordingGeneticAlgorithm):
                    result = run_fii_selection(
                        selection_config=self._config(fiis),
                        n_runs=1,
                        ga_config=dict(self.ANCHOR_GA_BUDGET),
                        random_seed=self.ANCHOR_SEED,
                    )
                self.assertEqual(_CapRecordingGeneticAlgorithm.constructed_with, [2])
                self._assert_capped_sleeve(result, n_assets)

    def _owner_with_limitar_concentracao(self):
        fiis = self._v4_fiis(0.7, ["limitar_concentracao"])
        self.assertEqual(
            (fiis.n_assets, fiis.max_per_segment, fiis.liquidity_and_size_filters["hhi_max"]),
            (5, 2, 0.25),
        )
        return fiis

    def _run_many(self, fiis, seed, incumbents=None):
        return run_fii_selection(
            selection_config=self._config(fiis),
            n_runs=self.MANY_RUNS,
            ga_config=dict(self.MANY_RUNS_GA_BUDGET),
            random_seed=seed,
            incumbents=incumbents,
        )

    def test_owner_profile_with_limitar_concentracao_runs_many_ga_runs(self) -> None:
        # Without the HHI skip (AD-028) seed 0 raises "FII selection violates
        # the configured HHI limit" (spec review r1 S1).
        fiis = self._owner_with_limitar_concentracao()
        for seed in self.SEEDS:
            with self.subTest(seed=seed):
                result = self._run_many(fiis, seed)
                self.assertEqual(result["n_runs"], self.MANY_RUNS)
                self._assert_capped_sleeve(result, 5, hhi_max=0.25)
                self.assertNotIn("protection_zone", result)

    def test_owner_profile_with_limitar_concentracao_and_13_incumbents_runs(self) -> None:
        # The 13 -> 5 migration: incumbents are the 13 best-SCORE funds of the
        # scored universe. Without the protection walk's HHI term (AD-029)
        # seed 0 falls back to the K = 0 consensus with status "fallback_hhi".
        fiis = self._owner_with_limitar_concentracao()
        ranked = build_fii_scores(
            preprocess_fii(
                _raw_snapshot(),
                eligibility_filters=dict(fiis.liquidity_and_size_filters),
            ),
            None,
            factor_weights=fiis.factor_weights,
        )
        incumbents = (
            ranked.sort_values(["SCORE", "TICKER"], ascending=[False, True])["TICKER"]
            .head(13)
            .tolist()
        )
        self.assertEqual(len(incumbents), 13)
        for seed in self.SEEDS:
            with self.subTest(seed=seed):
                result = self._run_many(fiis, seed, incumbents=incumbents)
                self._assert_capped_sleeve(result, 5, hhi_max=0.25)
                zone = result["protection_zone"]
                self.assertNotEqual(zone["status"], "fallback_hhi")
                self.assertEqual(zone["status"], "applied")
                kept = set(zone["kept"])
                departed = set(zone["departures"])
                self.assertEqual(kept & departed, set())
                self.assertEqual(kept | departed, set(incumbents))
                self.assertLessEqual(kept, set(result["selected_tickers"]))
                for ticker, reason in zone["departures"].items():
                    self.assertTrue(
                        reason in A6_DEPARTURE_REASONS
                        or reason.startswith("saiu: substituído por "),
                        (ticker, reason),
                    )


if __name__ == "__main__":
    unittest.main()
