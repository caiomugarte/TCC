import unittest
from datetime import datetime, timezone
from decimal import Decimal

from pydantic import ValidationError

from app.db.models import ProfileRecord
from app.routers.profile import _response
from app.schemas.profile import ProfileSubmission
from app.services.profile import CRYPTO_SUGGESTION_WARNING, compute_profile


def valid_answers() -> dict[str, str | list[str]]:
    return {
        "objetivo": "crescimento",
        "horizonte": "mais_de_10_anos",
        "capacidade": "30_a_60",
        "reacao": "manter",
        "perda": "10_a_20",
        "experiencia": "intermediaria",
        "liquidez": "mais_de_3_anos",
        "renda": "8_a_20k",
        "patrimonio": "200k_a_1m",
        "concentracao": "30_a_60",
        "necessidade_futura": "10_a_30",
        "produtos": "etf_fii_acoes",
        "operacoes": "ocasional",
        "formacao": "autodidata",
        "restricoes": ["nenhuma"],
    }


def conservative_answers() -> dict[str, str | list[str]]:
    answers = valid_answers()
    answers["objetivo"] = "preservacao"
    answers["reacao"] = "vender_tudo"
    answers["perda"] = "ate_5"
    return answers


class ProfileServiceTests(unittest.TestCase):
    def make_submission(self, answers=None) -> ProfileSubmission:
        return ProfileSubmission(
            answers=answers or valid_answers(),
            investableCapitalBrl=100_000,
            consented=True,
        )

    def test_profile_computation_is_deterministic_and_versioned(self):
        first = compute_profile(self.make_submission())
        second = compute_profile(self.make_submission())

        self.assertEqual(first.model_dump(), second.model_dump())
        self.assertEqual(first.version, 1)
        self.assertEqual(set(first.dimensions), {"apetite", "capacidade", "liquidez", "conhecimento"})
        self.assertGreaterEqual(first.score, 0)
        self.assertLessEqual(first.score, 1)

    def test_short_horizon_caps_profile(self):
        answers = valid_answers()
        answers["horizonte"] = "ate_2_anos"

        profile = compute_profile(self.make_submission(answers))

        self.assertLessEqual(profile.score, 0.25)
        self.assertIn("horizonte de até 2 anos", profile.rules[0])
        self.assertEqual(profile.generic_profile, "conservador")

    def test_invalid_or_incomplete_answers_are_rejected(self):
        answers = valid_answers()
        del answers["liquidez"]

        with self.assertRaises(ValidationError):
            self.make_submission(answers)

    def test_restrictions_are_preserved_as_warning(self):
        answers = valid_answers()
        answers["restricoes"] = ["evitar_cripto"]

        profile = compute_profile(self.make_submission(answers))

        self.assertTrue(any("restrições registradas" in warning for warning in profile.warnings))

    def test_conservative_profile_without_crypto_restriction_suggests_it(self):
        profile = compute_profile(self.make_submission(conservative_answers()))

        self.assertLess(profile.score, 0.25)
        self.assertIn(CRYPTO_SUGGESTION_WARNING, profile.warnings)

    def test_conservative_profile_with_crypto_restriction_has_no_suggestion(self):
        answers = conservative_answers()
        answers["restricoes"] = ["evitar_cripto"]

        profile = compute_profile(self.make_submission(answers))

        self.assertLess(profile.score, 0.25)
        self.assertNotIn(CRYPTO_SUGGESTION_WARNING, profile.warnings)

    def test_non_conservative_profile_has_no_crypto_suggestion(self):
        profile = compute_profile(self.make_submission())

        self.assertGreaterEqual(profile.score, 0.25)
        self.assertNotIn(CRYPTO_SUGGESTION_WARNING, profile.warnings)


class ProfileResponseTests(unittest.TestCase):
    def make_record(self, warnings_json) -> ProfileRecord:
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        return ProfileRecord(
            id="profile-1",
            account_id="account-1",
            version=1,
            answers=valid_answers(),
            dimensions={"apetite": 0.1, "capacidade": 0.5, "liquidez": 0.8, "conhecimento": 0.4},
            suitability_score=Decimal("0.1"),
            warnings_json=warnings_json,
            generic_profile="conservador",
            investable_capital_brl=Decimal("10000"),
            consented_at=now,
            created_at=now,
        )

    def test_response_exposes_profile_warnings(self):
        response = _response(self.make_record([CRYPTO_SUGGESTION_WARNING]))

        self.assertEqual(response.warnings, [CRYPTO_SUGGESTION_WARNING])
        self.assertEqual(
            response.model_dump(by_alias=True)["warnings"], [CRYPTO_SUGGESTION_WARNING]
        )

    def test_response_defaults_missing_warnings_to_empty_list(self):
        response = _response(self.make_record(None))

        self.assertEqual(response.warnings, [])


if __name__ == "__main__":
    unittest.main()
