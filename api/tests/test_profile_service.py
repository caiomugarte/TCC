import unittest

from pydantic import ValidationError

from app.schemas.profile import ProfileSubmission
from app.services.profile import compute_profile


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


if __name__ == "__main__":
    unittest.main()
