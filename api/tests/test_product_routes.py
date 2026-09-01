import unittest
from unittest.mock import patch

from fastapi import HTTPException
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from app.auth.dependencies import ClerkIdentity, get_current_account
from app.db.base import Base
from app.db.models import Account, PortfolioSnapshot, RecommendationRun
from app.entitlements.dependencies import require_premium
from app.routers.portfolio import read_portfolio, save_portfolio
from app.routers.profile import read_profile, save_profile
from app.routers.recommendations import create_recommendation, read_recommendation
from app.routers.review import read_review
from app.schemas.portfolio import PortfolioInput
from app.schemas.profile import ProfileSubmission
from app.schemas.recommendation import RecommendationRequest


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


class ProductRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.session = Session(self.engine)
        self.account = Account(email="one@example.com")
        self.other_account = Account(email="two@example.com")
        self.session.add_all([self.account, self.other_account])
        self.session.flush()

    def tearDown(self) -> None:
        self.session.close()
        Base.metadata.drop_all(self.engine)
        self.engine.dispose()

    def save_valid_profile(self):
        return save_profile(
            ProfileSubmission(
                answers=valid_answers(),
                investableCapitalBrl=100_000,
                consented=True,
            ),
            self.account,
            self.session,
        )

    def test_profile_is_versioned_and_reloaded_for_current_account(self):
        first = self.save_valid_profile()
        second = self.save_valid_profile()

        loaded = read_profile(self.account, self.session)

        self.assertEqual(first.version, 1)
        self.assertEqual(second.version, 2)
        self.assertEqual(loaded.id, second.id)
        self.assertEqual(loaded.suitability_score, second.suitability_score)
        self.assertIsNone(read_profile(self.other_account, self.session))

    def test_recommendation_is_persisted_and_cross_account_read_is_hidden(self):
        profile = self.save_valid_profile()
        engine_result = {
            "plan": "basic",
            "model_version": "allocation-v1",
            "snapshot_id": "fixture-v1",
            "snapshot_cutoff": "2026-07-21",
            "classes": [
                {"key": "brazilian_stocks", "label": "Ações brasileiras", "target_weight": 0.2, "target_amount_brl": 20_000},
                {"key": "fiis", "label": "FIIs", "target_weight": 0.2, "target_amount_brl": 20_000},
                {"key": "international", "label": "Exposição internacional", "target_weight": 0.2, "target_amount_brl": 20_000},
                {"key": "fixed_income", "label": "Renda fixa", "target_weight": 0.3, "target_amount_brl": 30_000},
                {"key": "crypto", "label": "Criptoativos", "target_weight": 0.1, "target_amount_brl": 10_000},
            ],
            "assumptions": ["fixture"],
            "risks": ["fixture"],
        }
        with patch("app.routers.recommendations.generate_basic_recommendation", return_value=engine_result):
            recommendation = create_recommendation(RecommendationRequest(), self.account, self.session)

        loaded = read_recommendation(recommendation.id, self.account, self.session)
        self.assertEqual(loaded.profile_version, profile.version)
        self.assertEqual(loaded.snapshot_id, "fixture-v1")
        with self.assertRaises(HTTPException) as error:
            read_recommendation(recommendation.id, self.other_account, self.session)
        self.assertEqual(error.exception.status_code, 404)

    def test_missing_recommendation_input_does_not_publish_partial_result(self):
        self.save_valid_profile()
        with patch(
            "app.routers.recommendations.generate_basic_recommendation",
            side_effect=ValueError("snapshot missing"),
        ):
            with self.assertRaises(HTTPException) as error:
                create_recommendation(RecommendationRequest(), self.account, self.session)

        self.assertEqual(error.exception.status_code, 409)
        self.assertEqual(
            self.session.scalar(select(func.count()).select_from(RecommendationRun)),
            0,
        )

    def test_portfolio_normalizes_values_and_preserves_history(self):
        first = save_portfolio(
            PortfolioInput(currency="BRL", classes={
                "brazilian_stocks": 100,
                "fiis": 200,
                "international": 300,
                "fixed_income": 400,
                "crypto": 0,
            }),
            self.account,
            self.session,
        )
        second = save_portfolio(
            PortfolioInput(currency="BRL", classes={
                "brazilian_stocks": 200,
                "fiis": 200,
                "international": 200,
                "fixed_income": 200,
                "crypto": 200,
            }),
            self.account,
            self.session,
        )

        loaded = read_portfolio(self.account, self.session)
        history_count = self.session.scalar(
            select(func.count()).select_from(PortfolioSnapshot).where(PortfolioSnapshot.account_id == self.account.id)
        )

        self.assertEqual(first.total_value_brl, 1000)
        self.assertEqual(sum(first.normalized_weights.values()), 1)
        self.assertEqual(loaded.id, second.id)
        self.assertEqual(history_count, 2)

    def test_zero_total_portfolio_is_rejected(self):
        with self.assertRaises(HTTPException) as error:
            save_portfolio(
                PortfolioInput(currency="BRL", classes={key: 0 for key in (
                    "brazilian_stocks", "fiis", "international", "fixed_income", "crypto"
                )}),
                self.account,
                self.session,
            )
        self.assertEqual(error.exception.status_code, 422)

    def test_signup_fixture_reaches_review_after_reload_without_db_edits(self):
        account = get_current_account(
            ClerkIdentity("fixture_user", "fixture@example.com"),
            self.session,
        )
        profile = save_profile(
            ProfileSubmission(
                answers=valid_answers(),
                investableCapitalBrl=100_000,
                consented=True,
            ),
            account,
            self.session,
        )
        engine_result = {
            "plan": "basic",
            "model_version": "allocation-v1",
            "snapshot_id": "fixture-v1",
            "snapshot_cutoff": "2026-07-21",
            "classes": [
                {"key": "brazilian_stocks", "label": "Ações brasileiras", "target_weight": 0.2, "target_amount_brl": 20_000},
                {"key": "fiis", "label": "FIIs", "target_weight": 0.2, "target_amount_brl": 20_000},
                {"key": "international", "label": "Exposição internacional", "target_weight": 0.2, "target_amount_brl": 20_000},
                {"key": "fixed_income", "label": "Renda fixa", "target_weight": 0.3, "target_amount_brl": 30_000},
                {"key": "crypto", "label": "Criptoativos", "target_weight": 0.1, "target_amount_brl": 10_000},
            ],
            "assumptions": ["fixture"],
            "risks": ["fixture"],
        }
        with patch("app.routers.recommendations.generate_basic_recommendation", return_value=engine_result):
            recommendation = create_recommendation(RecommendationRequest(), account, self.session)
        portfolio = save_portfolio(
            PortfolioInput(currency="BRL", classes={
                "brazilian_stocks": 20_000,
                "fiis": 20_000,
                "international": 20_000,
                "fixed_income": 30_000,
                "crypto": 10_000,
            }),
            account,
            self.session,
        )

        account_id = account.id
        self.session.close()
        self.session = Session(self.engine)
        reloaded_account = self.session.get(Account, account_id)
        self.assertIsNotNone(reloaded_account)
        self.assertEqual(read_profile(reloaded_account, self.session).id, profile.id)
        self.assertEqual(
            read_recommendation(recommendation.id, reloaded_account, self.session).id,
            recommendation.id,
        )
        self.assertEqual(read_portfolio(reloaded_account, self.session).id, portfolio.id)
        self.assertEqual(len(read_review(reloaded_account, self.session).items), 5)

        other_account = get_current_account(
            ClerkIdentity("other_fixture_user", "other@example.com"),
            self.session,
        )
        self.assertIsNone(read_profile(other_account, self.session))
        self.assertIsNone(read_portfolio(other_account, self.session))
        with self.assertRaises(HTTPException) as cross_account_error:
            read_recommendation(recommendation.id, other_account, self.session)
        self.assertEqual(cross_account_error.exception.status_code, 404)
        with self.assertRaises(HTTPException) as review_error:
            read_review(other_account, self.session)
        self.assertEqual(review_error.exception.status_code, 409)
        with self.assertRaises(HTTPException) as premium_error:
            require_premium(reloaded_account, self.session)
        self.assertEqual(premium_error.exception.status_code, 403)


if __name__ == "__main__":
    unittest.main()
