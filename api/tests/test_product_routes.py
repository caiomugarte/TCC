import unittest
from unittest.mock import patch

from fastapi import HTTPException
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from app.auth.dependencies import ClerkIdentity, get_current_account
from app.db.base import Base
from app.db.models import Account, Entitlement, PortfolioSnapshot, ProfileRecord, RecommendationRun
from app.entitlements.dependencies import require_premium
from app.routers.portfolio import read_portfolio, save_portfolio
from app.routers.profile import read_profile, save_profile
from app.routers.recommendations import (
    create_recommendation,
    read_latest_completed_recommendation,
    read_latest_recommendation,
    read_recommendation,
    router as recommendations_router,
)
from app.routers.review import read_review
from app.schemas.portfolio import PortfolioInput
from app.schemas.profile import ProfileSubmission
from app.schemas.recommendation import RecommendationRequest
from app.repositories.recommendations import RecommendationRepository
from app.services.premium_recommendation import PremiumRecommendationError
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

    class QueueCoordinator:
        def __init__(self):
            self.submitted = []

        def create_run(self, account, request, session, *, plan):
            profile = session.get(ProfileRecord, request.profile_id) if request.profile_id else session.scalar(
                select(ProfileRecord)
                .where(ProfileRecord.account_id == account.id)
                .order_by(ProfileRecord.version.desc())
                .limit(1)
            )
            if profile is None or profile.account_id != account.id:
                raise PremiumRecommendationError(
                    "profile_not_found",
                    "Perfil não encontrado.",
                    404,
                )
            run = RecommendationRepository(session).create_queued(
                account_id=account.id,
                profile_id=profile.id,
                policy={"plan": plan},
                provenance={"plan": plan},
                plan=plan,
                snapshot_id="fixture-v1",
                snapshot_cutoff="2026-07-21",
                model_version=f"{plan}-v1",
            )
            session.commit()
            session.refresh(run)
            self.submitted.append(run.id)
            return run

    def post_basic(self, request=None):
        coordinator = self.QueueCoordinator()
        with patch(
            "app.routers.recommendations.get_premium_recommendation_service",
            return_value=coordinator,
        ):
            response = create_recommendation(
                request or RecommendationRequest(),
                self.account,
                self.session,
            )
        return response, coordinator

    def save_completed_run(self, profile, *, plan="basic", result=None):
        repository = RecommendationRepository(self.session)
        run = repository.create_queued(
            account_id=profile.account_id,
            profile_id=profile.id,
            policy={"plan": plan},
            provenance={"plan": plan},
            plan=plan,
            snapshot_id=f"{plan}-fixture",
            snapshot_cutoff="2026-07-21",
            model_version=f"{plan}-v1",
        )
        repository.mark_running(run.id)
        repository.mark_completed(
            run.id,
            result or {"classes": [], "assumptions": [], "risks": []},
        )
        self.session.commit()
        return run

    def test_profile_is_versioned_and_reloaded_for_current_account(self):
        first = self.save_valid_profile()
        second = self.save_valid_profile()

        loaded = read_profile(self.account, self.session)

        self.assertEqual(first.version, 1)
        self.assertEqual(second.version, 2)
        self.assertEqual(loaded.id, second.id)
        self.assertEqual(loaded.suitability_score, second.suitability_score)
        self.assertIsNone(read_profile(self.other_account, self.session))

    def test_profile_provenance_is_persisted_with_canonical_restrictions(self):
        answers = valid_answers()
        answers["restricoes"] = ["evitar_exterior", "priorizar_renda", "evitar_exterior"]
        submission = ProfileSubmission(
            answers=answers,
            investableCapitalBrl=100_000,
            consented=True,
        )
        computed = compute_profile(submission)

        profile = save_profile(submission, self.account, self.session)
        stored = self.session.get(ProfileRecord, profile.id)

        self.assertIsNotNone(stored)
        self.assertEqual(float(stored.raw_score), computed.raw_score)
        self.assertEqual(stored.rules_json, computed.rules)
        self.assertEqual(stored.warnings_json, computed.warnings)
        self.assertEqual(stored.restrictions_json, ["priorizar_renda", "evitar_exterior"])
        self.assertEqual(stored.schema_version, 1)

    def test_recommendation_is_persisted_and_cross_account_read_is_hidden(self):
        profile = self.save_valid_profile()
        recommendation, coordinator = self.post_basic()

        loaded = read_recommendation(recommendation.id, self.account, self.session)
        self.assertEqual(loaded.profile_version, profile.version)
        self.assertEqual(loaded.snapshot_id, "fixture-v1")
        self.assertEqual(loaded.status, "queued")
        self.assertEqual(loaded.plan, "basic")
        self.assertEqual(coordinator.submitted, [recommendation.id])
        with self.assertRaises(HTTPException) as error:
            read_recommendation(recommendation.id, self.other_account, self.session)
        self.assertEqual(error.exception.status_code, 404)

    def test_basic_start_is_accepted_and_profile_is_account_scoped(self):
        self.save_valid_profile()
        recommendation, coordinator = self.post_basic()
        post_route = next(
            route for route in recommendations_router.routes
            if getattr(route, "endpoint", None) is create_recommendation
        )
        self.assertEqual(post_route.status_code, 202)
        self.assertEqual(recommendation.status, "queued")
        self.assertEqual(coordinator.submitted, [recommendation.id])

        other_profile = save_profile(
            ProfileSubmission(
                answers=valid_answers(),
                investableCapitalBrl=100_000,
                consented=True,
            ),
            self.other_account,
            self.session,
        )
        coordinator = self.QueueCoordinator()
        with patch(
            "app.routers.recommendations.get_premium_recommendation_service",
            return_value=coordinator,
        ):
            with self.assertRaises(HTTPException) as error:
                create_recommendation(
                    RecommendationRequest(profile_id=other_profile.id),
                    self.account,
                    self.session,
                )
        self.assertEqual(error.exception.status_code, 404)
        self.assertEqual(coordinator.submitted, [])

    def test_basic_and_premium_share_completed_result_shape(self):
        profile = self.save_valid_profile()
        result = {
            "classes": [
                {
                    "key": "brazilian_stocks",
                    "label": "Ações brasileiras",
                    "target_weight": 0.4,
                    "target_amount_brl": 40_000,
                }
            ],
            "assumptions": ["fixture"],
            "risks": ["fixture risk"],
            "stocks": [
                {
                    "ticker": "AAA3",
                    "sleeve_weight": 0.5,
                    "portfolio_weight": 0.2,
                    "target_amount_brl": 20_000,
                    "reasons": ["fixture"],
                }
            ],
            "fiis": [
                {
                    "ticker": "AAA11",
                    "sleeve_weight": 1.0,
                    "portfolio_weight": 0.2,
                    "target_amount_brl": 20_000,
                    "reasons": ["fixture"],
                }
            ],
            "provenance": {"selector_sources": {"stocks": {}, "fiis": {}}},
        }
        for plan in ("basic", "premium"):
            with self.subTest(plan=plan):
                record = self.save_completed_run(profile, plan=plan, result=result)
                response = read_recommendation(record.id, self.account, self.session)
                self.assertEqual(response.status, "completed")
                self.assertEqual(response.classes[0].target_amount_brl, 40_000)
                self.assertEqual(response.stocks[0].ticker, "AAA3")
                self.assertEqual(response.fiis[0].ticker, "AAA11")
                self.assertEqual(response.provenance["selector_sources"]["stocks"], {})

    def test_latest_recommendation_matches_current_profile_and_plan(self):
        first_profile = self.save_valid_profile()
        basic = self.save_completed_run(first_profile)

        self.save_valid_profile()
        self.assertIsNone(read_latest_recommendation(self.account, self.session))

        current_profile = self.session.scalar(
            select(ProfileRecord)
            .where(ProfileRecord.account_id == self.account.id)
            .order_by(ProfileRecord.version.desc())
            .limit(1)
        )
        self.session.add(
            Entitlement(
                account_id=self.account.id,
                plan="premium",
                status="active",
            )
        )
        premium = self.save_completed_run(current_profile, plan="premium")
        self.assertEqual(read_latest_recommendation(self.account, self.session).id, premium.id)

        self.session.scalar(
            select(Entitlement).where(Entitlement.account_id == self.account.id)
        ).status = "inactive"
        self.session.commit()
        self.assertIsNone(read_latest_recommendation(self.account, self.session))
        self.assertNotEqual(first_profile.version, current_profile.version)

    def test_missing_recommendation_input_does_not_publish_partial_result(self):
        self.save_valid_profile()
        class FailingCoordinator:
            def create_run(self, *_args, **_kwargs):
                raise PremiumRecommendationError(
                    "snapshot_unavailable",
                    "A recomendação não está disponível com os dados atuais.",
                    409,
                )

        with patch(
            "app.routers.recommendations.get_premium_recommendation_service",
            return_value=FailingCoordinator(),
        ):
            with self.assertRaises(HTTPException) as error:
                create_recommendation(RecommendationRequest(), self.account, self.session)

        self.assertEqual(error.exception.status_code, 409)
        self.assertEqual(
            self.session.scalar(select(func.count()).select_from(RecommendationRun)),
            0,
        )

    def test_failed_source_poll_exposes_only_sanitized_status(self):
        profile = self.save_valid_profile()
        repository = RecommendationRepository(self.session)
        run = repository.create_queued(
            account_id=self.account.id,
            profile_id=profile.id,
            policy={"plan": "basic"},
            provenance={"plan": "basic"},
            plan="basic",
        )
        repository.mark_running(run.id)
        repository.mark_failed(
            run.id,
            "snapshot_unavailable",
            "Os dados Premium não estão disponíveis para esta execução.",
        )
        self.session.commit()

        response = read_recommendation(run.id, self.account, self.session)

        self.assertEqual(response.status, "failed")
        self.assertEqual(response.failure_code, "snapshot_unavailable")
        self.assertEqual(response.classes, [])
        self.assertEqual(response.stocks, [])
        self.assertEqual(response.fiis, [])

    def test_queued_and_failed_responses_do_not_expose_snapshot_paths(self):
        profile = self.save_valid_profile()
        repository = RecommendationRepository(self.session)
        run = repository.create_queued(
            account_id=self.account.id,
            profile_id=profile.id,
            policy={"plan": "basic"},
            provenance={
                "manifest_path": "/private/snapshots/manifest.json",
                "manifest": {"sources": {"stocks": {"path": "/private/stocks.csv"}}},
            },
            plan="basic",
        )
        self.session.commit()

        self.assertIsNone(read_recommendation(run.id, self.account, self.session).provenance)
        repository.mark_running(run.id)
        repository.mark_failed(run.id, "snapshot_unavailable", "safe diagnostic")
        self.session.commit()
        failed = read_recommendation(run.id, self.account, self.session)
        self.assertEqual(failed.status, "failed")
        self.assertIsNone(failed.provenance)

    def test_latest_completed_skips_pending_and_follows_current_profile_and_plan(self):
        old_profile = self.save_valid_profile()
        self.save_completed_run(old_profile, plan="premium")
        current_profile = self.save_valid_profile()
        premium = self.save_completed_run(current_profile, plan="premium")
        RecommendationRepository(self.session).create_queued(
            account_id=self.account.id,
            profile_id=current_profile.id,
            policy={"plan": "premium"},
            provenance={"plan": "premium"},
            plan="premium",
        )
        self.save_completed_run(current_profile, plan="basic")
        self.session.add(Entitlement(account_id=self.account.id, plan="premium", status="active"))
        self.session.commit()

        latest = read_latest_completed_recommendation(self.account, self.session)

        self.assertEqual(latest.id, premium.id)
        self.assertEqual(latest.profile_version, current_profile.version)
        self.assertEqual(latest.plan, "premium")

        self.session.scalar(
            select(Entitlement).where(Entitlement.account_id == self.account.id)
        ).status = "inactive"
        self.session.commit()
        self.assertEqual(
            read_latest_completed_recommendation(self.account, self.session).plan,
            "basic",
        )

    def test_latest_completed_is_database_only_and_route_precedes_id_route(self):
        profile = self.save_valid_profile()
        completed = self.save_completed_run(profile)
        pending = RecommendationRepository(self.session).create_queued(
            account_id=self.account.id,
            profile_id=profile.id,
            policy={"plan": "basic"},
            provenance={"plan": "basic"},
            plan="basic",
        )
        self.session.commit()
        paths = [route.path for route in recommendations_router.routes]
        self.assertLess(
            paths.index("/v1/recommendations/latest-completed"),
            paths.index("/v1/recommendations/{recommendation_id}"),
        )

        with patch(
            "app.routers.recommendations.get_premium_recommendation_service",
            side_effect=AssertionError("GET must not start or refresh a run"),
        ):
            latest = read_latest_completed_recommendation(self.account, self.session)
            status = read_recommendation(pending.id, self.account, self.session)
        self.assertEqual(latest.id, completed.id)
        self.assertEqual(status.status, "queued")

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
        coordinator = self.QueueCoordinator()
        with patch(
            "app.routers.recommendations.get_premium_recommendation_service",
            return_value=coordinator,
        ):
            recommendation = create_recommendation(RecommendationRequest(), account, self.session)
        repository = RecommendationRepository(self.session)
        repository.mark_running(recommendation.id)
        repository.mark_completed(
            recommendation.id,
            {
                "classes": engine_result["classes"],
                "assumptions": engine_result["assumptions"],
                "risks": engine_result["risks"],
            },
        )
        self.session.commit()
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
