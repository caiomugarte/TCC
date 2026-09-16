import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.base import Base
from app.db.models import Account, Entitlement, ProfileRecord
from app.repositories.recommendations import RecommendationRepository
from app.routers.premium import start_premium_recommendation
from app.routers.recommendations import read_recommendation
from app.schemas.recommendation import PremiumRecommendationRequest
from app.services.premium_recommendation import PremiumRecommendationService


class ExecutorSpy:
    def submit(self, _run_id):
        return None


def policy(profile, **kwargs):
    return {
        "provenance": {
            "policy_version": kwargs["rules_version"],
            "source_snapshot_ids": list(kwargs["source_snapshot_ids"]),
            "source_snapshot_hashes": {},
            "cutoff_date": kwargs["cutoff_date"],
            "random_seed": kwargs["random_seed"],
            "model_versions": {},
        },
        "profile": {"profile_revision": profile.version},
    }


class PremiumRouteTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.session = Session(self.engine)
        self.account = Account(email="premium@example.com")
        self.other = Account(email="other@example.com")
        self.session.add_all([self.account, self.other])
        self.session.flush()
        self.profile = ProfileRecord(
            account_id=self.account.id,
            version=1,
            answers={},
            dimensions={},
            suitability_score=0.5,
            generic_profile="moderado",
            investable_capital_brl=10_000,
            consented_at=datetime.now(timezone.utc),
        )
        self.session.add(self.profile)
        self.entitlement = Entitlement(
            account_id=self.account.id,
            plan="premium",
            status="active",
        )
        self.session.add(self.entitlement)
        self.session.commit()

    def tearDown(self):
        self.session.close()
        Base.metadata.drop_all(self.engine)
        self.engine.dispose()

    def service(self):
        manifest = {"manifest_id": "fixture", "cutoff_date": "2026-07-21", "sources": {}}
        return PremiumRecommendationService(
            manifest_loader=lambda: manifest,
            manifest_validator=lambda value: value,
            policy_resolver=policy,
            executor=ExecutorSpy(),
        )

    def test_start_returns_queued_and_status_read_is_account_scoped(self):
        service = self.service()
        with patch(
            "app.routers.premium.create_premium_run",
            side_effect=lambda account, request, session: service.create_run(account, request, session),
        ):
            response = start_premium_recommendation(
                PremiumRecommendationRequest(),
                self.account,
                self.session,
                self.entitlement,
            )

        self.assertEqual(response.status, "queued")
        self.assertEqual(response.plan, "premium")
        self.assertEqual(response.classes, [])
        with self.assertRaises(HTTPException) as error:
            read_recommendation(response.id, self.other, self.session)
        self.assertEqual(error.exception.status_code, 404)

    def test_failed_status_hides_partial_result(self):
        repository = RecommendationRepository(self.session)
        run = repository.create_queued(
            account_id=self.account.id,
            profile_id=self.profile.id,
            policy={"provenance": {"policy_version": "fixture"}},
            provenance={"manifest_id": "fixture"},
        )
        repository.mark_failed(run.id, "engine_error", "fixture")
        self.session.commit()

        response = read_recommendation(run.id, self.account, self.session)
        self.assertEqual(response.status, "failed")
        self.assertEqual(response.classes, [])
        self.assertEqual(response.stocks, [])
        self.assertEqual(response.failure_code, "engine_error")


if __name__ == "__main__":
    unittest.main()
