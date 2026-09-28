import unittest
from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.base import Base
from app.db.models import Account, ProfileRecord, RecommendationRun
from app.repositories.recommendations import (
        RecommendationRepository,
        RecommendationStateError,
)


def result():
    return {
        "classes": [],
        "stocks": [],
        "fiis": [],
        "assumptions": ["fixture"],
        "risks": ["fixture"],
        "provenance": {"random_seed": 19},
    }


class RecommendationRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.session = Session(self.engine)
        self.account = Account(email="one@example.com")
        self.other = Account(email="two@example.com")
        self.session.add_all([self.account, self.other])
        self.session.flush()
        self.profile = ProfileRecord(
            account_id=self.account.id,
            version=1,
            answers={},
            dimensions={},
            suitability_score=0.5,
            generic_profile="moderado",
            investable_capital_brl=1000,
            consented_at=datetime.now(timezone.utc),
        )
        self.session.add(self.profile)
        self.session.flush()
        self.repository = RecommendationRepository(self.session)

    def tearDown(self):
        self.session.close()
        Base.metadata.drop_all(self.engine)
        self.engine.dispose()

    def create_run(
        self,
        *,
        plan="premium",
        account_id=None,
        profile_id=None,
    ):
        run = self.repository.create_queued(
            account_id=account_id or self.account.id,
            profile_id=profile_id or self.profile.id,
            policy={"provenance": {"policy_version": "fixture"}},
            provenance={"manifest_id": "manifest", "cutoff_date": "2026-07-21"},
            plan=plan,
        )
        self.session.commit()
        return run

    def complete_run(self, **kwargs):
        run = self.create_run(**kwargs)
        self.repository.mark_running(run.id)
        self.repository.mark_completed(run.id, result())
        self.session.commit()
        return run

    def test_valid_lifecycle_and_terminal_immutability(self):
        run = self.create_run()
        self.assertEqual(run.status, "queued")
        self.repository.mark_running(run.id)
        self.session.commit()
        self.repository.mark_completed(run.id, result())
        self.session.commit()

        loaded = self.session.get(RecommendationRun, run.id)
        self.assertEqual(loaded.status, "completed")
        self.assertIsNotNone(loaded.result_json)
        with self.assertRaises(RecommendationStateError):
            self.repository.mark_failed(run.id, "late", "must not overwrite")

    def test_invalid_transition_and_account_scoped_read(self):
        run = self.create_run()
        with self.assertRaises(RecommendationStateError):
            self.repository.mark_completed(run.id, result())
        self.assertIsNotNone(self.repository.get_owned_run(self.account.id, run.id))
        self.assertIsNone(self.repository.get_owned_run(self.other.id, run.id))

    def test_failed_run_clears_all_result_fields(self):
        run = self.create_run()
        self.repository.mark_failed(run.id, "engine_error", "fixture failure")
        self.session.commit()
        loaded = self.session.get(RecommendationRun, run.id)
        self.assertEqual(loaded.status, "failed")
        self.assertIsNone(loaded.result_json)
        self.assertEqual(loaded.classes, [])
        self.assertEqual(loaded.failure_code, "engine_error")

    def test_plan_and_completed_status_scope_latest_completed_lookup(self):
        basic = self.complete_run(plan="basic")
        premium = self.complete_run(plan="premium")
        failed_basic = self.create_run(plan="basic")
        self.repository.mark_failed(failed_basic.id, "fixture", "failed")
        self.session.commit()

        self.assertEqual(basic.plan, "basic")
        self.assertEqual(premium.plan, "premium")
        self.assertEqual(
            self.repository.get_latest_completed_for_profile(
                self.account.id, self.profile.id, plan="basic"
            ).id,
            basic.id,
        )
        self.assertEqual(
            self.repository.get_latest_completed_for_profile(
                self.account.id, self.profile.id, plan="premium"
            ).id,
            premium.id,
        )

    def test_latest_completed_lookup_is_account_and_profile_scoped(self):
        matched = self.complete_run(plan="basic")
        other_profile = ProfileRecord(
            account_id=self.account.id,
            version=2,
            answers={},
            dimensions={},
            suitability_score=0.5,
            generic_profile="moderado",
            investable_capital_brl=1000,
            consented_at=datetime.now(timezone.utc),
        )
        other_account_profile = ProfileRecord(
            account_id=self.other.id,
            version=1,
            answers={},
            dimensions={},
            suitability_score=0.5,
            generic_profile="moderado",
            investable_capital_brl=1000,
            consented_at=datetime.now(timezone.utc),
        )
        self.session.add_all([other_profile, other_account_profile])
        self.session.flush()
        self.complete_run(plan="basic", profile_id=other_profile.id)
        self.complete_run(
            plan="basic",
            account_id=self.other.id,
            profile_id=other_account_profile.id,
        )

        selected = self.repository.get_latest_completed_for_profile(
            self.account.id, self.profile.id, plan="basic"
        )
        self.assertEqual(selected.id, matched.id)


if __name__ == "__main__":
    unittest.main()
