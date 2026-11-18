import unittest
from datetime import datetime, timezone

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.db.base import Base
from app.db.models import Account, Entitlement, ProfileRecord, RecommendationRun
from app.repositories.recommendations import RecommendationRepository
from app.schemas.recommendation import PremiumRecommendationRequest
from app.services.premium_policy import POLICY_VERSION
from app.services.premium_recommendation import (
    PremiumRecommendationError,
    PremiumRecommendationService,
)


class ExecutorSpy:
    def __init__(self):
        self.submitted = []

    def submit(self, run_id):
        self.submitted.append(run_id)


def fake_policy(profile, **kwargs):
    return {
        "provenance": {
            "policy_version": kwargs["rules_version"],
            "model_versions": {"allocation": "fixture"},
            "source_snapshot_ids": list(kwargs["source_snapshot_ids"]),
            "source_snapshot_hashes": dict(kwargs["source_snapshot_hashes"]),
            "cutoff_date": kwargs["cutoff_date"],
            "random_seed": kwargs["random_seed"],
        },
        "profile": {"profile_revision": profile.version},
    }


class PremiumRecommendationServiceTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.session = Session(self.engine)
        self.account = Account(email="premium@example.com")
        self.session.add(self.account)
        self.session.flush()
        self.profile = ProfileRecord(
            account_id=self.account.id,
            version=1,
            answers={"restricoes": ["nenhuma"]},
            dimensions={"apetite": 0.5},
            suitability_score=0.5,
            generic_profile="moderado",
            investable_capital_brl=10_000,
            consented_at=datetime.now(timezone.utc),
        )
        self.session.add(self.profile)
        self.session.commit()

    def tearDown(self):
        self.session.close()
        Base.metadata.drop_all(self.engine)
        self.engine.dispose()

    def service(self, executor, *, manifest_id="manifest-fixture", rules_version=POLICY_VERSION):
        manifest = {
            "manifest_id": manifest_id,
            "cutoff_date": "2026-07-21",
            "sources": {},
        }
        return PremiumRecommendationService(
            manifest_loader=lambda: manifest,
            manifest_validator=lambda value: value,
            policy_resolver=fake_policy,
            executor=executor,
            rules_version=rules_version,
        )

    def create(self, service, **request):
        return service.create_run(
            self.account,
            PremiumRecommendationRequest(**request),
            self.session,
        )

    def test_persists_before_submit_and_uses_owned_profile(self):
        executor = ExecutorSpy()
        run = self.service(executor).create_run(
            self.account,
            PremiumRecommendationRequest(),
            self.session,
        )

        self.assertEqual(run.status, "queued")
        self.assertEqual(executor.submitted, [run.id])
        persisted = self.session.scalar(select(RecommendationRun).where(RecommendationRun.id == run.id))
        self.assertEqual(persisted.status, "queued")
        self.assertEqual(persisted.profile_id, self.profile.id)
        self.assertEqual(persisted.provenance_json["manifest_id"], "manifest-fixture")
        self.assertEqual(
            persisted.policy_json,
            fake_policy(
                self.profile,
                rules_version=persisted.policy_version,
                source_snapshot_ids=("manifest-fixture",),
                source_snapshot_hashes={},
                cutoff_date="2026-07-21",
                random_seed=persisted.provenance_json["random_seed"],
            ),
        )

    def test_missing_profile_rejects_before_manifest_or_executor(self):
        executor = ExecutorSpy()
        manifest_called = []
        service = PremiumRecommendationService(
            manifest_loader=lambda: manifest_called.append(True),
            manifest_validator=lambda value: value,
            policy_resolver=fake_policy,
            executor=executor,
        )
        other = Account(email="other@example.com")
        self.session.add(other)
        self.session.commit()

        with self.assertRaises(PremiumRecommendationError) as error:
            service.create_run(other, PremiumRecommendationRequest(), self.session)

        self.assertEqual(error.exception.code, "profile_required")
        self.assertEqual(manifest_called, [])
        self.assertEqual(executor.submitted, [])

    def test_submission_failure_is_persisted_as_failed_run(self):
        class BrokenExecutor:
            def submit(self, _run_id):
                raise RuntimeError("worker unavailable")

        run = self.service(BrokenExecutor()).create_run(
            self.account,
            PremiumRecommendationRequest(),
            self.session,
        )
        self.assertEqual(run.status, "failed")
        self.assertEqual(run.failure_code, "submission_failed")
        self.assertIsNone(run.result_json)

    def test_basic_plan_is_persisted_and_terminal_requests_create_fresh_runs(self):
        executor = ExecutorSpy()
        service = self.service(executor)

        first = service.create_run(
            self.account,
            PremiumRecommendationRequest(),
            self.session,
            plan="basic",
        )
        self.assertEqual(first.plan, "basic")
        self.assertEqual(first.model_version, "basic-v1")
        self.assertEqual(first.provenance_json["plan"], "basic")
        self.assertEqual(len(executor.submitted), 1)

        RecommendationRepository(self.session).mark_failed(first.id, "fixture", "failed")
        self.session.commit()
        second = service.create_run(
            self.account,
            PremiumRecommendationRequest(),
            self.session,
            plan="basic",
        )

        self.assertNotEqual(second.id, first.id)
        self.assertEqual(second.plan, "basic")
        self.assertEqual(executor.submitted, [first.id, second.id])

    def test_premium_same_key_reuses_queued_running_and_completed_run(self):
        executor = ExecutorSpy()
        service = self.service(executor)
        repository = RecommendationRepository(self.session)
        first = self.create(service)

        self.assertEqual(self.create(service, force=False).id, first.id)
        repository.mark_running(first.id)
        self.session.commit()
        self.assertEqual(self.create(service).id, first.id)
        repository.mark_completed(first.id, {"classes": [], "assumptions": [], "risks": []})
        self.session.commit()
        self.assertEqual(self.create(service).id, first.id)
        self.assertEqual(executor.submitted, [first.id])

    def test_premium_new_manifest_creates_new_run_even_while_previous_is_active(self):
        executor = ExecutorSpy()
        repository = RecommendationRepository(self.session)
        old = self.create(self.service(executor, manifest_id="manifest-agosto"))
        repository.mark_running(old.id)
        self.session.commit()

        new = self.create(self.service(executor, manifest_id="manifest-setembro"))

        self.assertNotEqual(new.id, old.id)
        self.assertEqual(new.snapshot_id, "manifest-setembro")
        self.assertEqual(executor.submitted, [old.id, new.id])
        self.assertEqual(
            self.create(self.service(executor, manifest_id="manifest-setembro")).id, new.id
        )

    def test_premium_new_policy_version_creates_new_run(self):
        executor = ExecutorSpy()
        old = self.create(self.service(executor, rules_version="premium-policy-old"))
        RecommendationRepository(self.session).mark_running(old.id)
        self.session.commit()

        new = self.create(self.service(executor, rules_version="premium-policy-new"))

        self.assertNotEqual(new.id, old.id)
        self.assertEqual(new.policy_version, "premium-policy-new")
        self.assertEqual(executor.submitted, [old.id, new.id])

    def test_premium_failed_run_with_same_key_is_not_reused(self):
        executor = ExecutorSpy()
        service = self.service(executor)
        first = self.create(service)
        RecommendationRepository(self.session).mark_failed(first.id, "fixture", "failed")
        self.session.commit()

        second = self.create(service, force=False)

        self.assertNotEqual(second.id, first.id)
        self.assertEqual(second.status, "queued")
        self.assertEqual(executor.submitted, [first.id, second.id])

    def test_premium_force_always_creates_new_run(self):
        executor = ExecutorSpy()
        service = self.service(executor)
        repository = RecommendationRepository(self.session)
        queued = self.create(service)

        forced_over_active = self.create(service, force=True)
        repository.mark_running(forced_over_active.id)
        repository.mark_completed(
            forced_over_active.id, {"classes": [], "assumptions": [], "risks": []}
        )
        self.session.commit()
        forced_over_completed = self.create(service, force=True)

        ids = [queued.id, forced_over_active.id, forced_over_completed.id]
        self.assertEqual(len(set(ids)), 3)
        self.assertEqual(executor.submitted, ids)

    def test_premium_reuse_needs_the_current_manifest(self):
        executor = ExecutorSpy()
        first = self.create(self.service(executor))

        def broken_loader():
            raise FileNotFoundError("registry missing")

        service = PremiumRecommendationService(
            manifest_loader=broken_loader,
            manifest_validator=lambda value: value,
            policy_resolver=fake_policy,
            executor=executor,
        )
        with self.assertRaises(PremiumRecommendationError) as error:
            self.create(service)

        self.assertEqual(error.exception.code, "snapshot_unavailable")
        self.assertEqual(executor.submitted, [first.id])

    def test_find_reusable_run_exposes_the_reuse_rule(self):
        executor = ExecutorSpy()
        service = self.service(executor)
        run = self.create(service)

        found = service.find_reusable_run(
            self.session, self.account.id, self.profile.id, "manifest-fixture"
        )
        missing = service.find_reusable_run(
            self.session, self.account.id, self.profile.id, "manifest-outro"
        )

        self.assertEqual(found.id, run.id)
        self.assertIsNone(missing)

    def test_active_run_reuse_is_scoped_to_plan(self):
        executor = ExecutorSpy()
        service = self.service(executor)
        premium = service.create_run(
            self.account,
            PremiumRecommendationRequest(),
            self.session,
        )
        basic = service.create_run(
            self.account,
            PremiumRecommendationRequest(),
            self.session,
            plan="basic",
        )

        self.assertNotEqual(premium.id, basic.id)
        self.assertEqual(premium.plan, "premium")
        self.assertEqual(basic.plan, "basic")
        self.assertEqual(executor.submitted, [premium.id, basic.id])


if __name__ == "__main__":
    unittest.main()
