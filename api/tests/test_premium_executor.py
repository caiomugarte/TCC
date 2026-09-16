import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.base import Base
from app.db.models import Account, ProfileRecord
from app.repositories.recommendations import RecommendationRepository
from app.services.premium_executor import PremiumExecutor


def create_run(session, account, profile, status="queued"):
    repository = RecommendationRepository(session)
    run = repository.create_queued(
        account_id=account.id,
        profile_id=profile.id,
        policy={"provenance": {"policy_version": "fixture"}},
        provenance={"manifest_id": "manifest-fixture", "manifest": {"fixture": True}},
    )
    if status == "running":
        repository.mark_running(run.id)
    session.commit()
    return run


class PremiumExecutorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.db_path = Path(self.directory.name) / "executor.sqlite"
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        Base.metadata.create_all(self.engine)
        self.sessions = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)
        self.session = self.sessions()
        self.account = Account(email="executor@example.com")
        self.session.add(self.account)
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
        self.session.commit()

    def tearDown(self):
        self.session.close()
        self.engine.dispose()
        self.directory.cleanup()

    @staticmethod
    def result():
        return {
            "classes": [],
            "stocks": [],
            "fiis": [],
            "assumptions": [],
            "risks": [],
        }

    def test_worker_uses_new_sessions_and_commits_terminal_result(self):
        run = create_run(self.session, self.account, self.profile)
        session_count = []

        def session_factory():
            session_count.append(True)
            return self.sessions()

        executor = PremiumExecutor(
            session_factory=session_factory,
            manifest_loader=lambda _provenance: {"fixture": True},
            optimization_runner=lambda *_args: self.result(),
            workspace_root=Path(self.directory.name) / "workspaces",
        )
        try:
            executor.execute_run(run.id)
        finally:
            executor.shutdown()

        with self.sessions() as check:
            stored = check.get(type(run), run.id)
            self.assertEqual(stored.status, "completed")
            self.assertIsNotNone(stored.result_json)
        self.assertGreaterEqual(len(session_count), 2)

    def test_engine_failure_publishes_no_partial_result(self):
        run = create_run(self.session, self.account, self.profile)
        executor = PremiumExecutor(
            session_factory=self.sessions,
            manifest_loader=lambda _provenance: {"fixture": True},
            optimization_runner=lambda *_args: (_ for _ in ()).throw(RuntimeError("fixture boom")),
        )
        try:
            executor.execute_run(run.id)
        finally:
            executor.shutdown()

        with self.sessions() as check:
            stored = check.get(type(run), run.id)
            self.assertEqual(stored.status, "failed")
            self.assertEqual(stored.failure_code, "engine_error")
            self.assertIsNone(stored.result_json)
            self.assertNotIn("fixture boom", stored.failure_message)

    def test_recovery_marks_queued_and_running_runs_failed(self):
        queued = create_run(self.session, self.account, self.profile)
        running = create_run(self.session, self.account, self.profile, status="running")
        executor = PremiumExecutor(session_factory=self.sessions)
        try:
            self.assertEqual(executor.recover_stale(), 2)
        finally:
            executor.shutdown()

        with self.sessions() as check:
            self.assertEqual(check.get(type(queued), queued.id).status, "failed")
            self.assertEqual(check.get(type(running), running.id).status, "failed")


if __name__ == "__main__":
    unittest.main()
