import hashlib
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.base import Base
from app.db.models import Account, ProfileRecord
from app.repositories.recommendations import RecommendationRepository, _output_hash
from app.services.premium_executor import PremiumExecutor
from app.services.status_invest_inputs import SourceSnapshot, StatusInvestInputError, StatusInvestInputs


def create_run(session, account, profile, status="queued", plan="premium"):
    repository = RecommendationRepository(session)
    run = repository.create_queued(
        account_id=account.id,
        profile_id=profile.id,
        plan=plan,
        policy={"plan": plan, "provenance": {"policy_version": "fixture"}},
        provenance={"manifest_id": "manifest-fixture", "manifest": {"fixture": True}},
    )
    if status == "running":
        repository.mark_running(run.id)
    session.commit()
    return run


def selector_inputs_for(workspace):
    source_root = Path(workspace) / "status-invest"
    source_root.mkdir(parents=True, exist_ok=True)
    snapshots = {}
    for role, ticker in (("stocks", "AAA3"), ("fiis", "AAA11")):
        path = source_root / f"{role}.csv"
        path.write_text(f"TICKER\n{ticker}\n", encoding="utf-8")
        snapshots[role] = SourceSnapshot(
            path=path.resolve(),
            provider="statusinvest",
            retrieved_at="2026-07-21T00:00:00Z",
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    return StatusInvestInputs(stocks=snapshots["stocks"], fiis=snapshots["fiis"])


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
            status_invest_inputs_loader=selector_inputs_for,
            optimization_runner=lambda *_args, **_kwargs: self.result(),
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
            status_invest_inputs_loader=selector_inputs_for,
            optimization_runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("fixture boom")),
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

    def test_basic_and_premium_runs_share_lifecycle_and_run_scoped_inputs(self):
        runs = [
            create_run(self.session, self.account, self.profile, plan=plan)
            for plan in ("basic", "premium")
        ]
        invocations = []

        def runner(policy, _manifest, workspace, _capital, *, selector_inputs):
            invocations.append((policy["plan"], Path(workspace), selector_inputs))
            result = self.result()
            result["provenance"] = {"selector_sources": selector_inputs.provenance()}
            return result

        executor = PremiumExecutor(
            session_factory=self.sessions,
            manifest_loader=lambda _provenance: {"fixture": True},
            status_invest_inputs_loader=selector_inputs_for,
            optimization_runner=runner,
            workspace_root=Path(self.directory.name) / "workspaces",
        )
        try:
            for run in runs:
                executor.execute_run(run.id)
        finally:
            executor.shutdown()

        self.assertEqual([entry[0] for entry in invocations], ["basic", "premium"])
        self.assertNotEqual(invocations[0][1], invocations[1][1])
        self.assertTrue(all(entry[2].stocks.path.is_file() for entry in invocations))
        with self.sessions() as check:
            for run, plan in zip(runs, ("basic", "premium")):
                stored = check.get(type(run), run.id)
                self.assertEqual(stored.plan, plan)
                self.assertEqual(stored.status, "completed")

    def test_partial_refresh_failure_skips_runner_and_keeps_snapshot(self):
        run = create_run(self.session, self.account, self.profile, plan="basic")
        calls = []
        workspace_root = Path(self.directory.name) / "workspaces"

        def partial_refresh(workspace):
            path = Path(workspace) / "status-invest" / "stocks.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("TICKER\nAAA3\n", encoding="utf-8")
            raise StatusInvestInputError("collector failed at /private/statusinvest/session.log")

        executor = PremiumExecutor(
            session_factory=self.sessions,
            manifest_loader=lambda _provenance: {"fixture": True},
            status_invest_inputs_loader=partial_refresh,
            optimization_runner=lambda *_args, **_kwargs: calls.append(True),
            workspace_root=workspace_root,
        )
        try:
            executor.execute_run(run.id)
        finally:
            executor.shutdown()

        with self.sessions() as check:
            stored = check.get(type(run), run.id)
            self.assertEqual(stored.status, "failed")
            self.assertEqual(stored.failure_code, "snapshot_unavailable")
            self.assertNotIn("/private/statusinvest", stored.failure_message)
            self.assertIsNone(stored.result_json)
        self.assertEqual(calls, [])
        self.assertTrue((workspace_root / run.id / "status-invest" / "stocks.csv").is_file())

    def test_success_persists_source_provenance_and_output_hash(self):
        run = create_run(self.session, self.account, self.profile)

        def runner(_policy, _manifest, _workspace, _capital, *, selector_inputs):
            result = self.result()
            result["provenance"] = {
                "allocation_history": {"manifest_id": "manifest-fixture"},
                "selector_sources": selector_inputs.provenance(),
            }
            return result

        executor = PremiumExecutor(
            session_factory=self.sessions,
            manifest_loader=lambda _provenance: {"fixture": True},
            status_invest_inputs_loader=selector_inputs_for,
            optimization_runner=runner,
            workspace_root=Path(self.directory.name) / "workspaces",
        )
        try:
            executor.execute_run(run.id)
        finally:
            executor.shutdown()

        with self.sessions() as check:
            stored = check.get(type(run), run.id)
            self.assertEqual(stored.status, "completed")
            self.assertEqual(stored.result_json["provenance"]["allocation_history"]["manifest_id"], "manifest-fixture")
            sources = stored.result_json["provenance"]["selector_sources"]
            self.assertEqual(sources["stocks"]["provider"], "statusinvest")
            self.assertEqual(sources["fiis"]["sha256"], hashlib.sha256(b"TICKER\nAAA11\n").hexdigest())
            self.assertNotIn("path", sources["stocks"])
            self.assertEqual(stored.output_hash, _output_hash(stored.result_json))

    def test_failed_rerun_preserves_previous_completed_result(self):
        previous = create_run(self.session, self.account, self.profile)
        repository = RecommendationRepository(self.session)
        repository.mark_running(previous.id)
        previous_result = self.result()
        previous_result["marker"] = "previous success"
        repository.mark_completed(previous.id, previous_result)
        self.session.commit()
        previous_hash = previous.output_hash
        failed_rerun = create_run(self.session, self.account, self.profile)

        executor = PremiumExecutor(
            session_factory=self.sessions,
            manifest_loader=lambda _provenance: {"fixture": True},
            status_invest_inputs_loader=lambda _workspace: (_ for _ in ()).throw(
                StatusInvestInputError("FII collector failed")
            ),
            optimization_runner=lambda *_args, **_kwargs: self.fail("runner must not execute"),
        )
        try:
            executor.execute_run(failed_rerun.id)
        finally:
            executor.shutdown()

        with self.sessions() as check:
            stored_previous = check.get(type(previous), previous.id)
            stored_failed = check.get(type(failed_rerun), failed_rerun.id)
            latest = RecommendationRepository(check).get_latest_completed_for_profile(
                self.account.id,
                self.profile.id,
                plan="premium",
            )
            self.assertEqual(stored_previous.status, "completed")
            self.assertEqual(stored_previous.result_json["marker"], "previous success")
            self.assertEqual(stored_previous.output_hash, previous_hash)
            self.assertEqual(stored_failed.status, "failed")
            self.assertEqual(latest.id, previous.id)

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
