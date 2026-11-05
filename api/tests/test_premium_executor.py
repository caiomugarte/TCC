import hashlib
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.base import Base
from app.db.models import Account, ProfileRecord
from app.adapters.premium_optimization import PremiumOptimizationError
from app.repositories.recommendations import RecommendationRepository, _output_hash
from app.services.premium_executor import PremiumExecutor
from app.services.status_invest_inputs import SourceSnapshot, StatusInvestInputError, StatusInvestInputs


SHORTAGE_FIELDS = {
    "profile": "conservador",
    "eligible_count": 7,
    "universe_count": 7,
    "required_count": 10,
    "size_min": 5_000_000_000.0,
    "liq_min": 2_000_000.0,
}

FROZEN_SHORTAGE_MESSAGE = (
    "Universo FII insuficiente: 7 ativos utilizáveis (7 elegíveis) "
    "para 10 exigidos; limiares size_min=5000000000, liq_min=2000000; "
    "perfil conservador."
)


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

    def test_universe_shortage_persists_the_frozen_diagnostic(self):
        run = create_run(self.session, self.account, self.profile, plan="basic")
        shortage = PremiumOptimizationError(
            "7 ativos utilizáveis (7 elegíveis) para 10 exigidos",
            "infeasible_universe",
            dict(SHORTAGE_FIELDS),
        )
        executor = PremiumExecutor(
            session_factory=self.sessions,
            manifest_loader=lambda _provenance: {"fixture": True},
            status_invest_inputs_loader=selector_inputs_for,
            optimization_runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(shortage),
        )
        try:
            executor.execute_run(run.id)
        finally:
            executor.shutdown()

        with self.sessions() as check:
            stored = check.get(type(run), run.id)
            self.assertEqual(stored.status, "failed")
            self.assertEqual(stored.failure_code, "infeasible_universe")
            self.assertEqual(stored.failure_message, FROZEN_SHORTAGE_MESSAGE)
            self.assertIsNone(stored.result_json)

    def test_shortage_message_never_leaks_raw_text_or_paths(self):
        run = create_run(self.session, self.account, self.profile)
        shortage = PremiumOptimizationError(
            "fixture boom at /private/statusinvest/session.log",
            "infeasible_universe",
            dict(SHORTAGE_FIELDS),
        )
        executor = PremiumExecutor(
            session_factory=self.sessions,
            manifest_loader=lambda _provenance: {"fixture": True},
            status_invest_inputs_loader=selector_inputs_for,
            optimization_runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(shortage),
        )
        try:
            executor.execute_run(run.id)
        finally:
            executor.shutdown()

        with self.sessions() as check:
            stored = check.get(type(run), run.id)
            self.assertEqual(stored.failure_code, "infeasible_universe")
            self.assertEqual(stored.failure_message, FROZEN_SHORTAGE_MESSAGE)
            self.assertNotIn("fixture boom", stored.failure_message)
            self.assertNotIn("/private/statusinvest", stored.failure_message)
            self.assertNotIn("/", stored.failure_message)
            self.assertLessEqual(len(stored.failure_message), 512)
            self.assertLessEqual(len(stored.failure_code), 64)

    def test_other_failure_codes_keep_their_fixed_strings(self):
        fixed = {
            "snapshot_changed": "Os dados Premium foram alterados e a execução foi cancelada.",
            "snapshot_unavailable": "Os dados Premium não estão disponíveis para esta execução.",
            "infeasible_constraints": "As restrições do perfil não produziram uma carteira factível.",
            "policy_invalid": "O perfil não produziu uma política Premium válida.",
            "submission_failed": "A execução Premium não pôde ser enviada ao worker.",
            "worker_restarted": "A execução foi interrompida e precisa ser solicitada novamente.",
            "engine_error": "A execução Premium não produziu um resultado.",
        }
        for code, message in fixed.items():
            with self.subTest(code=code):
                self.assertEqual(PremiumExecutor._safe_failure_message(code), message)
        self.assertEqual(
            PremiumExecutor._safe_failure_message(
                "infeasible_constraints", dict(SHORTAGE_FIELDS)
            ),
            fixed["infeasible_constraints"],
        )


ZONE_POLICY = {
    "plan": "premium",
    "stocks": {"n_assets": 2, "protection_zone": {"k_offset": 6, "K": 8, "delta": 0.11}},
    "fiis": {"n_assets": 2, "protection_zone": {"k_offset": 3, "K": 5, "delta": 0.11}},
    "provenance": {"policy_version": "premium-policy-v3"},
}


class PremiumExecutorIncumbentTests(unittest.TestCase):
    """PZ-24..PZ-27: the executor hands the previous Premium portfolio to the runner."""

    setUp = PremiumExecutorTests.setUp
    tearDown = PremiumExecutorTests.tearDown
    result = staticmethod(PremiumExecutorTests.result)

    def zone_run(self, *, plan="premium", account=None, profile=None, policy=None):
        repository = RecommendationRepository(self.session)
        run = repository.create_queued(
            account_id=(account or self.account).id,
            profile_id=(profile or self.profile).id,
            plan=plan,
            policy=ZONE_POLICY if policy is None else policy,
            provenance={"manifest_id": "manifest-fixture", "manifest": {"fixture": True}},
        )
        self.session.commit()
        return run

    def completed_run(self, result_json, **kwargs):
        run = self.zone_run(**kwargs)
        repository = RecommendationRepository(self.session)
        repository.mark_running(run.id)
        repository.mark_completed(run.id, self.result())
        self.session.commit()
        run.result_json = result_json
        self.session.commit()
        return run

    def portfolio(self, stocks, fiis):
        result = self.result()
        result["stocks"] = [{"ticker": ticker, "weight": 0.1} for ticker in stocks]
        result["fiis"] = [{"ticker": ticker, "weight": 0.1} for ticker in fiis]
        return result

    def execute(self, run, runner):
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
            return check.get(type(run), run.id).status

    def recording_runner(self, calls):
        def runner(_policy, _manifest, _workspace, _capital, **kwargs):
            calls.append(kwargs)
            return self.result()

        return runner

    def test_no_prior_run_calls_a_fixed_signature_runner_as_today(self):
        run = self.zone_run()
        calls = []

        def runner(_policy, _manifest, _workspace, _capital, *, selector_inputs):
            calls.append(selector_inputs)
            return self.result()

        self.assertEqual(self.execute(run, runner), "completed")
        self.assertEqual(len(calls), 1)
        self.assertIsInstance(calls[0], StatusInvestInputs)

    def test_prior_completed_premium_run_is_passed_normalized(self):
        previous = self.completed_run(
            self.portfolio([" bbse3", "BBSE3", "itsa4"], ["hglg11", "KNRI11 ", "hglg11"])
        )
        run = self.zone_run()
        calls = []

        self.assertEqual(self.execute(run, self.recording_runner(calls)), "completed")
        self.assertEqual(
            calls[0]["incumbents"],
            {"run_id": previous.id, "stocks": ["BBSE3", "ITSA4"], "fiis": ["HGLG11", "KNRI11"]},
        )
        self.assertIsInstance(calls[0]["selector_inputs"], StatusInvestInputs)

    def test_run_that_completed_while_this_one_was_queued_is_the_incumbent(self):
        first = self.zone_run()
        second = self.zone_run()
        calls = []

        def runner(_policy, _manifest, _workspace, _capital, **kwargs):
            calls.append(kwargs)
            return self.portfolio(["AAA3"], ["AAA11"])

        self.assertEqual(self.execute(first, runner), "completed")
        self.assertEqual(self.execute(second, runner), "completed")
        self.assertNotIn("incumbents", calls[0])
        self.assertEqual(
            calls[1]["incumbents"],
            {"run_id": first.id, "stocks": ["AAA3"], "fiis": ["AAA11"]},
        )

    def test_unreadable_result_shapes_give_none_for_that_class_and_complete(self):
        cases = {
            "result is null": (None, None, None),
            "result is not an object": (["BBSE3"], None, None),
            "class is not a list": ({"stocks": {"ticker": "BBSE3"}, "fiis": [{"ticker": "HGLG11"}]}, None, ["HGLG11"]),
            "entry is not an object": ({"stocks": ["BBSE3"], "fiis": [{"ticker": "HGLG11"}]}, None, ["HGLG11"]),
            "entry without ticker": ({"stocks": [{"weight": 1}], "fiis": [{"ticker": "HGLG11"}]}, None, ["HGLG11"]),
            "ticker not a string": ({"stocks": [{"ticker": 3}], "fiis": [{"ticker": "HGLG11"}]}, None, ["HGLG11"]),
            "ticker blank": ({"stocks": [{"ticker": "BBSE3"}], "fiis": [{"ticker": "  "}]}, ["BBSE3"], None),
            "class missing": ({"stocks": [{"ticker": "BBSE3"}]}, ["BBSE3"], None),
            "empty class": ({"stocks": [], "fiis": [{"ticker": "HGLG11"}]}, [], ["HGLG11"]),
        }
        for label, (result_json, stocks, fiis) in cases.items():
            with self.subTest(case=label):
                previous = self.completed_run(result_json)
                run = self.zone_run()
                calls = []
                self.assertEqual(self.execute(run, self.recording_runner(calls)), "completed")
                self.assertEqual(
                    calls[0]["incumbents"],
                    {"run_id": previous.id, "stocks": stocks, "fiis": fiis},
                )
                with self.sessions() as cleanup:
                    for stale in (previous, run):
                        cleanup.delete(cleanup.get(type(stale), stale.id))
                    cleanup.commit()

    def test_basic_run_never_queries_for_incumbents(self):
        self.completed_run(self.portfolio(["BBSE3"], ["HGLG11"]))
        run = self.zone_run(plan="basic")
        calls = []
        with mock.patch.object(
            RecommendationRepository,
            "get_latest_completed_for_account",
            side_effect=AssertionError("basic must not query"),
        ) as lookup:
            self.assertEqual(self.execute(run, self.recording_runner(calls)), "completed")
        lookup.assert_not_called()
        self.assertNotIn("incumbents", calls[0])

    def test_premium_policy_without_zone_never_queries_for_incumbents(self):
        self.completed_run(self.portfolio(["BBSE3"], ["HGLG11"]))
        run = self.zone_run(policy={"plan": "premium", "stocks": {}, "fiis": {}})
        calls = []
        with mock.patch.object(
            RecommendationRepository, "get_latest_completed_for_account"
        ) as lookup:
            self.assertEqual(self.execute(run, self.recording_runner(calls)), "completed")
        lookup.assert_not_called()
        self.assertNotIn("incumbents", calls[0])

    def test_other_account_and_non_completed_or_basic_runs_are_not_incumbents(self):
        other = Account(email="other@example.com")
        self.session.add(other)
        self.session.flush()
        other_profile = ProfileRecord(
            account_id=other.id,
            version=1,
            answers={},
            dimensions={},
            suitability_score=0.5,
            generic_profile="moderado",
            investable_capital_brl=10_000,
            consented_at=datetime.now(timezone.utc),
        )
        self.session.add(other_profile)
        self.session.commit()
        self.completed_run(self.portfolio(["BBSE3"], ["HGLG11"]), account=other, profile=other_profile)
        self.completed_run(self.portfolio(["ITSA4"], ["KNRI11"]), plan="basic")
        failed = self.zone_run()
        RecommendationRepository(self.session).mark_failed(failed.id, "engine_error", "fixture")
        self.session.commit()
        run = self.zone_run()
        calls = []

        self.assertEqual(self.execute(run, self.recording_runner(calls)), "completed")
        self.assertNotIn("incumbents", calls[0])


if __name__ == "__main__":
    unittest.main()
