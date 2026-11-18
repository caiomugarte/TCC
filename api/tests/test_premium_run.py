"""Issue #84: criar e executar um run Premium sem a API de pé."""

import contextlib
import importlib.util
import io
import json
import os
import signal
import socket
import tempfile
import time
import unittest
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.adapters.premium_optimization import run_premium_optimization
from app.db.base import Base
from app.db.models import Account, Entitlement, ProfileRecord, RecommendationRun
from app.routers.recommendations import read_latest_completed_recommendation, read_recommendation
from app.services.premium_executor import InProcessPremiumExecutor, PremiumExecutor
from app.services.premium_run import PremiumRunCommandError, run_premium

from fetch_status_invest import OUTPUT_COLUMNS as STOCK_COLUMNS
from fetch_status_invest_fii import OUTPUT_COLUMNS as FII_COLUMNS
from snapshot_manifest import SnapshotManifest, sha256_path, validate_manifest

from tests.test_premium_integration import zone_selector
from tests.test_premium_snapshot_only import _write_selector, build_snapshot


STOCK_TICKERS = [f"S{index:02d}" for index in range(40)]
FII_TICKERS = [f"F{index:02d}" for index in range(40)]


def build_zone_snapshot(root: Path) -> Path:
    """Manifesto de fixture cujos CSVs de seleção têm os tickers do seletor falso."""

    manifest_path = build_snapshot(root)
    _write_selector(root / "stocks.csv", STOCK_COLUMNS, STOCK_TICKERS, "Bens Industriais")
    _write_selector(root / "fiis.csv", FII_COLUMNS, FII_TICKERS, "Papéis")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["stock_source"]["sha256"] = sha256_path(root / "stocks.csv")
    payload["fii_source"]["sha256"] = sha256_path(root / "fiis.csv")
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest_path


def frequencies_for(prefix, n_assets, has_incumbents):
    # n_assets + 3 candidatos; com titulares o ranking se inverte, e a zona segura parte deles.
    tickers = [f"{prefix}{index:02d}" for index in range(n_assets + 3)]
    if has_incumbents:
        tickers.reverse()
    return {ticker: round(0.95 - 0.05 * index, 2) for index, ticker in enumerate(tickers)}


@contextmanager
def network_blocked():
    def no_network(*_args, **_kwargs):
        raise AssertionError("o run Premium abriu uma conexão de rede")

    with mock.patch.object(socket.socket, "connect", no_network), mock.patch.object(
        socket, "create_connection", no_network
    ), mock.patch.object(socket, "getaddrinfo", no_network):
        yield


class PremiumRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.engine = create_engine(f"sqlite:///{self.root / 'premium-run.sqlite'}")
        Base.metadata.create_all(self.engine)
        self.sessions = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)
        self.account_id = self.add_account("premium@example.com", plan="premium", status="active")
        registry = self.root / "registry"
        registry.mkdir()
        self.manifest_path = build_zone_snapshot(registry)
        self.runner_calls = []

    def tearDown(self):
        self.engine.dispose()
        self.temp.cleanup()

    def add_account(self, email, *, plan=None, status="active", profile=True):
        with self.sessions() as session:
            account = Account(email=email)
            session.add(account)
            session.flush()
            if plan is not None:
                session.add(Entitlement(account_id=account.id, plan=plan, status=status))
            if profile:
                session.add(
                    ProfileRecord(
                        account_id=account.id,
                        version=1,
                        answers={"restricoes": ["nenhuma"]},
                        dimensions={
                            "apetite": 0.5,
                            "capacidade": 0.5,
                            "liquidez": 0.5,
                            "conhecimento": 0.5,
                        },
                        restrictions_json=["nenhuma"],
                        suitability_score=0.5,
                        generic_profile="moderado",
                        investable_capital_brl=100_000,
                        consented_at=datetime.now(timezone.utc),
                    )
                )
            session.commit()
            return account.id

    def runner(self, policy, manifest, workspace, capital, *, selector_inputs, incumbents=None, whole_units=False):
        self.runner_calls.append(incumbents)
        kwargs = {} if incumbents is None else {"incumbents": incumbents}
        return run_premium_optimization(
            policy,
            manifest,
            workspace,
            capital,
            selector_inputs=selector_inputs,
            snapshot_loader=lambda _manifest: SimpleNamespace(rows=(), metadata={}),
            allocation_engine=lambda *_args, **_kwargs: {
                "weights": {
                    "brazilian_stocks": 0.4,
                    "fiis": 0.2,
                    "international_equity": 0.1,
                    "fixed_income": 0.2,
                    "crypto": 0.1,
                }
            },
            stock_engine=zone_selector("S", frequencies_for),
            fii_engine=zone_selector("F", frequencies_for),
            whole_units=whole_units,
            **kwargs,
        )

    def executor(self, runner=None):
        return InProcessPremiumExecutor(
            session_factory=self.sessions,
            optimization_runner=runner or self.runner,
            workspace_root=self.root / "workspaces",
        )

    def premium_run(self, account_id=None, *, force=False, executor=None):
        return run_premium(
            account_id,
            force=force,
            session_factory=self.sessions,
            manifest_loader=lambda: validate_manifest(SnapshotManifest.load(self.manifest_path)),
            executor=executor or self.executor(),
        )

    def stored(self, run_id):
        with self.sessions() as session:
            run = session.get(RecommendationRun, run_id)
            session.expunge(run)
            return run

    def account(self, session):
        return session.get(Account, self.account_id)

    def test_cria_e_conclui_um_run_que_a_api_lista_com_a_rede_bloqueada(self):
        with network_blocked():
            outcome = self.premium_run()

        stored = self.stored(outcome.run_id)
        self.assertEqual(stored.status, "completed", stored.failure_code)
        self.assertEqual(outcome.account_id, self.account_id)
        self.assertFalse(outcome.reused)
        self.assertEqual(outcome.snapshot_id, "snapshot-2026-10-30")
        with self.sessions() as session:
            latest = read_latest_completed_recommendation(self.account(session), session)
            by_id = read_recommendation(outcome.run_id, self.account(session), session)
        self.assertEqual(latest.id, outcome.run_id)
        self.assertEqual(by_id.status, "completed")
        self.assertEqual(stored.provenance_json["manifest_id"], "snapshot-2026-10-30")
        self.assertEqual(stored.provenance_json["manifest_path"], str(self.manifest_path))
        self.assertEqual(stored.snapshot_cutoff, "2026-10-30")

    def test_com_run_anterior_concluido_a_zona_de_protecao_usa_a_carteira_dele(self):
        with network_blocked():
            first = self.premium_run()
            second = self.premium_run(force=True)

        self.assertNotEqual(second.run_id, first.run_id)
        previous = self.stored(first.run_id)
        current = self.stored(second.run_id)
        self.assertEqual(current.status, "completed", current.failure_code)
        zone = current.result_json["provenance"]["protection_zone"]
        self.assertEqual(zone["incumbent_run_id"], first.run_id)
        for key in ("stocks", "fiis"):
            held = sorted(item["ticker"] for item in previous.result_json[key] if item.get("quantity") != 0)
            self.assertEqual(zone[key]["status"], "applied")
            self.assertEqual(zone[key]["incumbents"], held)

    def test_mesma_chave_devolve_o_run_existente_sem_recalcular(self):
        first = self.premium_run()
        again = self.premium_run()

        self.assertEqual(again.run_id, first.run_id)
        self.assertFalse(first.reused)
        self.assertTrue(again.reused)
        self.assertEqual(len(self.runner_calls), 1)

    def test_force_cria_e_executa_um_run_novo(self):
        first = self.premium_run()
        forced = self.premium_run(force=True)

        self.assertNotEqual(forced.run_id, first.run_id)
        self.assertFalse(forced.reused)
        self.assertEqual(len(self.runner_calls), 2)
        self.assertEqual(self.stored(forced.run_id).status, "completed")

    def test_run_reaproveitado_ainda_em_execucao_nao_e_executado_de_novo(self):
        first = self.premium_run()
        with self.sessions() as session:
            session.get(RecommendationRun, first.run_id).status = "running"
            session.commit()

        with self.assertRaises(PremiumRunCommandError) as caught:
            self.premium_run()

        self.assertEqual(caught.exception.code, "run_in_progress")
        self.assertEqual(caught.exception.run_id, first.run_id)
        self.assertEqual(len(self.runner_calls), 1)

    def assert_refused(self, code, account_id=None):
        with self.assertRaises(PremiumRunCommandError) as caught:
            self.premium_run(account_id)
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(self.runner_calls, [])
        with self.sessions() as session:
            self.assertEqual(session.scalars(select(RecommendationRun)).all(), [])
        return caught.exception

    def test_sem_account_ignora_contas_sem_premium_ativo_ou_sem_perfil(self):
        self.add_account("basic@example.com", plan="basic")
        self.add_account("canceled@example.com", plan="premium", status="canceled")
        self.add_account("sem-perfil@example.com", plan="premium", profile=False)
        self.add_account("sem-entitlement@example.com")

        self.assertEqual(self.premium_run().account_id, self.account_id)

    def test_sem_account_e_sem_conta_premium_com_perfil_falha(self):
        with self.sessions() as session:
            session.query(Entitlement).delete()
            session.commit()

        self.assert_refused("no_premium_account")

    def test_sem_account_e_com_duas_contas_premium_falha_listando_as_duas(self):
        other = self.add_account("outra@example.com", plan="premium", status="grace_period")

        error = self.assert_refused("multiple_premium_accounts")
        self.assertIn(self.account_id, error.message)
        self.assertIn(other, error.message)

    def test_account_escolhe_a_conta_entre_varias_premium(self):
        other = self.add_account("outra@example.com", plan="premium")

        self.assertEqual(self.premium_run(other).account_id, other)

    def test_account_inexistente_sem_premium_ou_sem_perfil_falha(self):
        basic = self.add_account("basic@example.com", plan="basic")
        no_profile = self.add_account("sem-perfil@example.com", plan="premium", profile=False)
        for account_id, code in (
            ("nao-existe", "account_not_found"),
            (basic, "account_not_premium"),
            (no_profile, "account_without_profile"),
        ):
            with self.subTest(code=code):
                self.assert_refused(code, account_id)

    def test_snapshot_indisponivel_recusa_sem_criar_run(self):
        self.manifest_path.unlink()

        error = self.assert_refused("run_refused")
        self.assertEqual(error.detail, "snapshot_unavailable")

    def test_ctrl_c_no_meio_do_calculo_marca_o_run_failed_e_propaga(self):
        def interrupted(*_args, **_kwargs):
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            self.premium_run(executor=self.executor(interrupted))

        with self.sessions() as session:
            (run,) = session.scalars(select(RecommendationRun)).all()
            self.assertEqual(run.status, "failed")
            self.assertEqual(run.failure_code, "interrupted")
            self.assertIsNone(run.result_json)
            self.assertEqual(
                run.failure_message, "A execução foi interrompida e precisa ser solicitada novamente."
            )

    def test_run_interrompido_nao_bloqueia_o_proximo_sem_force(self):
        def interrupted(*_args, **_kwargs):
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            self.premium_run(executor=self.executor(interrupted))

        self.assertEqual(self.stored(self.premium_run().run_id).status, "completed")

    def test_api_reiniciando_no_meio_do_run_vira_erro_claro_e_nao_sucesso(self):
        def api_restarts_then_runs(*args, **kwargs):
            # A API, ao subir, marca failed todo run queued/running (recover_stale).
            PremiumExecutor(session_factory=self.sessions).recover_stale()
            return self.runner(*args, **kwargs)

        with self.assertRaises(PremiumRunCommandError) as caught:
            self.premium_run(executor=self.executor(api_restarts_then_runs))

        error = caught.exception
        self.assertEqual(error.code, "run_failed")
        self.assertEqual(error.detail, "worker_restarted")
        self.assertIn("API", error.message)
        self.assertEqual(self.stored(error.run_id).status, "failed")

    def test_run_que_falha_no_calculo_vira_erro_com_o_codigo_da_falha(self):
        def broken(*_args, **_kwargs):
            raise RuntimeError("constraint infeasible")

        with self.assertRaises(PremiumRunCommandError) as caught:
            self.premium_run(executor=self.executor(broken))

        self.assertEqual(caught.exception.code, "run_failed")
        self.assertEqual(caught.exception.detail, "infeasible_constraints")


def _load_cli():
    path = Path(__file__).resolve().parents[2] / "scripts" / "premium_run.py"
    spec = importlib.util.spec_from_file_location("premium_run_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PremiumRunCliTests(unittest.TestCase):
    """Cada código de saída da docstring de scripts/premium_run.py pelo caminho real."""

    setUp = PremiumRunTests.setUp
    tearDown = PremiumRunTests.tearDown
    add_account = PremiumRunTests.add_account
    runner = PremiumRunTests.runner
    executor = PremiumRunTests.executor
    stored = PremiumRunTests.stored

    cli = _load_cli()

    def main(self, *argv, runner=None):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = self.cli.main(
                list(argv),
                session_factory=self.sessions,
                manifest_loader=lambda: validate_manifest(SnapshotManifest.load(self.manifest_path)),
                executor=self.executor(runner),
            )
        return code, stdout.getvalue(), stderr.getvalue()

    def runs(self):
        with self.sessions() as session:
            return [(run.id, run.status, run.failure_code) for run in session.scalars(select(RecommendationRun))]

    def test_0_cria_e_conclui_e_depois_reaproveita(self):
        code, out, _err = self.main()
        self.assertEqual(code, 0)
        ((run_id, status, _),) = self.runs()
        self.assertEqual(status, "completed")
        self.assertIn(f"run {run_id} criado e concluído", out)

        code, out, _err = self.main("--account", self.account_id)
        self.assertEqual(code, 0)
        self.assertIn(f"run {run_id} reaproveitado", out)
        self.assertEqual(len(self.runs()), 1)

    def test_0_force_cria_outro_run(self):
        self.assertEqual(self.main()[0], 0)
        self.assertEqual(self.main("--force")[0], 0)
        self.assertEqual([status for _id, status, _code in self.runs()], ["completed", "completed"])

    def test_2_argumento_invalido(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(self.main("--nao-existe")[0], 2)
        self.assertEqual(self.runs(), [])

    def test_3_conta_ambigua_ou_invalida(self):
        self.add_account("outra@example.com", plan="premium")
        code, _out, err = self.main()
        self.assertEqual(code, 3)
        self.assertIn("--account", err)
        self.assertEqual(self.main("--account", "nao-existe")[0], 3)
        self.assertEqual(self.runs(), [])

    def test_4_snapshot_indisponivel(self):
        self.manifest_path.unlink()
        code, _out, err = self.main()
        self.assertEqual(code, 4)
        self.assertIn("snapshot_unavailable", err)
        self.assertEqual(self.runs(), [])

    def test_5_run_failed(self):
        def broken(*_args, **_kwargs):
            raise RuntimeError("engine exploded")

        code, _out, err = self.main(runner=broken)
        self.assertEqual(code, 5)
        self.assertIn("engine_error", err)
        self.assertEqual([status for _id, status, _code in self.runs()], ["failed"])

    def test_6_run_reaproveitado_em_execucao(self):
        self.assertEqual(self.main()[0], 0)
        with self.sessions() as session:
            session.scalars(select(RecommendationRun)).one().status = "queued"
            session.commit()
        code, _out, err = self.main()
        self.assertEqual(code, 6)
        self.assertIn("--force", err)

    def test_130_ctrl_c_marca_o_run_failed(self):
        def interrupted(*_args, **_kwargs):
            raise KeyboardInterrupt

        self.assertEqual(self.main(runner=interrupted)[0], 130)
        self.assertEqual([(status, code) for _id, status, code in self.runs()], [("failed", "interrupted")])

    def test_130_sigterm_marca_o_run_failed_e_restaura_o_handler(self):
        def unhandled(_signum, _frame):
            raise AssertionError("o comando não tratou SIGTERM")

        before = signal.signal(signal.SIGTERM, unhandled)
        self.addCleanup(signal.signal, signal.SIGTERM, before)

        def terminated(*_args, **_kwargs):
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(1)
            raise AssertionError("SIGTERM não interrompeu o run")

        self.assertEqual(self.main(runner=terminated)[0], 130)
        self.assertEqual([(status, code) for _id, status, code in self.runs()], [("failed", "interrupted")])
        self.assertIs(signal.getsignal(signal.SIGTERM), unhandled)


if __name__ == "__main__":
    unittest.main()
