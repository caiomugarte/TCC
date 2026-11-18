"""Issue #84: criar e executar um run Premium sem a API de pé."""

import json
import socket
import tempfile
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
from app.services.premium_executor import InProcessPremiumExecutor
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


if __name__ == "__main__":
    unittest.main()
