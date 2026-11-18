"""Issue #87: o comando ciclo-mensal encadeia publicação, run Premium e teste para frente e grava o relatório."""

import contextlib
from datetime import date
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

from sqlalchemy import select

from app.db.models import RecommendationRun
from app.services.premium_run import run_premium

from snapshot_manifest import SnapshotManifest, validate_manifest

from tests import test_premium_run as premium_run_tests
from tests.test_premium_run import build_zone_snapshot


def _load_cli():
    path = Path(__file__).resolve().parents[2] / "scripts" / "ciclo_mensal.py"
    spec = importlib.util.spec_from_file_location("ciclo_mensal_cli", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve anotações pelo módulo registrado
    spec.loader.exec_module(module)
    return module


ciclo = _load_cli()

OUTUBRO = "2026-10-29"
NOVEMBRO = "2026-10-30"


def build_manifest(root: Path, manifest_id: str, cutoff: str) -> Path:
    """Manifesto de fixture com o id e o corte pedidos (as séries têm 2026-10-29 e 2026-10-30)."""
    root.mkdir(parents=True)
    path = build_zone_snapshot(root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["manifest_id"] = manifest_id
    payload["cutoff_date"] = cutoff
    payload["common_dates"] = [day for day in payload["common_dates"] if day <= cutoff]
    payload["state"] = "published"
    target = root / f"{manifest_id}.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    path.unlink()
    return target


class FakePublish:
    """Etapa de publicação de fixture: devolve o manifesto dado, ou falha com a etapa do #85."""

    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path
        self.calls = []
        self.error = None

    def __call__(self, day):
        self.calls.append(day)
        if self.error is not None:
            raise self.error
        manifest = validate_manifest(SnapshotManifest.load(self.manifest_path))
        return ciclo.PublishResult(manifest, self.manifest_path, reused=len(self.calls) > 1)


TESTE_PARA_FRENTE = {
    "versao": 1,
    "inicio": OUTUBRO,
    "fim": NOVEMBRO,
    "referencias": {
        "IBOV": {"retorno": 0.01, "data_inicio": OUTUBRO, "data_fim": NOVEMBRO},
        "IFIX": {"retorno": -0.005, "data_inicio": OUTUBRO, "data_fim": NOVEMBRO},
        "CDI": {"retorno": 0.0009, "data_inicio": OUTUBRO, "data_fim": NOVEMBRO},
    },
    "carteira": {"retorno_ponderado": 0.0123, "retorno_peso_igual": 0.0111, "peso_coberto": 1.0,
                 "classes_sem_retorno": []},
    "classes": {
        "brazilian_stocks": {"peso_alvo": 0.4, "retorno_ponderado": 0.02, "retorno_peso_igual": 0.019,
                             "referencia": "IBOV", "retorno_referencia": 0.01},
        "fiis": {"peso_alvo": 0.2, "retorno_ponderado": -0.01, "retorno_peso_igual": -0.012,
                 "referencia": "IFIX", "retorno_referencia": -0.005},
    },
    "ativos": {"acoes": [], "fiis": []},
    "cache_dir": "/home/alguem/.cache/prumo/teste-para-frente/2026-10-30",
}


class FakeForwardTest:
    def __init__(self):
        self.calls = []

    def __call__(self, run, manifest):
        self.calls.append((run["id"], manifest.manifest_id))
        return {**TESTE_PARA_FRENTE, "run_id": run["id"], "manifesto_run": run["snapshot_id"],
                "manifesto_posterior": manifest.manifest_id}


def frequencies_with_zone(prefix, n_assets, has_incumbents):
    """Sem titulares, os n primeiros. Com titulares, um novo muito melhor (substitui o pior titular) e um novo só um
    pouco melhor que o penúltimo titular: o consenso tiraria esse titular, a zona (delta 0,11) o mantém."""
    incumbents = [f"{prefix}{index:02d}" for index in range(n_assets)]
    strong, weak, tail = (f"{prefix}{n_assets + offset:02d}" for offset in range(3))
    if not has_incumbents:
        return {ticker: round(0.95 - 0.01 * index, 3) for index, ticker in enumerate([*incumbents, strong, weak, tail])}
    scores = {strong: 0.99, weak: 0.79, tail: 0.10}
    scores.update({ticker: round(0.80 - 0.005 * index, 3) for index, ticker in enumerate(incumbents)})
    return scores


class CicloMensalCase(unittest.TestCase):
    add_account = premium_run_tests.PremiumRunTests.add_account
    runner = premium_run_tests.PremiumRunTests.runner
    executor = premium_run_tests.PremiumRunTests.executor
    stored = premium_run_tests.PremiumRunTests.stored

    def setUp(self):
        patcher = mock.patch.object(premium_run_tests, "frequencies_for", frequencies_with_zone)
        patcher.start()
        self.addCleanup(patcher.stop)
        premium_run_tests.PremiumRunTests.setUp(self)
        self.registry = self.root / "registry"
        self.october = build_manifest(self.registry / "out", "mensal-2026-10-30", OUTUBRO)
        self.november = build_manifest(self.registry / "nov", "mensal-2026-10-31", NOVEMBRO)
        self.reports = self.root / "docs" / "validation" / "ciclos"
        self.publish = FakePublish(self.november)
        self.forward = FakeForwardTest()

    tearDown = premium_run_tests.PremiumRunTests.tearDown

    def run_october(self):
        """A carteira de outubro no banco, antes do ciclo de novembro."""
        return run_premium(
            session_factory=self.sessions,
            manifest_loader=lambda: validate_manifest(SnapshotManifest.load(self.october)),
            executor=self.executor(),
        )

    def rodar(self, account_id, manifest_path):
        return run_premium(
            account_id,
            session_factory=self.sessions,
            manifest_loader=lambda: validate_manifest(SnapshotManifest.load(manifest_path)),
            executor=getattr(self, "rodar_executor", None) or self.executor(),
        )

    def ciclo(self, account_id=None):
        return ciclo.ciclo_mensal(
            date(2026, 10, 31),
            account_id=account_id,
            session_factory=self.sessions,
            relatorio_dir=self.reports,
            publicar=self.publish,
            rodar=self.rodar,
            medir=self.forward,
        )

    def runs(self):
        with self.sessions() as session:
            return [(run.id, run.snapshot_id, run.status) for run in session.scalars(select(RecommendationRun))]

    def report(self, manifest_id="mensal-2026-10-31"):
        return json.loads((self.reports / f"{manifest_id}.json").read_text(encoding="utf-8"))

    def markdown(self, manifest_id="mensal-2026-10-31"):
        return (self.reports / f"{manifest_id}.md").read_text(encoding="utf-8")


class CicloMensalTests(CicloMensalCase):
    def test_caminho_feliz_sai_da_carteira_de_outubro_para_a_de_novembro_com_relatorio(self):
        october = self.run_october()

        result = self.ciclo()

        self.assertTrue(result.ok, result.erro)
        self.assertIsNone(result.etapa)
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.manifest_id, "mensal-2026-10-31")
        (new_run,) = [run for run in self.runs() if run[1] == "mensal-2026-10-31"]
        self.assertEqual(new_run[2], "completed")
        self.assertEqual(result.run_id, new_run[0])
        self.assertEqual(self.forward.calls, [(october.run_id, "mensal-2026-10-31")])

        report = self.report()
        stored = self.stored(new_run[0])
        self.assertEqual(report["manifesto"], {"id": "mensal-2026-10-31", "corte": NOVEMBRO})
        self.assertEqual(report["run"]["id"], new_run[0])
        self.assertEqual(report["run"]["politica"], stored.policy_version)
        self.assertEqual(report["run_anterior"]["id"], october.run_id)
        self.assertEqual(report["teste_para_frente"]["status"], "medido")
        self.assertEqual(report["teste_para_frente"]["resultado"]["carteira"]["retorno_ponderado"], 0.0123)
        self.assertNotIn("cache_dir", report["teste_para_frente"]["resultado"])
        self.assertEqual(result.relatorio, report)

    def test_relatorio_conta_o_que_ficou_entrou_e_saiu_por_classe_e_os_titulares_que_a_zona_manteve(self):
        october = self.run_october()

        self.ciclo()

        report = self.report()
        # 12 ações e 5 FIIs no perfil moderado: o novo forte entra no lugar do pior titular; o fraco não entra.
        self.assertEqual(report["carteira"]["acoes"]["contagem"], {"ficaram": 11, "entraram": 1, "sairam": 1})
        self.assertEqual(report["carteira"]["acoes"]["entraram"], ["S12"])
        self.assertEqual(report["carteira"]["acoes"]["sairam"], ["S11"])
        self.assertEqual(report["carteira"]["fiis"]["contagem"], {"ficaram": 4, "entraram": 1, "sairam": 1})
        zone = report["zona_de_protecao"]
        self.assertEqual(zone["run_titular"], october.run_id)
        self.assertEqual(zone["acoes"]["status"], "applied")
        self.assertEqual(zone["acoes"]["mantidos_pela_zona"], ["S10"])
        self.assertEqual(zone["fiis"]["mantidos_pela_zona"], ["F03"])
        self.assertEqual(zone["acoes"]["substituicoes"], [{"sai": "S11", "entra": "S12"}])

        markdown = self.markdown()
        self.assertIn("| Ações | 11 | 1 | 1 |", markdown)
        self.assertIn("| FIIs | 4 | 1 | 1 |", markdown)
        self.assertIn("S10", markdown)

    def test_sem_run_anterior_concluido_registra_sem_carteira_anterior_e_segue(self):
        result = self.ciclo()

        self.assertTrue(result.ok, result.erro)
        self.assertEqual(self.forward.calls, [])
        report = self.report()
        self.assertIsNone(report["run_anterior"])
        self.assertEqual(report["teste_para_frente"], {"status": "sem carteira anterior", "resultado": None})
        self.assertEqual(report["carteira"]["acoes"]["contagem"], {"ficaram": 0, "entraram": 12, "sairam": 0})
        self.assertIn("sem carteira anterior", self.markdown())

    def test_run_anterior_do_mesmo_manifesto_nao_e_a_carteira_anterior(self):
        october = self.run_october()
        with self.sessions() as session:  # um run concluído de outro manifesto com o mesmo corte
            run = session.get(RecommendationRun, october.run_id)
            run.snapshot_id, run.snapshot_cutoff = "outro-2026-10-30", NOVEMBRO
            session.commit()

        result = self.ciclo()

        self.assertTrue(result.ok, result.erro)
        self.assertEqual(self.forward.calls, [])
        self.assertEqual(self.report()["teste_para_frente"]["status"], "sem carteira anterior")

    def test_publicacao_falhada_para_o_ciclo_sem_chegar_ao_run(self):
        october = self.run_october()
        self.publish.error = ciclo.PublishSnapshotError("sanidade", "acoes_com_score: 3; mínimo 40", check="acoes_com_score")

        result = self.ciclo()

        self.assertFalse(result.ok)
        self.assertEqual(result.etapa, "publicacao")
        self.assertIn("acoes_com_score", result.erro)
        self.assertEqual(result.exit_code, 4)
        self.assertEqual([run_id for run_id, _manifest, _status in self.runs()], [october.run_id])
        self.assertEqual(self.forward.calls, [])
        self.assertFalse(self.reports.exists())

    def test_run_que_falha_para_o_ciclo_antes_do_teste_para_frente(self):
        self.run_october()

        def broken(*_args, **_kwargs):
            raise RuntimeError("engine exploded")

        self.rodar_executor = self.executor(broken)
        result = self.ciclo()

        self.assertEqual((result.etapa, result.exit_code), ("run", 5))
        self.assertIn("engine_error", result.erro)
        self.assertEqual(self.forward.calls, [])
        self.assertFalse(self.reports.exists())

    def test_teste_para_frente_que_falha_para_o_ciclo_sem_relatorio(self):
        self.run_october()

        def unavailable(_run, _manifest):
            raise ciclo.TesteParaFrenteError("o COTAHIST do cache vai até 2026-10-29")

        self.forward = unavailable
        result = self.ciclo()

        self.assertEqual((result.etapa, result.exit_code), ("teste_para_frente", 6))
        self.assertIn("COTAHIST", result.erro)
        self.assertIsNotNone(result.run_id)
        self.assertFalse(self.reports.exists())

    def test_conta_ambigua_para_antes_de_publicar(self):
        self.add_account("outra@example.com", plan="premium")

        result = self.ciclo()

        self.assertEqual((result.etapa, result.exit_code), ("conta", 3))
        self.assertIn("--account", result.erro)
        self.assertEqual(self.publish.calls, [])

    def test_retomada_com_a_mesma_data_reaproveita_manifesto_e_run_e_regrava_o_relatorio(self):
        self.run_october()
        first = self.ciclo()
        json_bytes = (self.reports / "mensal-2026-10-31.json").read_bytes()
        md_bytes = (self.reports / "mensal-2026-10-31.md").read_bytes()
        (self.reports / "mensal-2026-10-31.md").write_text("estragado", encoding="utf-8")
        calls = len(self.runner_calls)

        again = self.ciclo()

        self.assertTrue(again.ok, again.erro)
        self.assertTrue(again.manifesto_reaproveitado)
        self.assertEqual(again.run_id, first.run_id)
        self.assertTrue(again.run_reaproveitado)
        self.assertEqual(len(self.runner_calls), calls)
        self.assertEqual(len(self.runs()), 2)
        self.assertEqual((self.reports / "mensal-2026-10-31.json").read_bytes(), json_bytes)
        self.assertEqual((self.reports / "mensal-2026-10-31.md").read_bytes(), md_bytes)

    def test_retomada_depois_de_falha_no_teste_para_frente_nao_refaz_o_run(self):
        self.run_october()
        measured = self.forward

        def unavailable(_run, _manifest):
            raise ciclo.TesteParaFrenteError("COTAHIST desatualizado")

        self.forward = unavailable
        failed = self.ciclo()
        calls = len(self.runner_calls)
        self.forward = measured

        resumed = self.ciclo()

        self.assertTrue(resumed.ok, resumed.erro)
        self.assertEqual(resumed.run_id, failed.run_id)
        self.assertEqual(len(self.runner_calls), calls)
        self.assertEqual(self.report()["teste_para_frente"]["status"], "medido")

    def test_tabela_acumulada_vem_dos_jsons_de_ciclos_ordenada_pelo_corte(self):
        self.reports.mkdir(parents=True)
        later = {
            "versao": 1,
            "manifesto": {"id": "mensal-2026-12-15", "corte": "2026-12-14"},
            "run": {"id": "run-dez"},
            "teste_para_frente": {"status": "sem carteira anterior", "resultado": None},
        }
        earlier = {
            "versao": 1,
            "manifesto": {"id": "mensal-2026-09-15", "corte": "2026-09-14"},
            "run": {"id": "run-set"},
            "teste_para_frente": {"status": "medido", "resultado": {
                **TESTE_PARA_FRENTE, "inicio": "2026-08-14", "fim": "2026-09-14",
                "carteira": {"retorno_ponderado": -0.0425, "retorno_peso_igual": -0.04},
            }},
        }
        for report in (later, earlier):
            (self.reports / f"{report['manifesto']['id']}.json").write_text(json.dumps(report), encoding="utf-8")
        self.run_october()

        self.ciclo()

        table = self.markdown().split("## Tabela acumulada", 1)[1]
        rows = [line for line in table.splitlines() if line.startswith("| 20")]
        self.assertEqual([row.split(" | ")[0] for row in rows], ["| 2026-09-14", "| 2026-10-30", "| 2026-12-14"])
        self.assertIn("-4.25%", rows[0])
        self.assertIn("+1.23%", rows[1])  # carteira ponderada deste ciclo
        self.assertIn("+1.00%", rows[1])  # IBOV deste ciclo
        self.assertIn("sem carteira anterior", rows[2])

    def test_json_de_ciclo_ilegivel_falha_na_etapa_relatorio(self):
        self.reports.mkdir(parents=True)
        (self.reports / "lixo.json").write_text("{nao é json", encoding="utf-8")
        self.run_october()

        result = self.ciclo()

        self.assertEqual((result.etapa, result.exit_code), ("relatorio", 7))
        self.assertIn("lixo.json", result.erro)


if __name__ == "__main__":
    unittest.main()
