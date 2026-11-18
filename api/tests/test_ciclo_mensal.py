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
            executor=self.executor(),
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


if __name__ == "__main__":
    unittest.main()
