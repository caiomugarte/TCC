#!/usr/bin/env python3
"""Ciclo mensal do Premium de uma vez: publica o snapshot, refaz a carteira e mede a anterior (issue #87, spec #82).

Rode da raiz do repositório (o run leva de 81 a 157 minutos com dados reais; rode fora de uma sessão de agente e com
a API parada):

    DATABASE_URL=sqlite:////caminho/para/api/prumo-dev.db \\
        .venv/bin/python scripts/ciclo_mensal.py --date 2026-12-15 --registry /caminho/para/snapshots

Etapas, em ordem, paradas na primeira falha (o nome da etapa sai no erro):

    conta              --account, ou a única conta com Premium ativo e perfil (mesma regra de scripts/premium_run.py)
    publicacao         scripts/publish_snapshot.py: séries, seleção, build, validate, sanidade, publish
    run                scripts/premium_run.py com o manifesto recém-publicado (zona de proteção com a carteira do
                       último run Premium concluído da conta)
    teste_para_frente  py/teste_para_frente.py com o run Premium concluído anterior da conta (o mais recente com
                       data de corte menor que a do manifesto novo) até o corte do manifesto novo; sem ele, o
                       relatório registra "sem carteira anterior" e o ciclo segue
    relatorio          <relatorio-dir>/<manifest_id>.json e .md, regravados a cada execução

Retomada. Rodar de novo com a mesma --date reaproveita o manifesto publicado (publish_snapshot é idempotente pela
data de corte) e o run (mesma conta, perfil, manifesto e versão da política; issue #83), refaz o teste para frente e
regrava o relatório. O relatório não tem horário de execução nem caminho absoluto: com as mesmas entradas, os bytes
são os mesmos.

Tabela acumulada. O .md de cada ciclo termina com uma tabela de todos os ciclos registrados, montada a partir de
todos os <relatorio-dir>/*.json no momento da gravação (o .md de um ciclo antigo não é reescrito quando um novo
entra; a tabela completa está sempre no .md mais recente).

Caminhos:

- Registro: --registry (padrão $PREMIUM_SNAPSHOT_REGISTRY). Nada é gravado na raiz dele fora o manifesto do mês.
- Banco: --database-url (padrão $DATABASE_URL; sem os dois, o api/prumo-dev.db do checkout de onde o script roda,
  que num worktree é uma cópia velha). O banco usado é impresso no início.
- Relatório: --relatorio-dir (padrão docs/validation/ciclos/ do checkout de onde o script roda); não pode ser o
  próprio registro, onde todo *.json da raiz é lido como manifesto.

Aviso ao fim (issue #88). main() aceita ``notificar``: uma função chamada uma vez, no fim, com o CicloResult (sucesso
ou falha, com ``etapa`` e ``erro``). Uma exceção dela é impressa no stderr e não muda o código de saída.

Exit codes:

    0    ciclo concluído: manifesto publicado (ou reaproveitado), run concluído (ou reaproveitado), relatório gravado
    2    argumentos inválidos (inclusive --registry ausente ou --relatorio-dir igual ao registro)
    3    etapa conta: --account inexistente, sem Premium ativo ou sem perfil; ou, sem --account, zero ou mais de
         uma conta Premium com perfil
    4    etapa publicacao falhou; nenhum manifesto novo publicado, o anterior continua valendo
    5    etapa run falhou (recusado, terminou failed ou reaproveitado ainda em execução)
    6    etapa teste_para_frente falhou (entrada inválida, fonte indisponível)
    7    etapa relatorio falhou (JSON de ciclo ilegível em --relatorio-dir, erro de escrita)
    130  interrompido (Ctrl-C ou SIGTERM); o run em execução, se havia um, foi marcado failed (interrupted)
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
import json
import os
from pathlib import Path
import signal
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT / "py", PROJECT_ROOT / "api", PROJECT_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from publish_snapshot import PublishResult, PublishSnapshotError  # noqa: E402,F401
from teste_para_frente import TesteParaFrenteError  # noqa: E402,F401

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_ACCOUNT = 3
EXIT_PUBLISH = 4
EXIT_RUN = 5
EXIT_FORWARD = 6
EXIT_REPORT = 7
EXIT_INTERRUPTED = 130

ETAPA_CONTA = "conta"
ETAPA_PUBLICACAO = "publicacao"
ETAPA_RUN = "run"
ETAPA_TESTE = "teste_para_frente"
ETAPA_RELATORIO = "relatorio"

EXIT_BY_STAGE = {
    ETAPA_CONTA: EXIT_ACCOUNT,
    ETAPA_PUBLICACAO: EXIT_PUBLISH,
    ETAPA_RUN: EXIT_RUN,
    ETAPA_TESTE: EXIT_FORWARD,
    ETAPA_RELATORIO: EXIT_REPORT,
}

VERSAO_RELATORIO = 1
SEM_CARTEIRA_ANTERIOR = "sem carteira anterior"
DEFAULT_RELATORIO_DIR = PROJECT_ROOT / "docs" / "validation" / "ciclos"


@dataclass
class CicloResult:
    """O que o ciclo fez até parar. ``etapa`` e ``erro`` só vêm preenchidos quando ele falhou."""

    data: date
    ok: bool = False
    etapa: str | None = None
    erro: str | None = None
    exit_code: int = EXIT_OK
    account_id: str | None = None
    manifest_id: str | None = None
    corte: str | None = None
    manifesto_reaproveitado: bool | None = None
    run_id: str | None = None
    run_reaproveitado: bool | None = None
    run_anterior_id: str | None = None
    relatorio_json: Path | None = None
    relatorio_md: Path | None = None
    relatorio: dict | None = field(default=None, repr=False)


class _StageFailed(Exception):
    def __init__(self, etapa: str, erro: str, exit_code: int | None = None):
        super().__init__(erro)
        self.etapa = etapa
        self.erro = erro
        self.exit_code = EXIT_BY_STAGE[etapa] if exit_code is None else exit_code


def _stage(etapa: str, action: Callable[[], Any]) -> Any:
    try:
        return action()
    except (KeyboardInterrupt, _StageFailed):
        raise
    except Exception as exc:  # noqa: BLE001 - toda falha da etapa vira erro com o nome da etapa
        message = getattr(exc, "message", None) or str(exc) or type(exc).__name__
        raise _StageFailed(etapa, message) from exc


def ciclo_mensal(
    day: date,
    *,
    session_factory: Callable[[], Any],
    relatorio_dir: Path,
    publicar: Callable[[date], PublishResult],
    rodar: Callable[[str, Path], Any],
    medir: Callable[[Mapping, Any], dict],
    account_id: str | None = None,
) -> CicloResult:
    """Roda as etapas do ciclo de ``day`` e devolve o resultado; nunca levanta por falha de etapa.

    ``publicar(day)`` devolve o ``PublishResult`` do #85; ``rodar(account_id, manifest_path)`` cria ou reaproveita e
    executa o run Premium (``PremiumRunOutcome`` do #84); ``medir(run_anterior, manifesto)`` é o teste para frente do
    #86. ``run_anterior`` é um dict com os campos do run (id, plan, status, snapshot_id, snapshot_cutoff,
    result_json).
    """
    from app.services.premium_run import select_premium_account

    result = CicloResult(data=day)
    current = ETAPA_CONTA
    try:
        with session_factory() as session:
            result.account_id = _stage(ETAPA_CONTA, lambda: select_premium_account(session, account_id))

        current = ETAPA_PUBLICACAO
        published = _stage(ETAPA_PUBLICACAO, lambda: publicar(day))
        manifest = published.manifest
        result.manifest_id = manifest.manifest_id
        result.corte = manifest.cutoff_date.isoformat()
        result.manifesto_reaproveitado = published.reused

        current = ETAPA_RUN
        outcome = _stage(ETAPA_RUN, lambda: rodar(result.account_id, Path(published.manifest_path)))
        result.run_id = outcome.run_id
        result.run_reaproveitado = outcome.reused

        current = ETAPA_TESTE
        new_run, previous = _stage(
            ETAPA_TESTE, lambda: _runs(session_factory, result.account_id, outcome.run_id, result.corte)
        )
        if previous is None:
            forward = {"status": SEM_CARTEIRA_ANTERIOR, "resultado": None}
        else:
            result.run_anterior_id = previous["id"]
            measured = _stage(ETAPA_TESTE, lambda: medir(previous, manifest))
            forward = {"status": "medido", "resultado": _without_local_paths(measured)}

        current = ETAPA_RELATORIO
        report = _stage(ETAPA_RELATORIO, lambda: build_report(manifest, new_run, previous, forward))
        result.relatorio_json, result.relatorio_md = _stage(
            ETAPA_RELATORIO, lambda: write_report(Path(relatorio_dir), report)
        )
        result.relatorio = report
    except _StageFailed as failure:
        result.etapa, result.erro, result.exit_code = failure.etapa, failure.erro, failure.exit_code
        return result
    except KeyboardInterrupt:
        result.etapa, result.erro, result.exit_code = current, "interrompido", EXIT_INTERRUPTED
        return result
    result.ok = True
    return result


_RUN_FIELDS = ("id", "plan", "status", "snapshot_id", "snapshot_cutoff", "policy_version", "result_json",
               "provenance_json")


def _runs(session_factory, account_id: str, run_id: str, cutoff: str) -> tuple[dict, dict | None]:
    """O run novo e o run Premium concluído anterior da conta (corte menor que ``cutoff``, criado por último)."""
    from sqlalchemy import desc, select

    from app.db.models import RecommendationRun

    with session_factory() as session:
        new_run = session.get(RecommendationRun, run_id)
        previous = session.scalar(
            select(RecommendationRun)
            .where(
                RecommendationRun.account_id == account_id,
                RecommendationRun.plan == "premium",
                RecommendationRun.status == "completed",
                RecommendationRun.id != run_id,
                RecommendationRun.snapshot_cutoff < cutoff,
            )
            .order_by(desc(RecommendationRun.created_at), desc(RecommendationRun.id))
            .limit(1)
        )
        as_dict = lambda run: {name: getattr(run, name) for name in _RUN_FIELDS}  # noqa: E731
        return as_dict(new_run), (as_dict(previous) if previous is not None else None)


def _without_local_paths(measured: Mapping) -> dict:
    """O resultado do teste para frente sem o diretório de cache (caminho absoluto da máquina)."""
    return {key: value for key, value in measured.items() if key != "cache_dir"}


# --- relatório -------------------------------------------------------------------------------------------------------


CLASSES = (("acoes", "stocks", "Ações"), ("fiis", "fiis", "FIIs"))


def _tickers(run: Mapping | None, key: str) -> list[str]:
    from app.services.premium_executor import _incumbent_tickers

    if run is None:
        return []
    tickers = _incumbent_tickers(run.get("result_json"), key)
    if tickers is None:
        raise ValueError(f"carteira ilegível no run {run['id']} ({key})")
    return tickers


def _portfolio_changes(new_run: Mapping, previous: Mapping | None) -> dict:
    """Por classe, os tickers comprados do run novo contra os do run anterior (quantity 0 não conta)."""
    changes = {}
    for label, key, _title in CLASSES:
        now, before = set(_tickers(new_run, key)), set(_tickers(previous, key))
        groups = {"ficaram": sorted(now & before), "entraram": sorted(now - before), "sairam": sorted(before - now)}
        changes[label] = {**groups, "contagem": {name: len(items) for name, items in groups.items()}}
    return changes


def _protection_zone(new_run: Mapping) -> dict | None:
    """O que a zona de proteção do run novo fez; ``mantidos_pela_zona`` são os titulares mantidos que o consenso puro
    teria tirado."""
    result = new_run.get("result_json") or {}
    zone = (result.get("provenance") or {}).get("protection_zone")
    if not isinstance(zone, Mapping):
        return None
    summary: dict[str, Any] = {"run_titular": zone.get("incumbent_run_id")}
    for label, key, _title in CLASSES:
        record = zone.get(key)
        if not isinstance(record, Mapping):
            summary[label] = None
            continue
        summary[label] = {
            "status": record.get("status"),
            "titulares": list(record.get("incumbents") or []),
            "mantidos_pela_zona": sorted(set(record.get("kept") or []) - set(record.get("consensus_tickers") or [])),
            "substituicoes": [{"sai": item.get("out"), "entra": item.get("in")} for item in record.get("replaced") or []],
            "turnover": record.get("turnover"),
        }
    return summary


def build_report(manifest, new_run: Mapping, previous: Mapping | None, forward: Mapping) -> dict:
    provenance = new_run.get("provenance_json") or {}
    return {
        "versao": VERSAO_RELATORIO,
        "manifesto": {"id": manifest.manifest_id, "corte": manifest.cutoff_date.isoformat()},
        "run": {
            "id": new_run["id"],
            "politica": new_run.get("policy_version"),
            "model_versions": dict(provenance.get("model_versions") or {}),
            "random_seed": provenance.get("random_seed"),
        },
        "run_anterior": None if previous is None else {
            "id": previous["id"], "manifesto": previous["snapshot_id"], "corte": previous["snapshot_cutoff"],
        },
        "carteira": _portfolio_changes(new_run, previous),
        "zona_de_protecao": _protection_zone(new_run),
        "teste_para_frente": dict(forward),
    }


def write_report(directory: Path, report: Mapping) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    manifest_id = report["manifesto"]["id"]
    json_path = directory / f"{manifest_id}.json"
    md_path = directory / f"{manifest_id}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def _pct(value: Any) -> str:
    return "—" if value is None else f"{value * 100:+.2f}%"


def _list(items: Sequence[str]) -> str:
    return ", ".join(items) if items else "—"


def render_markdown(report: Mapping) -> str:
    manifest, run, previous = report["manifesto"], report["run"], report["run_anterior"]
    lines = [
        f"# Ciclo mensal {manifest['id']}",
        "",
        "Gerado por `scripts/ciclo_mensal.py` (issue #87); os números vêm do JSON ao lado, com o mesmo nome.",
        "",
        f"- Manifesto: `{manifest['id']}`, data de corte {manifest['corte']}",
        f"- Run Premium: `{run['id']}`, política `{run['politica']}`, semente {run['random_seed']}",
        "- Carteira anterior: "
        + (SEM_CARTEIRA_ANTERIOR if previous is None
           else f"run `{previous['id']}` (manifesto `{previous['manifesto']}`, corte {previous['corte']})"),
        "",
        "## O que mudou na carteira",
        "",
        "| Classe | Ficaram | Entraram | Saíram |",
        "|---|---:|---:|---:|",
    ]
    for label, _key, title in CLASSES:
        count = report["carteira"][label]["contagem"]
        lines.append(f"| {title} | {count['ficaram']} | {count['entraram']} | {count['sairam']} |")
    lines.append("")
    for label, _key, title in CLASSES:
        change = report["carteira"][label]
        lines.append(f"- {title}: entraram {_list(change['entraram'])}; saíram {_list(change['sairam'])}")
    lines += ["", "## Zona de proteção", ""]
    zone = report["zona_de_protecao"]
    if zone is None:
        lines.append("O run não registrou a zona de proteção.")
    else:
        lines.append(f"Titulares do run `{zone['run_titular']}`." if zone["run_titular"] else "Sem titulares.")
        lines.append("")
        for label, _key, title in CLASSES:
            record = zone[label]
            if record is None:
                lines.append(f"- {title}: sem registro")
                continue
            swaps = ", ".join(f"{item['sai']} → {item['entra']}" for item in record["substituicoes"])
            lines.append(
                f"- {title} ({record['status']}): titulares que a zona manteve e o consenso teria tirado: "
                f"{_list(record['mantidos_pela_zona'])}; substituições: {swaps or '—'}"
            )
    lines.append("")
    return "\n".join(lines)
