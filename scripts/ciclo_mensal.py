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

Aviso ao fim no Telegram (issue #88, scripts/aviso_telegram.py), ligado por padrão: no fim, também em falha ou
interrupção, o bot do Prumo manda o resumo (sucesso) ou a etapa e o erro (falha). Variáveis de ambiente:

    PRUMO_TELEGRAM_BOT_TOKEN   token do bot (BotFather)
    PRUMO_TELEGRAM_CHAT_ID     chat que recebe o aviso

Sem uma delas, o ciclo segue e o stderr diz que o aviso não foi enviado. Falha no envio (rede, timeout de 10 s, token
inválido) só vira uma linha no stderr, sem o token, e não muda o código de saída. main() aceita outro ``notificar``
(chamado uma vez com o CicloResult; ``None`` desliga o aviso); uma exceção dele também não muda o código de saída.

Exit codes:

    0    ciclo concluído: manifesto publicado (ou reaproveitado), run concluído (ou reaproveitado), relatório gravado
    2    etapa argumentos: argumentos inválidos (inclusive --registry ausente ou --relatorio-dir igual ao registro);
         avisa no Telegram como as outras falhas (--help não avisa)
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

from publish_snapshot import PublishResult  # noqa: E402

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_ACCOUNT = 3
EXIT_PUBLISH = 4
EXIT_RUN = 5
EXIT_FORWARD = 6
EXIT_REPORT = 7
EXIT_INTERRUPTED = 130

ETAPA_ARGUMENTOS = "argumentos"
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
    """O que o ciclo fez até parar. ``etapa`` e ``erro`` só vêm preenchidos quando ele falhou; ``data`` é ``None``
    quando --date faltou ou é ilegível."""

    data: date | None
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
    def __init__(self, etapa: str, erro: str):
        super().__init__(erro)
        self.etapa = etapa
        self.erro = erro
        self.exit_code = EXIT_BY_STAGE[etapa]


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
        new_run, previous, incumbent = _stage(
            ETAPA_TESTE, lambda: _runs(session_factory, result.account_id, outcome.run_id, result.corte)
        )
        if previous is None:
            forward = {"status": SEM_CARTEIRA_ANTERIOR, "resultado": None}
        else:
            result.run_anterior_id = previous["id"]
            measured = _stage(ETAPA_TESTE, lambda: medir(previous, manifest))
            forward = {"status": "medido", "resultado": _without_local_paths(measured)}

        current = ETAPA_RELATORIO
        report = _stage(ETAPA_RELATORIO, lambda: build_report(manifest, new_run, previous, forward, incumbent))
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


def _runs(session_factory, account_id: str, run_id: str, cutoff: str) -> tuple[dict, dict | None, dict | None]:
    """O run novo, o run Premium concluído anterior da conta (corte menor que ``cutoff``, criado por último; é o que o
    teste para frente mede) e o run titular que a zona de proteção do run novo usou (``None`` sem titular gravado)."""
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
        incumbent_id = _incumbent_run_id(new_run.result_json)
        incumbent = session.get(RecommendationRun, incumbent_id) if incumbent_id else None
        as_dict = lambda run: None if run is None else {name: getattr(run, name) for name in _RUN_FIELDS}  # noqa: E731
        return as_dict(new_run), as_dict(previous), as_dict(incumbent)


def _zone_record(result_json: Any) -> Mapping | None:
    zone = ((result_json or {}).get("provenance") or {}).get("protection_zone")
    return zone if isinstance(zone, Mapping) else None


def _incumbent_run_id(result_json: Any) -> str | None:
    return (_zone_record(result_json) or {}).get("incumbent_run_id")


def _without_local_paths(measured: Mapping) -> dict:
    """O resultado do teste para frente sem o diretório de cache (caminho absoluto da máquina)."""
    return {key: value for key, value in measured.items() if key != "cache_dir"}


# --- relatório -------------------------------------------------------------------------------------------------------


CLASSES = (("acoes", "stocks", "Ações"), ("fiis", "fiis", "FIIs"))


def _tickers(run: Mapping | None, key: str) -> list[str]:
    from app.services.premium_executor import incumbent_tickers

    if run is None:
        return []
    tickers = incumbent_tickers(run.get("result_json"), key)
    if tickers is None:
        raise ValueError(f"carteira ilegível no run {run['id']} ({key})")
    return tickers


def _portfolio_changes(new_run: Mapping, previous: Mapping | None) -> dict:
    """Por classe, os tickers comprados do run novo contra os de ``previous`` (quantity 0 não conta)."""
    changes: dict[str, Any] = {"run_comparado": None if previous is None else previous["id"]}
    for label, key, _title in CLASSES:
        now, before = set(_tickers(new_run, key)), set(_tickers(previous, key))
        groups = {"ficaram": sorted(now & before), "entraram": sorted(now - before), "sairam": sorted(before - now)}
        changes[label] = {**groups, "contagem": {name: len(items) for name, items in groups.items()}}
    return changes


def _protection_zone(new_run: Mapping) -> dict | None:
    """O que a zona de proteção do run novo fez; ``mantidos_pela_zona`` são os titulares mantidos que o consenso puro
    teria tirado."""
    zone = _zone_record(new_run.get("result_json"))
    if zone is None:
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


def build_report(
    manifest, new_run: Mapping, previous: Mapping | None, forward: Mapping, incumbent: Mapping | None = None
) -> dict:
    """``previous`` é a carteira medida no teste para frente; a mudança da carteira compara com ``incumbent``, o run
    titular que a zona de proteção do run novo usou, e só cai em ``previous`` quando o run novo não gravou titular."""
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
        "carteira": _portfolio_changes(new_run, incumbent if incumbent is not None else previous),
        "zona_de_protecao": _protection_zone(new_run),
        "teste_para_frente": dict(forward),
    }


def write_report(directory: Path, report: Mapping) -> tuple[Path, Path]:
    """Grava ``<manifest_id>.json`` e depois ``.md``, este com a tabela acumulada de todos os JSONs de ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    manifest_id = report["manifesto"]["id"]
    json_path = directory / f"{manifest_id}.json"
    md_path = directory / f"{manifest_id}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report, load_cycles(directory)), encoding="utf-8")
    return json_path, md_path


def load_cycles(directory: Path) -> list[dict]:
    """Todos os relatórios de ciclo de ``directory``, ordenados pela data de corte (depois pelo id do manifesto)."""
    cycles = []
    for path in sorted(directory.glob("*.json")):
        try:
            cycle = json.loads(path.read_text(encoding="utf-8"))
            key = (cycle["manifesto"]["corte"], cycle["manifesto"]["id"])
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"{path.name} não é um relatório de ciclo legível: {exc}") from exc
        cycles.append((key, cycle))
    return [cycle for _key, cycle in sorted(cycles, key=lambda item: item[0])]


def _pct(value: Any) -> str:
    return "—" if value is None else f"{value * 100:+.2f}%"


def _list(items: Sequence[str]) -> str:
    return ", ".join(items) if items else "—"


def _forward_section(forward: Mapping) -> list[str]:
    measured = forward.get("resultado")
    if forward.get("status") != "medido" or not measured:
        return [f"{SEM_CARTEIRA_ANTERIOR.capitalize()}: nada a medir neste ciclo."]
    references = measured.get("referencias") or {}
    portfolio = measured.get("carteira") or {}
    lines = [
        f"Carteira do run `{measured.get('run_id')}` de {measured.get('inicio')} a {measured.get('fim')}, retorno total "
        "com proventos: ponderado pelos pesos da carteira e com peso igual dentro de cada classe.",
        "",
        "| Classe | Retorno | Peso igual | Referência |",
        "|---|---:|---:|---|",
    ]
    for key, item in (measured.get("classes") or {}).items():
        reference = item.get("referencia")
        reference_text = f"{reference} {_pct(item.get('retorno_referencia'))}" if reference else item.get("fonte", "—")
        lines.append(f"| {key} | {_pct(item.get('retorno_ponderado'))} | {_pct(item.get('retorno_peso_igual'))} "
                     f"| {reference_text} |")
    lines.append(f"| carteira | {_pct(portfolio.get('retorno_ponderado'))} | {_pct(portfolio.get('retorno_peso_igual'))} "
                 f"| CDI {_pct((references.get('CDI') or {}).get('retorno'))} |")
    missing = [ticker for item in (measured.get("classes") or {}).values() for ticker in item.get("sem_retorno") or []]
    if missing:
        lines += ["", f"Sem retorno (fora das médias, peso renormalizado): {_list(missing)}."]
    return lines


def _compared_with(run_id: str | None) -> list[str]:
    if run_id is None:
        return []
    return [f"Comparada com a carteira do run `{run_id}`.", ""]


TABLE_HEADER = [
    "| Corte | Manifesto | Run | Período medido | Carteira | Peso igual | Ações | IBOV | FIIs | IFIX | CDI |",
    "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
]


def _table_row(cycle: Mapping) -> str:
    manifest = cycle["manifesto"]
    forward = cycle.get("teste_para_frente") or {}
    measured = forward.get("resultado") if forward.get("status") == "medido" else None
    run_id = (cycle.get("run") or {}).get("id")
    if not measured:
        return (f"| {manifest['corte']} | `{manifest['id']}` | `{run_id}` | {SEM_CARTEIRA_ANTERIOR} "
                "| — | — | — | — | — | — | — |")
    classes = measured.get("classes") or {}
    references = measured.get("referencias") or {}
    portfolio = measured.get("carteira") or {}
    cells = [
        _pct(portfolio.get("retorno_ponderado")),
        _pct(portfolio.get("retorno_peso_igual")),
        _pct((classes.get("brazilian_stocks") or {}).get("retorno_ponderado")),
        _pct((references.get("IBOV") or {}).get("retorno")),
        _pct((classes.get("fiis") or {}).get("retorno_ponderado")),
        _pct((references.get("IFIX") or {}).get("retorno")),
        _pct((references.get("CDI") or {}).get("retorno")),
    ]
    return (f"| {manifest['corte']} | `{manifest['id']}` | `{run_id}` | {measured.get('inicio')} a {measured.get('fim')} "
            f"| {' | '.join(cells)} |")


def render_markdown(report: Mapping, cycles: Sequence[Mapping] = ()) -> str:
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
        *_compared_with(report["carteira"].get("run_comparado")),
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
    lines += ["", "## Teste para frente da carteira anterior", ""]
    lines += _forward_section(report["teste_para_frente"])
    lines += [
        "",
        "## Tabela acumulada",
        "",
        "Todos os relatórios `*.json` deste diretório quando este relatório foi gravado; cada linha mede a carteira "
        "do ciclo anterior até o corte da linha (Carteira: ponderada; Ações e FIIs: ponderados na classe).",
        "",
        *TABLE_HEADER,
        *(_table_row(cycle) for cycle in cycles),
        "",
    ]
    return "\n".join(lines)


# --- CLI -------------------------------------------------------------------------------------------------------------


Notificador = Callable[[CicloResult], None]
_AVISO_TELEGRAM: Any = object()  # padrão de main(): o aviso do Telegram configurado pelo ambiente


class _ArgumentosInvalidos(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    """Erro de argumento vira exceção (para o aviso) em vez de ``SystemExit``; ``--help`` continua saindo com 0."""

    def error(self, message):
        self.print_usage(sys.stderr)
        raise _ArgumentosInvalidos(message)


def _build_parser() -> argparse.ArgumentParser:
    from publish_snapshot import DEFAULT_CACHE_ROOT
    from teste_para_frente import DEFAULT_CACHE_ROOT as FORWARD_CACHE_ROOT

    parser = _Parser(
        prog="ciclo_mensal",
        description="Publica o snapshot do mês, refaz a carteira Premium, mede a anterior e grava o relatório.",
    )
    parser.add_argument("--date", required=True, metavar="YYYY-MM-DD", help="data do snapshot do mês")
    parser.add_argument("--registry", default=os.environ.get("PREMIUM_SNAPSHOT_REGISTRY"),
                        help="raiz do registro de snapshots (padrão: $PREMIUM_SNAPSHOT_REGISTRY)")
    parser.add_argument("--account", help="id da conta; sem ele, a única conta Premium com perfil")
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"),
                        help="URL SQLAlchemy do banco (padrão: $DATABASE_URL, senão o banco padrão da API)")
    parser.add_argument("--relatorio-dir", default=str(DEFAULT_RELATORIO_DIR),
                        help=f"onde gravar o relatório do ciclo (padrão {DEFAULT_RELATORIO_DIR})")
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT),
                        help=f"cache de brutos da publicação, um por data (padrão {DEFAULT_CACHE_ROOT})")
    parser.add_argument("--seed-cache", help="cache de onde semear os anuais congelados (ver publish_snapshot)")
    parser.add_argument("--limits", metavar="JSON", help="limites das checagens de sanidade (ver publish_snapshot)")
    parser.add_argument("--cache-teste-para-frente", default=str(FORWARD_CACHE_ROOT),
                        help=f"cache do teste para frente (padrão {FORWARD_CACHE_ROOT})")
    return parser


def _session_factory(database_url: str | None):
    if database_url:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        return sessionmaker(bind=create_engine(database_url), autoflush=False, autocommit=False)
    from app.db.session import SessionLocal

    return SessionLocal


def _database_label(session_factory) -> str:
    bind = getattr(session_factory, "kw", {}).get("bind")
    url = getattr(bind, "url", None)
    return url.render_as_string(hide_password=True) if url is not None else "?"


def _interrupt(_signum, _frame):
    raise KeyboardInterrupt


def _print_result(result: CicloResult) -> None:
    if not result.ok:
        print(f"ciclo_mensal: etapa {result.etapa}: {result.erro}", file=sys.stderr)
        if result.etapa == ETAPA_PUBLICACAO:
            print("ciclo_mensal: nenhum manifesto novo publicado; o anterior continua valendo", file=sys.stderr)
        return
    manifest_origin = "reaproveitado" if result.manifesto_reaproveitado else "publicado"
    run_origin = "reaproveitado" if result.run_reaproveitado else "criado e concluído"
    print(f"manifesto {result.manifest_id} (corte {result.corte}) {manifest_origin}")
    print(f"run {result.run_id} {run_origin}")
    print(f"teste para frente: {f'run {result.run_anterior_id}' if result.run_anterior_id else SEM_CARTEIRA_ANTERIOR}")
    print(f"relatório {result.relatorio_json} e {result.relatorio_md.name}")


def main(
    argv: Sequence[str] | None = None,
    *,
    session_factory=None,
    executor=None,
    medir: Callable[[Mapping, Any], dict] | None = None,
    notificar: Notificador | None = _AVISO_TELEGRAM,
) -> int:
    """Roda o ciclo; devolve o código de saída. ``session_factory``, ``executor`` e ``medir`` trocam banco, cálculo e
    rede nos testes; ``notificar`` recebe o CicloResult no fim (padrão: o aviso no Telegram de
    ``aviso_telegram.notificador_telegram``, que lê PRUMO_TELEGRAM_BOT_TOKEN e PRUMO_TELEGRAM_CHAT_ID; ``None`` desliga)."""
    day = None
    try:
        try:
            args = _build_parser().parse_args(argv)
        except SystemExit as exc:  # --help
            return int(exc.code or 0)
        try:
            day = date.fromisoformat(args.date)
        except ValueError as exc:
            raise ValueError(f"--date {args.date!r}: {exc}") from exc
        if not args.registry:
            raise ValueError("informe --registry ou PREMIUM_SNAPSHOT_REGISTRY")
        registry, relatorio_dir = Path(args.registry).expanduser(), Path(args.relatorio_dir).expanduser()
        if registry.resolve() == relatorio_dir.resolve():
            raise ValueError("--relatorio-dir não pode ser o registro (todo *.json da raiz dele é lido como manifesto)")
        from publish_snapshot import load_limits

        limits = load_limits(Path(args.limits) if args.limits else None)
    except (_ArgumentosInvalidos, OSError, TypeError, ValueError) as exc:
        print(f"ciclo_mensal: argumento inválido: {exc}", file=sys.stderr)
        failure = CicloResult(data=day, etapa=ETAPA_ARGUMENTOS, erro=str(exc), exit_code=EXIT_USAGE)
        _notify(failure, notificar)
        return failure.exit_code

    from app.services.premium_run import run_premium
    from publish_snapshot import publish_snapshot
    from snapshot_manifest import validate_manifest
    from teste_para_frente import teste_para_frente

    session_factory = session_factory or _session_factory(args.database_url)
    print(f"ciclo_mensal: banco {_database_label(session_factory)}", file=sys.stderr)

    def publicar(cycle_day: date) -> PublishResult:
        return publish_snapshot(
            cycle_day,
            registry=registry,
            cache_root=Path(args.cache_root).expanduser(),
            seed_cache=Path(args.seed_cache).expanduser() if args.seed_cache else None,
            limits=limits,
        )

    def rodar(account_id: str, manifest_path: Path):
        return run_premium(
            account_id,
            session_factory=session_factory,
            manifest_loader=lambda: validate_manifest(manifest_path),
            executor=executor,
        )

    def medir_padrao(run: Mapping, manifest) -> dict:
        return teste_para_frente(run, manifest, cache_dir=Path(args.cache_teste_para_frente).expanduser())

    previous_sigterm = signal.signal(signal.SIGTERM, _interrupt)
    try:
        result = ciclo_mensal(
            day,
            session_factory=session_factory,
            relatorio_dir=relatorio_dir,
            publicar=publicar,
            rodar=rodar,
            medir=medir or medir_padrao,
            account_id=args.account,
        )
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
    _print_result(result)
    _notify(result, notificar)
    return result.exit_code


def _notify(result: CicloResult, notificar: Notificador | None) -> None:
    if notificar is _AVISO_TELEGRAM:
        from aviso_telegram import notificador_telegram

        notificar = notificador_telegram()
    if notificar is not None:
        try:
            notificar(result)
        except Exception as exc:  # noqa: BLE001 - o aviso nunca muda o resultado do ciclo
            print(f"ciclo_mensal: o aviso do fim do ciclo falhou: {type(exc).__name__}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
