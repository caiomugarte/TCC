#!/usr/bin/env python3
"""Cria e executa um run Premium de uma conta sem a API de pé (issue #84, spec #82).

Rode da raiz do repositório:

    DATABASE_URL=sqlite:////caminho/para/api/prumo-dev.db \\
    PREMIUM_SNAPSHOT_REGISTRY=/caminho/para/snapshots \\
        .venv/bin/python scripts/premium_run.py [--account <id>] [--force]

O run é criado pelo mesmo create_run do endpoint (mesma proveniência, mesma regra de reuso) e executado neste
processo até o fim; o comando só volta quando ele termina. Leva de 81 a 157 minutos com dados reais.

- Conta: --account, ou a única conta com Premium ativo (active/grace_period) e perfil preenchido. O piloto
  (PREMIUM_PILOT_ACCOUNT_IDS) não é exigido.
- Reuso: sem --force, um run da conta com o mesmo perfil, manifesto e versão da política, em qualquer estado menos
  failed, é devolvido sem recalcular. --force sempre cria e executa um run novo.
- Banco: DATABASE_URL, lido no import como pela API. Sem ele, api/prumo-dev.db do checkout de onde o script roda
  (num worktree, é uma cópia velha); o banco usado é impresso no início.
- Manifesto: o da API. PREMIUM_MANIFEST_PATH, ou o manifesto compatível mais recente de PREMIUM_SNAPSHOT_REGISTRY
  (padrão: snapshots/ do checkout).
- Não rode com a API subindo ou reiniciando: no startup ela marca failed todo run queued/running de qualquer conta
  (recover_stale), inclusive o deste comando, cujo resultado é então descartado (sai com 5).

Exit codes:

    0    run completed: criado e executado agora, ou reaproveitado
    1    erro inesperado (banco ilegível, travado ou sem as tabelas; falha de disco); a mensagem sai no stderr. Um
         run que estava executando foi marcado failed pelo executor
    2    argumentos inválidos
    3    conta: --account inexistente, sem Premium ativo ou sem perfil; ou, sem --account, zero ou mais de uma
         conta Premium com perfil
    4    o run não foi criado (snapshot indisponível, perfil que não gera política válida)
    5    o run terminou failed (inclusive marcado failed pela API reiniciando no meio)
    6    o run reaproveitado ainda está queued/running em outro processo; espere ou use --force
    130  interrompido (Ctrl-C ou SIGTERM); o run em execução, se havia um, foi marcado failed (interrupted)
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import signal
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT / "py", PROJECT_ROOT / "api"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

EXIT_OK = 0
EXIT_UNEXPECTED = 1
EXIT_USAGE = 2
EXIT_ACCOUNT = 3
EXIT_REFUSED = 4
EXIT_FAILED = 5
EXIT_IN_PROGRESS = 6
EXIT_INTERRUPTED = 130

_EXIT_BY_CODE = {
    "account_not_found": EXIT_ACCOUNT,
    "account_not_premium": EXIT_ACCOUNT,
    "account_without_profile": EXIT_ACCOUNT,
    "no_premium_account": EXIT_ACCOUNT,
    "multiple_premium_accounts": EXIT_ACCOUNT,
    "run_refused": EXIT_REFUSED,
    "run_failed": EXIT_FAILED,
    "run_in_progress": EXIT_IN_PROGRESS,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="premium_run", description="Cria e executa um run Premium sem a API."
    )
    parser.add_argument("--account", help="id da conta; sem ele, a única conta Premium com perfil")
    parser.add_argument("--force", action="store_true", help="cria um run novo mesmo havendo um reaproveitável")
    return parser


def _database_label(session_factory) -> str:
    bind = getattr(session_factory, "kw", {}).get("bind")
    url = getattr(bind, "url", None)
    return url.render_as_string(hide_password=True) if url is not None else "?"


def _interrupt(_signum, _frame):
    raise KeyboardInterrupt


def main(argv: Sequence[str] | None = None, *, session_factory=None, manifest_loader=None, executor=None) -> int:
    """Roda o comando; devolve o código de saída. As injeções trocam banco, manifesto e executor nos testes."""

    try:
        args = _build_parser().parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)

    from app.services.premium_run import PremiumRunCommandError, run_premium

    if session_factory is None:
        from app.db.session import SessionLocal

        session_factory = SessionLocal
    print(f"premium_run: banco {_database_label(session_factory)}", file=sys.stderr)

    previous_sigterm = signal.signal(signal.SIGTERM, _interrupt)
    try:
        outcome = run_premium(
            args.account,
            force=args.force,
            session_factory=session_factory,
            manifest_loader=manifest_loader,
            executor=executor,
        )
    except PremiumRunCommandError as exc:
        print(f"premium_run: {exc.message}", file=sys.stderr)
        return _EXIT_BY_CODE[exc.code]
    except KeyboardInterrupt:
        print("premium_run: interrompido; o run em execução, se havia um, foi marcado failed (interrupted)", file=sys.stderr)
        return EXIT_INTERRUPTED
    except Exception as exc:  # noqa: BLE001 - todo erro fora dos previstos sai com 1, da tabela, sem traceback
        print(f"premium_run: erro inesperado: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_UNEXPECTED
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)

    origin = "reaproveitado (mesma chave; nada recalculado)" if outcome.reused else "criado e concluído"
    print(f"run {outcome.run_id} {origin}")
    print(f"  conta {outcome.account_id}, perfil {outcome.profile_id}")
    print(f"  manifesto {outcome.snapshot_id} (corte {outcome.snapshot_cutoff}), política {outcome.policy_version}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
