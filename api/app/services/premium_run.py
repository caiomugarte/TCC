"""Cria e executa um run Premium no processo do chamador, sem a API (issue #84).

``run_premium`` é a etapa "run Premium" do ciclo mensal e o miolo do CLI
``scripts/premium_run.py``: escolhe a conta, cria o run pelo mesmo ``create_run``
da API (mesma regra de reuso e mesma proveniência) e o executa com
``InProcessPremiumExecutor``, que só devolve no fim.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from app.db.models import Account, Entitlement, ProfileRecord, RecommendationRun
from app.entitlements.dependencies import is_premium_active
from app.schemas.recommendation import PremiumRecommendationRequest
from app.services.premium_executor import InProcessPremiumExecutor
from app.services.premium_recommendation import (
    PremiumRecommendationError,
    PremiumRecommendationService,
)


class PremiumRunCommandError(RuntimeError):
    """O comando não terminou com um run ``completed``; ``code`` diz por quê.

    Códigos: ``account_not_found``, ``account_not_premium``, ``account_without_profile``,
    ``no_premium_account``, ``multiple_premium_accounts`` (escolha da conta);
    ``run_refused`` (``create_run`` recusou, ``detail`` traz o código do serviço);
    ``run_failed`` (o run terminou ``failed``, ``detail`` traz o ``failure_code``);
    ``run_in_progress`` (o run reaproveitado ainda está ``queued``/``running``).
    """

    def __init__(self, code: str, message: str, *, run_id: str | None = None, detail: str | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.run_id = run_id
        self.detail = detail


@dataclass(frozen=True)
class PremiumRunOutcome:
    run_id: str
    account_id: str
    profile_id: str
    snapshot_id: str | None
    snapshot_cutoff: str | None
    policy_version: str | None
    reused: bool


def _has_profile(session: Any, account_id: str) -> bool:
    return session.scalar(
        select(ProfileRecord.id).where(ProfileRecord.account_id == account_id).limit(1)
    ) is not None


def select_premium_account(session: Any, account_id: str | None = None) -> str:
    """A conta pedida, conferida; sem ``account_id``, a única conta Premium com perfil."""

    if account_id is not None:
        if session.get(Account, account_id) is None:
            raise PremiumRunCommandError("account_not_found", f"Conta {account_id} não existe.")
        entitlement = session.scalar(select(Entitlement).where(Entitlement.account_id == account_id))
        if not is_premium_active(entitlement):
            raise PremiumRunCommandError(
                "account_not_premium", f"A conta {account_id} não tem Premium ativo."
            )
        if not _has_profile(session, account_id):
            raise PremiumRunCommandError(
                "account_without_profile", f"A conta {account_id} não tem perfil preenchido."
            )
        return account_id

    candidates = sorted(
        entitlement.account_id
        for entitlement in session.scalars(select(Entitlement))
        if is_premium_active(entitlement) and _has_profile(session, entitlement.account_id)
    )
    if not candidates:
        raise PremiumRunCommandError(
            "no_premium_account", "Nenhuma conta Premium ativa com perfil; passe --account."
        )
    if len(candidates) > 1:
        raise PremiumRunCommandError(
            "multiple_premium_accounts",
            "Há mais de uma conta Premium ativa com perfil; escolha com --account: "
            + ", ".join(candidates),
        )
    return candidates[0]


def _failure_message(run: RecommendationRun) -> str:
    if run.failure_code == "worker_restarted":
        return (
            f"O run {run.id} foi marcado failed (worker_restarted) por outro processo, "
            "provavelmente a API reiniciando durante a execução; o resultado foi descartado. "
            "Rode de novo com a API parada."
        )
    return f"O run {run.id} terminou failed ({run.failure_code}): {run.failure_message}"


def run_premium(
    account_id: str | None = None,
    *,
    force: bool = False,
    session_factory: Callable[[], Any] | None = None,
    manifest_loader: Callable[[], object] | None = None,
    executor: InProcessPremiumExecutor | None = None,
) -> PremiumRunOutcome:
    """Cria (ou reaproveita) e executa o run Premium da conta; devolve o run ``completed``.

    Sem ``force``, um run com a mesma chave (conta, perfil, manifesto, versão da política)
    em estado não ``failed`` é devolvido sem recalcular (``reused=True``). Levanta
    ``PremiumRunCommandError`` quando o run não termina ``completed``. Um Ctrl-C no meio
    marca o run ``failed`` (``interrupted``) e propaga o ``KeyboardInterrupt``.

    ``session_factory`` padrão é o ``SessionLocal`` da API, que lê ``DATABASE_URL`` no import;
    ``manifest_loader`` padrão é o da API (``PREMIUM_MANIFEST_PATH`` ou o manifesto compatível
    mais recente de ``PREMIUM_SNAPSHOT_REGISTRY``).
    """

    if session_factory is None:
        from app.db.session import SessionLocal

        session_factory = SessionLocal
    executor = executor or InProcessPremiumExecutor(session_factory=session_factory)
    service_options: dict[str, Any] = {"executor": executor}
    if manifest_loader is not None:
        service_options["manifest_loader"] = manifest_loader
    service = PremiumRecommendationService(**service_options)

    with session_factory() as session:
        account_id = select_premium_account(session, account_id)
        try:
            run = service.create_run(
                account_id, PremiumRecommendationRequest(force=force), session, plan="premium"
            )
        except PremiumRecommendationError as exc:
            raise PremiumRunCommandError(
                "run_refused", f"O run não foi criado ({exc.code}): {exc.message}", detail=exc.code
            ) from exc
        run_id = run.id

    with session_factory() as session:
        stored = session.get(RecommendationRun, run_id)
        reused = run_id not in executor.submitted
        if stored.status == "failed":
            raise PremiumRunCommandError(
                "run_failed", _failure_message(stored), run_id=run_id, detail=stored.failure_code
            )
        if stored.status != "completed":
            raise PremiumRunCommandError(
                "run_in_progress",
                f"Já existe o run {run_id} em {stored.status} com a mesma chave; "
                "espere ele terminar ou use --force.",
                run_id=run_id,
            )
        return PremiumRunOutcome(
            run_id=run_id,
            account_id=account_id,
            profile_id=stored.profile_id,
            snapshot_id=stored.snapshot_id,
            snapshot_cutoff=stored.snapshot_cutoff,
            policy_version=stored.policy_version,
            reused=reused,
        )
