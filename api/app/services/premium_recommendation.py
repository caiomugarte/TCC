from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
import hashlib
import os
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.db.models import Account, ProfileRecord, RecommendationRun
from app.repositories.recommendations import RecommendationRepository
from app.schemas.recommendation import PremiumRecommendationRequest
from app.services.premium_policy import POLICY_VERSION, PremiumPolicyError, resolve_premium_policy


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class PremiumRecommendationError(ValueError):
    """Validation error safe to expose at the Premium API boundary."""

    def __init__(self, code: str, message: str, status_code: int = 409, details: object = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details


def _default_manifest_loader() -> object:
    from snapshot_manifest import latest_compatible_manifest

    configured_path = os.getenv("PREMIUM_MANIFEST_PATH")
    if configured_path:
        from snapshot_manifest import validate_manifest

        return validate_manifest(Path(configured_path))
    registry = Path(
        os.getenv("PREMIUM_SNAPSHOT_REGISTRY", str(PROJECT_ROOT / "snapshots"))
    )
    return latest_compatible_manifest(registry)


def _default_manifest_validator(manifest: object) -> object:
    from snapshot_manifest import validate_manifest

    return validate_manifest(manifest)


def _plain(value: object) -> object:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _manifest_value(manifest: object, name: str, default: object = None) -> object:
    if isinstance(manifest, Mapping):
        return manifest.get(name, default)
    return getattr(manifest, name, default)


def _manifest_payload(manifest: object) -> dict[str, Any]:
    if hasattr(manifest, "as_dict"):
        payload = manifest.as_dict()
    elif isinstance(manifest, Mapping):
        payload = dict(manifest)
    else:
        raise PremiumRecommendationError("snapshot_unavailable", "O snapshot Premium é inválido.")
    normalized = _plain(payload)
    if not isinstance(normalized, dict):
        raise PremiumRecommendationError("snapshot_unavailable", "O snapshot Premium é inválido.")
    manifest_path = _manifest_value(manifest, "manifest_path")
    if manifest_path:
        normalized["manifest_path"] = str(manifest_path)
    return normalized


def _manifest_hashes(manifest: object) -> dict[str, str]:
    sources = _manifest_value(manifest, "sources", {})
    if isinstance(sources, Mapping):
        result = {}
        for key, source in sources.items():
            if isinstance(source, Mapping):
                digest = source.get("sha256") or source.get("hash")
            else:
                digest = getattr(source, "sha256", None)
            if digest:
                result[str(key)] = str(digest)
        if result:
            return result
    return {
        str(key): str(value)
        for key, value in dict(_manifest_value(manifest, "source_snapshot_hashes", {})).items()
    }


def derive_premium_seed(account_id: str, profile_id: str, manifest_id: str, policy_version: str) -> int:
    payload = "|".join((account_id, profile_id, manifest_id, policy_version)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 2_147_483_647


class PremiumRecommendationService:
    """Validate and enqueue one immutable account-owned Premium run."""

    def __init__(
        self,
        *,
        manifest_loader: Callable[[], object] = _default_manifest_loader,
        manifest_validator: Callable[[object], object] = _default_manifest_validator,
        policy_resolver: Callable[..., object] = resolve_premium_policy,
        repository_factory: Callable[[Session], RecommendationRepository] = RecommendationRepository,
        executor: object | None = None,
        rules_version: str = POLICY_VERSION,
    ) -> None:
        self.manifest_loader = manifest_loader
        self.manifest_validator = manifest_validator
        self.policy_resolver = policy_resolver
        self.repository_factory = repository_factory
        self.executor = executor
        self.rules_version = rules_version

    def create_run(
        self,
        account: Account | str,
        request: PremiumRecommendationRequest | None,
        session: Session,
    ) -> RecommendationRun:
        account_id = account.id if isinstance(account, Account) else str(account)
        profile = self._profile(session, account_id, getattr(request, "profile_id", None))
        try:
            manifest = self.manifest_validator(self.manifest_loader())
        except PremiumRecommendationError:
            raise
        except Exception as exc:
            raise PremiumRecommendationError(
                "snapshot_unavailable",
                "Os dados Premium não estão disponíveis para uma execução segura.",
            ) from exc

        manifest_id = str(_manifest_value(manifest, "manifest_id") or "premium-snapshot")
        seed = derive_premium_seed(account_id, profile.id, manifest_id, self.rules_version)
        hashes = _manifest_hashes(manifest)
        cutoff = _manifest_value(manifest, "cutoff_date")
        cutoff_value = cutoff.isoformat() if hasattr(cutoff, "isoformat") else str(cutoff or "")
        try:
            policy = self.policy_resolver(
                profile,
                rules_version=self.rules_version,
                source_snapshot_ids=(manifest_id,),
                source_snapshot_hashes=hashes,
                cutoff_date=cutoff_value,
                random_seed=seed,
            )
        except (PremiumPolicyError, ValueError, TypeError) as exc:
            raise PremiumRecommendationError(
                "profile_invalid",
                "O perfil não pode gerar uma política Premium válida.",
            ) from exc

        policy_json = _plain(policy)
        if not isinstance(policy_json, dict):
            raise PremiumRecommendationError("profile_invalid", "A política Premium é inválida.")
        policy_provenance = policy_json.get("provenance", {})
        if not isinstance(policy_provenance, Mapping):
            policy_provenance = {}
        provenance = {
            **dict(policy_provenance),
            "manifest_id": manifest_id,
            "manifest_path": _manifest_payload(manifest).get("manifest_path"),
            "manifest": _manifest_payload(manifest),
            "source_snapshot_ids": [manifest_id],
            "source_snapshot_hashes": hashes,
            "cutoff_date": cutoff_value,
            "random_seed": seed,
            "model_versions": dict(policy_provenance.get("model_versions", {})),
        }
        repository = self.repository_factory(session)
        run = repository.create_queued(
            account_id=account_id,
            profile_id=profile.id,
            policy=policy_json,
            provenance=provenance,
            policy_version=str(policy_provenance.get("policy_version", self.rules_version)),
            snapshot_id=manifest_id,
            snapshot_cutoff=cutoff_value,
            model_version="premium-v1",
        )
        session.commit()
        if hasattr(session, "refresh"):
            session.refresh(run)

        executor = self.executor or _default_executor()
        try:
            executor.submit(run.id)
        except Exception as exc:
            failure = repository.mark_failed(
                run.id,
                "submission_failed",
                "A execução Premium não pôde ser enviada ao worker.",
            )
            session.commit()
            if hasattr(session, "refresh"):
                session.refresh(failure)
            return failure
        return run

    def execute_run(self, run_id: str) -> None:
        (self.executor or _default_executor()).execute_run(run_id)

    @staticmethod
    def _profile(session: Session, account_id: str, profile_id: str | None) -> ProfileRecord:
        statement = select(ProfileRecord).where(ProfileRecord.account_id == account_id)
        if profile_id:
            statement = statement.where(ProfileRecord.id == profile_id)
        else:
            statement = statement.order_by(desc(ProfileRecord.version)).limit(1)
        profile = session.scalar(statement)
        if profile is None:
            if profile_id:
                raise PremiumRecommendationError("profile_not_found", "Perfil não encontrado.", 404)
            raise PremiumRecommendationError(
                "profile_required",
                "Complete o perfil antes de gerar a recomendação Premium.",
            )
        return profile


_service: PremiumRecommendationService | None = None


def _default_executor() -> object:
    from app.services.premium_executor import get_default_executor

    return get_default_executor()


def get_premium_recommendation_service() -> PremiumRecommendationService:
    global _service
    if _service is None:
        _service = PremiumRecommendationService()
    return _service


def create_premium_run(
    account: Account,
    request: PremiumRecommendationRequest,
    session: Session,
) -> RecommendationRun:
    return get_premium_recommendation_service().create_run(account, request, session)
