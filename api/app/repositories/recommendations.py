from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import RecommendationRun, utc_now


RUN_STATES = frozenset({"queued", "running", "completed", "failed"})
TERMINAL_RUN_STATES = frozenset({"completed", "failed"})
VALID_TRANSITIONS = {
    "queued": frozenset({"running", "failed"}),
    "running": frozenset({"completed", "failed"}),
    "completed": frozenset(),
    "failed": frozenset(),
}


class RecommendationStateError(ValueError):
    """Raised when a recommendation run transition is invalid."""


def _json_copy(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    try:
        return json.loads(
            json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, default=str)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("recommendation payload must be JSON serializable") from exc


def _output_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class RecommendationRepository:
    """Account-scoped run persistence and its terminal state machine."""

    def __init__(self, session: Session):
        self.session = session

    def create_queued(
        self,
        *,
        account_id: str,
        profile_id: str,
        policy: object,
        provenance: Mapping[str, Any],
        policy_version: str | None = None,
        snapshot_id: str | None = None,
        snapshot_cutoff: str | None = None,
        model_version: str = "premium-v1",
    ) -> RecommendationRun:
        policy_json = _json_copy(policy)
        provenance_json = _json_copy(provenance)
        if not isinstance(provenance_json, dict):
            raise ValueError("recommendation provenance must be an object")
        policy_data = policy_json if isinstance(policy_json, dict) else {}
        policy_provenance = policy_data.get("provenance", {})
        if not isinstance(policy_provenance, Mapping):
            policy_provenance = {}
        snapshot_ids = policy_provenance.get("source_snapshot_ids", [])
        if not isinstance(snapshot_ids, (list, tuple)) or not snapshot_ids:
            snapshot_ids = ["premium"]
        resolved_policy_version = policy_version or str(
            policy_provenance.get("policy_version") or "premium-policy-v1"
        )
        resolved_snapshot_id = snapshot_id or str(
            provenance_json.get("manifest_id") or snapshot_ids[0]
        )
        resolved_cutoff = snapshot_cutoff or str(
            provenance_json.get("cutoff_date")
            or policy_provenance.get("cutoff_date")
            or "unknown"
        )
        run = RecommendationRun(
            account_id=account_id,
            profile_id=profile_id,
            plan="premium",
            model_version=model_version,
            snapshot_id=resolved_snapshot_id,
            snapshot_cutoff=resolved_cutoff,
            classes=[],
            assumptions=[],
            risks=[],
            status="queued",
            policy_version=resolved_policy_version,
            policy_json=policy_json,
            provenance_json=provenance_json,
            result_json=None,
            output_hash=None,
        )
        self.session.add(run)
        self.session.flush()
        return run

    def get(self, run_id: str) -> RecommendationRun | None:
        return self.session.get(RecommendationRun, run_id)

    def get_owned_run(self, account_id: str, run_id: str) -> RecommendationRun | None:
        return self.session.scalar(
            select(RecommendationRun).where(
                RecommendationRun.id == run_id,
                RecommendationRun.account_id == account_id,
            )
        )

    def mark_running(self, run_id: str) -> RecommendationRun:
        run = self._required(run_id)
        self._transition(run, "running")
        run.status = "running"
        run.started_at = run.started_at or utc_now()
        self.session.flush()
        return run

    def mark_completed(
        self,
        run_id: str,
        result: Mapping[str, Any],
    ) -> RecommendationRun:
        run = self._required(run_id)
        self._transition(run, "completed")
        payload = _json_copy(result)
        if not isinstance(payload, dict):
            raise ValueError("recommendation result must be an object")
        classes = payload.get("classes")
        assumptions = payload.get("assumptions", [])
        risks = payload.get("risks", [])
        if not isinstance(classes, list) or not isinstance(assumptions, list) or not isinstance(risks, list):
            raise ValueError("recommendation result has invalid summary fields")
        run.result_json = payload
        run.classes = classes
        run.assumptions = assumptions
        run.risks = risks
        run.status = "completed"
        run.failure_code = None
        run.failure_message = None
        run.completed_at = utc_now()
        run.output_hash = _output_hash(payload)
        self.session.flush()
        return run

    def mark_failed(
        self,
        run_id: str,
        failure_code: str,
        failure_message: str,
    ) -> RecommendationRun:
        run = self._required(run_id)
        self._transition(run, "failed")
        run.status = "failed"
        run.failure_code = str(failure_code)[:64]
        run.failure_message = str(failure_message)[:512]
        run.result_json = None
        run.classes = []
        run.assumptions = []
        run.risks = []
        run.output_hash = None
        run.completed_at = utc_now()
        self.session.flush()
        return run

    def fail_stale_runs(
        self,
        failure_code: str = "worker_restarted",
        failure_message: str = "A execução foi interrompida e precisa ser solicitada novamente.",
    ) -> list[RecommendationRun]:
        runs = list(
            self.session.scalars(
                select(RecommendationRun).where(
                    RecommendationRun.status.in_(("queued", "running"))
                )
            )
        )
        for run in runs:
            self.mark_failed(run.id, failure_code, failure_message)
        return runs

    recover_stale = fail_stale_runs

    def _required(self, run_id: str) -> RecommendationRun:
        run = self.get(run_id)
        if run is None:
            raise RecommendationStateError("recommendation run not found")
        if run.status not in RUN_STATES:
            raise RecommendationStateError(f"unknown recommendation state: {run.status}")
        return run

    @staticmethod
    def _transition(run: RecommendationRun, target: str) -> None:
        if target not in RUN_STATES:
            raise RecommendationStateError(f"unknown recommendation state: {target}")
        if target not in VALID_TRANSITIONS.get(run.status, frozenset()):
            raise RecommendationStateError(
                f"invalid recommendation transition: {run.status} -> {target}"
            )
