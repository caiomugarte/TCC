from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from threading import RLock
from typing import Any

from app.adapters.premium_optimization import PremiumOptimizationError, run_premium_optimization
from app.db.models import ProfileRecord
from app.db.session import SessionLocal
from app.repositories.recommendations import RecommendationRepository, RecommendationStateError


def _open_session(factory: Callable[[], Any]) -> Any:
    return factory()


def _close_session(session: Any) -> None:
    close = getattr(session, "close", None)
    if callable(close):
        close()


class PremiumExecutor:
    """Single-process bounded worker for the private Premium pilot."""

    def __init__(
        self,
        *,
        session_factory: Callable[[], Any] = SessionLocal,
        optimization_runner: Callable[..., Mapping[str, Any]] = run_premium_optimization,
        manifest_loader: Callable[[Mapping[str, Any]], object] | None = None,
        manifest_validator: Callable[[object], object] | None = None,
        workspace_root: Path | None = None,
        max_workers: int = 1,
        max_pending: int = 2,
    ) -> None:
        if max_workers != 1:
            raise ValueError("Premium pilot executor must use one worker")
        if max_pending < 1:
            raise ValueError("Premium pilot executor queue must allow one run")
        self.session_factory = session_factory
        self.optimization_runner = optimization_runner
        self.manifest_loader = manifest_loader
        self.manifest_validator = manifest_validator
        self.workspace_root = Path(workspace_root or Path(tempfile.gettempdir()) / "prumo-premium")
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="premium")
        self.futures: dict[str, Future[None]] = {}
        self.max_pending = max_pending
        self._futures_lock = RLock()
        self.closed = False

    def submit(self, run_id: str) -> Future[None]:
        with self._futures_lock:
            if self.closed:
                raise RuntimeError("Premium worker is shut down")
            self.futures = {
                current_id: future
                for current_id, future in self.futures.items()
                if not future.done()
            }
            if len(self.futures) >= self.max_pending:
                raise RuntimeError("Premium worker queue is full")
            future = self.pool.submit(self.execute_run, run_id)
            self.futures[run_id] = future
            future.add_done_callback(lambda _future: self._forget_future(run_id))
            return future

    def _forget_future(self, run_id: str) -> None:
        with self._futures_lock:
            self.futures.pop(run_id, None)

    def execute_run(self, run_id: str) -> None:
        session = _open_session(self.session_factory)
        try:
            repository = RecommendationRepository(session)
            run = repository.get(run_id)
            if run is None or run.status != "queued":
                _close_session(session)
                return
            repository.mark_running(run_id)
            session.commit()
            policy = run.policy_json
            provenance = dict(run.provenance_json or {})
            profile = session.get(ProfileRecord, run.profile_id)
            if profile is None:
                raise RuntimeError("profile for Premium run is no longer available")
            capital = float(profile.investable_capital_brl)
        except Exception as exc:
            _close_session(session)
            self._fail(run_id, self._failure_code(exc), str(exc))
            return
        _close_session(session)

        try:
            manifest = self._load_manifest(provenance)
            workspace = self.workspace_root / run_id
            workspace.mkdir(parents=True, exist_ok=True)
            result = self.optimization_runner(policy, manifest, workspace, capital)
            if not isinstance(result, Mapping):
                raise RuntimeError("Premium optimization returned an invalid result")
        except Exception as exc:
            self._fail(run_id, self._failure_code(exc), str(exc))
            return

        session = _open_session(self.session_factory)
        try:
            repository = RecommendationRepository(session)
            repository.mark_completed(run_id, result)
            session.commit()
        except RecommendationStateError:
            session.rollback()
        except Exception as exc:
            session.rollback()
            _close_session(session)
            self._fail(run_id, self._failure_code(exc), str(exc))
            return
        finally:
            _close_session(session)

    def recover_stale(self) -> int:
        session = _open_session(self.session_factory)
        try:
            count = len(RecommendationRepository(session).fail_stale_runs())
            session.commit()
            return count
        finally:
            _close_session(session)

    def shutdown(self, wait: bool = True) -> None:
        self.pool.shutdown(wait=wait)
        self.closed = True

    def _load_manifest(self, provenance: Mapping[str, Any]) -> object:
        if self.manifest_loader is not None:
            manifest = self.manifest_loader(provenance)
        else:
            manifest_data = provenance.get("manifest")
            manifest_path = provenance.get("manifest_path")
            if manifest_data is None and manifest_path:
                from snapshot_manifest import SnapshotManifest

                manifest = SnapshotManifest.load(Path(str(manifest_path)))
            elif isinstance(manifest_data, Mapping):
                from snapshot_manifest import SnapshotManifest

                base_dir = Path(str(manifest_path)).parent if manifest_path else None
                manifest = SnapshotManifest.from_dict(manifest_data, base_dir=base_dir)
            else:
                raise RuntimeError("Premium run has no persisted snapshot manifest")
        if self.manifest_validator is not None:
            return self.manifest_validator(manifest)
        if hasattr(manifest, "source_for"):
            from snapshot_manifest import validate_manifest

            return validate_manifest(manifest)
        return manifest

    @staticmethod
    def _failure_code(exc: Exception) -> str:
        code = getattr(exc, "code", None)
        if isinstance(code, str) and code:
            return code[:64]
        text = str(exc).lower()
        if "snapshot" in text or "manifest" in text:
            return "snapshot_changed"
        if "infeasible" in text or "constraint" in text:
            return "infeasible_constraints"
        if isinstance(exc, PremiumOptimizationError):
            return exc.code
        return "engine_error"

    def _fail(self, run_id: str, code: str, message: str) -> None:
        session = _open_session(self.session_factory)
        try:
            repository = RecommendationRepository(session)
            try:
                repository.mark_failed(run_id, code, self._safe_failure_message(code))
            except RecommendationStateError:
                session.rollback()
                return
            session.commit()
        finally:
            _close_session(session)

    @staticmethod
    def _safe_failure_message(code: str) -> str:
        return {
            "snapshot_changed": "Os dados Premium foram alterados e a execução foi cancelada.",
            "snapshot_unavailable": "Os dados Premium não estão disponíveis para esta execução.",
            "infeasible_constraints": "As restrições do perfil não produziram uma carteira factível.",
            "policy_invalid": "O perfil não produziu uma política Premium válida.",
            "submission_failed": "A execução Premium não pôde ser enviada ao worker.",
            "worker_restarted": "A execução foi interrompida e precisa ser solicitada novamente.",
        }.get(code, "A execução Premium não produziu um resultado.")


_default: PremiumExecutor | None = None


def get_default_executor() -> PremiumExecutor:
    global _default
    if _default is None or _default.closed:
        _default = PremiumExecutor()
    return _default
