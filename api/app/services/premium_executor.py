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
from app.services.premium_policy import WHOLE_UNITS_MODEL_VERSION
from app.services.status_invest_inputs import StatusInvestInputError, StatusInvestInputs, refresh_inputs


def _open_session(factory: Callable[[], Any]) -> Any:
    return factory()


def _close_session(session: Any) -> None:
    close = getattr(session, "close", None)
    if callable(close):
        close()


def _incumbent_tickers(result_json: object, key: str) -> list[str] | None:
    """Tickers of one class of a completed run; None when the data is unreadable.

    Entries without a quantity (pre-change) or with an int quantity >= 1 are kept;
    quantity 0 is skipped; any other quantity makes the class unreadable (LOT-18, LOT-19).
    """

    if not isinstance(result_json, Mapping):
        return None
    entries = result_json.get(key)
    if not isinstance(entries, list):
        return None
    tickers: list[str] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            return None
        quantity = entry.get("quantity")
        if quantity is not None:
            if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity < 0:
                return None
            if quantity == 0:
                continue
        ticker = entry.get("ticker")
        if not isinstance(ticker, str) or not ticker.strip():
            return None
        normalized = ticker.strip().upper()
        if normalized not in tickers:
            tickers.append(normalized)
    return tickers


def _declares_protection_zone(policy: object) -> bool:
    if not isinstance(policy, Mapping):
        return False
    return any(
        isinstance(policy.get(key), Mapping)
        and policy[key].get("protection_zone") is not None
        for key in ("stocks", "fiis")
    )


def _declares_whole_units(policy: object) -> bool:
    if not isinstance(policy, Mapping):
        return False
    provenance = policy.get("provenance")
    if not isinstance(provenance, Mapping):
        return False
    model_versions = provenance.get("model_versions")
    return (
        isinstance(model_versions, Mapping)
        and model_versions.get("whole_units") == WHOLE_UNITS_MODEL_VERSION
    )


class PremiumExecutor:
    """Single-process bounded worker for the private Premium pilot."""

    def __init__(
        self,
        *,
        session_factory: Callable[[], Any] = SessionLocal,
        optimization_runner: Callable[..., Mapping[str, Any]] = run_premium_optimization,
        manifest_loader: Callable[[Mapping[str, Any]], object] | None = None,
        manifest_validator: Callable[[object], object] | None = None,
        status_invest_inputs_loader: Callable[[Path], StatusInvestInputs] = refresh_inputs,
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
        self.status_invest_inputs_loader = status_invest_inputs_loader
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
            whole_units = run.plan == "premium" and _declares_whole_units(policy)
            provenance = dict(run.provenance_json or {})
            profile = session.get(ProfileRecord, run.profile_id)
            if profile is None:
                raise RuntimeError("profile for Premium run is no longer available")
            capital = float(profile.investable_capital_brl)
            incumbents = self._incumbents(repository, run)
        except Exception as exc:
            _close_session(session)
            self._fail(run_id, self._failure_code(exc), exc)
            return
        _close_session(session)

        try:
            manifest = self._load_manifest(provenance)
            workspace = self.workspace_root / run_id
            workspace.mkdir(parents=True, exist_ok=True)
            selector_inputs = self.status_invest_inputs_loader(workspace)
            if not isinstance(selector_inputs, StatusInvestInputs):
                raise StatusInvestInputError("Status Invest inputs were not validated")
            runner_kwargs: dict[str, Any] = {"selector_inputs": selector_inputs}
            if incumbents is not None:
                runner_kwargs["incumbents"] = incumbents
            if whole_units:
                runner_kwargs["whole_units"] = True
            result = self.optimization_runner(
                policy,
                manifest,
                workspace,
                capital,
                **runner_kwargs,
            )
            if not isinstance(result, Mapping):
                raise RuntimeError("Premium optimization returned an invalid result")
        except Exception as exc:
            self._fail(run_id, self._failure_code(exc), exc)
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
            self._fail(run_id, self._failure_code(exc), exc)
            return
        finally:
            _close_session(session)

    @staticmethod
    def _incumbents(
        repository: RecommendationRepository, run: Any
    ) -> dict[str, Any] | None:
        """The account's previous Premium portfolio, read at execution time (A-7)."""

        if run.plan != "premium" or not _declares_protection_zone(run.policy_json):
            return None
        previous = repository.get_latest_completed_for_account(
            run.account_id, plan="premium", exclude_run_id=run.id
        )
        if previous is None:
            return None
        return {
            "run_id": previous.id,
            "stocks": _incumbent_tickers(previous.result_json, "stocks"),
            "fiis": _incumbent_tickers(previous.result_json, "fiis"),
        }

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
        if isinstance(exc, StatusInvestInputError):
            return "snapshot_unavailable"
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

    def _fail(self, run_id: str, code: str, exc: Exception) -> None:
        session = _open_session(self.session_factory)
        try:
            repository = RecommendationRepository(session)
            try:
                fields = getattr(exc, "details", None)
                repository.mark_failed(
                    run_id, code, self._safe_failure_message(code, fields)
                )
            except RecommendationStateError:
                session.rollback()
                return
            session.commit()
        finally:
            _close_session(session)

    @staticmethod
    def _safe_failure_message(
        code: str, fields: Mapping[str, Any] | None = None
    ) -> str:
        if code == "infeasible_universe" and isinstance(fields, Mapping):
            return (
                "Universo FII insuficiente: "
                f"{float(fields['universe_count']):.0f} ativos utilizáveis "
                f"({float(fields['eligible_count']):.0f} elegíveis) "
                f"para {float(fields['required_count']):.0f} exigidos; "
                f"limiares size_min={float(fields['size_min']):.0f}, "
                f"liq_min={float(fields['liq_min']):.0f}; "
                f"perfil {fields['profile']}."
            )
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
