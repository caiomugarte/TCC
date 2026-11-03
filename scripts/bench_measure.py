"""RSS sampling and per-stage measurement for scripts/benchmark_premium.py.

Linux only (``/proc`` and ``resource``). Kept apart from the CLI so the
benchmark entry point stays small; it is part of the benchmark's code
fingerprint (``CODE_PATHS`` in benchmark_premium.py).
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import resource
import threading
import time
from typing import Any, Callable, Iterator, Mapping, Sequence

ENGINE_STAGES = ("snapshot_load", "allocation", "stocks", "fiis")
RSS_INTERVAL_S = 1.0
# The sampler may miss a sub-second spike; below this ratio the run is flagged.
RSS_MIN_RATIO = 0.90
# ru_maxrss and VmHWM both come from approximate kernel RSS counters and may
# differ slightly; beyond this ratio of the peak they are said to disagree.
RSS_PEAK_TOLERANCE = 0.02

TIME_FIELDS = (
    "wall_s",
    "cpu_process_time_s",
    "cpu_user_s",
    "cpu_system_s",
    "cpu_seconds",
    "cpu_children_seconds",
    "cpu_seconds_total",
)


# --------------------------------------------------------------------------
# RSS sampling
# --------------------------------------------------------------------------


def read_proc_status_kb(field: str, status_path: Path = Path("/proc/self/status")) -> int | None:
    """Return a ``/proc/<pid>/status`` field in kB (``VmRSS``, ``VmHWM``)."""

    try:
        text = status_path.read_text(encoding="ascii", errors="replace")
    except OSError:
        return None
    prefix = f"{field}:"
    for line in text.splitlines():
        if line.startswith(prefix):
            parts = line[len(prefix):].split()
            if parts:
                return int(parts[0])
    return None


def ru_maxrss_kb() -> int:
    """Process peak RSS from getrusage; Linux reports kilobytes."""

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


class RssSampler:
    """Background thread reading VmRSS every ``interval`` seconds."""

    def __init__(
        self,
        interval: float = RSS_INTERVAL_S,
        reader: Callable[[], int | None] = lambda: read_proc_status_kb("VmRSS"),
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.interval = float(interval)
        self.reader = reader
        self.clock = clock
        self.samples: list[tuple[float, int]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def sample_now(self) -> int | None:
        value = self.reader()
        if value is not None:
            with self._lock:
                self.samples.append((self.clock(), int(value)))
        return value

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.sample_now()

    def start(self) -> "RssSampler":
        self.sample_now()
        self._thread = threading.Thread(target=self._run, name="rss-sampler", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.sample_now()

    def __enter__(self) -> "RssSampler":
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    def snapshot(self) -> list[tuple[float, int]]:
        with self._lock:
            return list(self.samples)

    def peak_kb(self, start: float | None = None, end: float | None = None) -> int | None:
        values = [
            kb
            for at, kb in self.snapshot()
            if (start is None or at >= start) and (end is None or at <= end)
        ]
        return max(values) if values else None

    def count(self, start: float | None = None, end: float | None = None) -> int:
        return sum(
            1
            for at, _kb in self.snapshot()
            if (start is None or at >= start) and (end is None or at <= end)
        )

    def max_gap_s(self) -> float | None:
        times = sorted(at for at, _kb in self.snapshot())
        if len(times) < 2:
            return None
        return max(b - a for a, b in zip(times, times[1:]))


def validate_rss(
    sampled_peak_kb: int | None,
    ru_maxrss: int,
    vm_hwm_kb: int | None = None,
    *,
    min_ratio: float = RSS_MIN_RATIO,
    tolerance: float = RSS_PEAK_TOLERANCE,
) -> dict[str, Any]:
    """Cross-check the sampled peak against the kernel's process peak.

    The process peak is ``max(ru_maxrss, VmHWM)``: ``getrusage`` reads the
    approximate per-CPU RSS counters, so it can trail the exact ``VmHWM`` by
    a little; both must agree within ``tolerance``.

    ``invalid``: the sampler saw more than the process peak, or the two
    kernel peaks disagree (the measurement is broken). ``undersampled``: the
    1 s sampler missed more than ``1 - min_ratio`` of the peak. ``ok``
    otherwise.
    """

    peaks = [value for value in (ru_maxrss, vm_hwm_kb) if value]
    if sampled_peak_kb is None or not peaks:
        return {"status": "invalid", "ratio": None, "process_peak_kb": None, "reason": "no RSS samples or no process peak"}
    process_peak = max(peaks)
    ratio = sampled_peak_kb / process_peak
    if sampled_peak_kb > process_peak * (1 + tolerance):
        status, reason = "invalid", "sampled peak exceeds the process peak"
    elif vm_hwm_kb and ru_maxrss and abs(ru_maxrss - vm_hwm_kb) > tolerance * process_peak:
        status, reason = "invalid", "ru_maxrss and VmHWM disagree"
    elif ratio < min_ratio:
        status, reason = "undersampled", f"sampled peak below {min_ratio:.0%} of the process peak"
    else:
        status, reason = "ok", None
    return {"status": status, "ratio": round(ratio, 4), "process_peak_kb": process_peak, "reason": reason}


# --------------------------------------------------------------------------
# Stage measurement
# --------------------------------------------------------------------------


def _usage() -> dict[str, float]:
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "wall": time.perf_counter(),
        "process": time.process_time(),
        "user": own.ru_utime,
        "system": own.ru_stime,
        "children": children.ru_utime + children.ru_stime,
    }


def _usage_delta(start: Mapping[str, float], end: Mapping[str, float]) -> dict[str, float]:
    user = end["user"] - start["user"]
    system = end["system"] - start["system"]
    children = end["children"] - start["children"]
    wall = end["wall"] - start["wall"]
    total = user + system + children
    return {
        "wall_s": wall,
        "cpu_process_time_s": end["process"] - start["process"],
        "cpu_user_s": user,
        "cpu_system_s": system,
        "cpu_seconds": user + system,
        "cpu_children_seconds": children,
        "cpu_seconds_total": total,
        "cpu_utilization": (total / wall) if wall > 0 else None,
    }


def _empty_stage() -> dict[str, Any]:
    stage: dict[str, Any] = {field: 0.0 for field in TIME_FIELDS}
    stage.update(
        {
            "executed": False,
            "cpu_utilization": None,
            "rss_peak_sampled_kb": None,
            "rss_samples": 0,
            "vm_hwm_kb_at_end": None,
        }
    )
    return stage


class StageMeter:
    """Measures named stages around injected engine callables."""

    def __init__(self, sampler: RssSampler) -> None:
        self.sampler = sampler
        self.stages: dict[str, dict[str, Any]] = {}

    @contextmanager
    def measure(self, name: str) -> Iterator[dict[str, Any]]:
        record: dict[str, Any] = {"executed": True}
        self.stages[name] = record
        start = _usage()
        self.sampler.sample_now()
        try:
            yield record
        finally:
            self.sampler.sample_now()
            end = _usage()
            record.update(_usage_delta(start, end))
            record["rss_peak_sampled_kb"] = self.sampler.peak_kb(start["wall"], end["wall"] + 1e-9)
            record["rss_samples"] = self.sampler.count(start["wall"], end["wall"] + 1e-9)
            record["vm_hwm_kb_at_end"] = read_proc_status_kb("VmHWM")

    def wrap(
        self,
        name: str,
        engine: Callable[..., Any],
        inspect: Callable[[Any], Mapping[str, Any]] | None = None,
        around: Callable[[], Any] | None = None,
    ) -> Callable[..., Any]:
        """Return ``engine`` timed as stage ``name``.

        ``around`` is an optional context-manager factory entered inside the
        measured window; its ``__enter__`` value is passed to ``inspect``.
        """

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self.measure(name) as record:
                if around is None:
                    result = engine(*args, **kwargs)
                    observed: Any = result
                else:
                    with around() as captured:
                        result = engine(*args, **kwargs)
                    observed = (result, captured)
            if inspect is not None:
                record["engine"] = dict(inspect(observed))
            return result

        return wrapper


def compose_stages(stages: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Fill missing engine stages and derive ``composition = total - engines``."""

    if "total" not in stages:
        raise ValueError("total stage was not measured")
    result = {name: dict(stages.get(name) or _empty_stage()) for name in ENGINE_STAGES}
    total = dict(stages["total"])
    composition: dict[str, Any] = {"executed": True, "derived": "total - engines"}
    for field in TIME_FIELDS:
        engines = sum(float(result[name].get(field, 0.0)) for name in ENGINE_STAGES)
        composition[field] = float(total[field]) - engines
    wall = composition["wall_s"]
    composition["cpu_utilization"] = (
        composition["cpu_seconds_total"] / wall if wall > 0 else None
    )
    composition["rss_peak_sampled_kb"] = None
    composition["rss_samples"] = 0
    composition["vm_hwm_kb_at_end"] = None
    result["composition"] = composition
    result["total"] = total
    return result


def generation_summary(runs: Sequence[Mapping[str, Any]], reported_n_runs: object) -> dict[str, Any]:
    generations = [int(run["generations_run"]) for run in runs if run.get("generations_run") is not None]
    return {
        "n_runs_effective": int(reported_n_runs) if reported_n_runs is not None else len(runs),
        "runs_observed": len(runs),
        "generations_total": sum(generations) if generations else None,
        "generations_mean": (sum(generations) / len(generations)) if generations else None,
        "generations_min": min(generations) if generations else None,
        "generations_max": max(generations) if generations else None,
        "converged_early_runs": sum(1 for run in runs if run.get("converged_early")),
    }


def stock_inspect(result: Any) -> dict[str, Any]:
    result = result if isinstance(result, Mapping) else {}
    runs = result.get("all_runs") or []
    return generation_summary(list(runs), result.get("n_runs"))


def fii_inspect(payload: Any) -> dict[str, Any]:
    result, runs = payload
    result = result if isinstance(result, Mapping) else {}
    return generation_summary(list(runs), result.get("n_runs"))


@contextmanager
def capture_fii_runs() -> Iterator[list[Mapping[str, Any]]]:
    """Record each FII execution result (observation only).

    ``run_fii_selection`` returns ``n_runs`` but not the per-run
    ``generations_run``; it resolves ``_run_fii_execution`` at call time, so
    a recording proxy sees every sequential run. Restored on exit.
    """

    import fii_selection

    captured: list[Mapping[str, Any]] = []
    original = fii_selection._run_fii_execution

    def recording(*args: Any, **kwargs: Any) -> Any:
        execution_result, portfolio = original(*args, **kwargs)
        captured.append(dict(execution_result))
        return execution_result, portfolio

    fii_selection._run_fii_execution = recording
    try:
        yield captured
    finally:
        fii_selection._run_fii_execution = original
