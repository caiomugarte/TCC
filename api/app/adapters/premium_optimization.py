from __future__ import annotations

from dataclasses import asdict, is_dataclass
import math
from pathlib import Path
import sys
from typing import Any, Callable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PY_ROOT = PROJECT_ROOT / "py"
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from allocation_config import ASSET_CLASSES  # noqa: E402


def _default_snapshot_loader(manifest: object) -> object:
    from allocation_data import load_snapshot_bundle

    return load_snapshot_bundle(manifest)


def _default_allocation_engine(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    from pipelines.asset_allocation import run_allocation

    return run_allocation(*args, **kwargs)


def _default_stock_engine(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    from pipelines.multi_run import run_multi_execution_profile

    config = kwargs["config"]
    settings = config.get("system_ga_config", {})
    run_count = int(settings.get("run_count", 1)) if isinstance(settings, Mapping) else 1
    return run_multi_execution_profile(
        profile=None,
        n_runs=run_count,
        parallel=False,
        selection_config=config,
        workspace=kwargs["workspace"],
        random_seed=int(kwargs["seed"]),
        input_path=kwargs["input_path"],
    )


def _default_fii_engine(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    from fii_selection import run_fii_selection

    return run_fii_selection(*args, **kwargs)


API_CLASS_KEYS = {
    "brazilian_stocks": "brazilian_stocks",
    "fiis": "fiis",
    "international_equity": "international",
    "fixed_income": "fixed_income",
    "crypto": "crypto",
}

CLASS_LABELS = {
    "brazilian_stocks": "Ações brasileiras",
    "fiis": "FIIs",
    "international": "Exposição internacional",
    "fixed_income": "Renda fixa",
    "crypto": "Criptoativos",
}


class PremiumOptimizationError(ValueError):
    """Raised when a Premium engine cannot produce a complete result."""

    def __init__(self, message: str, code: str = "engine_error") -> None:
        super().__init__(message)
        self.code = code


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _policy_section(policy: object, name: str) -> dict[str, Any]:
    if isinstance(policy, Mapping):
        value = policy.get(name)
    else:
        value = getattr(policy, name, None)
    if value is None:
        raise PremiumOptimizationError(f"policy section is missing: {name}", "policy_invalid")
    normalized = _json_value(value)
    if not isinstance(normalized, dict):
        raise PremiumOptimizationError(f"policy section is invalid: {name}", "policy_invalid")
    return normalized


def _manifest_value(manifest: object, name: str, default: object = None) -> object:
    if isinstance(manifest, Mapping):
        return manifest.get(name, default)
    return getattr(manifest, name, default)


def _source_path(manifest: object, key: str) -> Path:
    source: object | None = None
    source_for = getattr(manifest, "source_for", None)
    if callable(source_for):
        try:
            source = source_for(key)
        except (KeyError, ValueError):
            source = None

    if source is None and isinstance(manifest, Mapping):
        sources = manifest.get("sources", {})
        if isinstance(sources, Mapping):
            aliases = {"stock": "stocks", "fii": "fiis"}
            source = sources.get(key) or sources.get(aliases.get(key, key))
        source = source or manifest.get(f"{key}_source")

    if source is None:
        raise PremiumOptimizationError(
            f"manifest source is missing: {key}", "snapshot_unavailable"
        )
    resolved_path = getattr(source, "resolved_path", None)
    if callable(resolved_path):
        return Path(resolved_path())
    if isinstance(source, Mapping):
        raw_path = source.get("path") or source.get("source_path")
    else:
        raw_path = source
    if not raw_path:
        raise PremiumOptimizationError(
            f"manifest source path is missing: {key}", "snapshot_unavailable"
        )
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path
    manifest_path = _manifest_value(manifest, "manifest_path")
    if manifest_path:
        return (Path(str(manifest_path)).parent / path).resolve()
    return path.resolve()


def _seed(policy: object, explicit_seed: int | None) -> int:
    if explicit_seed is not None:
        return int(explicit_seed)
    provenance = _policy_section(policy, "provenance")
    value = provenance.get("random_seed")
    return 42 if value is None else int(value)


def _allocation_config(section: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    constraints = section.get("class_constraints")
    if not isinstance(constraints, Mapping):
        raise PremiumOptimizationError("allocation class constraints are missing", "policy_invalid")
    config = {
        "volatility_cap": section.get("volatility_cap"),
        "drawdown_cap": section.get("drawdown_cap"),
        "crypto_risk_contribution_cap": section.get("crypto_risk_contribution_cap"),
        "hhi_penalty": section.get("hhi_penalty"),
        "risk_adjusted_weights": section.get("risk_adjusted_weights", {}),
        "minimum_class_weight": constraints.get("minimum_class_weight", 0.0),
    }
    return config, dict(constraints)


def _extract_allocation_candidate(result: Mapping[str, Any]) -> Mapping[str, Any]:
    current_target = result.get("current_target")
    selected = current_target.get("selected") if isinstance(current_target, Mapping) else None
    if isinstance(selected, Mapping):
        for key in ("profile_winner", "knee", "selected"):
            candidate = selected.get(key)
            if isinstance(candidate, Mapping):
                return candidate
    candidate = result.get("selected")
    if isinstance(candidate, Mapping):
        return candidate
    return result


def _allocation_weights(result: Mapping[str, Any]) -> tuple[dict[str, float], Mapping[str, Any]]:
    candidate = _extract_allocation_candidate(result)
    raw_weights = candidate.get("weights")
    if raw_weights is None:
        raw_classes = candidate.get("classes")
        if isinstance(raw_classes, list):
            raw_weights = {
                item.get("key"): item.get("target_weight")
                for item in raw_classes
                if isinstance(item, Mapping)
            }
    if not isinstance(raw_weights, Mapping):
        raise PremiumOptimizationError("allocation result has no class weights", "engine_error")

    aliases = {"international": "international_equity"}
    weights: dict[str, float] = {}
    for key, value in raw_weights.items():
        engine_key = aliases.get(str(key), str(key))
        if engine_key not in ASSET_CLASSES:
            raise PremiumOptimizationError(
                f"allocation result has unknown class: {key}", "engine_error"
            )
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise PremiumOptimizationError(
                f"allocation result has invalid weight: {key}", "engine_error"
            )
        weights[engine_key] = float(value)
    if set(weights) != set(ASSET_CLASSES):
        raise PremiumOptimizationError("allocation result is missing a class", "engine_error")
    if any(value < -1e-9 or value > 1.0 + 1e-9 for value in weights.values()):
        raise PremiumOptimizationError("allocation weights are outside 0..1", "engine_error")
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise PremiumOptimizationError(
            f"class weights must sum to 1; got {total:.8f}", "engine_error"
        )
    return weights, candidate


def _validate_constraints(weights: Mapping[str, float], constraints: Mapping[str, Any]) -> None:
    minimums = constraints.get("minimum_weights", constraints.get("minimums", {}))
    maximums = constraints.get("maximum_weights", constraints.get("maximums", {}))
    if not isinstance(minimums, Mapping) or not isinstance(maximums, Mapping):
        raise PremiumOptimizationError("allocation class constraints are invalid", "policy_invalid")
    for key, minimum in minimums.items():
        if weights.get(str(key), -1.0) < float(minimum) - 1e-9:
            raise PremiumOptimizationError(
                f"allocation violates minimum weight for {key}", "infeasible_constraints"
            )
    for key, maximum in maximums.items():
        if weights.get(str(key), 2.0) > float(maximum) + 1e-9:
            raise PremiumOptimizationError(
                f"allocation violates maximum weight for {key}", "infeasible_constraints"
            )
    hhi_max = constraints.get("hhi_max")
    if hhi_max is not None and sum(value * value for value in weights.values()) > float(hhi_max) + 1e-9:
        raise PremiumOptimizationError("allocation violates class HHI limit", "infeasible_constraints")


def _rounded_weights(weights: Mapping[str, float]) -> dict[str, float]:
    rounded = {key: round(float(value), 6) for key, value in weights.items()}
    difference = round(1.0 - sum(rounded.values()), 6)
    if difference:
        target = max(rounded, key=rounded.get)
        rounded[target] = round(rounded[target] + difference, 6)
    return rounded


def _rounded_amounts(weights: Mapping[str, float], capital: float) -> dict[str, float]:
    """Allocate whole BRL cents without making a zero-weight item negative."""

    capital_cents = int(round(float(capital) * 100))
    if capital_cents < 0 or not weights:
        raise PremiumOptimizationError("capital allocation is invalid", "engine_error")
    raw_cents = {
        key: max(0.0, float(value)) * capital_cents
        for key, value in weights.items()
    }
    cents = {key: int(math.floor(value + 1e-9)) for key, value in raw_cents.items()}
    remainder = capital_cents - sum(cents.values())
    if remainder > 0:
        order = sorted(
            cents,
            key=lambda key: (-(raw_cents[key] - cents[key]), str(key)),
        )
        for index in range(remainder):
            cents[order[index % len(order)]] += 1
        remainder = 0
    elif remainder < 0:
        order = sorted(cents, key=lambda key: (-cents[key], str(key)))
        for key in order:
            while remainder < 0 and cents[key] > 0:
                cents[key] -= 1
                remainder += 1
    if remainder != 0 or any(value < 0 for value in cents.values()):
        raise PremiumOptimizationError("capital allocation cannot be rounded safely", "engine_error")
    return {key: value / 100 for key, value in cents.items()}


def _class_targets(weights: Mapping[str, float], capital: float, candidate: Mapping[str, Any]) -> list[dict[str, Any]]:
    classes = []
    metrics = candidate.get("metrics", {})
    if not isinstance(metrics, Mapping):
        metrics = {}
    amounts = _rounded_amounts(
        {engine_key: max(0.0, float(weights[engine_key])) for engine_key in ASSET_CLASSES},
        capital,
    )
    for engine_key in ASSET_CLASSES:
        weight = max(0.0, float(weights[engine_key]))
        api_key = API_CLASS_KEYS[engine_key]
        classes.append(
            {
                "key": api_key,
                "label": CLASS_LABELS[api_key],
                "target_weight": weight,
                "target_amount_brl": amounts[engine_key],
                "metrics": dict(_json_value(metrics)),
            }
        )
    return classes


def _selector_config(section: Mapping[str, Any]) -> dict[str, Any]:
    ga = section.get("system_ga_config")
    if not isinstance(ga, Mapping):
        raise PremiumOptimizationError("selector GA configuration is missing", "policy_invalid")
    settings = dict(ga)
    settings.setdefault("n_assets", section.get("n_assets"))
    settings.setdefault("lambda", section.get("lambda_hhi"))
    if "population" in settings:
        settings.setdefault("pop_size", settings["population"])
    for key in ("n_assets", "lambda", "generations", "pop_size"):
        if settings.get(key) is None:
            raise PremiumOptimizationError(
                f"selector GA configuration is missing: {key}", "policy_invalid"
            )
    return {
        "selection_preset": section.get("selection_preset", "explicit"),
        "n_assets": int(section["n_assets"]),
        "factor_weights": dict(section.get("factor_weights", {})),
        "liquidity_and_size_filters": dict(section.get("liquidity_and_size_filters", {})),
        "lambda_hhi": float(section.get("lambda_hhi", settings["lambda"])),
        "system_ga_config": settings,
    }


def _selector_output(result: Mapping[str, Any]) -> tuple[list[str], dict[str, float], dict[str, Any]]:
    raw_tickers = result.get("selected_tickers", result.get("tickers"))
    if raw_tickers is None:
        portfolio = result.get("portfolio")
        if portfolio is not None and hasattr(portfolio, "__getitem__"):
            try:
                raw_tickers = list(portfolio["TICKER"])
            except (KeyError, TypeError):
                raw_tickers = None
    if not isinstance(raw_tickers, (list, tuple)) or not raw_tickers:
        raise PremiumOptimizationError("selector returned no constituents", "engine_error")
    tickers = [str(ticker).strip().upper() for ticker in raw_tickers]
    if any(not ticker for ticker in tickers) or len(set(tickers)) != len(tickers):
        raise PremiumOptimizationError("selector returned invalid constituents", "engine_error")

    raw_weights = result.get("sleeve_weights", result.get("weights"))
    weights: dict[str, float] = {}
    if isinstance(raw_weights, Mapping):
        for ticker in tickers:
            value = raw_weights.get(ticker, raw_weights.get(ticker.lower()))
            if value is None:
                raise PremiumOptimizationError(
                    f"selector has no sleeve weight for {ticker}", "engine_error"
                )
            try:
                weights[ticker] = float(value)
            except (TypeError, ValueError) as exc:
                raise PremiumOptimizationError(
                    f"selector returned invalid sleeve weight for {ticker}", "engine_error"
                ) from exc
    else:
        equal = 1.0 / len(tickers)
        weights = {ticker: equal for ticker in tickers}
    if any(not math.isfinite(value) or value < 0 for value in weights.values()):
        raise PremiumOptimizationError("selector returned invalid sleeve weights", "engine_error")
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise PremiumOptimizationError(
            f"sleeve weights must sum to 1; got {total:.8f}", "engine_error"
        )
    return tickers, _rounded_weights(weights), dict(result)


def _selector_hhi(result: Mapping[str, Any]) -> float | None:
    candidates = [result.get("hhi")]
    metrics = result.get("metrics")
    if isinstance(metrics, Mapping):
        candidates.append(metrics.get("hhi"))
    for value in candidates:
        if value is None:
            continue
        try:
            result_value = float(value)
        except (TypeError, ValueError) as exc:
            raise PremiumOptimizationError(
                "selector returned an invalid HHI", "engine_error"
            ) from exc
        if not math.isfinite(result_value):
            raise PremiumOptimizationError("selector returned an invalid HHI", "engine_error")
        return result_value
    portfolio = result.get("portfolio")
    if portfolio is not None:
        try:
            from core.metrics import hhi_sector

            result_value = float(hhi_sector(portfolio))
        except (TypeError, ValueError, KeyError) as exc:
            raise PremiumOptimizationError(
                "selector returned an invalid portfolio HHI", "engine_error"
            ) from exc
        if math.isfinite(result_value):
            return result_value
    return None


def _validate_selector_constraints(
    result: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    filters = config.get("liquidity_and_size_filters", {})
    if not isinstance(filters, Mapping) or filters.get("hhi_max") is None:
        return
    hhi_max = float(filters["hhi_max"])
    hhi = _selector_hhi(result)
    if hhi is None:
        raise PremiumOptimizationError(
            "selector result does not expose HHI for the configured limit",
            "engine_error",
        )
    if hhi > hhi_max + 1e-9:
        raise PremiumOptimizationError(
            f"selector HHI {hhi:.6f} exceeds {hhi_max:.6f}",
            "infeasible_constraints",
        )


def _constituents(
    result: Mapping[str, Any],
    class_weight: float,
    class_amount: float,
) -> list[dict[str, Any]]:
    tickers, sleeve_weights, normalized = _selector_output(result)
    reasons = normalized.get("exclusion_reasons", {})
    if not isinstance(reasons, Mapping):
        reasons = {}
    amounts = _rounded_amounts(sleeve_weights, class_amount)
    constituents = []
    for ticker in tickers:
        reason_value = reasons.get(ticker, [])
        if isinstance(reason_value, (list, tuple)):
            ticker_reasons = [str(item) for item in reason_value]
        elif reason_value:
            ticker_reasons = [str(reason_value)]
        else:
            ticker_reasons = []
        sleeve_weight = sleeve_weights[ticker]
        portfolio_weight = class_weight * sleeve_weight
        constituents.append(
            {
                "ticker": ticker,
                "sleeve_weight": round(sleeve_weight, 6),
                "portfolio_weight": round(portfolio_weight, 6),
                "target_amount_brl": amounts[ticker],
                "reasons": ticker_reasons,
            }
        )
    return constituents


def run_premium_optimization(
    policy: object,
    manifest: object,
    workspace: Path,
    investable_capital_brl: float,
    *,
    snapshot_loader: Callable[..., object] = _default_snapshot_loader,
    allocation_engine: Callable[..., Mapping[str, Any]] = _default_allocation_engine,
    stock_engine: Callable[..., Mapping[str, Any]] = _default_stock_engine,
    fii_engine: Callable[..., Mapping[str, Any]] = _default_fii_engine,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run the three explicit Premium engines and compose one API result."""

    if not math.isfinite(float(investable_capital_brl)) or investable_capital_brl <= 0:
        raise PremiumOptimizationError("investable capital must be positive", "policy_invalid")
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    allocation_section = _policy_section(policy, "allocation")
    stock_section = _policy_section(policy, "stocks")
    fii_section = _policy_section(policy, "fiis")
    policy_dict = _json_value(policy)
    if not isinstance(policy_dict, dict):
        raise PremiumOptimizationError("policy is not serializable", "policy_invalid")
    run_seed = _seed(policy, seed)

    try:
        bundle = snapshot_loader(manifest)
        allocation_config, class_constraints = _allocation_config(allocation_section)
        allocation_result = allocation_engine(
            bundle.rows,
            metadata=bundle.metadata,
            allocation_config=allocation_config,
            class_constraints=class_constraints,
        )
    except PremiumOptimizationError:
        raise
    except Exception as exc:
        raise PremiumOptimizationError(str(exc), "engine_error") from exc
    if not isinstance(allocation_result, Mapping):
        raise PremiumOptimizationError("allocation engine returned an invalid result", "engine_error")
    if allocation_result.get("status") == "unavailable" or allocation_result.get("diagnostics"):
        diagnostic = allocation_result.get("diagnostics") or "allocation is unavailable"
        raise PremiumOptimizationError(str(diagnostic), "infeasible_constraints")
    weights, allocation_candidate = _allocation_weights(allocation_result)
    _validate_constraints(weights, class_constraints)
    weights = _rounded_weights(weights)
    classes = _class_targets(weights, float(investable_capital_brl), allocation_candidate)
    class_by_engine = {
        engine_key: item for engine_key, item in zip(ASSET_CLASSES, classes)
    }
    stock_config = _selector_config(stock_section)
    fii_config = _selector_config(fii_section)
    selector_results: dict[str, Mapping[str, Any]] = {}

    stock_weight = weights["brazilian_stocks"]
    if stock_weight > 1e-9:
        stock_workspace = workspace / "stocks"
        try:
            stock_result = stock_engine(
                input_path=_source_path(manifest, "stock"),
                config=stock_config,
                workspace=stock_workspace,
                seed=run_seed,
            )
        except PremiumOptimizationError:
            raise
        except Exception as exc:
            code = (
                "infeasible_constraints"
                if stock_config["liquidity_and_size_filters"].get("hhi_max") is not None
                and any(term in str(exc).lower() for term in ("converg", "hhi", "constraint"))
                else "engine_error"
            )
            raise PremiumOptimizationError("stock selector failed", code) from exc
        if not isinstance(stock_result, Mapping):
            raise PremiumOptimizationError("stock engine returned an invalid result", "engine_error")
        _validate_selector_constraints(stock_result, stock_config)
        selector_results["stocks"] = stock_result
        stocks = _constituents(
            stock_result,
            stock_weight,
            float(class_by_engine["brazilian_stocks"]["target_amount_brl"]),
        )
    else:
        stocks = []

    fii_weight = weights["fiis"]
    if fii_weight > 1e-9:
        fii_workspace = workspace / "fiis"
        try:
            fii_result = fii_engine(
                input_path=_source_path(manifest, "fii"),
                selection_config=fii_config,
                n_runs=max(1, int(fii_config["system_ga_config"].get("run_count", 1))),
                parallel=False,
                workspace=fii_workspace,
                seed=run_seed,
            )
        except PremiumOptimizationError:
            raise
        except Exception as exc:
            code = (
                "infeasible_constraints"
                if fii_config["liquidity_and_size_filters"].get("hhi_max") is not None
                and any(term in str(exc).lower() for term in ("converg", "hhi", "constraint"))
                else "engine_error"
            )
            raise PremiumOptimizationError("FII selector failed", code) from exc
        if not isinstance(fii_result, Mapping):
            raise PremiumOptimizationError("FII engine returned an invalid result", "engine_error")
        _validate_selector_constraints(fii_result, fii_config)
        selector_results["fiis"] = fii_result
        fiis = _constituents(
            fii_result,
            fii_weight,
            float(class_by_engine["fiis"]["target_amount_brl"]),
        )
    else:
        fiis = []

    provenance = _policy_section(policy, "provenance")
    provenance = {
        **provenance,
        "manifest_id": _manifest_value(manifest, "manifest_id"),
        "manifest_path": str(_manifest_value(manifest, "manifest_path"))
        if _manifest_value(manifest, "manifest_path")
        else None,
        "workspace": str(workspace),
        "random_seed": run_seed,
    }
    return {
        "policy": policy_dict,
        "classes": classes,
        "stocks": stocks,
        "fiis": fiis,
        "assumptions": [
            "Resultado Premium bruto, baseado em snapshots imutáveis e regras versionadas.",
            "Ações brasileiras e FIIs são sleeves selecionados pelos parâmetros resolvidos do perfil.",
            "Valores em BRL representam alvos de capital investível, não ordens de execução.",
        ],
        "risks": [
            "Dados históricos e proxies não garantem resultados futuros.",
            "Custos, impostos, spreads e execução não fazem parte deste cálculo.",
            "Restrições de perfil ou dados insuficientes podem tornar uma execução indisponível.",
        ],
        "provenance": _json_value(provenance),
        "diagnostics": {
            "allocation": _json_value(allocation_result.get("diagnostics")),
            "selectors": {
                key: _json_value(value.get("metrics", {}))
                for key, value in selector_results.items()
            },
        },
    }
