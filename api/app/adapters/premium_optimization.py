from __future__ import annotations

from dataclasses import asdict, is_dataclass
import math
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PY_ROOT = PROJECT_ROOT / "py"
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

# Import after the sys.path mutation: status_invest_inputs imports modules
# that live in py/ (snapshot_manifest), which must already be importable.
from app.services.status_invest_inputs import (  # noqa: E402
    SelectorPriceError,
    SelectorSnapshotChangedError,
    SourceSnapshot,
    StatusInvestInputs,
    load_selector_prices,
)
from app.services.whole_units import (  # noqa: E402
    FII_MINIMUM_HOLDINGS,
    REASON_DID_NOT_FIT,
    REASON_FII_MINIMUM,
    allocate_whole_units,
)

from allocation_config import ASSET_CLASSES  # noqa: E402
from fii_selection import FiiUniverseInfeasibleError  # noqa: E402


def _default_snapshot_loader(manifest: object) -> object:
    from allocation_data import load_snapshot_bundle

    return load_snapshot_bundle(manifest)


def _default_allocation_engine(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    from pipelines.equal_weight_risky import run_equal_weight_risky

    return run_equal_weight_risky(*args, **kwargs)


def _default_stock_engine(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    from pipelines.multi_run import run_multi_execution_profile

    config = kwargs["config"]
    settings = config.get("system_ga_config", {})
    run_count = int(settings.get("run_count", 1)) if isinstance(settings, Mapping) else 1
    return run_multi_execution_profile(
        profile=None,
        n_runs=run_count,
        parallel=False,
        adaptive_mode=bool(settings.get("adaptive_mode", False)) if isinstance(settings, Mapping) else False,
        min_runs=int(settings.get("min_runs", 30)) if isinstance(settings, Mapping) else 30,
        target_cv=float(settings.get("target_cv", 0.03)) if isinstance(settings, Mapping) else 0.03,
        target_jaccard=float(settings.get("target_jaccard", 0.70)) if isinstance(settings, Mapping) else 0.70,
        selection_config=config,
        workspace=kwargs["workspace"],
        random_seed=int(kwargs["seed"]),
        input_path=kwargs["input_path"],
        incumbents=kwargs.get("incumbents"),
    )


def _default_fii_engine(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    from fii_selection import run_fii_selection

    config = kwargs.get("selection_config", kwargs.get("config", {}))
    settings = config.get("system_ga_config", {}) if isinstance(config, Mapping) else {}
    if isinstance(settings, Mapping):
        kwargs.setdefault("adaptive_mode", bool(settings.get("adaptive_mode", False)))
        kwargs.setdefault("min_runs", int(settings.get("min_runs", 30)))
        kwargs.setdefault("target_cv", float(settings.get("target_cv", 0.03)))
        kwargs.setdefault("target_jaccard", float(settings.get("target_jaccard", 0.70)))
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

    def __init__(
        self,
        message: str,
        code: str = "engine_error",
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = (
            MappingProxyType(dict(details)) if details is not None else None
        )


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
    config = {
        "selection_preset": section.get("selection_preset", "explicit"),
        "n_assets": int(section["n_assets"]),
        "factor_weights": dict(section.get("factor_weights", {})),
        "liquidity_and_size_filters": dict(section.get("liquidity_and_size_filters", {})),
        "lambda_hhi": float(section.get("lambda_hhi", settings["lambda"])),
        "system_ga_config": settings,
    }
    zone = section.get("protection_zone")
    if zone is not None:
        if not isinstance(zone, Mapping):
            raise PremiumOptimizationError("selector protection zone is invalid", "policy_invalid")
        config["protection_zone"] = dict(zone)
    cap = section.get("max_per_segment")
    if cap is not None:
        if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
            raise PremiumOptimizationError("selector segment cap is invalid", "policy_invalid")
        config["max_per_segment"] = int(cap)
    return config


def _class_incumbents(
    incumbents: Mapping[str, Any] | None,
    key: str,
    config: Mapping[str, Any],
) -> list[str] | None:
    """Incumbents of one class, only when its config declares a protection zone."""

    if not incumbents or config.get("protection_zone") is None:
        return None
    value = incumbents.get(key)
    if not isinstance(value, (list, tuple)):
        return None
    return [str(ticker) for ticker in value]


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


ZONE_SELECTOR_STATUSES = frozenset({"applied", "fallback_hhi"})


def _zone_record(
    key: str,
    config: Mapping[str, Any],
    incumbents: Mapping[str, Any] | None,
    class_incumbents: list[str] | None,
    result: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Per-class protection-zone record (PZ-28); None when the class declares no zone."""

    zone = config.get("protection_zone")
    if zone is None:
        return None
    if result is not None and class_incumbents:
        record = result.get("protection_zone")
        if not isinstance(record, Mapping) or record.get("status") not in ZONE_SELECTOR_STATUSES:
            raise PremiumOptimizationError(
                f"{key} selector returned no protection-zone record", "engine_error"
            )
        return _json_value(record)
    if result is None:
        status = "class_not_selected"
    elif incumbents is not None and not isinstance(incumbents.get(key), (list, tuple)):
        status = "incumbents_unreadable"
    else:
        status = "no_incumbents"
    return {
        "status": status,
        "K": int(zone["K"]),
        "k_offset": int(zone["k_offset"]),
        "delta": float(zone["delta"]),
        "n_assets": int(config["n_assets"]),
    }


def _zone_reasons(record: Mapping[str, Any] | None) -> dict[str, str]:
    """Reason per selected ticker (PZ-29); empty when the class had no incumbents."""

    if not record or record.get("status") not in ZONE_SELECTOR_STATUSES:
        return {}
    incumbents = set(record.get("incumbents", []))
    swapped_in = {item["in"]: item["out"] for item in record.get("replaced", [])}
    reasons = {}
    for ticker in [*record.get("kept", []), *record.get("entered", [])]:
        if ticker in swapped_in:
            reasons[ticker] = f"substituiu {swapped_in[ticker]}"
        elif ticker in incumbents:
            reasons[ticker] = "mantido"
        else:
            reasons[ticker] = "novo"
    return reasons


def _constituents(
    result: Mapping[str, Any],
    class_weight: float,
    class_amount: float,
    zone_reasons: Mapping[str, str] | None = None,
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
        if zone_reasons and ticker in zone_reasons:
            ticker_reasons.append(zone_reasons[ticker])
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


WHOLE_UNITS_ASSUMPTION = (
    "Quantidades calculadas com o preço da coleta Status Invest desta execução; "
    "a cotação na hora da compra pode ser diferente."
)


def _whole_unit_rows(
    rows: list[dict[str, Any]],
    class_target_brl: float,
    capital_brl: float,
    prices: Mapping[str, int],
    *,
    minimum_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    """Split one class's target rows into bought and dropped whole-unit rows."""

    try:
        allocation = allocate_whole_units(
            [row["ticker"] for row in rows],
            {row["ticker"]: int(round(float(row["target_amount_brl"]) * 100)) for row in rows},
            prices,
            int(round(float(class_target_brl) * 100)),
            minimum_count=minimum_count,
        )
    except ValueError as exc:
        raise PremiumOptimizationError(f"whole-unit allocation failed: {exc}", "engine_error") from exc
    bought: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for row, line in zip(rows, allocation.lines):
        price_brl = line.price_cents / 100
        if line.quantity == 0:
            dropped.append(
                {
                    "ticker": row["ticker"],
                    "sleeve_weight": row["sleeve_weight"],
                    "portfolio_weight": row["portfolio_weight"],
                    "target_amount_brl": row["target_amount_brl"],
                    "price_brl": price_brl,
                    "reasons": [*row["reasons"], REASON_DID_NOT_FIT],
                }
            )
            continue
        amount_brl = line.quantity * line.price_cents / 100
        bought.append(
            {
                **row,
                "reasons": [*row["reasons"], REASON_FII_MINIMUM] if line.forced else row["reasons"],
                "price_brl": price_brl,
                "quantity": line.quantity,
                "amount_brl": amount_brl,
                "effective_portfolio_weight": round(amount_brl / float(capital_brl), 6),
            }
        )
    residual = {
        "cash_residual_brl": allocation.cash_residual_cents / 100,
        "over_target_brl": allocation.over_target_cents / 100,
    }
    return bought, dropped, residual


def _buy_list(
    class_item: dict[str, Any],
    rows: list[dict[str, Any]],
    snapshot: SourceSnapshot,
    price_loader: Callable[[SourceSnapshot, Sequence[str]], Mapping[str, int]],
    capital_brl: float,
    *,
    minimum_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Whole-unit rows of one class; writes the class residual and excess in place."""

    if not rows:
        class_item.update(
            {"cash_residual_brl": float(class_item["target_amount_brl"]), "over_target_brl": 0.0}
        )
        return [], []
    try:
        prices = price_loader(snapshot, [row["ticker"] for row in rows])
    except SelectorSnapshotChangedError as exc:
        raise PremiumOptimizationError(str(exc), "snapshot_changed") from exc
    except SelectorPriceError as exc:
        raise PremiumOptimizationError(str(exc), "engine_error") from exc
    bought, dropped, residual = _whole_unit_rows(
        rows,
        float(class_item["target_amount_brl"]),
        capital_brl,
        prices,
        minimum_count=minimum_count,
    )
    class_item.update(residual)
    return bought, dropped


def run_premium_optimization(
    policy: object,
    manifest: object,
    workspace: Path,
    investable_capital_brl: float,
    *,
    selector_inputs: StatusInvestInputs,
    snapshot_loader: Callable[..., object] = _default_snapshot_loader,
    allocation_engine: Callable[..., Mapping[str, Any]] = _default_allocation_engine,
    stock_engine: Callable[..., Mapping[str, Any]] = _default_stock_engine,
    fii_engine: Callable[..., Mapping[str, Any]] = _default_fii_engine,
    seed: int | None = None,
    incumbents: Mapping[str, Any] | None = None,
    whole_units: bool = False,
    price_loader: Callable[[SourceSnapshot, Sequence[str]], Mapping[str, int]] = load_selector_prices,
) -> dict[str, Any]:
    """Run the three explicit Premium engines and compose one API result.

    With ``whole_units`` the stock and FII rows become a whole-unit buy list
    priced from the run's own hashed selector CSVs (whole-units-v1).
    """

    if not math.isfinite(float(investable_capital_brl)) or investable_capital_brl <= 0:
        raise PremiumOptimizationError("investable capital must be positive", "policy_invalid")
    if not isinstance(selector_inputs, StatusInvestInputs):
        raise PremiumOptimizationError(
            "validated Status Invest selector inputs are required", "snapshot_unavailable"
        )
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    allocation_section = _policy_section(policy, "allocation")
    stock_section = _policy_section(policy, "stocks")
    fii_section = _policy_section(policy, "fiis")
    to_dict = getattr(policy, "to_dict", None)
    policy_dict = _json_value(to_dict() if callable(to_dict) else policy)
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
    class_incumbents = {
        "stocks": _class_incumbents(incumbents, "stocks", stock_config),
        "fiis": _class_incumbents(incumbents, "fiis", fii_config),
    }
    engine_incumbents = {
        key: {"incumbents": value} if value else {}
        for key, value in class_incumbents.items()
    }
    selector_results: dict[str, Mapping[str, Any]] = {}
    zone_records: dict[str, dict[str, Any] | None] = {}

    stock_weight = weights["brazilian_stocks"]
    if stock_weight > 1e-9:
        stock_workspace = workspace / "stocks"
        try:
            stock_result = stock_engine(
                input_path=selector_inputs.stocks.path,
                config=stock_config,
                workspace=stock_workspace,
                seed=run_seed,
                **engine_incumbents["stocks"],
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
        zone_records["stocks"] = _zone_record(
            "stocks", stock_config, incumbents, class_incumbents["stocks"], stock_result
        )
        stocks = _constituents(
            stock_result,
            stock_weight,
            float(class_by_engine["brazilian_stocks"]["target_amount_brl"]),
            _zone_reasons(zone_records["stocks"]),
        )
    else:
        zone_records["stocks"] = _zone_record(
            "stocks", stock_config, incumbents, class_incumbents["stocks"], None
        )
        stocks = []

    fii_weight = weights["fiis"]
    if fii_weight > 1e-9:
        fii_workspace = workspace / "fiis"
        try:
            fii_result = fii_engine(
                input_path=selector_inputs.fiis.path,
                selection_config=fii_config,
                n_runs=max(1, int(fii_config["system_ga_config"].get("run_count", 1))),
                parallel=False,
                workspace=fii_workspace,
                seed=run_seed,
                **engine_incumbents["fiis"],
            )
        except PremiumOptimizationError:
            raise
        except FiiUniverseInfeasibleError as exc:
            raise PremiumOptimizationError(
                str(exc),
                "infeasible_universe",
                {
                    "profile": exc.profile,
                    "eligible_count": exc.eligible_count,
                    "universe_count": exc.universe_count,
                    "required_count": exc.required_count,
                    "size_min": exc.size_min,
                    "liq_min": exc.liq_min,
                },
            ) from exc
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
        zone_records["fiis"] = _zone_record(
            "fiis", fii_config, incumbents, class_incumbents["fiis"], fii_result
        )
        fiis = _constituents(
            fii_result,
            fii_weight,
            float(class_by_engine["fiis"]["target_amount_brl"]),
            _zone_reasons(zone_records["fiis"]),
        )
    else:
        zone_records["fiis"] = _zone_record(
            "fiis", fii_config, incumbents, class_incumbents["fiis"], None
        )
        fiis = []

    whole_unit_result: dict[str, Any] = {}
    if whole_units:
        stocks, stocks_dropped = _buy_list(
            class_by_engine["brazilian_stocks"],
            stocks,
            selector_inputs.stocks,
            price_loader,
            float(investable_capital_brl),
            minimum_count=0,
        )
        fiis, fiis_dropped = _buy_list(
            class_by_engine["fiis"],
            fiis,
            selector_inputs.fiis,
            price_loader,
            float(investable_capital_brl),
            minimum_count=FII_MINIMUM_HOLDINGS,
        )
        whole_unit_result = {"stocks_dropped": stocks_dropped, "fiis_dropped": fiis_dropped}

    policy_provenance = _policy_section(policy, "provenance")
    manifest_id = _manifest_value(manifest, "manifest_id")
    allocation_history = {
        "manifest_id": manifest_id,
        "manifest_version": _manifest_value(manifest, "manifest_version"),
        "cutoff_date": policy_provenance.get("cutoff_date"),
        "source_snapshot_ids": policy_provenance.get("source_snapshot_ids", []),
        "source_snapshot_hashes": policy_provenance.get("source_snapshot_hashes", {}),
    }
    provenance = {
        key: value
        for key, value in policy_provenance.items()
        if key not in {"cutoff_date", "source_snapshot_ids", "source_snapshot_hashes"}
    }
    provenance.update(
        {
            "manifest_id": manifest_id,
            "random_seed": run_seed,
            "allocation_history": allocation_history,
            "allocation_method": allocation_result.get("provenance"),
            "selector_sources": selector_inputs.provenance(),
        }
    )
    if any(record is not None for record in zone_records.values()):
        provenance["protection_zone"] = {
            "incumbent_run_id": incumbents.get("run_id") if incumbents else None,
            **{key: record for key, record in zone_records.items() if record is not None},
        }
    assumptions = [
        "Resultado Premium bruto, baseado em snapshots imutáveis e regras versionadas.",
        "Ações brasileiras e FIIs são sleeves selecionados pelos parâmetros resolvidos do perfil.",
        "Valores em BRL representam alvos de capital investível, não ordens de execução.",
    ]
    if whole_units:
        assumptions.append(WHOLE_UNITS_ASSUMPTION)
    return {
        "policy": policy_dict,
        "classes": classes,
        "stocks": stocks,
        "fiis": fiis,
        **whole_unit_result,
        "assumptions": assumptions,
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
