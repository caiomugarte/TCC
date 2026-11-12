"""The 19 stock fields of the selector for one ticker at d, from declared formulas (IND-01, IND-03, IND-04, MAP-03, PIT-02, A-8).

The inputs are assembled by the caller from the CVM, COTAHIST and dividend modules, and each
carries the date that dates it (DT_RECEB of the filing, session date). compute asserts that no
input is dated after d (PIT-02). A field with a zero divisor or a missing input has no value and
a reason (IND-03). Reasons: no_share_count, no_quote, no_dividend_source, short_history,
zero_divisor, non_positive_base, missing_input: <name>, stale_filing (STL-01).

Per-share fields divide the ticker price by its unit divisor k (1 for a share, the composition
total for a unit, MAP-03). VALOR DE MERCADO takes its class prices from the company (A-17). Share
counts arrive already multiplied by the A-18 scale and by the B3 events of the class with ex-date in (DT_REFER, d] (AD-033).
"""

from dataclasses import dataclass, fields
from datetime import date

from .dividends import Dividend, dps_12m
from .errors import PITLeakError
from .fields import ACCOUNTING_FIELDS, ALL_FIELDS

FORMULAS: dict[str, str] = {
    "PRECO": "p: último pregão em COTAHIST com data <= d",
    "VALOR DE MERCADO": "N_ON × p_ON + N_PREF × p_PREF (A-17): p_ON do ticker ON; p_PREF do preferencial mais líquido; classe com N > 0 e sem cotação usa a outra (vm_price_fallback); N = arquivado × f (A-18) × eventos B3 da classe com data-ex em (DT_REFER, d] (AD-033)",
    "LIQUIDEZ MEDIA DIARIA": "média de VOLTOT das últimas 21 sessões <= d, CODBDI 02, 07 e 08 (A-7)",
    "P/L": "p_eq / (LL / N_total), p_eq = p / k, N_total = N_ON + N_PREF",
    "P/VP": "p_eq / (PL_ctrl / N_total), PL_ctrl = 2.03 − 2.03.09 (A-21)",
    "PSR": "p_eq / (REC / N_total)",
    "EV/EBIT": "(VALOR DE MERCADO + DL) / EBIT",
    "DY": "100 × DPS_12m / p, DPS_12m = soma dos dividendos com d − 365 dias < data-ex <= d (por unidade para units)",
    "ROE": "100 × LL / abs(PL_ctrl), PL_ctrl = 2.03 − 2.03.09 (A-21)",
    "ROA": "100 × LL / AT",
    "ROIC": "100 × (EBIT − abs(IR)) / (PL + DB)",
    "MARG. LIQUIDA": "100 × LL / REC",
    "MARGEM EBIT": "100 × EBIT / REC",
    "LIQ. CORRENTE": "AC / PC",
    "DIVIDA LIQUIDA / EBIT": "DL / EBIT",
    "DIV. LIQ. / PATRI.": "DL / PL_ctrl (A-21)",
    "CAGR RECEITAS 5 ANOS": "100 × ((REC_FY / REC_FY−5)^(1/5) − 1), exercícios fiscais da DFP; vazio se algum <= 0",
    "CAGR LUCROS 5 ANOS": "100 × ((LL_TTM / LL_FY−5)^(1/5) − 1), LL_FY−5 = exercício fiscal cinco anos antes da DFP usável mais recente; vazio se algum <= 0 (A-21)",
    "PEG RATIO": "P/L / (100 × (LL_TTM / LL_TTM,1 ano antes − 1)); vazio se LL_TTM,1 ano antes <= 0",
}


@dataclass(frozen=True)
class Input:
    """A value and the date that dates it: DT_RECEB of the filing, the session date or the ex-date."""

    value: float
    when: date


@dataclass(frozen=True)
class TickerInputs:
    """Everything one ticker's 19 fields need at d. None means the input is absent."""

    ticker: str
    unit_divisor: int = 1  # k: 1 for a share, the composition total for a unit (MAP-03)
    price: Input | None = None  # PRECO
    price_on: Input | None = None  # class price of the company, ON ticker (A-17)
    price_pref: Input | None = None  # class price of the company, most liquid preferred ticker (A-17)
    n_on: int | None = None  # issued ON shares, A-18 scale applied; None: no share count
    n_pref: int | None = None  # issued PREF shares, A-18 scale applied
    liquidity: Input | None = None  # LIQUIDEZ MEDIA DIARIA
    ll: Input | None = None  # TTM net income per DRE row: 3.11.01 when non-zero, else 3.11 (A-21, F7)
    rec: Input | None = None  # TTM 3.01
    ebit: Input | None = None  # TTM 3.05
    ir: Input | None = None  # TTM 3.08, sign as filed
    ll_1y: Input | None = None  # TTM net income one year before (PEG)
    pl: Input | None = None  # BPP 2.03
    pl_ctrl: Input | None = None  # BPP 2.03 less 2.03.09: the controllers' equity (A-21, F8)
    at: Input | None = None  # BPA 1
    ac: Input | None = None  # BPA 1.01
    pc: Input | None = None  # BPP 2.01
    debt: Input | None = None  # BPP 2.01.04 + 2.02.01
    cash: Input | None = None  # BPA 1.01.01 + 1.01.02
    rec_fy: Input | None = None  # fiscal-year 3.01 (CAGR)
    rec_fy5: Input | None = None  # fiscal-year 3.01, five years before (CAGR)
    ll_fy5: Input | None = None  # fiscal-year net income, five years before (CAGR LUCROS, A-21)
    dividends: list[Dividend] | None = None  # per unit for a unit; None: no dividend source


@dataclass(frozen=True)
class Value:
    value: float | None
    reason: str | None = None
    note: str | None = None


def compute(inputs: TickerInputs, d: date) -> dict[str, Value]:
    """The 19 fields for one ticker at d. Raises PITLeakError when an input is dated after d (PIT-02)."""
    _assert_point_in_time(inputs, d)
    market_value = _market_value(inputs)
    price = _price(inputs)
    net_debt = _sub(_get(inputs.debt, "DB"), _get(inputs.cash, "CX"))
    ll = _get(inputs.ll, "LL")
    rec = _get(inputs.rec, "REC")
    ebit = _get(inputs.ebit, "EBIT")
    ir = _get(inputs.ir, "IR")
    pl = _get(inputs.pl, "PL")
    pl_ctrl = _get(inputs.pl_ctrl, "PL_ctrl")
    at = _get(inputs.at, "AT")
    ac = _get(inputs.ac, "AC")
    pc = _get(inputs.pc, "PC")
    p_l = _price_per(inputs, ll)
    results = {
        "PRECO": price,
        "VALOR DE MERCADO": market_value,
        "LIQUIDEZ MEDIA DIARIA": _get_quote(inputs.liquidity),
        "P/L": p_l,
        "P/VP": _price_per(inputs, pl_ctrl),
        "PSR": _price_per(inputs, rec),
        "EV/EBIT": _ev_ebit(market_value, net_debt, ebit),
        "DY": _dividend_yield(inputs, d, price),
        "ROE": _ratio(ll, _abs(pl_ctrl), 100),
        "ROA": _ratio(ll, at, 100),
        "ROIC": _ratio(_sub(ebit, _abs(ir)), _sum(pl, _get(inputs.debt, "DB")), 100),
        "MARG. LIQUIDA": _ratio(ll, rec, 100),
        "MARGEM EBIT": _ratio(ebit, rec, 100),
        "LIQ. CORRENTE": _ratio(ac, pc),
        "DIVIDA LIQUIDA / EBIT": _ratio(net_debt, ebit),
        "DIV. LIQ. / PATRI.": _ratio(net_debt, pl_ctrl),
        "CAGR RECEITAS 5 ANOS": _cagr(inputs.rec_fy, inputs.rec_fy5, "REC_FY"),
        "CAGR LUCROS 5 ANOS": _cagr(inputs.ll, inputs.ll_fy5, "LL_TTM"),
        "PEG RATIO": _peg(inputs, p_l, ll),
    }
    return {field: results[field] for field in ALL_FIELDS}


STALE_FILING_REASON = "stale_filing"  # STL-01: the reason of an accounting field whose newest filing ends more than 270 days before d


def mask_accounting(values: dict[str, Value], reason: str) -> dict[str, Value]:
    """A copy of values in which every accounting field (B-7) has no value and the given reason (STL-01).

    The market fields and DY keep their value, reason and note (STL-02). The input dict is not changed.
    """
    return {field: Value(None, reason) if field in ACCOUNTING_FIELDS else value for field, value in values.items()}


def _assert_point_in_time(inputs: TickerInputs, d: date) -> None:
    """PIT-02: an input or a dividend dated after d is a leak, not something to drop silently."""
    for item in fields(inputs):
        value = getattr(inputs, item.name)
        if isinstance(value, Input) and value.when > d:
            raise PITLeakError(inputs.ticker, d, value.when)
    for dividend in inputs.dividends or ():
        if dividend.ex_date is not None and dividend.ex_date > d:
            raise PITLeakError(inputs.ticker, d, dividend.ex_date)


def _get(item: Input | None, name: str) -> Value:
    if item is None:
        return Value(None, f"missing_input: {name}")
    return Value(item.value)


def _get_quote(item: Input | None) -> Value:
    return Value(None, "no_quote") if item is None else Value(item.value)


def _price(inputs: TickerInputs) -> Value:
    return _get_quote(inputs.price)


def _sub(left: Value, right: Value) -> Value:
    if left.reason or right.reason:
        return Value(None, left.reason or right.reason)
    return Value(left.value - right.value)


def _sum(*parts: Value) -> Value:
    for part in parts:
        if part.reason:
            return Value(None, part.reason)
    return Value(sum(part.value for part in parts))


def _abs(part: Value) -> Value:
    return part if part.reason else Value(abs(part.value))


def _ratio(num: Value, den: Value, scale: float = 1.0) -> Value:
    if num.reason:
        return Value(None, num.reason)
    if den.reason:
        return Value(None, den.reason)
    if den.value == 0:
        return Value(None, "zero_divisor")
    return Value(scale * num.value / den.value)


def _share_total(inputs: TickerInputs) -> int | None:
    if inputs.n_on is None or inputs.n_pref is None:
        return None
    return inputs.n_on + inputs.n_pref


def _market_value(inputs: TickerInputs) -> Value:
    """N_ON × p_ON + N_PREF × p_PREF (A-17); a class with shares and no quote takes the other class's price."""
    total = _share_total(inputs)
    if total is None or total <= 0:
        return Value(None, "no_share_count")
    quotes = {"ON": inputs.price_on, "PREF": inputs.price_pref}
    shares = {"ON": inputs.n_on, "PREF": inputs.n_pref}
    value = 0.0
    fallback = False
    for cls in ("ON", "PREF"):
        if shares[cls] == 0:
            continue
        other = "PREF" if cls == "ON" else "ON"
        own_quote = quotes[cls]
        quote = own_quote or quotes[other]
        if quote is None:
            return Value(None, "no_quote")
        fallback = fallback or own_quote is None
        value += shares[cls] * quote.value
    return Value(value, None, "vm_price_fallback" if fallback else None)


def _price_per(inputs: TickerInputs, flow: Value) -> Value:
    """p_eq / (flow / N_total) with p_eq = p / k: P/L, P/VP and PSR."""
    total = _share_total(inputs)
    if total is None or total <= 0:
        return Value(None, "no_share_count")
    if flow.reason:
        return Value(None, flow.reason)
    if inputs.price is None:
        return Value(None, "no_quote")
    per_share = flow.value / total
    if per_share == 0:
        return Value(None, "zero_divisor")
    return Value((inputs.price.value / inputs.unit_divisor) / per_share)


def _ev_ebit(market_value: Value, net_debt: Value, ebit: Value) -> Value:
    if market_value.reason:
        return Value(None, market_value.reason)
    if net_debt.reason:
        return Value(None, net_debt.reason)
    if ebit.reason:
        return Value(None, ebit.reason)
    if ebit.value == 0:
        return Value(None, "zero_divisor")
    return Value((market_value.value + net_debt.value) / ebit.value)


def _dividend_yield(inputs: TickerInputs, d: date, price: Value) -> Value:
    if inputs.dividends is None:
        return Value(None, "no_dividend_source")
    if price.reason:
        return Value(None, price.reason)
    if price.value == 0:
        return Value(None, "zero_divisor")
    return Value(100 * dps_12m(inputs.dividends, d, inputs.ticker) / price.value)


def _cagr(end: Input | None, start: Input | None, name: str) -> Value:
    if end is None:
        return Value(None, f"missing_input: {name}")
    if start is None:
        return Value(None, "short_history")
    if end.value <= 0 or start.value <= 0:
        return Value(None, "non_positive_base")
    return Value(100 * ((end.value / start.value) ** (1 / 5) - 1))


def _peg(inputs: TickerInputs, p_l: Value, ll: Value) -> Value:
    if p_l.reason:
        return Value(None, p_l.reason)
    if ll.reason:
        return Value(None, ll.reason)
    if inputs.ll_1y is None:
        return Value(None, "short_history")
    if inputs.ll_1y.value <= 0:
        return Value(None, "non_positive_base")
    growth = ll.value / inputs.ll_1y.value - 1
    if growth == 0:
        return Value(None, "zero_divisor")
    return Value(p_l.value / (100 * growth))
