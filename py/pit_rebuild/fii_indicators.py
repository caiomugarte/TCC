"""The 13 indicators of the FII selector rebuilt at d (issue #60).

Formulas, written before the measurement:

- PRECO: COTAHIST close of the last session on or before d, at most 30 days old (cotahist.quote_at).
- LIQUIDEZ MEDIA DIARIA: mean VOLTOT of the last 21 sessions on or before d (cotahist.avg_volume).
- N COTISTAS, PATRIMONIO, N COTAS: Total_Numero_Cotistas, Patrimonio_Liquido, Cotas_Emitidas of the report at d.
- PERCENTUAL EM CAIXA: Total_Necessidades_Liquidez / Patrimonio_Liquido x 100 of the report at d.
- P/VP: PRECO / Valor_Patrimonial_Cotas of the report at d.
- CAGR VALOR COTA 3 ANOS: (Valor_Patrimonial_Cotas / that of the report 36 months older, known at d) ** (1/3) - 1, in %.
- DY: distributions with d - 365 days < ex-date <= d over PRECO, in %.
- ULTIMO DIVIDENDO: the newest distribution with ex-date on or before d.
- CAGR DIVIDENDOS 3 ANOS: fii_dividends.cagr_3y.
- SETOR, GESTAO: Segmento_Atuacao and Tipo_Gestao of the report at d (mapped in fii_compare).

The report at d is the newest one delivered on or before d (fii_informes.latest). A report whose
month ended more than 270 days before d gives no informe value (stale_informe), as the stock rule.
The ticker reaches its fund by the ISIN of its last session on or before d; an ISIN that more than
one fund reports goes to the fund whose quota count is nearest the B3 count (ambiguous_isin without B3).
"""

import calendar
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
import math

from . import cotahist, fii_dividends, fii_informes
from .dividends import Dividend
from .fii_informes import Informe, Informes

STALE_DAYS = 270
VP_CAGR_MONTHS = 36
INFORME_FIELDS = (
    "N COTISTAS",
    "PATRIMONIO",
    "N COTAS",
    "PERCENTUAL EM CAIXA",
    "P/VP",
    "CAGR VALOR COTA 3 ANOS",
    "SETOR",
    "GESTAO",
)
DIVIDEND_FIELDS = ("DY", "ULTIMO DIVIDENDO", "CAGR DIVIDENDOS 3 ANOS")


@dataclass
class Rebuilt:
    ticker: str
    d: date
    isin: str | None = None
    cnpj: str | None = None
    informe: Informe | None = None
    informe_3y: Informe | None = None
    values: dict[str, object] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    sources: dict[str, tuple[str, str | None]] = field(default_factory=dict)
    b3_only_dy: float | None = None
    delivered_after: list[date] = field(default_factory=list)
    quote_day: date | None = None


def rebuild(
    ticker: str,
    d: date,
    quotes: cotahist.Quotes,
    informes: Informes,
    b3_fund: fii_dividends.B3Fund | None,
    yahoo: Sequence[Dividend] | None,
) -> Rebuilt:
    result = Rebuilt(ticker=ticker, d=d)
    quote = cotahist.quote_at(quotes, ticker, d)
    price = quote[1] if quote else None
    result.quote_day = quote[0] if quote else None
    _put(result, "PRECO", price, "no_quote")
    _put(result, "LIQUIDEZ MEDIA DIARIA", cotahist.avg_volume(quotes, ticker, d), "no_quote")

    result.isin = cotahist.isin(quotes, ticker, d)
    result.cnpj, reason = resolve_fund(informes, result.isin, d, b3_fund)
    report = fii_informes.latest(informes, result.cnpj, d) if result.cnpj else None
    if result.cnpj and report is None:
        reason = "no_informe_by_d"
    elif report is not None and (d - _month_end(report.ref)).days > STALE_DAYS:
        reason, report = "stale_informe", None
    if result.cnpj:
        result.delivered_after = fii_informes.delivered_after(informes, result.cnpj, d)
    result.informe = report
    _informe_values(result, informes, report, price, d, reason)
    _dividend_values(result, quotes, b3_fund, yahoo, price, d)
    return result


def resolve_fund(
    informes: Informes, isin: str | None, d: date, b3_fund: fii_dividends.B3Fund | None
) -> tuple[str | None, str | None]:
    """(CNPJ, None) of the fund behind the ISIN at d, or (None, reason)."""
    if not isin:
        return None, "no_isin"
    candidates = fii_informes.funds_of_isin(informes, isin, d)
    if not candidates:
        return None, "no_cvm_match"
    if len(candidates) == 1:
        return candidates[0], None
    if b3_fund is None or not b3_fund.quantity:
        return None, "ambiguous_isin"
    distances = []
    for cnpj in candidates:
        report = fii_informes.latest(informes, cnpj, d)
        if report is not None and report.shares:
            distances.append((abs(math.log(report.shares / b3_fund.quantity)), cnpj))
    if not distances:
        return None, "ambiguous_isin"
    return min(distances)[1], None


def _informe_values(result: Rebuilt, informes: Informes, report: Informe | None, price: float | None, d: date, reason: str | None) -> None:
    if report is None:
        for name in INFORME_FIELDS:
            _put(result, name, None, reason or "no_informe_by_d")
        return
    _put(result, "N COTISTAS", report.holders, "informe_blank")
    _put(result, "PATRIMONIO", report.equity, "informe_blank")
    _put(result, "N COTAS", report.shares, "informe_blank")
    cash = report.liquidity / report.equity * 100 if report.liquidity is not None and report.equity else None
    _put(result, "PERCENTUAL EM CAIXA", cash, "informe_blank")
    if price is None:
        _put(result, "P/VP", None, "no_quote")
    else:
        _put(result, "P/VP", price / report.nav_per_share if report.nav_per_share else None, "informe_blank")
    old = fii_informes.months_back(informes, report, VP_CAGR_MONTHS, d)
    result.informe_3y = old
    if old is None or not old.nav_per_share or not report.nav_per_share or report.nav_per_share / old.nav_per_share <= 0:
        _put(result, "CAGR VALOR COTA 3 ANOS", None, "no_informe_36_months_before")
    else:
        _put(result, "CAGR VALOR COTA 3 ANOS", ((report.nav_per_share / old.nav_per_share) ** (1 / 3) - 1) * 100, None)
    _put(result, "SETOR", report.segment or None, "informe_blank")
    _put(result, "GESTAO", report.management or None, "informe_blank")


def _dividend_values(
    result: Rebuilt, quotes: cotahist.Quotes, b3_fund: fii_dividends.B3Fund | None,
    yahoo: Sequence[Dividend] | None, price: float | None, d: date,
) -> None:
    b3 = None
    if b3_fund is not None and result.isin:
        b3 = fii_dividends.b3_dividends(b3_fund, result.isin, lambda day: cotahist.next_session(quotes, day))
    known_b3 = [item for item in b3 or [] if item.ex_date <= d]

    source, why, chosen = fii_dividends.choose(b3, yahoo, fii_dividends.window_start(d))
    result.sources["DY"] = (source, why)
    if price is None:
        _put(result, "DY", None, "no_quote")
    elif chosen is None:
        _put(result, "DY", None, why)
    else:
        _put(result, "DY", fii_dividends.sum_12m(chosen, d) / price * 100, None)
    if known_b3 and price:
        result.b3_only_dy = fii_dividends.sum_12m(known_b3, d) / price * 100

    if known_b3:
        result.sources["ULTIMO DIVIDENDO"] = ("b3", None)
        _put(result, "ULTIMO DIVIDENDO", fii_dividends.last_dividend(known_b3, d), None)
    else:
        reason = "b3_failed" if b3 is None else "b3_empty"
        if yahoo is None:
            result.sources["ULTIMO DIVIDENDO"] = ("none", f"{reason}_yahoo_failed")
            _put(result, "ULTIMO DIVIDENDO", None, f"{reason}_yahoo_failed")
        else:
            result.sources["ULTIMO DIVIDENDO"] = ("yahoo", reason)
            _put(result, "ULTIMO DIVIDENDO", fii_dividends.last_dividend(yahoo, d), "no_distribution_by_d")

    source, why, chosen = fii_dividends.choose(b3, yahoo, fii_dividends.window_start(d, fii_dividends.CAGR_YEARS))
    result.sources["CAGR DIVIDENDOS 3 ANOS"] = (source, why)
    if chosen is None:
        _put(result, "CAGR DIVIDENDOS 3 ANOS", None, why)
    else:
        _put(result, "CAGR DIVIDENDOS 3 ANOS", fii_dividends.cagr_3y(chosen, d), "short_or_zero_history")


def _put(result: Rebuilt, name: str, value: object, reason: str | None) -> None:
    result.values[name] = value
    if value is None:
        result.reasons[name] = reason or "no_value"


def _month_end(ref: date) -> date:
    return ref.replace(day=calendar.monthrange(ref.year, ref.month)[1])
