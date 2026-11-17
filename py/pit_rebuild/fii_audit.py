"""Classification of the worst FII cases from the evidence the sources hold (issue #60, criterion part 2).

Each case is the Status Invest value of one ticker, indicator and date, outside the tolerance. The
rules look for a source value equal to the reference at its two decimals (a value merely within the
tolerance is not evidence), in this order:

- another fund that reports the same ISIN gives it: "erro nosso" (the ISIN went to the wrong fund);
- an older report of the fund, delivered on or before d, gives it (newest first): "erro do Status
  Invest" (it shows a report older than the one available at d);
- a report delivered after d gives it (earliest delivered first): "diferença de método" (point in
  time; the CVM file keeps only the newest version, so an earlier version may have been out by d);
- the close or the 21-session mean of another session up to 10 days around d (nearest first) gives it, for PRECO
  and LIQUIDEZ MEDIA DIARIA: "diferença de método" (reference session);
- the other distribution source gives it, for DY, ULTIMO DIVIDENDO and CAGR DIVIDENDOS 3 ANOS:
  "diferença de método" (source);
- nothing gives it: "sem classificação". Such a case needs a reading by hand.

A case without a rebuilt value is classified by its reason (classify_missing). The evidence is the
report the value came from (or the session, or the source).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta

from . import cotahist, fii_dividends, fii_informes
from .dividends import Dividend
from .compare import ROUNDING_DECIMALS
from .fii_indicators import Rebuilt, VP_CAGR_MONTHS, nav_cagr
from .fii_informes import Informe, Informes

OURS = "erro nosso"
SI_ERROR = "erro do Status Invest"
METHOD = "diferença de método"
OPEN = "sem classificação"
CLASSES = (OURS, SI_ERROR, METHOD, OPEN)
SESSION_SPAN_DAYS = 10
# reason of a missing rebuilt value -> (class, note)
MISSING = {
    "no_isin": (OURS, "o ticker não tem ISIN no COTAHIST até d"),
    "no_cvm_match": (OURS, "nenhum informe entregue até d traz o ISIN do ticker"),
    "ambiguous_isin": (OURS, "mais de um fundo informa o ISIN e a B3 não desempata"),
    "no_informe_by_d": (OURS, "o fundo não tem informe entregue até d"),
    "stale_informe": (METHOD, "o informe mais recente em d é de um mês encerrado há mais de 270 dias"),
    "informe_blank": (OURS, "o informe em d não traz a coluna"),
    "no_quote": (METHOD, "sem pregão nos 30 dias até d"),
    "no_informe_36_months_before": (OURS, "não há informe 36 meses antes conhecido em d"),
    "short_or_zero_history": (OURS, "os proventos não alcançam a janela de 3 anos atrás ou somam zero nela"),
    "no_distribution_by_d": (OURS, "a fonte não tem provento até d"),
}
INFORME_NUMERIC = ("N COTISTAS", "PATRIMONIO", "N COTAS", "PERCENTUAL EM CAIXA", "P/VP", "CAGR VALOR COTA 3 ANOS")


@dataclass
class Context:
    rebuilt: Rebuilt
    informes: Informes
    quotes: cotahist.Quotes
    b3_fund: fii_dividends.B3Fund | None
    yahoo: Sequence[Dividend] | None


def classify(field: str, reference: float, ctx: Context) -> dict:
    used = ctx.rebuilt.informe.label() if ctx.rebuilt.informe else None
    if field in INFORME_NUMERIC:
        return _informe_case(field, reference, ctx, used)
    if field in ("PRECO", "LIQUIDEZ MEDIA DIARIA"):
        return _session_case(field, reference, ctx)
    if field in ("DY", "ULTIMO DIVIDENDO", "CAGR DIVIDENDOS 3 ANOS"):
        return _source_case(field, reference, ctx)
    raise ValueError(f"not a numeric FII indicator: {field}")


def classify_missing(reason: str) -> dict:
    """The class of a case with no rebuilt value; a failed or empty distribution source is ours."""
    if reason in MISSING:
        kind, note = MISSING[reason]
        return _case(kind, reason, note)
    if reason.startswith("b3_"):
        return _case(OURS, reason, "nenhuma fonte de proventos respondeu para o ticker")
    return _case(OPEN, reason, "motivo sem regra de classificação")


def same(reference: float | None, value: float | None) -> bool:
    """Whether a source value equals the reference at the two decimals of the CSV."""
    return reference is not None and value is not None and round(value, ROUNDING_DECIMALS) == round(reference, ROUNDING_DECIMALS)


def _informe_case(field: str, reference: float, ctx: Context, used: str | None) -> dict:
    rebuilt = ctx.rebuilt
    d = rebuilt.d
    price = rebuilt.values.get("PRECO")
    if rebuilt.isin:
        for cnpj in fii_informes.funds_of_isin(ctx.informes, rebuilt.isin, d):
            if cnpj == rebuilt.cnpj:
                continue
            other = fii_informes.latest(ctx.informes, cnpj, d)
            if other is not None and same(reference, _value(field, other, price, ctx.informes, d)):
                return _case(OURS, f"{cnpj}, {other.label()}", "o valor é o de outro fundo que informa o mesmo ISIN")
    reports = ctx.informes.by_fund.get(rebuilt.cnpj or "", [])
    floor = rebuilt.informe.ref if rebuilt.informe else date.max
    older = sorted((item for item in reports if item.delivered <= d and item.ref < floor), key=lambda item: item.ref, reverse=True)
    later = sorted((item for item in reports if item.delivered > d), key=lambda item: (item.delivered, item.ref))
    for report in older:
        if same(reference, _value(field, report, price, ctx.informes, d)):
            return _case(SI_ERROR, report.label(), "o valor é o de um informe mais antigo que o disponível em d")
    for report in later:
        if same(reference, _value(field, report, price, ctx.informes, report.delivered)):
            return _case(METHOD, report.label(), "o valor é o de um informe entregue depois de d")
    return _case(OPEN, used or "sem informe em d", "nenhum informe do fundo dá o valor")


def _value(field: str, report: Informe, price: float | None, informes: Informes, horizon: date) -> float | None:
    if field == "N COTISTAS":
        return report.holders
    if field == "PATRIMONIO":
        return report.equity
    if field == "N COTAS":
        return report.shares
    if field == "PERCENTUAL EM CAIXA":
        return report.liquidity / report.equity * 100 if report.liquidity is not None and report.equity else None
    if field == "P/VP":
        return price / report.nav_per_share if price and report.nav_per_share else None
    return nav_cagr(informes, report, fii_informes.months_back(informes, report, VP_CAGR_MONTHS, horizon), horizon)


def _session_case(field: str, reference: float, ctx: Context) -> dict:
    ticker, d = ctx.rebuilt.ticker, ctx.rebuilt.d
    nearest_first = sorted(ctx.quotes.rows.get(ticker, []), key=lambda row: (abs((row.day - d).days), row.day))
    for row in nearest_first:
        if abs((row.day - d).days) > SESSION_SPAN_DAYS or row.day == ctx.rebuilt.quote_day:
            continue
        value = row.close if field == "PRECO" else cotahist.avg_volume(ctx.quotes, ticker, row.day)
        if same(reference, value):
            return _case(METHOD, f"pregão de {row.day.isoformat()}", "o valor é o de outra sessão")
    return _case(OPEN, f"pregão de {ctx.rebuilt.quote_day.isoformat()}" if ctx.rebuilt.quote_day else "sem pregão", "nenhuma sessão próxima dá o valor")


def _source_case(field: str, reference: float, ctx: Context) -> dict:
    rebuilt, d = ctx.rebuilt, ctx.rebuilt.d
    used = rebuilt.sources.get(field, ("none", None))[0]
    b3 = None
    if ctx.b3_fund is not None and rebuilt.isin:
        b3 = [item for item in fii_dividends.b3_dividends(ctx.b3_fund, rebuilt.isin, lambda day: cotahist.next_session(ctx.quotes, day)) if item.ex_date <= d]
    yahoo = [item for item in ctx.yahoo or [] if item.ex_date is not None and item.ex_date <= d] if ctx.yahoo is not None else None
    price = rebuilt.values.get("PRECO")
    for name, dividends in (("b3", b3), ("yahoo", yahoo)):
        if name == used or not dividends:
            continue
        if field == "DY":
            value = fii_dividends.sum_12m(dividends, d) / price * 100 if price else None
        elif field == "ULTIMO DIVIDENDO":
            value = fii_dividends.last_dividend(dividends, d)
        else:
            value = fii_dividends.cagr_3y(dividends, d)
        if same(reference, value):
            return _case(METHOD, f"proventos da fonte {name}", "o valor é o da outra fonte de proventos")
    return _case(OPEN, f"proventos da fonte {used}", "a outra fonte não dá o valor")


def _case(kind: str, evidence: str, note: str) -> dict:
    return {"class": kind, "evidence": evidence, "note": note}
