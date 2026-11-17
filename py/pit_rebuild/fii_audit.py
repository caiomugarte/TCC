"""Classification of the worst FII cases from the evidence the sources hold (issue #60, criterion part 2).

Each case is the Status Invest value of one ticker, indicator and date. The rules only look for a
source value equal to the reference within the tolerance of the indicator (fii_compare.within):

- another fund that reports the same ISIN gives it: "erro nosso" (the ISIN went to the wrong fund);
- an older report of the fund, delivered on or before d, gives it: "erro do Status Invest" (it
  shows a report older than the one available at d);
- a report delivered after d gives it: "diferença de método" (point in time);
- the close or the 21-session mean of another session up to 10 days around d (nearest first) gives it, for PRECO
  and LIQUIDEZ MEDIA DIARIA: "diferença de método" (reference session);
- the other distribution source gives it, for DY, ULTIMO DIVIDENDO and CAGR DIVIDENDOS 3 ANOS:
  "diferença de método" (source);
- nothing gives it: "sem classificação". Such a case needs a reading by hand.

The evidence is the report the value came from (or the session, or the source).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta

from . import cotahist, fii_dividends, fii_informes
from .dividends import Dividend
from .fii_compare import within
from .fii_indicators import Rebuilt, VP_CAGR_MONTHS
from .fii_informes import Informe, Informes

OURS = "erro nosso"
SI_ERROR = "erro do Status Invest"
METHOD = "diferença de método"
OPEN = "sem classificação"
CLASSES = (OURS, SI_ERROR, METHOD, OPEN)
SESSION_SPAN_DAYS = 10
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


def _informe_case(field: str, reference: float, ctx: Context, used: str | None) -> dict:
    rebuilt = ctx.rebuilt
    d = rebuilt.d
    price = rebuilt.values.get("PRECO")
    if rebuilt.isin:
        for cnpj in fii_informes.funds_of_isin(ctx.informes, rebuilt.isin, d):
            if cnpj == rebuilt.cnpj:
                continue
            other = fii_informes.latest(ctx.informes, cnpj, d)
            if other is not None and within(field, reference, _value(field, other, price, ctx.informes, d)):
                return _case(OURS, f"{cnpj}, {other.label()}", "o valor é o de outro fundo que informa o mesmo ISIN")
    reports = sorted(ctx.informes.by_fund.get(rebuilt.cnpj or "", []), key=lambda item: item.ref, reverse=True)
    for report in reports:
        if rebuilt.informe is not None and report == rebuilt.informe:
            continue
        horizon = report.delivered if report.delivered > d else d
        if not within(field, reference, _value(field, report, price, ctx.informes, horizon)):
            continue
        if report.delivered > d:
            return _case(METHOD, report.label(), "o valor é o de um informe entregue depois de d")
        if rebuilt.informe is None or report.ref < rebuilt.informe.ref:
            return _case(SI_ERROR, report.label(), "o valor é o de um informe mais antigo que o disponível em d")
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
    old = fii_informes.months_back(informes, report, VP_CAGR_MONTHS, horizon)
    if old is None or not old.nav_per_share or not report.nav_per_share:
        return None
    return ((report.nav_per_share / old.nav_per_share) ** (1 / 3) - 1) * 100


def _session_case(field: str, reference: float, ctx: Context) -> dict:
    ticker, d = ctx.rebuilt.ticker, ctx.rebuilt.d
    nearest_first = sorted(ctx.quotes.rows.get(ticker, []), key=lambda row: (abs((row.day - d).days), row.day))
    for row in nearest_first:
        if abs((row.day - d).days) > SESSION_SPAN_DAYS or row.day == ctx.rebuilt.quote_day:
            continue
        value = row.close if field == "PRECO" else cotahist.avg_volume(ctx.quotes, ticker, row.day)
        if within(field, reference, value):
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
        if within(field, reference, value):
            return _case(METHOD, f"proventos da fonte {name}", "o valor é o da outra fonte de proventos")
    return _case(OPEN, f"proventos da fonte {used}", "a outra fonte não dá o valor")


def _case(kind: str, evidence: str, note: str) -> dict:
    return {"class": kind, "evidence": evidence, "note": note}
