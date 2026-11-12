#!/usr/bin/env python3
"""Classifies the worst rows of the adjusted verdict and generates the #64 audit document.

Run from the repository root. It reads `.cache/pit-rebuild` (offline, never written), the committed
`docs/validation/pit-rebuild-stocks.json`, and the two Status Invest CSVs that JSON names (its
`inputs.status_invest`, checked by sha256). It writes one file, the `--out` document:

    python3 scripts/pit_rebuild_audit.py --out docs/validation/pit-rebuild-stocks-audit.md

If a CSV is not at the path the JSON records, pass it with `--si-2026-11-05 PATH` (or `--si-2026-08-07`).

The search (`search_date`, `company_values`, `dy_search`) is the one the document states in its
"Reproduzir", "Proventos (DY)" and "Ordem das regras" sections: every version of every DFP and ITR of
the cache, with no age limit and no mask, and every B3 dividend line of the cache with no date window.
The tests are in tests/test_pit_rebuild_audit.py (fixtures, no network, no cache) and
tests/test_pit_rebuild_artifact.py (the committed document against the JSON).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import zipfile
from collections import Counter, OrderedDict, defaultdict
from contextlib import contextmanager
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "py"))

from pit_rebuild import cache as cache_mod  # noqa: E402
from pit_rebuild import cli, compare, cotahist, cvm, indicators, mapping  # noqa: E402
from pit_rebuild import shares as share_counts  # noqa: E402
from pit_rebuild.indicators import Input, TickerInputs, compute  # noqa: E402

CACHE = REPO / ".cache" / "pit-rebuild"
JSON_PATH = REPO / "docs" / "validation" / "pit-rebuild-stocks.json"
OUT_DEFAULT = REPO / "docs" / "validation" / "pit-rebuild-stocks-audit.md"
DATES = ("2026-08-07", "2026-11-05")
DAYS = [date(2026, 8, 7), date(2026, 11, 5)]
R_DATE = date(2026, 11, 9)  # B3 retrieval: the last day the cache reflects (every cached filing was received by then)
KINDS = ("DFP", "ITR")

ERRO_NOSSO = "erro nosso"
ERRO_SI = "erro do Status Invest"
METODO = "diferença de método"
WITHIN = "dentro da tolerância"
NC = "não classificado"
ORDER = [ERRO_NOSSO, ERRO_SI, METODO, WITHIN, NC]
SEM_BASE_PREFIXES = ("nenhum dos ", "nenhum subconjunto dos proventos")  # motive prefixes of `sem base`
SHARE_FIELDS = ("P/L", "P/VP", "PSR", "EV/EBIT")
FIELD_ACCOUNTS = {
    "P/L": ["3.11.01", "3.11"],
    "CAGR LUCROS 5 ANOS": ["3.11.01", "3.11"],
    "ROE": ["3.11.01", "3.11", "2.03", "2.03.09"],
    "ROA": ["3.11.01", "3.11", "1"],
    "MARG. LIQUIDA": ["3.11.01", "3.11", "3.01"],
    "MARGEM EBIT": ["3.05", "3.01"],
    "P/VP": ["2.03", "2.03.09"],
    "PSR": ["3.01"],
    "EV/EBIT": ["3.05", "2.01.04", "2.02.01", "1.01.01", "1.01.02"],
    "DIVIDA LIQUIDA / EBIT": ["3.05", "2.01.04", "2.02.01", "1.01.01", "1.01.02"],
    "DIV. LIQ. / PATRI.": ["2.01.04", "2.02.01", "1.01.01", "1.01.02", "2.03", "2.03.09"],
    "LIQ. CORRENTE": ["1.01", "2.01"],
    "CAGR RECEITAS 5 ANOS": ["3.01"],
}
ACCOUNTS = ["3.01", "3.05", "3.08", "3.11", "3.11.01", "1", "1.01", "1.01.01", "1.01.02", "2.01", "2.01.04", "2.02.01", "2.03", "2.03.09"]
DRE_ACCOUNTS = {"3.01", "3.05", "3.08", "3.11", "3.11.01"}
METHODS = [(), ("LL_TOTAL",), ("PL_TOTAL",), ("LL_TOTAL", "PL_TOTAL")]

# DY rows with a decision taken from the B3 records (the records are the cache's; the motive is the
# reading). A row with cls NC has no decision: it goes through `dy_search`. `trading` names the B3 files.
DY_DECISIONS = {
    ("2026-08-07", "HBRE3"): dict(cls=ERRO_NOSSO, trading="HBR REALTY", records=[("30/12/2025", "0,87407147387"), ("30/12/2025", "0,29135715796")], yahoo="Yahoo yahoo_HBRE3_2026-11-07.json: 2026-01-02 0.485595", motive="fonte Yahoo (histórico B3 curto, `b3_short_history`) dá um registro de 0,4856; a B3 lista dois registros de 30/12/2025 que somam 1,1654, que é o DY do SI (52,5 %)"),
    ("2026-08-07", "RIAA3"): dict(cls=ERRO_SI, trading="RIACHUELO", records=[("22/12/2025", "0,82658586042")], motive="o SI deixa de fora o JCP de 0,8266 (lastDatePrior 22/12/2025): soma 2,428 de 3,255"),
    ("2026-11-05", "RIAA3"): dict(cls=ERRO_SI, trading="RIACHUELO", records=[("22/12/2025", "0,82658586042")], motive="o SI deixa de fora o JCP de 0,8266 (lastDatePrior 22/12/2025): soma 2,488 de 3,315"),
    ("2026-08-07", "MDNE3"): dict(cls=ERRO_SI, trading="MOURA DUBEUX", records=[("30/12/2025", "0,59183509277")], motive="o SI soma o dividendo de 0,5918 quatro vezes; a B3 lista seis linhas idênticas e a regra A-15 mantém uma"),
    ("2026-11-05", "MDNE3"): dict(cls=ERRO_SI, trading="MOURA DUBEUX", records=[("30/12/2025", "0,59183509277")], motive="o SI soma o dividendo de 0,5918 quatro vezes; a B3 lista seis linhas idênticas e a regra A-15 mantém uma"),
    ("2026-08-07", "VLID3"): dict(cls=ERRO_SI, trading="VALID", records=[("28/11/2025", "0,37646831702")], motive="o SI soma o JCP de 0,3765 quatro vezes; a B3 lista quatro linhas idênticas e a regra A-15 mantém uma"),
    ("2026-11-05", "VLID3"): dict(cls=ERRO_SI, trading="VALID", records=[("28/11/2025", "0,37646831702")], motive="o SI soma o JCP de 0,3765 quatro vezes; a B3 lista quatro linhas idênticas e a regra A-15 mantém uma"),
    ("2026-08-07", "EQPA3"): dict(cls=ERRO_SI, trading="EQTL PARA", records=[("25/06/2026", "0,31687485244")], motive="o SI deixa de fora o dividendo ON de 0,3169 (lastDatePrior 25/06/2026); o DY do SI (2,53 %) é só o de 30/04/2026"),
    ("2026-11-05", "EQPA3"): dict(cls=ERRO_SI, trading="EQTL PARA", records=[("25/06/2026", "0,31687485244")], motive="o SI deixa de fora o dividendo ON de 0,3169 (lastDatePrior 25/06/2026); o DY do SI (4,65 %) soma 0,1346 e 0,0958"),
    ("2026-08-07", "PFRM3"): dict(cls=ERRO_SI, trading="PROFARMA", records=[("30/12/2025", "0,31808023603")], motive="o SI deixa de fora o dividendo ON de 0,3181 (lastDatePrior 30/12/2025); o DY do SI (4,63 %) é 2 × 0,1631"),
    ("2026-11-05", "PFRM3"): dict(cls=ERRO_SI, trading="PROFARMA", records=[("30/12/2025", "0,31808023603")], motive="o SI deixa de fora o dividendo ON de 0,3181 (lastDatePrior 30/12/2025); o DY do SI (2,42 %) é 0,1631"),
    ("2026-08-07", "COCE5"): dict(cls=ERRO_SI, trading="COELCE", records=[("29/04/2026", "1,48326563")], motive="o SI traz DY em branco (tratado como 0,00, A-4); a B3 lista o dividendo PNA de 1,4833 (lastDatePrior 29/04/2026)"),
    ("2026-11-05", "COCE5"): dict(cls=ERRO_SI, trading="COELCE", records=[("29/04/2026", "1,48326563")], motive="o SI traz DY em branco (tratado como 0,00, A-4); a B3 lista o dividendo PNA de 1,4833 (lastDatePrior 29/04/2026)"),
    ("2026-11-05", "PASS3"): dict(cls=ERRO_SI, trading="COMPASS", records=[("19/12/2025", "0,70009371944")], motive="o SI deixa de fora o dividendo ON de 0,7001 (lastDatePrior 19/12/2025); o DY do SI (2,90 %) é só o de 04/05/2026"),
    ("2026-08-07", "TGMA3"): dict(cls=METODO, trading="TEGMA", records=[("07/08/2025", "1,21"), ("07/08/2025", "0,14")], motive="o SI deixa de fora 1,21 + 0,14 com lastDatePrior 07/08/2025 (= d − 365, data-ex 08/08/2025); o DY do SI (10,84 %) soma 3,63 com o resto: o SI parece contar pelo último dia com direito e não pela data-ex (inferência de um caso)"),
    ("2026-08-07", "HBOR3"): dict(cls=NC, trading="HELBOR"),
    ("2026-08-07", "CLSC4"): dict(cls=NC, trading="CELESC"),
    ("2026-11-05", "JSLG3"): dict(cls=NC, trading="JSL"),
    ("2026-11-05", "EQTL3"): dict(cls=NC, trading="EQUATORIAL"),
    ("2026-11-05", "MBRF3"): dict(cls=NC, trading="MARFRIG"),
}


def fmt(value):
    if value is None:
        return "None"
    if isinstance(value, float):
        if value == int(value) and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


# ---------------------------------------------------------------- evidence (B-10) and methods text

def head(kind, dt, versao, rec):
    return f"CVM {kind} {dt.isoformat()} v{versao} rec {rec.isoformat()}"


def acc_items(kind, dt, versao, rec, nums, accounts):
    return [f"{head(kind, dt, versao, rec)}: {a} = {fmt(nums[a])}" for a in accounts if a in nums]


def comp_item(kind, dt, versao, rec, comp):
    on, pref, total = comp
    return f"{head(kind, dt, versao, rec)}: QT_ACAO_ORDIN_CAP_INTEGR = {on}, QT_ACAO_PREF_CAP_INTEGR = {pref}, QT_ACAO_TOTAL_CAP_INTEGR = {total}"


def how_text(method, view):
    if view == "COUNT":
        return "com a contagem de ações desse demonstrativo"
    if "LL_TOTAL" in method and "PL_TOTAL" in method:
        return "com os demonstrativos desse filing e com os totais (LL 3.11 e PL 2.03, não os do controlador)"
    if "LL_TOTAL" in method:
        return "com os demonstrativos desse filing e com o total da LL (3.11)"
    if "PL_TOTAL" in method:
        return "com os demonstrativos desse filing e com o total do PL (2.03)"
    return "com os demonstrativos desse filing"


def event_item(event):
    return f"B3 evento {event.isin} {event.label} {event.last_date_prior.isoformat()} ×{fmt(event.multiplier)}"


def cotahist_item(ticker, session, close):
    return f"COTAHIST {ticker} {session.isoformat()} {fmt(round(close, 2))}"


def same_filing(p, f0):
    return f0 is not None and (p["kind"], p["dt"], p["versao"]) == (f0.kind, f0.dt_refer, f0.versao)


def oldest(reprod):
    """The reproducer with the smallest DT_RECEB, then DT_REFER, then VERSAO (stable: view order breaks ties)."""
    return min(reprod, key=lambda p: (p["dt_receb"], p["dt"], p["versao"]))


def reproduces(field, reference, value):
    """A view reproduces the Status Invest value when it has one and it is within the report's tolerance (CMP-01)."""
    return value is not None and compare.within(field, reference, value)


# ---------------------------------------------------------------- acquisition of the rebuild's own inputs

CAPTURED: dict = defaultdict(list)
_ORIG_COMPUTE = cli.compute


def _capture(inputs, d):
    CAPTURED[(d, inputs.ticker)].append(inputs)
    return _ORIG_COMPUTE(inputs, d)


@contextmanager
def capturing():
    """Records the TickerInputs cli.compute receives: the first call of a ticker and a day is the report's own."""
    cli.compute = _capture
    try:
        yield
    finally:
        cli.compute = _ORIG_COMPUTE


@contextmanager
def pit_guard_off():
    """The search reads filings received after d at R, as the lote-5 audit did. The PIT-02 guard of the
    report (`indicators._assert_point_in_time`) is off only inside this block, never in the report."""
    original = indicators._assert_point_in_time
    indicators._assert_point_in_time = lambda inputs, d: None
    try:
        yield
    finally:
        indicators._assert_point_in_time = original


def current(day, ticker):
    return CAPTURED[(day, ticker)][0]


def members_of(acq, day, ticker):
    """cnpj, cd_cvm and the tickers of the company of `ticker` at `day`, as cli._evaluate_day groups them."""
    item = acq.mapped[day][ticker]
    tickers = [
        t for t in acq.eligible[day]
        if isinstance(acq.mapped[day][t], mapping.Mapped) and acq.mapped[day][t].cnpj == item.cnpj
        and cli._unmapped_reason(acq, day, t, day) is None
    ]
    return item.cnpj, item.cd_cvm, sorted(tickers)


def share_detail(acq, day, ticker):
    """The share-count rule of cli._counts, unpacked: filed count, B3 total, events, factor and residual."""
    cnpj, cd_cvm, tickers = members_of(acq, day, ticker)
    shares = cvm.shares(acq.dfp, acq.itr, cnpj, day)
    prices = cotahist.class_prices(acq.quotes, cli._quote_candidates(acq, cnpj, tickers, day), day)
    b3_total, retrieved_on, b3_root = None, None, None
    for t in sorted(tickers):
        company = acq.b3.get(t[:4])
        if company is not None and company.total_shares:
            b3_total, retrieved_on, b3_root = company.total_shares, company.retrieved_on, t[:4]
            break
    roots = sorted({t[:4] for t in tickers})
    records = [acq.b3[r] for r in roots if acq.b3.get(r) is not None]
    events = tuple(e for c in records for e in c.stock_events)

    def next_session(value):
        return cotahist.next_session(acq.quotes, value)

    sets = {cls: cli._class_isin_sets(acq.quotes, None if prices[cls] is None else prices[cls][0], shares.dt_refer, day)
            for cls in ("ON", "PREF")}
    adj = share_counts.adjust(shares, {c: sets[c][0] for c in sets}, {c: sets[c][1] for c in sets}, events, b3_total,
                              retrieved_on, day, next_session)
    return {"cnpj": cnpj, "cd_cvm": cd_cvm, "tickers": tickers, "shares": shares, "prices": prices,
            "b3_total": b3_total, "retrieved_on": retrieved_on, "b3_root": b3_root, "events": events,
            "next_session": next_session, "adj": adj}


def alternatives(acq, day, ticker, f, ref, reb):
    """Each alternative definition the rules test: {name: (value rounded to 2, within the report's tolerance)}."""
    cur = current(day, ticker)
    cnpj, _cd, _tickers = members_of(acq, day, ticker)
    alts = {}
    alts["LL_TOTAL"] = replace(cur, ll=cli._dated(cvm.ttm(acq.dfp, acq.itr, cnpj, day, "3.11")),
                               ll_fy5=cli._dated(cvm.annual(acq.dfp, cnpj, day, "3.11", 5)))
    alts["PL_TOTAL"] = replace(cur, pl_ctrl=cur.pl)
    det = share_detail(acq, day, ticker) if f in SHARE_FIELDS else None
    if det is not None and det["b3_total"]:
        total_ours = (cur.n_on or 0) + (cur.n_pref or 0)
        if total_ours > 0:
            n_on = round(det["b3_total"] * ((cur.n_on or 0) / total_ours))
            alts["B3_TOTAL"] = replace(cur, n_on=n_on, n_pref=det["b3_total"] - n_on)
    if f in SHARE_FIELDS and cur.price is not None:
        before = [s for s in acq.quotes.dates.get(ticker, []) if s < cur.price.when]
        if before:
            prev_day = before[-1]
            _d, close = cotahist.quote_at(acq.quotes, ticker, prev_day, max_age_days=None)
            alts["PRICE_PREV"] = replace(cur, price=Input(close, prev_day))
    out = {}
    for name, inp in alts.items():
        v = indicators.compute(inp, day)[f].value
        if v is None:
            continue
        v2 = round(v, 2)
        out[name] = (v2, compare.within(f, ref, v2))
    return out


def older_filing(acq, day, cnpj):
    older = cli._without_newest(acq, cnpj, day)
    if older is None:
        return None
    return cvm.newest_filing(older.dfp, older.itr, cnpj, day)


def period_of(acq, filing):
    store = acq.dfp if filing.kind == "DFP" else acq.itr
    return store.period(filing.cnpj, filing.dt_refer)


def cvm_item(acq, filing, account):
    """The B-10 CVM item of one account as filed (ÚLTIMO for DRE rows, the balance for BPA/BPP), or None."""
    period = period_of(acq, filing)
    if period is None:
        return None
    if account.startswith("3."):
        row = period.dre.get((account, cvm.ULTIMO))
        value = None if row is None else row.value
    else:
        value = period.bal.get(account)
    if value is None:
        return None
    return f"{head(filing.kind, filing.dt_refer, filing.versao, filing.dt_receb)}: {account} = {fmt(value)}"


def composition_item(acq, filing):
    if filing is None:
        return None
    period = period_of(acq, filing)
    if period is None or period.composition is None:
        return None
    on, pref, total = period.composition
    return comp_item(filing.kind, filing.dt_refer, filing.versao, filing.dt_receb, (on, pref, total))


# ---------------------------------------------------------------- filings: every version of the cache

def _members_iter(archive):
    return cvm._members(archive)


def load_versions(acq, cnpjs):
    """{kind: {cnpj: {(dt_refer, versao): Period}}} for every version of every period of the companies."""
    out = {kind: defaultdict(dict) for kind in KINDS}
    for kind in KINDS:
        prefix = kind.lower()
        index, statements, comps = {}, defaultdict(dict), {}
        for path in acq.files[kind]:
            with zipfile.ZipFile(path) as archive:
                for member, full in _members_iter(archive):
                    m = cvm._INDEX.match(member)
                    if m and m.group(1).lower() == prefix:
                        for row in cvm._rows(archive, full, member, cvm._INDEX_COLUMNS):
                            c = row["CNPJ_CIA"].strip()
                            if c in cnpjs:
                                index[(c, cvm._date(row["DT_REFER"], member), cvm._int(row["VERSAO"], member))] = cvm._date(row["DT_RECEB"], member)
                        continue
                    st = cvm._STATEMENT.match(member)
                    if st and st.group(1).lower() == prefix:
                        stmt, scope = st.group(2).upper(), st.group(3).lower()
                        columns = cvm._DRE_COLUMNS if stmt == "DRE" else cvm._BALANCE_COLUMNS
                        for row in cvm._rows(archive, full, member, columns):
                            c = row["CNPJ_CIA"].strip()
                            if c in cnpjs:
                                key = (c, cvm._date(row["DT_REFER"], member), cvm._int(row["VERSAO"], member))
                                statements[key].setdefault((scope, stmt), []).append((member, row))
                        continue
                    co = cvm._COMPOSITION.match(member)
                    if co and co.group(1).lower() == prefix:
                        for row in cvm._rows(archive, full, member, cvm._COMPOSITION_COLUMNS):
                            c = row["CNPJ_CIA"].strip()
                            if c in cnpjs:
                                comps[(c, cvm._date(row["DT_REFER"], member), cvm._int(row["VERSAO"], member))] = (member, row)
        firsts = {}
        for (c, dt, _v), rec in index.items():
            firsts[(c, dt)] = min(firsts.get((c, dt), rec), rec)
        for (c, dt, v), rec in sorted(index.items()):
            out[kind][c][(dt, v)] = _reconstruct(kind, c, dt, v, rec, firsts[(c, dt)], statements.get((c, dt, v), {}), comps.get((c, dt, v)))
    return out


def _reconstruct(kind, cnpj, dt, versao, rec, first, by_scope, comp):
    statement, scope = cvm._statement(by_scope)
    filing = cvm.Filing(kind=kind, cnpj=cnpj, dt_refer=dt, dt_receb=rec, first_receb=first, versao=versao, statement=statement)
    period = cvm._Period(filing=filing)
    for stmt in ("BPA", "BPP"):
        for member, row in by_scope.get((scope, stmt), []):
            if row["ORDEM_EXERC"].strip() != cvm.ULTIMO:
                continue
            value = cvm._value(row, member)
            if value is not None:
                period.bal[row["CD_CONTA"].strip()] = value
    for member, row in by_scope.get((scope, "DRE"), []):
        value = cvm._value(row, member)
        if value is None:
            continue
        account_key = (row["CD_CONTA"].strip(), row["ORDEM_EXERC"].strip())
        dt_ini = cvm._date(row["DT_INI_EXERC"], member)
        current_row = period.dre.get(account_key)
        if current_row is None or dt_ini < current_row.dt_ini:
            period.dre[account_key] = cvm._DreRow(dt_ini=dt_ini, value=value, desc=row["DS_CONTA"].strip())
    cvm._add_net_income(period)
    if comp is not None:
        period.composition = cvm._composition(*comp)
    return period


def check_against_store(acq, versions, cnpjs):
    """The reconstructed latest version of every period equals the store's period on every value it keeps."""
    problems = []
    for kind, store in (("DFP", acq.dfp), ("ITR", acq.itr)):
        for c in cnpjs:
            for period in store.periods(c):
                dt = period.filing.dt_refer
                mine = versions[kind][c].get((dt, period.filing.versao))
                if mine is None:
                    problems.append((kind, c, dt, "missing version"))
                    continue
                for attr in ("dt_receb", "first_receb", "versao", "statement"):
                    if getattr(period.filing, attr) != getattr(mine.filing, attr):
                        problems.append((kind, c, dt, attr))
                for k, value in period.bal.items():
                    if mine.bal.get(k) != value:
                        problems.append((kind, c, dt, "bal", k))
                for k, row in period.dre.items():
                    m = mine.dre.get(k)
                    if m is None or m.value != row.value or m.dt_ini != row.dt_ini:
                        problems.append((kind, c, dt, "dre", k))
                if period.composition != mine.composition:
                    problems.append((kind, c, dt, "composition"))
    return problems


def filings_of(versions, cnpj):
    """Every cached (kind, dt_refer, versao, DT_RECEB) of cnpj received on or before R, oldest receipt first."""
    out = []
    for kind in KINDS:
        for (dt, v), p in versions[kind].get(cnpj, {}).items():
            if p.filing.dt_receb <= R_DATE:
                out.append((kind, dt, v, p.filing.dt_receb))
    return sorted(out, key=lambda x: (x[3], x[1], x[2]))


def version_nums(p):
    """The archived numbers of one version: the accounts of the audit and its composition."""
    nums = {}
    for acc in ACCOUNTS:
        if acc in DRE_ACCOUNTS:
            row = p.dre.get((acc, cvm.ULTIMO))
            if row is not None:
                nums[acc] = row.value
        elif acc in p.bal:
            nums[acc] = p.bal[acc]
    return nums, (None if p.composition is None else tuple(p.composition))


def numbers(versions, cnpj):
    out = []
    for kind, dt, v, rec in filings_of(versions, cnpj):
        p = versions[kind][cnpj][(dt, v)]
        nums, comp = version_nums(p)
        out.append({"kind": kind, "dt_refer": dt, "versao": v, "dt_receb": rec, "nums": nums, "composition": comp})
    return out


def full_acq(acq, versions, cnpj, kind, dt, versao):
    """acq with (kind, dt, versao) as the newest filing of cnpj; periods received later are removed."""
    dfp = dict(acq.dfp._periods)
    itr = dict(acq.itr._periods)
    dfp[cnpj] = {d_: p for d_, p in dfp.get(cnpj, {}).items() if d_ <= dt}
    itr[cnpj] = {d_: p for d_, p in itr.get(cnpj, {}).items() if d_ <= dt}
    target = dfp if kind == "DFP" else itr
    target[cnpj] = dict(target[cnpj])
    target[cnpj][dt] = versions[kind][cnpj][(dt, versao)]
    return replace(acq, dfp=cvm.FilingStore("DFP", dfp), itr=cvm.FilingStore("ITR", itr))


def shares_of(versions, cnpj, kind, dt, versao):
    """The cvm.Shares of one version (its composition as filed), or None without a composition or with total 0."""
    p = versions[kind][cnpj][(dt, versao)]
    if p.composition is None or p.composition[2] <= 0:
        return None
    on, pref, total = p.composition
    return cvm.Shares(on=on, pref=pref, total=total, dt_refer=dt, dt_receb=p.filing.dt_receb)


def apply_method(view, inputs, cnpj, fday, method):
    if "LL_TOTAL" in method:
        inputs = replace(inputs, ll=cli._dated(cvm.ttm(view.dfp, view.itr, cnpj, fday, "3.11")),
                         ll_fy5=cli._dated(cvm.annual(view.dfp, cnpj, fday, "3.11", 5)))
    if "PL_TOTAL" in method:
        inputs = replace(inputs, pl_ctrl=inputs.pl)
    return inputs


def company_values(acq, view, day, fday, cnpj, cd_cvm, tickers, *, count_from=None, method=()):
    """{ticker: {field: value rounded to 2 decimals or None}} of one company under one view.

    FULL view: `view` is the acq in which the company's newest filing is the version under test, read at fday
    (d when it was received by d, else R). COUNT view (`count_from`): only the share count is that version's;
    every other input is the report's own at d. No stale-filing mask: an old filing is read whole."""
    if count_from is not None:
        company = cli._company(acq, day, cnpj, cd_cvm, tickers, day)
        prices = cotahist.class_prices(acq.quotes, cli._quote_candidates(acq, cnpj, tickers, day), day)
        counts = cli._counts(acq, day, cnpj, cd_cvm, tickers, day, count_from, prices)
        company = replace(company, n_on=counts.n_on, n_pref=counts.n_pref)
    else:
        company = cli._company(view, day, cnpj, cd_cvm, tickers, fday)
    candidates = cli._quote_candidates(acq, cnpj, tickers, day)
    out = {}
    for ticker in tickers:
        item = acq.mapped[day][ticker]
        found = cli._dividends_of(acq, day, ticker, item, candidates)
        close_day, close = cotahist.quote_at(acq.quotes, ticker, day)
        last_day = cotahist.last_trade(acq.quotes, ticker, day)[0]
        inputs = TickerInputs(
            ticker=ticker,
            unit_divisor=sum(item.composition.values()) if item.composition else 1,
            price=Input(close, close_day),
            price_on=company.price_on,
            price_pref=company.price_pref,
            n_on=company.n_on,
            n_pref=company.n_pref,
            liquidity=Input(cotahist.avg_volume(acq.quotes, ticker, day), last_day),
            dividends=found.amounts,
            **company.inputs,
        )
        inputs = apply_method(view if count_from is None else acq, inputs, cnpj, fday, method)
        vals = compute(inputs, day)
        out[ticker] = {name: (None if v.value is None else round(v.value, 2)) for name, v in vals.items()}
    return out


def candidate_reads(versions, cnpj, day):
    """The reads of the search for one company at day: (kind, dt_refer, versao, DT_RECEB, fday) of every cached version
    received on or before R. A version received on or before d is read at d; one received after d is read at R (the
    lote-5 read). No age limit: a period that ended years before d is read whole (no STL-01 mask)."""
    return [(kind, dt, v, rec, day if rec <= day else R_DATE) for kind, dt, v, rec in filings_of(versions, cnpj)]


def search_date(acq, versions, iso, rows):
    """Reproducers of each worst row of one date: {(field, ticker): [version or count views that reproduce]}.

    A view reproduces the row when compare.within(field, SI, value) holds (CMP-01). Every version of every
    filing of the row's company is tried, as the newest filing, in four accounting views and, when it has a
    composition, in the count view."""
    day = date.fromisoformat(iso)
    out = defaultdict(list)
    companies = OrderedDict()
    for r in rows:
        cnpj, cd, tickers = members_of(acq, day, r["ticker"])
        companies.setdefault(cnpj, (cd, tickers, []))[2].append(r)
    for cnpj, (cd, tickers, crows) in companies.items():
        for kind, dt, v, rec, fday in candidate_reads(versions, cnpj, day):
            nums, comp = version_nums(versions[kind][cnpj][(dt, v)])
            views = []
            for m in METHODS:
                view = full_acq(acq, versions, cnpj, kind, dt, v)
                views.append(("FULL", m, company_values(acq, view, day, fday, cnpj, cd, tickers, method=m)))
            if shares_of(versions, cnpj, kind, dt, v) is not None:
                views.append(("COUNT", (), company_values(acq, acq, day, fday, cnpj, cd, tickers,
                                                           count_from=shares_of(versions, cnpj, kind, dt, v))))
            for view_name, m, vals in views:
                for r in crows:
                    value = vals[r["ticker"]][r["field"]]
                    if reproduces(r["field"], r["reference"], value):
                        out[(r["field"], r["ticker"])].append({
                            "view": view_name, "method": m, "kind": kind, "dt": dt, "versao": v, "dt_receb": rec,
                            "fday": fday, "value": value, "nums": nums, "composition": comp,
                        })
    return out, companies


# ---------------------------------------------------------------- DY: B3 records and the subset search

def b3_file_index(trading):
    """{(lastDatePriorEx, valueCash): (file name, typeStock, corporateAction)} over the company's B3 cash files."""
    index = {}
    page = 1
    while True:
        name = cli.b3_cash_name(trading, page)
        path = CACHE / name
        if not path.is_file():
            break
        data = json.load(open(path, encoding="utf-8"))
        for r in data["results"]:
            index.setdefault((r.get("lastDatePriorEx"), r.get("valueCash")), (name, r.get("typeStock"), r.get("corporateAction")))
        if page >= data["page"]["totalPages"]:
            break
        page += 1
    return index


def b3_records(trading, cls_type, quotes):
    """Every B3 cash line of the class (one record per line, duplicates kept: the SI sums them), with its ex-date."""
    recs = []
    page = 1
    while True:
        name = cli.b3_cash_name(trading, page)
        path = CACHE / name
        if not path.is_file():
            break
        data = json.load(open(path, encoding="utf-8"))
        for r in data["results"]:
            if r.get("typeStock") != cls_type:
                continue
            last = datetime.strptime(r["lastDatePriorEx"], "%d/%m/%Y").date()
            recs.append({"file": name, "last": last, "ex": cotahist.next_session(quotes, last),
                         "amount": float(r["valueCash"].replace(",", "."))})
        if page >= data["page"]["totalPages"]:
            break
        page += 1
    return recs


GRID = 1e-5  # R$ unit of the DY subset search: amounts are rounded to it; every subset found is checked with its exact sum


def dy_search(lines, ref, prices):
    """One reproducing subset of the cache's B3 lines per price: {name: indices (ascending), or None}.

    Every line is an item, identical lines included (the SI sums them), with or without an ex-date. Sums are
    counted in GRID units: `reach[i]` is the bitset of the sums of the subsets of the first i lines, so the
    subsets are read back from the snapshots without enumerating them. The window is widened by the rounding
    error, and each candidate sum is confirmed with the exact sum of its lines (two decimals, as the SI)."""
    units = [round(x["amount"] / GRID) for x in lines]
    reach = [1]
    for u in units:
        reach.append(reach[-1] | (reach[-1] << u))
    total = reach[-1]
    margin = len(lines) // 2 + 2
    out = {}
    for name, price in prices:
        out[name] = None
        if price is None:
            continue
        lo = (ref - 0.005) * price / 100 / GRID
        hi = (ref + 0.005) * price / 100 / GRID
        base = max(0, math.floor(lo) - margin)
        width = math.ceil(hi) + margin - base + 1
        window = (total >> base) & ((1 << width) - 1)
        candidates = [base + k for k in range(width) if (window >> k) & 1]
        candidates.sort(key=lambda s: abs(s - (lo + hi) / 2))
        for s in candidates:
            idx = _witness(reach, units, s)
            exact = sum(lines[i]["amount"] for i in idx)
            if idx and round(100 * exact / price, 2) == ref:
                out[name] = idx
                break
    return out


def _witness(reach, units, s):
    """A subset with grid sum s: line i is taken only when s is not reachable without it (later lines left out first)."""
    idx = []
    for i in range(len(units) - 1, -1, -1):
        if (reach[i] >> s) & 1:
            continue
        idx.append(i)
        s -= units[i]
    return sorted(idx)


# ---------------------------------------------------------------- post-d reads (filings received after d, read at R)

def future_reads(acq, rows_by_date):
    """(iso, field, ticker) -> (newest filing at R, value) where the reads at R reproduce the SI and d does not."""
    with pit_guard_off():
        res = {day: cli._evaluate_day(acq, day, filings_day=R_DATE) for day in DAYS}
    out = {}
    for iso, rows in rows_by_date.items():
        day = date.fromisoformat(iso)
        for r in rows:
            if r["field"] == "DY":
                continue
            pair = next((p for p in res[day].pairs[r["field"]]["non_financial"] if p[0] == r["ticker"]), None)
            if pair is None or pair[2] is None:
                continue
            if compare.within(r["field"], r["reference"], round(pair[2], 2)) and not compare.within(r["field"], r["reference"], r["rebuilt"]):
                cnpj, _cd, _tickers = members_of(acq, day, r["ticker"])
                out[(iso, r["field"], r["ticker"])] = (cvm.newest_filing(acq.dfp, acq.itr, cnpj, R_DATE), round(pair[2], 2))
    return out


def fut_filing(acq, cnpj, fut):
    """The filing of a composition read at R: the store's period whose latest version was received at fut.dt_receb."""
    for store in (acq.dfp, acq.itr):
        period = store.period(cnpj, fut.dt_refer)
        if period is not None and period.filing.dt_receb == fut.dt_receb:
            return period.filing
    return None


def post_count(acq, day, cnpj, f, ref, cur, adj, det):
    """Share field reproduced with the composition of a filing received after d (count read at R, flows at d)."""
    fut = cvm.shares(acq.dfp, acq.itr, cnpj, R_DATE)
    if fut is None or det["shares"] is None or (fut.on, fut.pref) == (det["shares"].on, det["shares"].pref):
        return None
    m = {"ON": 1.0, "PREF": 1.0}
    for a in adj.applied:
        m[a["class"]] = a["factor"]
    n_on, n_pref = round(fut.on * adj.factor * m["ON"]), round(fut.pref * adj.factor * m["PREF"])
    val = indicators.compute(replace(cur, n_on=n_on, n_pref=n_pref), day)[f].value
    if val is None or not compare.within(f, ref, round(val, 2)):
        return None
    filing = fut_filing(acq, cnpj, fut)
    motive = (f"o Status Invest usa a composição de ações de {fut.dt_refer.isoformat()} recebida em {fut.dt_receb.isoformat()}, depois de d = {day.isoformat()} "
              f"(a primeira entrega é posterior à data de coleta do CSV): com essa contagem e os demais insumos de d reproduz {fmt(round(val, 2))}")
    items = [composition_item(acq, filing)] if filing is not None else []
    return motive, [i for i in items if i]


def post_flow(acq, day, f, fut_entry):
    """Accounting field reproduced with the filing that R reads and d does not."""
    filing, _value = fut_entry
    if filing is None:
        return None
    items = [it for it in (cvm_item(acq, filing, acc) for acc in FIELD_ACCOUNTS[f][:2]) if it]
    motive = (f"o Status Invest reproduz o valor com o {filing.kind} de {filing.dt_refer.isoformat()} v{filing.versao} recebido em {filing.dt_receb.isoformat()}, "
              f"depois de d = {day.isoformat()}; com os demonstrativos usáveis em d o valor fica fora da tolerância")
    return motive, items


# ---------------------------------------------------------------- classification (rule order: see HEADER)

def dedup(recs):
    """A-15: identical lines (same ex-date and value) count once in the window and in the evidence."""
    seen = OrderedDict()
    for x in recs:
        seen.setdefault((x["ex"], round(x["amount"], 9)), x)
    return list(seen.values())


def dy_item(x):
    if x["ex"] is None:
        return f"B3 proventos {x['file']}: sem cotação para a data-ex (lastDatePrior {x['last']:%d/%m/%Y}) {fmt(round(x['amount'], 6))}"
    return f"B3 proventos {x['file']}: {x['ex'].isoformat()} {fmt(round(x['amount'], 6))}"


def classify_dy(ctx, iso, day, row):
    acq = ctx["acq"]
    t, ref, reb = row["ticker"], row["reference"], row["rebuilt"]
    decision = DY_DECISIONS[(iso, t)]
    close_day, price = cotahist.quote_at(acq.quotes, t, day)
    price_item = cotahist_item(t, close_day, price)
    if decision["cls"] != NC:
        index = b3_file_index(decision["trading"])
        evid = []
        for lastprior, value in decision["records"]:
            key = (lastprior, value)
            if key not in index:
                raise RuntimeError(f"record not in B3 files: {iso} {t} {key}")
            ex = cotahist.next_session(acq.quotes, datetime.strptime(lastprior, "%d/%m/%Y").date())
            evid.append(f"B3 proventos {index[key][0]}: {ex.isoformat()} {fmt(round(float(value.replace(',', '.')), 6))}")
        if decision.get("yahoo"):
            evid.append(decision["yahoo"])
        return {"class": decision["cls"], "motive": decision["motive"], "evidence": evid + [price_item], "rule": "DY manual", "exact": None}

    cls_type = "PN" if t[-1] in "456" else "ON"
    lines = sorted(b3_records(decision["trading"], cls_type, acq.quotes), key=lambda x: (x["last"], x["amount"]))
    before = [s for s in acq.quotes.dates.get(t, []) if s < close_day]
    prev_day = before[-1] if before else None
    prev_close = [x.close for x in acq.quotes.rows[t] if x.day == prev_day][0] if before else None
    names = {"d": (close_day, price), "d − 1": (prev_day, prev_close)}
    found = dy_search(lines, ref, [(n, names[n][1]) for n in names])
    name = next((n for n in names if found.get(n) is not None), None)
    if name is not None:
        chosen = [lines[i] for i in found[name]]
        desc = "; ".join(
            (f"data-ex {x['ex']:%d/%m/%Y}" if x["ex"] else f"lastDatePrior {x['last']:%d/%m/%Y}") + f" {x['amount']:.6f}" for x in chosen)
        later = any(x["ex"] and x["ex"] > day for x in chosen)
        motive = (f"o DY do SI ({ref:.2f} %) é exato com {len(chosen)} registro(s) da B3 do cache ({desc}; classe {cls_type}, "
                  f"sobre o fechamento de {name}); o subconjunto é achado pela ordem das datas-ex, sem os registros mais novos quando não são necessários; "
                  f"a janela do projeto dá {reb:.2f} %" + ("; o subconjunto usa registro com data-ex posterior a d" if later else ""))
        evid = [dy_item(x) for x in chosen] + [cotahist_item(t, names[name][0], names[name][1])]
        return {"class": ERRO_SI, "motive": motive, "evidence": evid, "rule": "DY exato", "exact": None}

    window = dedup([x for x in lines if x["ex"] and day - timedelta(days=365) < x["ex"] <= day])
    outside = [x for x in lines if x["ex"] and x["ex"] <= day - timedelta(days=365)]
    near = ""
    if outside:
        x = max(outside, key=lambda y: y["ex"])
        near = (f"; o provento fora da janela mais recente ({x['amount']:.6f}, data-ex {x['ex']:%d/%m/%Y}, "
                f"{(day - x['ex']).days} dias antes de d) daria {100 * x['amount'] / price:.2f} % com o fechamento de d")
    motive = (f"nenhum subconjunto dos proventos da B3 do cache reproduz o DY do SI ({ref:.2f} %) a duas casas, nem com o fechamento de d nem com o de d − 1 "
              f"({len(lines)} registros testados, cada linha um registro); a janela do projeto dá {reb:.2f} % ({len(window)} provento(s) sem repetição, "
              f"somando {sum(x['amount'] for x in window):.4f} sobre {price:.2f}); o SI implica {ref / 100 * price:.4f} por ação{near}")
    evid = [dy_item(x) for x in window] or [f"B3 proventos {cli.b3_cash_name(decision['trading'], 1)}: nenhum registro com d − 365 < data-ex ≤ d"]
    return {"class": ERRO_SI, "motive": motive, "evidence": evid + [price_item], "rule": "DY sem base", "exact": None}


def reproducer_items(p, f):
    args = (p["kind"], p["dt"], p["versao"], p["dt_receb"])
    if p["view"] == "COUNT":
        return [comp_item(*args, p["composition"])]
    items = acc_items(*args, p["nums"], FIELD_ACCOUNTS[f][:3])
    if f in SHARE_FIELDS and p["composition"] is not None:
        items.append(comp_item(*args, p["composition"]))
    return items


def reproduced_motive(best, n_rep, day, ref):
    when = "antes" if best["dt_receb"] < day else ("no dia" if best["dt_receb"] == day else "depois")
    return (f"o Status Invest reproduz o valor com {best['kind']} {best['dt'].isoformat()} v{best['versao']}, recebido em {best['dt_receb'].isoformat()}, "
            f"{when} de d = {day.isoformat()}: {how_text(best['method'], best['view'])} "
            f"(SI {ref}, recalculado {best['value']}; {n_rep} demonstrativo(s) reproduzem, o mais antigo é citado)")


def base_items(acq, t, f, f0, det, rule, lag, older, price):
    """The B-10 items of the rebuild's own inputs: its filing's accounts, the share count, the events and the close."""
    items = []
    accs = FIELD_ACCOUNTS.get(f)
    if accs and f0 is not None:
        if rule == "defasagem (B-9)" and older is not None:
            items += [x for x in (cvm_item(acq, older, a) for a in accs[:2]) if x]
        items += [x for x in (cvm_item(acq, f0, a) for a in accs[:3]) if x]
    if f in SHARE_FIELDS:
        ci = composition_item(acq, f0)
        if ci:
            items.append(ci)
        for a in det["adj"].applied:
            for e in a["events"]:
                ev = next((x for x in det["events"] if x.isin == e["isin"] and x.label == e["label"]), None)
                if ev is not None:
                    items.append(event_item(ev))
        if price is not None:
            items.append(cotahist_item(t, price.when, price.value))
    return items


def decide(f):
    """The class and rule of one accounting row: the first rule of `Ordem das regras` (HEADER) whose facts hold.

    `f` holds booleans: lag (B-9); post (a filing received after d reproduces); erro_nosso (A-18 without support);
    b3_count (the B3 total count reproduces, the filed count is at most twice it, no suspicion); cited_af (a reproducer
    as filed or by count, F0 excluded); ll_method and pl_method (the total 3.11 or 2.03 reproduces and differs from
    ours); cited_any (a method view reproduces); price_two (the close of d − 1 reproduces two count fields)."""
    if f["lag"]:
        return ERRO_SI, "defasagem (B-9)"
    if f["post"]:
        return ERRO_SI, "post-d"
    if f["erro_nosso"]:
        return ERRO_NOSSO, "erro nosso"
    if f["b3_count"] and f["cited_af"]:
        return ERRO_SI, "reproduzido (sobre a contagem da B3)"
    if f["b3_count"]:
        return METODO, "contagem da B3 de R"
    if f["ll_method"]:
        return METODO, "lucro total 3.11"
    if f["pl_method"]:
        return METODO, "patrimônio total 2.03"
    if f["cited_af"]:
        return ERRO_SI, "reproduzido"
    if f["cited_any"]:
        return ERRO_SI, "reproduzido (visão de método)"
    if f["price_two"]:
        return ERRO_SI, "preço d−1"
    return ERRO_SI, "sem base"


def classify_accounting(ctx, iso, day, row):
    acq = ctx["acq"]
    f, t, ref, reb, lag = row["field"], row["ticker"], row["reference"], row["rebuilt"], row["si_lag"]
    cnpj, cd, _tickers = members_of(acq, day, t)
    cur = current(day, t)
    f0 = ctx["results"][day].filings.get(t)
    det = share_detail(acq, day, t) if f in SHARE_FIELDS else None
    adj = None if det is None else det["adj"]
    residual = None if adj is None else adj.residual
    b3_total = None if det is None else det["b3_total"]
    suspect = (iso, cd) in ctx["suspect"]
    alts = alternatives(acq, day, t, f, ref, reb)
    reproduces = {k for k, (v, ok) in alts.items() if ok}
    others = [p for p in ctx["reproducers"].get((iso, f, t), []) if not same_filing(p, f0)]
    as_filed = [p for p in others if p["view"] == "COUNT" or p["method"] == ()]
    best_af = oldest(as_filed) if as_filed else None
    best_any = oldest(others) if others else None
    n_af = len({(p["kind"], p["dt"], p["versao"]) for p in as_filed})
    n_any = len({(p["kind"], p["dt"], p["versao"]) for p in others})
    older = older_filing(acq, day, cnpj) if lag else None
    post = None
    if f in SHARE_FIELDS and not lag:
        post = post_count(acq, day, cnpj, f, ref, cur, adj, det)
    elif f in FIELD_ACCOUNTS and not lag and (iso, f, t) in ctx["fut"] and not ({"LL_TOTAL", "PL_TOTAL"} & reproduces):
        post = post_flow(acq, day, f, ctx["fut"][(iso, f, t)])
    price_fields = []
    if f in SHARE_FIELDS and "PRICE_PREV" in reproduces:
        for r in ctx["by_ticker"][(iso, t)]:
            if r["field"] in SHARE_FIELDS:
                ref_r = compare.reference(r["field"], acq.si[day].rows[t][r["field"]])[0]
                if alternatives(acq, day, t, r["field"], ref_r, r["rebuilt"]).get("PRICE_PREV", (None, False))[1]:
                    price_fields.append(r["field"])
    facts = {
        "lag": bool(lag),
        "post": post is not None,
        "erro_nosso": f in SHARE_FIELDS and residual is not None and residual > 2.0 and not suspect,
        "b3_count": f in SHARE_FIELDS and "B3_TOTAL" in reproduces and residual is not None and residual <= 2.0 and not suspect,
        "cited_af": best_af is not None,
        "ll_method": f in ("P/L", "CAGR LUCROS 5 ANOS", "ROE", "ROA", "MARG. LIQUIDA") and "LL_TOTAL" in reproduces and abs(alts["LL_TOTAL"][0] - reb) > 0.01,
        "pl_method": f in ("ROE", "P/VP", "DIV. LIQ. / PATRI.") and "PL_TOTAL" in reproduces and abs(alts["PL_TOTAL"][0] - reb) > 0.01,
        "cited_any": best_any is not None,
        "price_two": f in SHARE_FIELDS and "PRICE_PREV" in reproduces and len(price_fields) >= 2,
    }
    cls, rule = decide(facts)
    cited, extra = None, []

    def cite(pool, n):
        return reproduced_motive(pool, n, day, ref), reproducer_items(pool, f)

    if rule == "defasagem (B-9)":
        motive = "o filing anterior reproduz o Status Invest (B-9): a defasagem é do SI"
    elif rule == "post-d":
        motive, extra = post
        if best_af is not None:
            (motive, extra), cited = cite(best_af, n_af), best_af
    elif rule == "erro nosso":
        motive = (f"fator A-18 sem suporte: a contagem em d ({fmt((cur.n_on or 0) + (cur.n_pref or 0))}) é {residual:.2f} vezes a contagem B3 de R "
                  f"({fmt(b3_total)}, `b3_company_{det['b3_root']}.json`); a potência de dez escolhida (×{fmt(adj.factor)}) não corresponde a erro de unidade")
    elif rule in ("reproduzido (sobre a contagem da B3)", "reproduzido"):
        (motive, extra), cited = cite(best_af, n_af), best_af
    elif rule == "contagem da B3 de R":
        motive = (f"o Status Invest usa a contagem total da B3 de R ({fmt(b3_total)}, `b3_company_{det['b3_root']}.json`), posterior a d; "
                  f"a contagem arquivada em d com eventos é {residual:.2f} vezes a da B3 (AD-033); "
                  f"nenhum demonstrativo do cache, além do usado em d, reproduz o valor do SI ({ref})")
    elif rule == "lucro total 3.11":
        motive = (f"o Status Invest usa o lucro total do DRE (3.11), não o do controlador (3.11.01, A-21/F7): "
                  f"com 3.11 o valor fica em {alts['LL_TOTAL'][0]:.2f} (SI {ref})")
    elif rule == "patrimônio total 2.03":
        motive = (f"o Status Invest usa o patrimônio total (2.03), não o do controlador (2.03 − 2.03.09, A-21/F8): "
                  f"com 2.03 o valor fica em {alts['PL_TOTAL'][0]:.2f} (SI {ref})")
    elif rule == "reproduzido (visão de método)":
        (motive, extra), cited = cite(best_any, n_any), best_any
    elif rule == "preço d−1":
        motive = f"o Status Invest usa o fechamento da sessão anterior: reproduz {', '.join(price_fields)} com esse preço"
    else:
        n_checked, n_index = ctx["counts"][cnpj]
        pp = "; o fechamento de d − 1 reproduz este campo (um só campo, sem regra de preço)" if "PRICE_PREV" in reproduces else ""
        motive = (f"nenhum dos {n_checked} demonstrativos do cache com linhas de demonstrativo (DFP e ITR, todas as versões recebidas até 2026-11-09; "
                  f"{n_index} versões só têm o índice e não entram) reproduz o valor do SI ({ref}), "
                  f"testados o demonstrativo inteiro, com os totais da LL e do PL, e só a contagem de ações de cada um{pp}")
        extra = [f"Conferidos: {n_checked} demonstrativos (cd_cvm {cd}, seção Demonstrativos conferidos)"]
    price = cur.price
    items = base_items(acq, t, f, f0, det, rule, lag, older, price)
    evid = items + extra if rule == "sem base" else extra + items
    if rule == "preço d−1" and price is not None:
        before = [s for s in acq.quotes.dates.get(t, []) if s < price.when]
        if before:
            s0 = before[-1]
            evid.append(cotahist_item(t, s0, [x.close for x in acq.quotes.rows[t] if x.day == s0][0]))
    exact = None if cited is None else round(cited["value"], 2) == round(ref, 2)
    return {"class": cls, "motive": motive, "evidence": [e for e in evid if e], "rule": rule, "exact": exact}


def classify_within(acq, day, t, f, diff, reference, rebuilt, f0):
    if f == "DY":
        close_day, price = cotahist.quote_at(acq.quotes, t, day)
        evid = [cotahist_item(t, close_day, price)]
    else:
        item = cvm_item(acq, f0, FIELD_ACCOUNTS[f][0]) if f0 is not None else None
        evid = [item] if item else []
    motive = (f"dentro da tolerância (diferença {diff:.2f}); a lista de piores inclui linhas sem divergência "
              f"porque o campo tem menos de 10 linhas fora da tolerância")
    return {"class": WITHIN, "motive": motive, "evidence": evid, "rule": "dentro da tolerância", "exact": None}


def classify_all(ctx, av):
    rows = []
    acq = ctx["acq"]
    for iso in DATES:
        day = date.fromisoformat(iso)
        for f, blk in av["dates"][iso]["fields"].items():
            for rank, w in enumerate(blk["worst"], 1):
                t = w["ticker"]
                _cnpj, cd, _tickers = members_of(acq, day, t)
                row = {"date": iso, "field": f, "rank": rank, "ticker": t, "reference": w["reference"], "rebuilt": w["rebuilt"],
                       "diff": w["diff"], "si_lag": bool(w["si_lag"]), "previous_value": w["previous_value"], "filing": w["filing"],
                       "cd_cvm": cd}
                if compare.within(f, w["reference"], w["rebuilt"]):
                    res = classify_within(acq, day, t, f, w["diff"], w["reference"], w["rebuilt"], ctx["results"][day].filings.get(t))
                elif f == "DY":
                    res = classify_dy(ctx, iso, day, row)
                else:
                    res = classify_accounting(ctx, iso, day, row)
                rows.append({**row, **res})
    return rows


# ---------------------------------------------------------------- inputs, cache guard, document

def resolve_si(doc, overrides):
    """{date: (path, collected_on)} of the Status Invest CSVs the JSON names, each checked by sha256."""
    entries = {e["date"]: e for e in doc["inputs"]["status_invest"]}
    out = {}
    for iso in DATES:
        e = entries[iso]
        path = overrides.get(iso) or Path(e["path"])
        if not path.is_absolute() and overrides.get(iso) is None:
            path = REPO / path
        if not path.is_file():
            raise SystemExit(f"Status Invest CSV of {iso} not found at {path}; pass --si-{iso} PATH")
        if hashlib.sha256(path.read_bytes()).hexdigest() != e["sha256"]:
            raise SystemExit(f"sha256 of {path} differs from the JSON entry of {iso}")
        out[iso] = (path, date.fromisoformat(e["collected_on"]))
    return out


def cache_digest():
    """Names, sizes and mtimes of every file of the cache: unchanged after the run means nothing was written."""
    h = hashlib.sha256()
    for p in sorted(CACHE.rglob("*")):
        if p.is_file():
            st = p.stat()
            h.update(f"{p.relative_to(CACHE)}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def is_sem_base(r):
    return r["class"] == ERRO_SI and r["motive"].startswith(SEM_BASE_PREFIXES)


def class_counts(rows, iso):
    c = Counter(r["class"] for r in rows if r["date"] == iso)
    sub = Counter("sem base" if is_sem_base(r) else "reproduzido" for r in rows if r["date"] == iso and r["class"] == ERRO_SI)
    return c, sub


def criterion(rows, av):
    crit = av["criterion"]
    part1 = bool(crit["met"])
    pct = [(d, f, st["pct_within"]) for d, blk in av["dates"].items() for f, st in blk["fields"].items()]
    low = min(pct, key=lambda x: x[2])
    out_nc = sum(1 for r in rows if r["class"] == NC)
    erro = sum(1 for r in rows if r["class"] == ERRO_NOSSO)
    part2 = out_nc == 0
    verdict = "cumprido" if (part1 and part2) else "não cumprido"
    p1 = (f"Parte 1 (≥ 70 % nas duas datas): {'cumprida' if part1 else 'não cumprida'}, segundo `adjusted_verdict.criterion` do JSON; "
          f"a menor marca é {low[2]:.2f} % ({low[1]}, {low[0]}).")
    p2 = (f"Parte 2 (10 piores conferidos e explicados): {'cumprida' if part2 else 'não cumprida'}; "
          f"{out_nc} linhas fora da tolerância sem classe (`não classificado`) nas duas datas.")
    line = f"**Critério de liberação (#64): {verdict}.** Erro nosso: {erro} linhas nas duas datas."
    return line, p1, p2, verdict, out_nc, erro


HEADER = """# Auditoria dos piores casos do veredito ajustado (#64)

Este documento classifica os 10 piores casos de cada indicador do veredito ajustado, nas duas datas (AUD-01, DEC-2). Os números são os de `docs/validation/pit-rebuild-stocks.json` (`adjusted_verdict.dates[d].fields[f].worst`): Status Invest (SI), recalculado e diferença. As evidências foram lidas do cache `.cache/pit-rebuild` (demonstrativos CVM, composição de capital, proventos da B3 e do Yahoo, cotações COTAHIST) e dos dois CSV do Status Invest que o JSON cita (`inputs.status_invest`, com sha256).

O documento é gerado por `scripts/pit_rebuild_audit.py`, que lê o cache e o JSON e só grava o arquivo de `--out`:

```
python3 scripts/pit_rebuild_audit.py --out docs/validation/pit-rebuild-stocks-audit.md
```

## Como uma linha é lida

- **Classe** (uma por linha, AUD-02):
  - `erro nosso`: o nosso valor contradiz a fonte ou a regra A-18 (contagem de ações com fator sem suporte da B3).
  - `erro do Status Invest`: o SI traz um número que o cache não sustenta. Há duas origens, separadas no resumo: (a) `reproduzido`: uma visão de um demonstrativo do cache, ou um subconjunto dos proventos da B3, reproduz o valor do SI; cita-se o mais antigo (defasagem B-9 e demonstrativo recebido depois de d entram aqui); (b) `sem base`: nenhuma visão de nenhum demonstrativo do cache, nem subconjunto de proventos, reproduz o valor do SI. A regra de (b) é a decisão do usuário sobre o AC9 (linha fora da tolerância sem base no cache vira `erro do Status Invest`, como na conferência da #59).
  - `diferença de método`: o SI usa outra definição que reproduz o valor dele (contagem total da B3 de R, LL total 3.11, PL total 2.03).
  - `dentro da tolerância`: a linha entra na lista dos piores porque o campo tem menos de 10 linhas fora da tolerância; não há erro a explicar, e a linha não é um dos três erros.
  - `não classificado`, com o motivo: linha fora da tolerância sem demonstrativo no cache para conferir. Nenhuma linha desta versão fica nessa classe.
- **Ordem das regras** (a primeira que vale):
  1. dentro da tolerância (CMP-01);
  2. proventos (DY): decisão manual de 15 linhas (`DY_DECISIONS` no script, com os registros da B3 citados); nas demais, subconjunto que reproduz a duas casas, entre todos os registros da B3 do cache da classe do ticker (PN para 4, 5 e 6, ON para 3), sobre o fechamento de d ou o de d − 1; sem subconjunto, `sem base`;
  3. defasagem do filing anterior (B-9, LAG-01): `erro do Status Invest`;
  4. demonstrativo recebido depois de d que reproduz o valor, lido em R (contagem ou contas): `erro do Status Invest`, citado o mais antigo que reproduz;
  5. fator de contagem sem suporte da B3: a contagem em d passa de duas vezes a contagem B3 de R, sem suspeita (SHR-05): `erro nosso`;
  6. contagem total da B3 de R reproduz o valor e a contagem arquivada em d com eventos é no máximo duas vezes a da B3 (sem suspeita SHR-05): `diferença de método`, salvo se um demonstrativo do cache, diferente do usado em d, reproduz o valor como arquivado (contagem ou demonstrativo inteiro): então `erro do Status Invest`, citado o mais antigo (caso EGIE3 PSR da #64);
  7. lucro total 3.11 ou patrimônio total 2.03 reproduz o valor e difere do nosso em mais de 0,01: `diferença de método`;
  8. demonstrativo do cache, diferente do usado em d, que reproduz o valor como arquivado: `erro do Status Invest`, citado o mais antigo;
  9. demonstrativo do cache que reproduz o valor só com o total da LL ou do PL: `erro do Status Invest` (visão de método), citado o mais antigo;
  10. preço de d − 1 reproduz pelo menos dois campos de contagem: `erro do Status Invest`;
  11. o resto: `sem base`.
- **Reproduzir**: um demonstrativo reproduz o valor do SI quando, tratado como o filing mais novo do ticker, o valor recalculado fica dentro da tolerância do relatório (CMP-01: 5 % para os relativos, 0,5 p.p. para os percentuais; `compare.within`). Para cada versão de cada DFP e ITR do cache são testadas: o demonstrativo inteiro, como arquivado; com a LL 3.11 total; com o PL 2.03 total; com os dois; e só a contagem de ações da versão, com o fator A-18 e os eventos de d. Todas as versões contam, sem limite de idade: um período terminado há mais de 270 dias é lido inteiro (o relatório zera esses valores pela máscara STL-01; a busca não aplica a máscara, porque pergunta de qual demonstrativo o SI tirou o valor). O cache não tem versão recebida depois de R = 2026-11-09 (data de recuperação da B3), então "todas as versões do cache" e "recebidas até R" são o mesmo conjunto. Para um filing recebido depois de d, as leituras da CVM são feitas em R, como no lote 5; a contagem sozinha usa sempre os demais insumos de d. Nessas leituras a guarda de ponto no tempo do relatório (PIT-02, `indicators._assert_point_in_time`) fica desligada; o relatório não a desliga.
- **Mais antigo**: entre os demonstrativos que reproduzem (exceto o que o relatório usou em d, F0), o de menor DT_RECEB; empate: DT_REFER, depois VERSAO. Quando o mesmo demonstrativo reproduz por mais de uma visão, cita-se a primeira na ordem: as contas como arquivadas, depois os totais LL e PL, depois a contagem.
- **Proventos (DY)**: reprodução exata a duas casas (a precisão do SI), com o fechamento de d ou o de d − 1. Os registros são todas as linhas da B3 do cache da classe do ticker, sem janela de data nem exclusão por falta de cotação; cada linha é um registro, e linhas idênticas contam separadamente (o SI somou linhas repetidas nos casos MDNE3 e VLID3). As somas são feitas em unidades de 0,00001 R$; a janela é alargada pelo erro de arredondamento, e cada subconjunto achado é conferido com a soma exata. Quando existe subconjunto que reproduz, a evidência cita um deles, achado pela ordem das datas-ex (registros mais novos ficam de fora quando não são necessários); o motivo não conta os subconjuntos.
- **Demonstrativos conferidos**: cada empresa com linha `erro do Status Invest` sem base ou reproduzida, ou `diferença de método` pela contagem da B3, tem a lista completa dos demonstrativos conferidos na seção `Demonstrativos conferidos`, com o número arquivado de cada conta do campo e a composição de ações. Uma versão marcada `(só o índice no cache)` tem a linha do índice (DT_RECEB), mas nenhuma linha de BPA, BPP, DRE ou composição no cache: nos zips da CVM do cache, as linhas de demonstrativo de alguns períodos vêm só para a versão mais nova. Essa versão não reproduz nada e não é contada.
- **Defasagem**: `sim` quando a linha é `si_lag` no JSON (B-9): o filing anterior reproduz o SI dentro da tolerância.
- **Evidências (B-10)**, separadas por `; `:
  - `CVM <DFP|ITR> <DT_REFER> v<VERSAO> rec <DT_RECEB>: <conta> = <valor>`: o número arquivado, em R$, na coluna ÚLTIMO (em ITR, o acumulado do ano); `QT_ACAO_*` é a composição de ações do mesmo demonstrativo;
  - `B3 evento <ISIN> <label> <lastDatePrior> ×<multiplicador>`: evento de ações (desdobramento, grupamento, bonificação) do registro da empresa na B3;
  - `COTAHIST <ticker> <sessão> <fechamento>`: cotação de fechamento na sessão usada;
  - `B3 proventos <arquivo>: <data-ex> <valor>` e `Yahoo <arquivo>: <data-ex> <valor>`: registro de provento, com o arquivo do cache em que está (um registro por linha, sem repetição, A-15);
  - `Conferidos: N demonstrativos (cd_cvm X)`: o total de demonstrativos conferidos para a linha sem base; a lista está na seção `Demonstrativos conferidos`.
- **Contagem da B3** (`totalNumberShares`) não é um evento. Quando ela explica a diferença, aparece no motivo com o arquivo `b3_company_<raiz>.json` e a data de recuperação R (2026-11-09).
- **Valores dentro da tolerância**: a lista de piores é o top 10 por diferença. Quando um campo tem menos de 10 linhas fora da tolerância (5 % relativo, 0,5 p.p. nos indicadores percentuais), ela traz linhas com diferença 0,00 ou menor. Essas linhas ficam como `dentro da tolerância` e não contam em `erro nosso` nem em parte 2 do critério.

## Limites declarados

- A lista de piores deixa de fora os tickers sem valor reconstruído (`missing`, com o motivo do JSON; contagem por data, indicador e motivo na seção "Sem valor reconstruído") e, nos campos relativos, os tickers cuja referência do SI é zero. A auditoria não conta os tickers de referência zero: o JSON também não os conta, e são um limite declarado, não uma contagem.
- Nos campos de DY, uma célula em branco do SI é tratada como 0,00 (A-4) e fica no denominador; o JSON conta essas células em `reference_blank_as_zero`.
- O Status Invest é a referência do projeto. Quando uma classe é `erro do Status Invest`, a evidência mostra que o número do SI não sai do cache; ela não prova o que o SI fez internamente.
- `sem base` quer dizer: nenhuma visão de nenhum demonstrativo do cache reproduz o valor, e nenhum subconjunto de proventos reproduz o DY. Não quer dizer que o SI usou uma combinação que não está no cache: combinações de dois filings (uma conta de um e a contagem de outro) não são testadas, e um preço que não está no cache não é testado.
- Reproduzir é comparação dentro da tolerância do relatório, e a busca testa muitas visões. Das {EXACT_N} citações contábeis de reprodução, {EXACT_Y} têm o valor do demonstrativo igual ao do SI a duas casas; as outras {NEAR_N} estão só dentro da tolerância do relatório (5 % relativos, 0,5 p.p.). Essas aproximações são citadas como reprodução pela regra; a contagem de reproduções é um fato do cache, não prova do que o SI fez.
- A reprodução de DY por subconjunto é exata a duas casas. Quando mais de um subconjunto reproduz, a classe diz só que o valor do SI cabe em algum deles, não qual o SI usou; o citado é um deles. A busca usa a grade de 0,00001 R$ (ver Proventos), com confirmação exata de cada subconjunto achado: uma soma em que a grade e a soma exata divergem só numa borda da janela pode não ser achada.
- Versões sem linhas de demonstrativo no cache (só o índice): {INDEX_ONLY} das {ALL_VERSIONS} versões recebidas até R de {N_COMP} empresas com linhas fora da tolerância nas duas datas (sem DY). Empresas e versões são contadas uma vez cada, mesmo com linhas nas duas datas. Elas não reproduzem valor e não entram na contagem de demonstrativos conferidos.
- Algumas linhas de `erro do Status Invest` reproduzem o SI com um filing recebido depois de 2026-08-07, a data de coleta declarada do CSV do Status Invest. Ou o CSV foi coletado depois dessa data, ou o SI usa dado posterior à coleta. O cache não distingue as duas coisas: a classe diz só que o valor do SI depende de um filing recebido depois de d.

"""


def index_only_counts(versions, cnpjs):
    total = index_only = 0
    for cnpj in cnpjs:
        for n in numbers(versions, cnpj):
            total += 1
            index_only += 0 if (n["nums"] or n["composition"] is not None) else 1
    return index_only, total, len(cnpjs)


def render(rows, av, filings, index_counts):
    line, p1, p2, verdict, out_nc, erro = criterion(rows, av)
    io_n, all_n, comp_n = index_counts
    fields = av["fields"]
    exact_y = sum(1 for r in rows if r["exact"] is True)
    exact_n = sum(1 for r in rows if r["exact"] is not None)
    out = [HEADER.replace("{INDEX_ONLY}", str(io_n)).replace("{ALL_VERSIONS}", str(all_n)).replace("{N_COMP}", str(comp_n))
           .replace("{EXACT_N}", str(exact_n)).replace("{EXACT_Y}", str(exact_y)).replace("{NEAR_N}", str(exact_n - exact_y))]
    out.append("## Resumo\n")
    out.append("| Data | erro nosso | erro do Status Invest | diferença de método | dentro da tolerância | não classificado (fora da tolerância) | total |")
    out.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    grand = Counter()
    for d in DATES:
        c, _sub = class_counts(rows, d)
        out.append(f"| {d} | {c[ERRO_NOSSO]} | {c[ERRO_SI]} | {c[METODO]} | {c[WITHIN]} | {c[NC]} | {sum(c.values())} |")
        grand.update(c)
    out.append(f"| **Total** | **{grand[ERRO_NOSSO]}** | **{grand[ERRO_SI]}** | **{grand[METODO]}** | **{grand[WITHIN]}** | **{grand[NC]}** | **{sum(grand.values())}** |")
    out.append("")
    out.append("Dentro de `erro do Status Invest`, a origem da classe:\n")
    out.append("| Data | reproduzido por um demonstrativo ou provento | sem base em nenhum demonstrativo ou provento do cache |")
    out.append("| --- | ---: | ---: |")
    for d in DATES:
        _c, sub = class_counts(rows, d)
        out.append(f"| {d} | {sub['reproduzido']} | {sub['sem base']} |")
    out.append("")
    out.append("## Critério de liberação (#64)\n")
    out.append("A regra de DEC-2 tem duas partes. A parte 2 é cumprida quando cada um dos 10 piores de cada indicador, nas duas datas, está classificado numa das três classes ou está dentro da tolerância. `não classificado` fora da tolerância não explica a linha.\n")
    out.append(line + "\n")
    out.append("- " + p1)
    out.append("- " + p2)
    out.append("")
    n_sem_base = sum(1 for r in rows if is_sem_base(r))
    out.append(f"A parte 2 não separa `erro do Status Invest` reproduzido de `sem base` (decisão do usuário sobre o AC9): as duas são classes explicadas. "
               f"São {n_sem_base} linhas `sem base` (nenhum demonstrativo do cache reproduz o valor do SI, DY inclusive); as contagens por data estão no resumo.\n")
    for d in DATES:
        blk = av["dates"][d]
        rows_d = [r for r in rows if r["date"] == d]
        out.append(f"## {d}\n")
        out.append("### Contagem por indicador\n")
        out.append("| Indicador | erro nosso | erro do Status Invest | diferença de método | dentro da tolerância | não classificado | total |")
        out.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        totals = Counter()
        totn = 0
        for f in fields:
            c = Counter(r["class"] for r in rows_d if r["field"] == f)
            totals.update(c)
            n = len(blk["fields"][f]["worst"])
            totn += n
            out.append(f"| {f} | {c[ERRO_NOSSO]} | {c[ERRO_SI]} | {c[METODO]} | {c[WITHIN]} | {c[NC]} | {n} |")
        out.append(f"| **Total** | **{totals[ERRO_NOSSO]}** | **{totals[ERRO_SI]}** | **{totals[METODO]}** | **{totals[WITHIN]}** | **{totals[NC]}** | **{totn}** |")
        out.append("")
        out.append("### Linhas\n")
        for f in fields:
            st = blk["fields"][f]
            out.append(f"#### {f}\n")
            out.append("| # | Ticker | SI | Recalculado | Diferença | Defasagem | Classe | Motivo | Evidência |")
            out.append("| --- | --- | ---: | ---: | ---: | :---: | --- | --- | --- |")
            for w in st["worst"]:
                r = next(x for x in rows_d if x["field"] == f and x["ticker"] == w["ticker"])
                ev = "; ".join(r["evidence"]) if r["evidence"] else "—"
                lag = "sim" if w["si_lag"] else "não"
                out.append(f"| {r['rank']} | {w['ticker']} | {fmt(w['reference'])} | {fmt(w['rebuilt'])} | {w['diff']:.2f} | {lag} | {r['class']} | {r['motive']} | {ev} |")
            out.append("")
        out.append("### Sem valor reconstruído (AUD-05)\n")
        out.append("Contagem de `adjusted_verdict.dates[d].fields[f].missing` por motivo (o JSON), por indicador. Fora desta lista ficam os tickers de referência zero nos campos relativos (limite declarado acima).\n")
        out.append("| Indicador | Motivo | Contagem |")
        out.append("| --- | --- | ---: |")
        for f in fields:
            miss = Counter(m["reason"] for m in blk["fields"][f]["missing"])
            if not miss:
                out.append(f"| {f} | — | 0 |")
            for reason in sorted(miss):
                out.append(f"| {f} | {reason} | {miss[reason]} |")
        out.append("")
    companies = OrderedDict()
    for r in rows:
        if r["field"] != "DY" and ((r["class"] == ERRO_SI and (is_sem_base(r) or r["rule"].startswith("reproduzido") or r["rule"] == "post-d"))
                                   or (r["class"] == METODO and r["rule"] == "contagem da B3 de R")):
            companies.setdefault(str(r["cd_cvm"]), set()).add(r["field"])
    out.append("## Demonstrativos conferidos\n")
    out.append("Para cada empresa com linha `erro do Status Invest` (sem base ou reproduzida por outro demonstrativo) ou `diferença de método` pela contagem da B3, estão todos os demonstrativos do cache conferidos (DFP e ITR, todas as versões recebidas até 2026-11-09). Cada linha é um demonstrativo: tipo, DT_REFER, VERSAO, DT_RECEB e o número arquivado nas contas dos campos da empresa (ÚLTIMO; em ITR, o acumulado do ano) e a composição de ações (ON, PREF, TOTAL). Um `—` é conta que o demonstrativo não traz.\n")
    for cd in sorted(companies, key=int):
        info = filings[cd]
        flds = companies[cd]
        accs = [a for a in ACCOUNTS if any(a in FIELD_ACCOUNTS[f] for f in flds)]
        share = any(f in SHARE_FIELDS for f in flds)
        out.append(f"### cd_cvm {cd} ({', '.join(info['tickers'])}); {len(info['numbers'])} demonstrativos\n")
        head_cells = ["Demonstrativo", "DT_RECEB"] + accs + (["ON", "PREF", "TOTAL"] if share else [])
        out.append("| " + " | ".join(head_cells) + " |")
        out.append("| " + " | ".join(["---", "---"] + ["---:" for _ in head_cells[2:]]) + " |")
        for n in info["numbers"]:
            label = f"{n['kind']} {n['dt_refer'].isoformat()} v{n['versao']}" + ("" if (n["nums"] or n["composition"] is not None) else " (só o índice no cache)")
            cells = [label, n["dt_receb"].isoformat()]
            cells += [fmt(n["nums"][a]) if a in n["nums"] else "—" for a in accs]
            if share:
                comp = n["composition"]
                cells += ["—", "—", "—"] if comp is None else [fmt(comp[0]), fmt(comp[1]), fmt(comp[2])]
            out.append("| " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out) + "\n", verdict, out_nc, erro


def main(argv=None):
    parser = argparse.ArgumentParser(description="Classify the worst rows of the adjusted verdict and write the #64 audit document.")
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT, help="the document to write (the only file this script writes)")
    parser.add_argument("--si-2026-08-07", type=Path, default=None, help="Status Invest CSV of 2026-08-07 when not at the JSON's path")
    parser.add_argument("--si-2026-11-05", type=Path, default=None, help="Status Invest CSV of 2026-11-05 when not at the JSON's path")
    args = parser.parse_args(argv)
    doc = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    av = doc["adjusted_verdict"]
    si = resolve_si(doc, {"2026-08-07": args.si_2026_08_07, "2026-11-05": args.si_2026_11_05})
    before = cache_digest()
    t0 = time.time()
    cache = cache_mod.Cache(CACHE, offline=True)
    with capturing():
        acq = cli._acquire(cache, DAYS, [si[d][0] for d in DATES], time.perf_counter, [si[d][1] for d in DATES])
        results = {day: cli._evaluate_day(acq, day, previous=True) for day in DAYS}
    print(f"rebuild inputs read: {time.time() - t0:.1f} s", flush=True)

    search_rows = {}
    company = {}  # cnpj -> (cd_cvm, tickers), as the report groups them
    for iso in DATES:
        day = date.fromisoformat(iso)
        search_rows[iso] = [
            {"date": iso, "field": f, "ticker": w["ticker"], "reference": w["reference"], "rebuilt": w["rebuilt"]}
            for f, blk in av["dates"][iso]["fields"].items() if f != "DY"
            for w in blk["worst"] if not compare.within(f, w["reference"], w["rebuilt"])
        ]
        for r in search_rows[iso]:
            cnpj, cd, tickers = members_of(acq, day, r["ticker"])
            company.setdefault(cnpj, (cd, tickers))
    cnpjs = set(company)
    versions = load_versions(acq, cnpjs)
    problems = check_against_store(acq, versions, cnpjs)
    if problems:
        raise SystemExit(f"the reconstructed filings differ from the store: {problems[:5]}")
    print(f"versions loaded for {len(cnpjs)} companies: {time.time() - t0:.1f} s", flush=True)

    reproducers = {}
    with pit_guard_off():
        for iso in DATES:
            found, _companies = search_date(acq, versions, iso, search_rows[iso])
            for (f, t), lst in found.items():
                reproducers[(iso, f, t)] = lst
            print(f"search {iso}: {time.time() - t0:.1f} s", flush=True)
    fut = future_reads(acq, search_rows)
    print(f"post-d reads: {time.time() - t0:.1f} s", flush=True)

    counts = {}
    for cnpj in cnpjs:
        nums = numbers(versions, cnpj)
        n_checked = sum(1 for n in nums if n["nums"] or n["composition"] is not None)
        counts[cnpj] = (n_checked, len(nums) - n_checked)
    suspect = {(iso, row["cd_cvm"]) for iso, blk in doc["dates"].items() for row in blk["mapping"]["share_count_suspect"]}
    by_ticker = defaultdict(list)
    for iso in DATES:
        for f, blk in av["dates"][iso]["fields"].items():
            for w in blk["worst"]:
                by_ticker[(iso, w["ticker"])].append({"field": f, "rebuilt": w["rebuilt"]})
    ctx = {"acq": acq, "results": results, "reproducers": reproducers, "fut": fut, "suspect": suspect,
           "counts": counts, "by_ticker": by_ticker}
    rows = classify_all(ctx, av)

    filings = {str(cd): {"tickers": tickers, "numbers": numbers(versions, cnpj)} for cnpj, (cd, tickers) in company.items()}
    text, verdict, out_nc, erro = render(rows, av, filings, index_only_counts(versions, cnpjs))
    args.out.write_text(text, encoding="utf-8")
    after = cache_digest()
    if after != before:
        raise SystemExit("the cache changed during the run")
    for d in DATES:
        c, sub = class_counts(rows, d)
        print(d, {k: c[k] for k in ORDER}, "reproduzido", sub["reproduzido"], "sem base", sub["sem base"])
    print("rules", dict(Counter(r["rule"] for r in rows)))
    print(f"written {args.out}; criterion {verdict}; out_nc {out_nc}; erro nosso {erro}; {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
