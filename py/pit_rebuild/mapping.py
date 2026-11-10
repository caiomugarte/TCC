"""Ticker to CVM company, units, financial group and the tickers whose quotes are read (MAP-01..MAP-03, A-9, A-10, A-17).

The primary source of the mapping is the FCA valor mobiliario table: the row with the
latest Data_Referencia on or before d wins, over every loaded year (A-9, A-19). A ticker
absent from every FCA year maps through the B3 company record of its root (first four
characters), which gives the codeCVM; the CNPJ comes from the cadastro.

Reasons of an unmapped ticker (MAP-01): not_in_fca_or_b3 (no FCA row and no B3 record
for its root), no_cvm_code (the company has no CVM code, or the code is not in the
cadastro), unit_composition_unparsed (a unit whose Composicao_BDR_Unit is not read).
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
import csv
import io
import re
import zipfile
from pathlib import Path

from .dividends import B3Company
from .errors import IntegrityError

UNIT = "Units"
FINANCIAL_SETORES = frozenset(
    {
        "Bancos",
        "Emp. Adm. Part. - Bancos",
        "Seguradoras e Corretoras",
        "Emp. Adm. Part. - Seguradoras e Corretoras",
        "Intermediação Financeira",
        "Emp. Adm. Part. - Intermediação Financeira",
        "Arrendamento Mercantil",
    }
)  # A-10: the cadastro SETOR_ATIV values of the financial group
DRE_FINANCIAL_TERM = "Intermediação Financeira"  # A-10: DRE 3.01 description
SI_FINANCIAL_SETOR = "Financeiro e Outros"  # Status Invest SETOR, too coarse to decide a mapped ticker (A-10)
_VALOR_MOBILIARIO = re.compile(r"^fca_cia_aberta_valor_mobiliario_(\d{4})\.csv$", re.IGNORECASE)
_COMPOSITION_SPLIT = re.compile(r"\s*(?:/|\+|(?<!\S)[eE](?!\S))\s*")  # A-9 (F2): "/", "+" or a standalone "e"/"E"
_COMPOSITION_PART = re.compile(r"(\d+)\s*(.+)")  # a count, then a share class or a ticker
_CLASS_WORDS = (  # A-9 share classes; the words match case-insensitively
    (re.compile(r"ON|(?:AÇÃO|AÇÕES) ON|(?:AÇÃO|AÇÕES) ORDINÁRIAS?", re.IGNORECASE), "ON"),
    (re.compile(r"PN|PNS|(?:AÇÃO|AÇÕES) PN|(?:AÇÃO|AÇÕES) PREFERENCIAIS?", re.IGNORECASE), "PN"),
    (re.compile(r"PNA", re.IGNORECASE), "PNA"),
    (re.compile(r"PNB", re.IGNORECASE), "PNB"),
)
_TICKER = re.compile(r"^[A-Z]{3,4}\d{1,2}$")  # a ticker key stays as written; 3 letters too: FCA "ENG3" (T22)
_REQUIRED = ("CNPJ_Companhia", "Data_Referencia", "Versao", "Valor_Mobiliario", "Codigo_Negociacao", "Composicao_BDR_Unit")


@dataclass(frozen=True)
class FcaRow:
    ticker: str
    cnpj: str
    valor_mobiliario: str
    composition: str | None
    data_referencia: date
    versao: int


@dataclass(frozen=True)
class Mapped:
    cnpj: str
    cd_cvm: str
    via: str  # "fca" or "b3"
    composition: dict[str, int] | None = None  # units only: class or ticker -> count per unit


@dataclass(frozen=True)
class Unmapped:
    reason: str


class FcaIndex:
    """Every FCA row of the loaded years, by ticker and by company."""

    def __init__(self, rows: Iterable[FcaRow]):
        self._by_ticker: dict[str, list[FcaRow]] = {}
        self._tickers_by_cnpj: dict[str, set[str]] = {}
        for row in rows:
            self._by_ticker.setdefault(row.ticker, []).append(row)
            self._tickers_by_cnpj.setdefault(row.cnpj, set()).add(row.ticker)
        for rows_of_ticker in self._by_ticker.values():
            rows_of_ticker.sort(key=lambda row: (row.data_referencia, row.versao))

    def latest(self, ticker: str, d: date) -> FcaRow | None:
        """The row with the latest Data_Referencia on or before d (highest VERSAO on a tie)."""
        candidates = [row for row in self._by_ticker.get(ticker, []) if row.data_referencia <= d]
        return candidates[-1] if candidates else None

    def tickers_of(self, cnpj: str) -> set[str]:
        return set(self._tickers_by_cnpj.get(cnpj, set()))


def load_fca(zip_paths: Iterable[Path]) -> FcaIndex:
    """Read the valor mobiliario member of every FCA zip given (A-9)."""
    rows: list[FcaRow] = []
    for path in zip_paths:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                member = info.filename.rsplit("/", 1)[-1]
                if info.is_dir() or not _VALOR_MOBILIARIO.match(member):
                    continue
                text = archive.read(info).decode("latin-1").lstrip("﻿")
                reader = csv.DictReader(io.StringIO(text), delimiter=";")
                for column in _REQUIRED:
                    if column not in (reader.fieldnames or []):
                        raise IntegrityError(member, f"column {column} missing")
                for cells in reader:
                    ticker = cells["Codigo_Negociacao"].strip()
                    if not ticker:
                        continue
                    rows.append(
                        FcaRow(
                            ticker=ticker,
                            cnpj=cells["CNPJ_Companhia"].strip(),
                            valor_mobiliario=cells["Valor_Mobiliario"].strip(),
                            composition=cells["Composicao_BDR_Unit"].strip() or None,
                            data_referencia=_date(cells["Data_Referencia"], member),
                            versao=_int(cells["Versao"], member),
                        )
                    )
    return FcaIndex(rows)


def parse_composition(text: str | None) -> dict[str, int] | None:
    """Read a unit's composition by A-9's grammar (F2).

    "1 ON / 2 PN" -> {"ON": 1, "PN": 2}; "1 KLBN3 + 4 KLBN4" -> {"KLBN3": 1, "KLBN4": 4};
    "2 ações preferenciais e 1 ação ordinária" -> {"PN": 2, "ON": 1}. Parts are split on "/", "+" or a
    standalone "e"/"E"; repeated keys add up; any other part makes the whole composition None.
    """
    if text is None or not text.strip():
        return None
    composition: dict[str, int] = {}
    for part in _COMPOSITION_SPLIT.split(text.strip()):
        match = _COMPOSITION_PART.fullmatch(part.strip())
        if match is None:
            return None
        key = _class_key(match.group(2))
        if key is None:
            return None
        composition[key] = composition.get(key, 0) + int(match.group(1))
    return composition


def _class_key(text: str) -> str | None:
    """The key a composition part names: a ticker as written, or the share class of its words; None otherwise."""
    words = " ".join(text.split())
    if _TICKER.match(words):
        return words
    for pattern, key in _CLASS_WORDS:
        if pattern.fullmatch(words):
            return key
    return None


def map_tickers(
    eligible: Iterable[str],
    fca: FcaIndex,
    cadastro: Mapping[str, Mapping[str, str]],
    b3_companies: Mapping[str, B3Company | None],
    d: date,
) -> dict[str, Mapped | Unmapped]:
    """Map each eligible ticker to its company as of d, or give the reason it is not mapped (MAP-01)."""
    cd_cvm_to_cnpj = {info["cd_cvm"]: cnpj for cnpj, info in cadastro.items() if info["cd_cvm"]}
    mapping: dict[str, Mapped | Unmapped] = {}
    for ticker in sorted(set(eligible)):
        row = fca.latest(ticker, d)
        if row is not None:
            mapping[ticker] = _from_fca(row, cadastro)
        else:
            mapping[ticker] = _from_b3(b3_companies.get(ticker[:4]), cd_cvm_to_cnpj)
    return mapping


def quote_tickers(
    eligible: Iterable[str],
    mapped: Mapping[str, Mapped | Unmapped],
    fca: FcaIndex,
    si_tickers: Iterable[str],
) -> set[str]:
    """Every ticker whose COTAHIST lines are read (A-17): eligible, the company's other classes and units, same-root SI tickers, unit components."""
    eligible = set(eligible)
    wanted = set(eligible)
    for item in mapped.values():
        if isinstance(item, Mapped):
            wanted |= fca.tickers_of(item.cnpj)
            wanted |= {name for name in (item.composition or {}) if _TICKER.match(name)}
    roots = {ticker[:4] for ticker in eligible}
    wanted |= {ticker for ticker in si_tickers if ticker[:4] in roots}
    return wanted


def classify(
    mapping: Mapped | Unmapped,
    cadastro: Mapping[str, Mapping[str, str]],
    dre_301_desc: str | None,
    si_setor: str | None,
) -> dict[str, str | None]:
    """{"group": financial | non_financial | unclassified, "rule": ...} by A-10.

    A mapped ticker is financial by its cadastro SETOR_ATIV or its DRE 3.01 description; the Status
    Invest SETOR never puts a mapped ticker in the financial group. An unmapped ticker is non-financial
    when its SETOR is not "Financeiro e Outros", and unclassified when it is.
    """
    if isinstance(mapping, Mapped):
        if cadastro.get(mapping.cnpj, {}).get("setor_ativ") in FINANCIAL_SETORES:
            return {"group": "financial", "rule": "setor_ativ"}
        if dre_301_desc and DRE_FINANCIAL_TERM in dre_301_desc:
            return {"group": "financial", "rule": "dre_3_01"}
        return {"group": "non_financial", "rule": None}
    if si_setor == SI_FINANCIAL_SETOR:
        return {"group": "unclassified", "rule": None}
    return {"group": "non_financial", "rule": "si_setor"}


def _from_fca(row: FcaRow, cadastro: Mapping[str, Mapping[str, str]]) -> Mapped | Unmapped:
    info = cadastro.get(row.cnpj)
    if info is None or not info.get("cd_cvm"):
        return Unmapped("no_cvm_code")
    composition = None
    if row.valor_mobiliario == UNIT:
        composition = parse_composition(row.composition)
        if composition is None:
            return Unmapped("unit_composition_unparsed")
    return Mapped(cnpj=row.cnpj, cd_cvm=info["cd_cvm"], via="fca", composition=composition)


def _from_b3(company: B3Company | None, cd_cvm_to_cnpj: Mapping[str, str]) -> Mapped | Unmapped:
    if company is None:
        return Unmapped("not_in_fca_or_b3")
    cnpj = cd_cvm_to_cnpj.get(company.code_cvm or "")
    if cnpj is None:
        return Unmapped("no_cvm_code")
    return Mapped(cnpj=cnpj, cd_cvm=company.code_cvm, via="b3")


def _date(text: str, member: str) -> date:
    try:
        return date.fromisoformat(text.strip()[:10])
    except ValueError as exc:
        raise IntegrityError(member, f"Data_Referencia is not YYYY-MM-DD: {text!r}") from exc


def _int(text: str, member: str) -> int:
    try:
        return int(text.strip())
    except ValueError as exc:
        raise IntegrityError(member, f"Versao is not an integer: {text!r}") from exc
