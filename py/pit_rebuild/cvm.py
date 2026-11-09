"""CVM DFP and ITR filings and the cadastro, read point-in-time (PIT-01, PIT-03, A-6).

A filing is one (CNPJ, DT_REFER). The yearly zips hold every version of a filing in
the index (DT_RECEB per VERSAO) but the statements of the latest version only (A-6).
A filing is usable at d when its latest version was received on or before d, and
every value comes from usable filings. Each value carries the DT_RECEB of the filing
it comes from, so the indicator computation can assert PIT-02.

Statements: a filing is consolidated when it has consolidated income rows (DRE_con),
individual otherwise (_ind). The rows of one statement type are never mixed.

DRE rows of an ITR can repeat a key (year-to-date and three-month). The row with the
earliest DT_INI_EXERC is the fiscal year-to-date, whatever the row order in the file.
Dates are read from their first ten characters, so a time after the date is accepted.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
import csv
import io
from pathlib import Path
import re
import zipfile

from .errors import IntegrityError, MissingPathError

DFP = "DFP"
ITR = "ITR"
CONSOLIDATED = "consolidated"
INDIVIDUAL = "individual"
ULTIMO = "ÚLTIMO"
SCALE = {"MIL": 1000, "UNIDADE": 1}

_INDEX = re.compile(r"^(dfp|itr)_cia_aberta_(\d{4})\.csv$", re.IGNORECASE)
_STATEMENT = re.compile(r"^(dfp|itr)_cia_aberta_(BPA|BPP|DRE)_(con|ind)_(\d{4})\.csv$", re.IGNORECASE)
_COMPOSITION = re.compile(r"^(dfp|itr)_cia_aberta_composicao_capital_(\d{4})\.csv$", re.IGNORECASE)
_INDEX_COLUMNS = ("CNPJ_CIA", "DT_REFER", "VERSAO", "DT_RECEB")
_BALANCE_COLUMNS = ("CNPJ_CIA", "DT_REFER", "VERSAO", "ESCALA_MOEDA", "ORDEM_EXERC", "CD_CONTA", "VL_CONTA")
_DRE_COLUMNS = (*_BALANCE_COLUMNS, "DT_INI_EXERC", "DS_CONTA")
_COMPOSITION_COLUMNS = (
    "CNPJ_CIA",
    "DT_REFER",
    "VERSAO",
    "QT_ACAO_ORDIN_CAP_INTEGR",
    "QT_ACAO_PREF_CAP_INTEGR",
    "QT_ACAO_TOTAL_CAP_INTEGR",
)


@dataclass(frozen=True)
class Filing:
    """The filing a value comes from: kind, period end, DT_RECEB of its latest version, statement."""

    kind: str
    cnpj: str
    dt_refer: date
    dt_receb: date
    versao: int
    statement: str


@dataclass(frozen=True)
class _DreRow:
    dt_ini: date
    value: float
    desc: str


@dataclass
class _Period:
    filing: Filing
    bal: dict[str, float] = field(default_factory=dict)  # account -> ÚLTIMO value
    dre: dict[tuple[str, str], _DreRow] = field(default_factory=dict)  # (account, ORDEM_EXERC) -> row
    composition: tuple[int, int, int] | None = None  # ON, PREF, TOTAL as filed


class FilingStore:
    """The filings of one kind, by CNPJ and period end, as load_filings read them."""

    def __init__(self, kind: str, periods: dict[str, dict[date, _Period]]):
        self.kind = kind
        self._periods = periods

    def period(self, cnpj: str, dt_refer: date) -> _Period | None:
        return self._periods.get(cnpj, {}).get(dt_refer)

    def periods(self, cnpj: str) -> list[_Period]:
        return sorted(self._periods.get(cnpj, {}).values(), key=lambda p: p.filing.dt_refer)


def load_filings(zip_paths: Iterable[Path], kind: str) -> FilingStore:
    """Read the DFP or ITR zips of one kind (PIT-01, A-6, IND-01)."""
    if kind not in (DFP, ITR):
        raise ValueError(f"unknown filing kind: {kind}")
    prefix = kind.lower()
    paths = [Path(p) for p in zip_paths]
    latest = _latest_versions(paths, prefix)
    statements, compositions = _collect_statements(paths, prefix, latest)
    periods: dict[str, dict[date, _Period]] = {}
    for (cnpj, dt_refer), (versao, dt_receb) in latest.items():
        key = (cnpj, dt_refer)
        by_scope = statements.get(key, {})
        statement = CONSOLIDATED if ("con", "DRE") in by_scope else INDIVIDUAL
        scope = "con" if statement == CONSOLIDATED else "ind"
        filing = Filing(kind=kind, cnpj=cnpj, dt_refer=dt_refer, dt_receb=dt_receb, versao=versao, statement=statement)
        period = _Period(filing=filing)
        for stmt in ("BPA", "BPP"):
            for member, row in by_scope.get((scope, stmt), []):
                if row["ORDEM_EXERC"].strip() != ULTIMO:
                    continue
                value = _value(row, member)
                if value is not None:
                    period.bal[row["CD_CONTA"].strip()] = value
        for member, row in by_scope.get((scope, "DRE"), []):
            value = _value(row, member)
            if value is None:
                continue
            account_key = (row["CD_CONTA"].strip(), row["ORDEM_EXERC"].strip())
            dt_ini = _date(row["DT_INI_EXERC"], member)
            current = period.dre.get(account_key)
            if current is None or dt_ini < current.dt_ini:
                period.dre[account_key] = _DreRow(dt_ini=dt_ini, value=value, desc=row["DS_CONTA"].strip())
        if key in compositions:
            period.composition = _composition(*compositions[key])
        periods.setdefault(cnpj, {})[dt_refer] = period
    return FilingStore(kind, periods)


def usable(store: FilingStore, cnpj: str, d: date) -> list[Filing]:
    """The filings of store whose latest version was received on or before d, newest period first."""
    found = [p.filing for p in store.periods(cnpj) if p.filing.dt_receb <= d]
    return sorted(found, key=lambda f: f.dt_refer, reverse=True)


def refiled_after_d(dfp: FilingStore, itr: FilingStore, cnpj: str, d: date) -> bool:
    """True when a newer period than the newest usable filing was received after d (A-6, PIT-03)."""
    newest = _newest_usable(dfp, itr, cnpj, d)
    if newest is None:
        return False
    return any(
        p.filing.dt_refer > newest.filing.dt_refer and p.filing.dt_receb > d
        for store in (dfp, itr)
        for p in store.periods(cnpj)
    )


def row_at(store: FilingStore, cnpj: str, dt_refer: date, account: str, ordem: str = ULTIMO) -> tuple[date, float] | None:
    """(DT_INI_EXERC, value) of the DRE row kept for the key of one filing, or None."""
    period = store.period(cnpj, dt_refer)
    if period is None:
        return None
    row = period.dre.get((account, ordem))
    return None if row is None else (row.dt_ini, row.value)


def load_cadastro(path: Path) -> dict[str, dict[str, str]]:
    """CNPJ -> {cd_cvm, setor_ativ, sit} from the CVM cadastro (A-9, A-10)."""
    path = Path(path)
    if not path.is_file():
        raise MissingPathError(str(path), "cadastro CSV does not exist")
    text = path.read_bytes().decode("latin-1").lstrip("﻿")
    reader = csv.DictReader(io.StringIO(text), delimiter=";")
    _require_columns(reader.fieldnames, ("CNPJ_CIA", "CD_CVM", "SETOR_ATIV", "SIT"), path.name)
    return {
        row["CNPJ_CIA"].strip(): {
            "cd_cvm": row["CD_CVM"].strip(),
            "setor_ativ": row["SETOR_ATIV"].strip(),
            "sit": row["SIT"].strip(),
        }
        for row in reader
    }


def _newest_usable(dfp: FilingStore, itr: FilingStore, cnpj: str, d: date) -> _Period | None:
    candidates = [p for store in (dfp, itr) for p in store.periods(cnpj) if p.filing.dt_receb <= d]
    return max(candidates, key=lambda p: p.filing.dt_refer, default=None)


def _latest_versions(paths: list[Path], prefix: str) -> dict[tuple[str, date], tuple[int, date]]:
    latest: dict[tuple[str, date], tuple[int, date]] = {}
    for path in paths:
        with zipfile.ZipFile(path) as archive:
            for member, full_name in _members(archive):
                match = _INDEX.match(member)
                if not match or match.group(1).lower() != prefix:
                    continue
                for row in _rows(archive, full_name, member, _INDEX_COLUMNS):
                    key = (row["CNPJ_CIA"].strip(), _date(row["DT_REFER"], member))
                    versao = _int(row["VERSAO"], member)
                    dt_receb = _date(row["DT_RECEB"], member)
                    if key not in latest or versao > latest[key][0]:
                        latest[key] = (versao, dt_receb)
    return latest


def _collect_statements(paths: list[Path], prefix: str, latest: dict[tuple[str, date], tuple[int, date]]):
    statements: dict[tuple[str, date], dict[tuple[str, str], list[tuple[str, dict[str, str]]]]] = {}
    compositions: dict[tuple[str, date], tuple[str, dict[str, str]]] = {}
    for path in paths:
        with zipfile.ZipFile(path) as archive:
            for member, full_name in _members(archive):
                statement = _STATEMENT.match(member)
                if statement and statement.group(1).lower() == prefix:
                    stmt, scope = statement.group(2).upper(), statement.group(3).lower()
                    columns = _DRE_COLUMNS if stmt == "DRE" else _BALANCE_COLUMNS
                    for row in _rows(archive, full_name, member, columns):
                        key = (row["CNPJ_CIA"].strip(), _date(row["DT_REFER"], member))
                        if key not in latest or _int(row["VERSAO"], member) != latest[key][0]:
                            continue
                        statements.setdefault(key, {}).setdefault((scope, stmt), []).append((member, row))
                    continue
                composition = _COMPOSITION.match(member)
                if composition and composition.group(1).lower() == prefix:
                    for row in _rows(archive, full_name, member, _COMPOSITION_COLUMNS):
                        key = (row["CNPJ_CIA"].strip(), _date(row["DT_REFER"], member))
                        if key not in latest or _int(row["VERSAO"], member) != latest[key][0]:
                            continue
                        compositions[key] = (member, row)
    return statements, compositions


def _composition(member: str, row: dict[str, str]) -> tuple[int, int, int] | None:
    counts = [
        _count(row[column], member)
        for column in ("QT_ACAO_ORDIN_CAP_INTEGR", "QT_ACAO_PREF_CAP_INTEGR", "QT_ACAO_TOTAL_CAP_INTEGR")
    ]
    if any(count is None for count in counts):
        return None
    return counts[0], counts[1], counts[2]


def _members(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    return [(info.filename.rsplit("/", 1)[-1], info.filename) for info in archive.infolist() if not info.is_dir()]


def _rows(archive: zipfile.ZipFile, full_name: str, member: str, required: tuple[str, ...]) -> list[dict[str, str]]:
    text = archive.read(full_name).decode("latin-1").lstrip("﻿")
    reader = csv.DictReader(io.StringIO(text), delimiter=";")
    _require_columns(reader.fieldnames, required, member)
    return list(reader)


def _require_columns(header: list[str] | None, required: tuple[str, ...], member: str) -> None:
    present = set(header or [])
    for column in required:
        if column not in present:
            raise IntegrityError(member, f"column {column} missing")


def _value(row: dict[str, str], member: str) -> float | None:
    text = row["VL_CONTA"].strip()
    if not text:
        return None
    scale = row["ESCALA_MOEDA"].strip()
    if scale not in SCALE:
        raise IntegrityError(member, f"unknown ESCALA_MOEDA {scale!r}")
    try:
        number = float(text)
    except ValueError as exc:
        raise IntegrityError(member, f"VL_CONTA is not a number: {text!r}") from exc
    return number * SCALE[scale]


def _count(text: str, member: str) -> int | None:
    text = text.strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError as exc:
        raise IntegrityError(member, f"share count is not a number: {text!r}") from exc


def _int(text: str, member: str) -> int:
    try:
        return int(text.strip())
    except ValueError as exc:
        raise IntegrityError(member, f"VERSAO is not an integer: {text!r}") from exc


def _date(text: str, member: str) -> date:
    try:
        return date.fromisoformat(text.strip()[:10])
    except ValueError as exc:
        raise IntegrityError(member, f"date is not YYYY-MM-DD: {text!r}") from exc
