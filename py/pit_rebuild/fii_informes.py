"""CVM FII monthly reports (informe mensal estruturado) read point in time (issue #60).

The yearly zip of dados.cvm.gov.br/dados/FII/DOC/INF_MENSAL/DADOS/ holds three CSVs keyed by
(CNPJ_Fundo_Classe, Data_Referencia, Versao): geral (delivery date, ISIN, segment, management),
complemento (holders, equity, shares, net asset value per share) and ativo_passivo (liquidity
items). Each file carries only the newest version of a report, so a report whose newest version
was delivered after d is not known at d, and the fund falls back to its previous report.
"""

from bisect import bisect_right
from collections.abc import Iterable
import csv
from dataclasses import dataclass, field
from datetime import date
import io
from pathlib import Path
import zipfile

from .errors import IntegrityError

COVERAGE_COLUMNS = (
    "Total_Numero_Cotistas",
    "Patrimonio_Liquido",
    "Cotas_Emitidas",
    "Valor_Patrimonial_Cotas",
    "Total_Necessidades_Liquidez",
    "Segmento_Atuacao",
    "Tipo_Gestao",
    "Codigo_ISIN",
)


@dataclass(frozen=True)
class Informe:
    """One monthly report of one fund, newest version only."""

    cnpj: str
    ref: date
    version: int
    delivered: date
    name: str
    isin: str
    segment: str
    management: str
    holders: float | None = None
    assets: float | None = None
    equity: float | None = None
    shares: float | None = None
    nav_per_share: float | None = None
    liquidity: float | None = None  # Total_Necessidades_Liquidez
    cash: float | None = None  # Disponibilidades

    def label(self) -> str:
        """How the report names the report it used: reference month, version and delivery date."""
        return f"informe {self.ref.strftime('%Y-%m')} v{self.version}, entregue em {self.delivered.isoformat()}"


@dataclass
class Informes:
    by_fund: dict[str, list[Informe]] = field(default_factory=dict)  # sorted by delivery date, then reference month
    first_year: dict[str, int] = field(default_factory=dict)
    by_isin: dict[str, list[tuple[date, str]]] = field(default_factory=dict)  # ISIN -> (delivery date, CNPJ)


def load(zip_paths: Iterable[Path]) -> Informes:
    """Read the yearly zips; a report missing from complemento or ativo_passivo keeps None in those values."""
    geral: dict[tuple[str, str, str], dict[str, str]] = {}
    complemento: dict[tuple[str, str, str], dict[str, str]] = {}
    ativo: dict[tuple[str, str, str], dict[str, str]] = {}
    first_year: dict[str, int] = {}
    for path in zip_paths:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                name = info.filename.lower()
                if "_geral_" in name:
                    target = geral
                elif "_complemento_" in name:
                    target = complemento
                elif "_ativo_passivo_" in name:
                    target = ativo
                else:
                    continue
                with archive.open(info) as raw:
                    for row in csv.DictReader(io.TextIOWrapper(raw, encoding="latin-1"), delimiter=";"):
                        key = (row["CNPJ_Fundo_Classe"].strip(), row["Data_Referencia"].strip(), row["Versao"].strip())
                        target[key] = row
                        _note_coverage(first_year, row)
    by_fund: dict[str, list[Informe]] = {}
    for key, row in geral.items():
        cnpj, ref, version = key
        comp = complemento.get(key, {})
        assets_row = ativo.get(key, {})
        try:
            informe = Informe(
                cnpj=cnpj,
                ref=date.fromisoformat(ref),
                version=int(version),
                delivered=date.fromisoformat(row["Data_Entrega"].strip()),
                name=(row.get("Nome_Fundo_Classe") or "").strip(),
                isin=(row.get("Codigo_ISIN") or "").strip(),
                segment=(row.get("Segmento_Atuacao") or "").strip(),
                management=(row.get("Tipo_Gestao") or "").strip(),
                holders=_number(comp.get("Total_Numero_Cotistas")),
                assets=_number(comp.get("Valor_Ativo")),
                equity=_number(comp.get("Patrimonio_Liquido")),
                shares=_number(comp.get("Cotas_Emitidas")),
                nav_per_share=_number(comp.get("Valor_Patrimonial_Cotas")),
                liquidity=_number(assets_row.get("Total_Necessidades_Liquidez")),
                cash=_number(assets_row.get("Disponibilidades")),
            )
        except ValueError as exc:
            raise IntegrityError(cnpj, f"informe mensal {ref} v{version} has an invalid date: {exc}") from exc
        by_fund.setdefault(cnpj, []).append(informe)
    by_isin: dict[str, list[tuple[date, str]]] = {}
    for reports in by_fund.values():
        reports.sort(key=lambda item: (item.delivered, item.ref))
        for item in reports:
            if item.isin:
                by_isin.setdefault(item.isin, []).append((item.delivered, item.cnpj))
    return Informes(by_fund=by_fund, first_year=first_year, by_isin=by_isin)


def known(informes: Informes, cnpj: str, d: date) -> list[Informe]:
    """The fund's reports delivered on or before d."""
    reports = informes.by_fund.get(cnpj, [])
    return reports[: bisect_right([item.delivered for item in reports], d)]


def latest(informes: Informes, cnpj: str, d: date) -> Informe | None:
    """The report of the newest reference month among those delivered on or before d."""
    reports = known(informes, cnpj, d)
    return max(reports, key=lambda item: item.ref) if reports else None


def delivered_after(informes: Informes, cnpj: str, d: date) -> list[date]:
    """Reference months newer than the report at d whose newest version was delivered after d."""
    current = latest(informes, cnpj, d)
    floor = current.ref if current else date.min
    return sorted(item.ref for item in informes.by_fund.get(cnpj, []) if item.delivered > d and item.ref > floor)


def months_back(informes: Informes, report: Informe, months: int, d: date) -> Informe | None:
    """The report of the same fund `months` before report.ref, if it was delivered on or before d."""
    total = report.ref.year * 12 + report.ref.month - 1 - months
    target = date(total // 12, total % 12 + 1, 1)
    return next((item for item in known(informes, report.cnpj, d) if item.ref == target), None)


def funds_of_isin(informes: Informes, isin: str, d: date) -> list[str]:
    """Every CNPJ with a report delivered on or before d that gives this ISIN, sorted."""
    return sorted({cnpj for delivered, cnpj in informes.by_isin.get(isin, []) if delivered <= d})


def first_years(informes: Informes) -> dict[str, int]:
    """The first report year in which each COVERAGE_COLUMNS column has a value for some fund."""
    return dict(informes.first_year)


def _note_coverage(first_year: dict[str, int], row: dict[str, str]) -> None:
    year = int(row["Data_Referencia"][:4])
    for column in COVERAGE_COLUMNS:
        if (row.get(column) or "").strip() and year < first_year.get(column, 10_000):
            first_year[column] = year


def _number(cell: str | None) -> float | None:
    text = (cell or "").strip()
    if not text:
        return None
    if "," in text:
        text = text.replace(".", "").replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None
