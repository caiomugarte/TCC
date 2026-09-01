#!/usr/bin/env python3
"""Manually refresh Status Invest equity fundamentals."""

import argparse
import csv
from collections import Counter
import io
import json
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from typing import Dict, List, Mapping, Union
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


PY_ROOT = Path(__file__).resolve().parent
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from config import RAW_DATA_FILE  # noqa: E402


SOURCE_COLUMNS = (
    "TICKER", "PRECO", "DY", "P/L", "P/VP", "P/ATIVOS", "MARGEM BRUTA",
    "MARGEM EBIT", "MARG. LIQUIDA", "P/EBIT", "EV/EBIT",
    "DIVIDA LIQUIDA / EBIT", "DIV. LIQ. / PATRI.", "PSR", "P/CAP. GIRO",
    "P. AT CIR. LIQ.", "LIQ. CORRENTE", "ROE", "ROA", "ROIC",
    "PATRIMONIO / ATIVOS", "PASSIVOS / ATIVOS", "GIRO ATIVOS",
    "CAGR RECEITAS 5 ANOS", "CAGR LUCROS 5 ANOS", "LIQUIDEZ MEDIA DIARIA",
    "VPA", "LPA", "PEG RATIO", "VALOR DE MERCADO",
)
OUTPUT_COLUMNS = SOURCE_COLUMNS + ("SETOR",)

SECTORS = (
    (2, "Consumo Cíclico"),
    (3, "Consumo não Cíclico"),
    (10, "Utilidade Pública"),
    (1, "Bens Industriais"),
    (5, "Materiais Básicos"),
    (4, "Financeiro e Outros"),
    (8, "Tecnologia da Informação"),
    (7, "Saúde"),
    (6, "Petróleo, Gás e Biocombustíveis"),
    (9, "Comunicações"),
)

EXPORT_URL = "https://statusinvest.com.br/category/AdvancedSearchResultExport"
REFERER = "https://statusinvest.com.br/acoes/busca-avancada"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36"
)
EMPTY_VALUES = {"", "-", "--", "nan", "null", "n/a"}


class StatusInvestFetchError(RuntimeError):
    """Raised when a sector cannot produce valid data."""


def _header_name(value: str) -> str:
    return value.replace("\ufeff", "").strip().upper()


def _number(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if text.casefold() in EMPTY_VALUES:
        return ""
    if "," in text:
        text = text.replace(".", "").replace(",", ".")
    try:
        float(text)
    except ValueError as exc:
        raise StatusInvestFetchError(f"invalid numeric value: {value!r}") from exc
    return text


def parse_sector_csv(payload: Union[bytes, str], sector_name: str) -> List[Dict[str, str]]:
    """Parse one Status Invest export and attach its sector."""

    if isinstance(payload, bytes):
        try:
            text = payload.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = payload.decode("cp1252")
    else:
        text = payload.lstrip("\ufeff")

    lines = text.splitlines()
    if not lines:
        raise StatusInvestFetchError(f"empty CSV for sector {sector_name}")
    delimiter = ";" if ";" in lines[0] else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    if not reader.fieldnames:
        raise StatusInvestFetchError(f"empty CSV for sector {sector_name}")

    names = [_header_name(name) for name in reader.fieldnames]
    missing = [column for column in SOURCE_COLUMNS if column not in names]
    if missing:
        raise StatusInvestFetchError(
            f"sector {sector_name} missing columns: {', '.join(missing)}"
        )

    rows = []
    for raw in reader:
        normalized = {_header_name(key): value for key, value in raw.items() if key}
        ticker = str(normalized.get("TICKER", "")).strip().upper()
        if not ticker:
            raise StatusInvestFetchError(f"sector {sector_name} contains a blank ticker")
        row = {"TICKER": ticker}
        for column in SOURCE_COLUMNS[1:]:
            row[column] = _number(normalized.get(column))
        row["SETOR"] = sector_name
        rows.append(row)

    if not rows:
        raise StatusInvestFetchError(f"sector {sector_name} returned no companies")
    return rows


def validate_rows(rows: List[Mapping[str, str]]) -> None:
    if not rows:
        raise StatusInvestFetchError("no rows downloaded")
    missing = [column for column in OUTPUT_COLUMNS if column not in rows[0]]
    if missing:
        raise StatusInvestFetchError(f"output missing columns: {', '.join(missing)}")
    duplicates = [
        ticker for ticker, count in Counter(row["TICKER"] for row in rows).items()
        if count > 1
    ]
    if duplicates:
        raise StatusInvestFetchError(f"duplicate tickers: {', '.join(duplicates[:5])}")


def fetch_sector(sector_id: int, sector_name: str, timeout: int = 30) -> List[Dict[str, str]]:
    search = json.dumps({"Sector": str(sector_id)}, separators=(",", ":"))
    url = f"{EXPORT_URL}?{urlencode({'search': search, 'CategoryType': '1'})}"
    request = Request(
        url,
        headers={
            "Accept": "text/csv,text/plain;q=0.9,*/*;q=0.8",
            "Referer": REFERER,
            "User-Agent": USER_AGENT,
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise StatusInvestFetchError(f"could not fetch {sector_name}: {exc}") from exc
    return parse_sector_csv(payload, sector_name)


def write_dataset(rows: List[Mapping[str, str]], output_path: Path) -> None:
    validate_rows(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(
            "w", encoding="utf-8", newline="", dir=output_path.parent,
            prefix=f".{output_path.name}.", suffix=".tmp", delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            writer = csv.DictWriter(temporary, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()


def refresh(output_path: Path = RAW_DATA_FILE, timeout: int = 30) -> int:
    all_rows = []
    for index, (sector_id, sector_name) in enumerate(SECTORS, start=1):
        rows = fetch_sector(sector_id, sector_name, timeout)
        all_rows.extend(rows)
        print(f"[{index}/{len(SECTORS)}] {sector_name}: {len(rows)} tickers")
    validate_rows(all_rows)
    write_dataset(all_rows, output_path)
    print(f"Saved {len(all_rows)} tickers to {output_path}")
    return len(all_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh Status Invest equity fundamentals")
    parser.add_argument("--output", type=Path, default=RAW_DATA_FILE)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()
    try:
        refresh(args.output, args.timeout)
    except StatusInvestFetchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
