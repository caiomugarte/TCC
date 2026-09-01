#!/usr/bin/env python3
"""Refresh Status Invest FII fundamentals by segment."""

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


RAW_DATA_FILE = PY_ROOT.parent / "data" / "raw" / "status_invest_fii.csv"

SOURCE_COLUMNS = (
    "TICKER",
    "PRECO",
    "ULTIMO DIVIDENDO",
    "DY",
    "VALOR PATRIMONIAL COTA",
    "P/VP",
    "LIQUIDEZ MEDIA DIARIA",
    "PERCENTUAL EM CAIXA",
    "CAGR DIVIDENDOS 3 ANOS",
    "CAGR VALOR COTA 3 ANOS",
    "PATRIMONIO",
    "N COTISTAS",
    "GESTAO",
    "N COTAS",
)
OUTPUT_COLUMNS = SOURCE_COLUMNS + ("SETOR",)

# Verified against Status Invest's current FII advanced-search export.
FII_SEGMENTS = (
    (23, "Incorporações"),
    (62, "Serviços Financeiros Diversos"),
    (87, "Fundo de Desenvolvimento"),
    (88, "Fundo de Fundos"),
    (89, "Hospitalar"),
    (90, "Hotéis"),
    (91, "Agências de Bancos"),
    (92, "Lajes Corporativas"),
    (93, "Varejo"),
    (94, "Imóveis Comerciais - Outros"),
    (95, "Imóveis Industriais e Logísticos"),
    (96, "Misto"),
    (97, "Papéis"),
    (98, "Shoppings"),
    (99, "Educacional"),
    (100, "Indefinido"),
    (103, "Imóveis Residenciais"),
    (108, "Logística"),
)

EXPORT_URL = "https://statusinvest.com.br/category/AdvancedSearchResultExport"
REFERER = "https://statusinvest.com.br/fundos-imobiliarios/busca-avancada"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36"
)
EMPTY_VALUES = {"", "-", "--", "nan", "null", "n/a"}
HEADER_ALIASES = {
    "CAGR VALOR CORA 3 ANOS": "CAGR VALOR COTA 3 ANOS",
}


class StatusInvestFiiFetchError(RuntimeError):
    """Raised when an FII segment cannot produce valid data."""


def _header_name(value: str) -> str:
    normalized = value.replace("\ufeff", "").strip().upper()
    return HEADER_ALIASES.get(normalized, normalized)


def _number(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if text.casefold() in EMPTY_VALUES:
        return ""
    if "," in text:
        text = text.replace(".", "").replace(",", ".")
    try:
        float(text)
    except ValueError as exc:
        raise StatusInvestFiiFetchError(f"invalid numeric value: {value!r}") from exc
    return text


def parse_segment_csv(
    payload: Union[bytes, str],
    segment_name: str,
) -> List[Dict[str, str]]:
    """Parse one FII export, normalize values, and attach `SETOR`."""

    if isinstance(payload, bytes):
        try:
            text = payload.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = payload.decode("cp1252")
    else:
        text = payload.lstrip("\ufeff")

    lines = text.splitlines()
    if not lines:
        raise StatusInvestFiiFetchError(f"empty CSV for segment {segment_name}")
    delimiter = ";" if ";" in lines[0] else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    if not reader.fieldnames:
        raise StatusInvestFiiFetchError(f"empty CSV for segment {segment_name}")

    names = [_header_name(name) for name in reader.fieldnames]
    missing = [column for column in SOURCE_COLUMNS if column not in names]
    if missing:
        raise StatusInvestFiiFetchError(
            f"segment {segment_name} missing columns: {', '.join(missing)}"
        )

    rows = []
    for raw in reader:
        normalized = {
            _header_name(key): value for key, value in raw.items() if key
        }
        ticker = str(normalized.get("TICKER", "")).strip().upper()
        if not ticker:
            raise StatusInvestFiiFetchError(
                f"segment {segment_name} contains a blank ticker"
            )
        row = {"TICKER": ticker}
        for column in SOURCE_COLUMNS[1:]:
            if column == "GESTAO":
                row[column] = str(normalized.get(column, "")).strip()
            else:
                row[column] = _number(normalized.get(column))
        row["SETOR"] = segment_name
        rows.append(row)

    if not rows:
        raise StatusInvestFiiFetchError(f"segment {segment_name} returned no FIIs")
    return rows


def validate_rows(rows: List[Mapping[str, str]]) -> None:
    if not rows:
        raise StatusInvestFiiFetchError("no FII rows downloaded")
    missing = [column for column in OUTPUT_COLUMNS if column not in rows[0]]
    if missing:
        raise StatusInvestFiiFetchError(f"output missing columns: {', '.join(missing)}")
    duplicates = [
        ticker
        for ticker, count in Counter(row["TICKER"] for row in rows).items()
        if count > 1
    ]
    if duplicates:
        raise StatusInvestFiiFetchError(
            f"duplicate FII tickers across segments: {', '.join(duplicates[:5])}"
        )


def fetch_segment(
    segment_id: int,
    segment_name: str,
    timeout: int = 30,
) -> List[Dict[str, str]]:
    search = json.dumps({"Segment": str(segment_id)}, separators=(",", ":"))
    url = f"{EXPORT_URL}?{urlencode({'search': search, 'CategoryType': '2'})}"
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
        raise StatusInvestFiiFetchError(
            f"could not fetch FII segment {segment_name}: {exc}"
        ) from exc
    return parse_segment_csv(payload, segment_name)


def write_dataset(rows: List[Mapping[str, str]], output_path: Path) -> None:
    """Atomically replace the FII raw dataset after complete validation."""

    validate_rows(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
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
    for index, (segment_id, segment_name) in enumerate(FII_SEGMENTS, start=1):
        rows = fetch_segment(segment_id, segment_name, timeout)
        all_rows.extend(rows)
        print(f"[{index}/{len(FII_SEGMENTS)}] {segment_name}: {len(rows)} FIIs")
    validate_rows(all_rows)
    write_dataset(all_rows, output_path)
    print(f"Saved {len(all_rows)} FIIs to {output_path}")
    return len(all_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh Status Invest FII fundamentals")
    parser.add_argument("--output", type=Path, default=RAW_DATA_FILE)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()
    try:
        refresh(args.output, args.timeout)
    except StatusInvestFiiFetchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
