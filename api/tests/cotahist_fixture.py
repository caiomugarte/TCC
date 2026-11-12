"""Tiny COTAHIST zips for the API tests (issue #63). No network.

Lines follow the real 245-column layout read by py/pit_rebuild/cotahist.py:
TIPREG 1-2, DATA 3-10, CODBDI 11-12, CODNEG 13-24, TPMERC 25-27, ESPECI 40-49,
PREULT 109-121, VOLTOT 171-188, FATCOT 211-217, CODISI 231-242.
"""

from __future__ import annotations

from datetime import date
import io
import zipfile

COLLECTED_ON = date(2026, 11, 12)


def trade(ticker: str, day: date, *, codbdi: str = "02", tpmerc: str = "010") -> str:
    line = [" "] * 245

    def put(start: int, text: str) -> None:
        line[start - 1 : start - 1 + len(text)] = list(text)

    put(1, "01")
    put(3, day.strftime("%Y%m%d"))
    put(11, codbdi)
    put(13, ticker.ljust(12))
    put(25, tpmerc)
    put(40, "ON".ljust(10))
    put(109, "1000".zfill(13))
    put(171, "100000".zfill(18))
    put(211, "1".zfill(7))
    put(231, "BRXXXXCNOR00")
    return "".join(line)


def cotahist_zip(lines: list[str], member: str = "COTAHIST_A2026.TXT") -> bytes:
    header = "00COTAHIST.2026BOVESPA 20261112".ljust(245)
    text = "\r\n".join([header, *lines]) + "\r\n"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, text.encode("latin-1"))
    return buffer.getvalue()
