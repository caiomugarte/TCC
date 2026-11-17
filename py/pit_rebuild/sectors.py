"""The B3 sector classification of a company, from its GetDetail record (#78).

GetDetail carries industryClassification as "setor / subsetor / segmento", e.g.
"Financeiro / Intermediários Financeiros / Bancos". B3 writes the commas of a name as
periods ("Petróleo. Gás e Biocombustíveis"); the names are kept as B3 writes them.
"""

from dataclasses import dataclass
import json


@dataclass(frozen=True)
class Classification:
    setor: str
    subsetor: str
    segmento: str


def parse_b3_detail(payload: bytes) -> Classification | None:
    """The classification of one GetDetail response; None when it has no three-level industryClassification."""
    body = json.loads(payload)
    text = body.get("industryClassification") if isinstance(body, dict) else None
    if not isinstance(text, str):
        return None
    parts = [part.strip() for part in text.split("/")]
    if len(parts) != 3 or not all(parts):
        return None
    return Classification(*parts)
