"""URLs and cache names of the FII sources (issue #60). COTAHIST and Yahoo come from sources.py."""

import re

from .sources import _b64_json

INFORME_LISTING = "https://dados.cvm.gov.br/dados/FII/DOC/INF_MENSAL/DADOS/"
B3_FUNDS_BASE = "https://sistemaswebb3-listados.b3.com.br/fundsProxy/fundsCall"
B3_FII_TYPE = 7  # the typeFund of the FII page of the B3 site
_YEAR = re.compile(r"inf_mensal_fii_(\d{4})\.zip")


def informe_url(year: int) -> str:
    return f"{INFORME_LISTING}inf_mensal_fii_{year}.zip"


def informe_name(year: int) -> str:
    return f"inf_mensal_fii_{year}.zip"


def informe_listing_name() -> str:
    return "listing_inf_mensal_fii.html"


def listed_years(listing: str) -> list[int]:
    """Years of the yearly zips the CVM listing offers, sorted."""
    return sorted({int(year) for year in _YEAR.findall(listing)})


def b3_fund_url(root: str) -> str:
    payload = {"cnpj": "0", "identifierFund": root, "typeFund": B3_FII_TYPE}
    return f"{B3_FUNDS_BASE}/GetListedSupplementFunds/{_b64_json(payload)}"


def b3_fund_name(root: str) -> str:
    return f"b3_fund_{root}.json"
