"""The 19 fields the comparison covers and their tolerance class (IND-02, A-2).

The 16 score fields are read from METRIC_GROUPS in py/config.py with ast. They
are neither copied nor imported: importing config creates directories under
data/ and outputs/ (design, Risks).
"""

import ast
from pathlib import Path

from .safety import REPO_ROOT

CONFIG_PATH = REPO_ROOT / "py" / "config.py"
EXTRA_FIELDS = ("PRECO", "VALOR DE MERCADO", "LIQUIDEZ MEDIA DIARIA")
PERCENT_FIELDS = frozenset(
    {
        "DY",
        "ROE",
        "ROA",
        "ROIC",
        "MARG. LIQUIDA",
        "MARGEM EBIT",
        "CAGR RECEITAS 5 ANOS",
        "CAGR LUCROS 5 ANOS",
    }
)


def metric_groups(config_path: Path = CONFIG_PATH) -> dict[str, list[str]]:
    """Return METRIC_GROUPS as written in the config source, without importing it."""
    tree = ast.parse(Path(config_path).read_text(encoding="utf-8"), filename=str(config_path))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and node.value is not None:
            if isinstance(node.target, ast.Name) and node.target.id == "METRIC_GROUPS":
                return ast.literal_eval(node.value)
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "METRIC_GROUPS" for t in node.targets):
                return ast.literal_eval(node.value)
    raise ValueError(f"METRIC_GROUPS is not a literal assignment in {config_path}")


SCORE_FIELDS: tuple[str, ...] = tuple(
    field for fields in metric_groups().values() for field in fields
)
ALL_FIELDS: tuple[str, ...] = SCORE_FIELDS + EXTRA_FIELDS


def tolerance_kind(field: str) -> str:
    """Return "pp" for the percent indicators (A-2) and "relative" for the other fields."""
    if field not in ALL_FIELDS:
        raise ValueError(f"unknown field: {field}")
    return "pp" if field in PERCENT_FIELDS else "relative"
