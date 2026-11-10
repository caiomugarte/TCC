"""The report object of the rebuild and its Markdown (RPT-02, RPT-03, RPT-04, VER-03, VER-04, MAP-02, COV-02, IND-04).

build assembles the object from the per-date statistics and the provenance of the run. The verdicts
are derived here from those statistics, so each verdict comes from one place. render_markdown reads the
object only (RPT-04): a number changed in the object changes the document. encode_json writes the object
as sorted, indented JSON with a final newline.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from datetime import date
from functools import reduce
import json

from . import compare
from .cotahist import LIQUIDITY_SESSIONS
from .fields import ALL_FIELDS, PERCENT_FIELDS
from .indicators import FORMULAS
from .status_invest import ELIGIBILITY

SCHEMA = "pit-rebuild-stocks/v1"
ISSUE = 59
GROUPS = (
    ("non_financial", "não financeiro"),
    ("non_financial_without_unclassified", "não financeiro sem não classificados"),
    ("financial", "financeiro"),
)
DETAILED_GROUPS = ("non_financial", "financial")  # the sem-não-classificados group repeats the first one
CLASS_SCOPE = "16 indicadores de METRIC_GROUPS, grupo não financeiro incluindo os não classificados"
WINDOW = "d − 365 dias < data-ex ≤ d"
CAVEAT = (
    "O Yahoo Finance não tem empresas que saíram da bolsa. O DY de uma empresa extinta "
    "não é medido por essa fonte, e o resultado deve ser lido com esse limite."
)
ELIGIBILITY_SOURCE = ["py/core/preprocessing.py:103-106", "api/app/services/premium_policy.py:56"]
SHARE_SCALE_RULE = (
    "A CVM não informa escala em `composicao_capital`. Cada contagem é multiplicada por 1000 quando o fator "
    "que deixa `N_total × f` mais perto da `totalNumberShares` da B3 é 1000. A contagem da B3 só escolhe a "
    "escala e nunca entra num valor. Sem registro da B3, ou com contagem zero, `f` é 1 e a empresa aparece em "
    "`share_scale.unverified`. A escolha erra só se a contagem mudou 31,6 vezes ou mais entre o arquivamento "
    "e a consulta à B3."
)
NON_PIT_INPUTS: list[dict[str, str]] = [
    {
        "input": "B3 `totalNumberShares`",
        "role": "unit_only",
        "use": "escolhe a escala das contagens (A-18); nunca entra num valor",
        "full_rebuild_check": "escala datada por arquivamento, com a contagem da B3 de cada data",
    },
    {
        "input": "Cadastro CNPJ → CD_CVM",
        "role": "identity",
        "use": "identifica a empresa (A-9)",
        "full_rebuild_check": "empresas canceladas no cadastro",
    },
    {
        "input": "FCA `valor_mobiliario` (ticker → CNPJ, composição da unidade)",
        "role": "identity",
        "use": "mapeia ticker e unidade (A-9, MAP-03); a linha vem da maior Data_Referencia até d",
        "full_rebuild_check": "linhas escolhidas pela data de entrega, não pela data de referência",
    },
    {
        "input": "B3 `codeCVM` da empresa",
        "role": "identity",
        "use": "fallback de mapeamento para tickers fora da FCA (A-9)",
        "full_rebuild_check": "empresas sem registro atual na B3",
    },
    {
        "input": "Cadastro `SETOR_ATIV`",
        "role": "classification",
        "use": "decide o grupo financeiro e a classe (A-10)",
        "full_rebuild_check": "setor datado; reclassificações ao longo dos anos",
    },
    {
        "input": "B3 `tradingName`",
        "role": "source_selection",
        "use": "escolhe o histórico de dividendos buscado (A-15)",
        "full_rebuild_check": "empresas renomeadas e canceladas sem registro atual",
    },
    {
        "input": "Valores de dividendos (B3 `valueCash`, Yahoo)",
        "role": "dated_value_current_amount",
        "use": "a data-ex é PIT (A-16); o valor é o atual",
        "full_rebuild_check": "valores como eram conhecidos em cada data de decisão",
    },
]


def build(
    *,
    dates: Sequence[date],
    tool_commit: str | None,
    status_invest: Sequence[Mapping[str, object]],
    raw_files: Sequence[Mapping[str, object]],
    per_date: Mapping[str, Mapping[str, object]],
    dividends: Mapping[str, Mapping[str, object]],
    coverage: Mapping[str, object],
) -> dict[str, object]:
    """The report object of one run (RPT-02, RPT-03).

    per_date maps each ISO date to {"eligible": int, "mapping": dict, "fields": {field: {group: FieldStats}}}.
    dividends holds by_source, yahoo_reasons and b3_empty_yahoo_failed, each keyed by ISO date.
    """
    iso_dates = [day.isoformat() for day in sorted(dates)]
    if not iso_dates:
        raise ValueError("the report needs at least one decision date")
    if sorted(per_date) != iso_dates:
        raise ValueError("per_date must hold exactly the decision dates of the run")
    dates_out: dict[str, dict[str, object]] = {}
    for iso in iso_dates:
        entry = per_date[iso]
        fields_out: dict[str, dict[str, object]] = {}
        verdicts_out: dict[str, dict[str, str]] = {}
        for field in ALL_FIELDS:
            fields_out[field] = {}
            verdicts_out[field] = {}
            for group, _label in GROUPS:
                stats = entry["fields"][field][group]
                fields_out[field][group] = asdict(stats)
                verdicts_out[field][group] = compare.verdict(stats.pct_within, stats.denominator)
        dates_out[iso] = {
            "eligible": entry["eligible"],
            "mapping": entry["mapping"],
            "fields": fields_out,
            "verdicts": verdicts_out,
        }
    overall = {
        field: {
            group: reduce(compare.overall, [dates_out[iso]["verdicts"][field][group] for iso in iso_dates])
            for group, _label in GROUPS
        }
        for field in ALL_FIELDS
    }
    class_result = compare.class_verdict(overall)
    verdicts = {
        "fields": overall,
        "class": class_result["class"],
        "class_without_unclassified": class_result["class_without_unclassified"],
        "class_differs": class_result["differs"],
        "financial_group": class_result["financial"],
        "class_scope": CLASS_SCOPE,
    }
    yahoo_used = any(dividends["by_source"][iso].get("yahoo", 0) > 0 for iso in iso_dates)
    dividends_out = {
        "source": "secondary" if yahoo_used else "primary",
        "by_source": {iso: dict(dividends["by_source"][iso]) for iso in iso_dates},
        "yahoo_reasons": {iso: dict(dividends["yahoo_reasons"][iso]) for iso in iso_dates},
        "b3_empty_yahoo_failed": {iso: dividends["b3_empty_yahoo_failed"][iso] for iso in iso_dates},
        "window": WINDOW,
        "caveat": CAVEAT,
    }
    parameters = {
        "dates": iso_dates,
        "eligibility": {
            "cap_min": ELIGIBILITY["cap_min"],
            "liq_min": ELIGIBILITY["liq_min"],
            "source": list(ELIGIBILITY_SOURCE),
        },
        "tolerance": {
            "relative": compare.RELATIVE_TOLERANCE_PCT / 100,
            "pp": compare.PP_TOLERANCE,
            "rounding_decimals": compare.ROUNDING_DECIMALS,
        },
        "thresholds": {"reconstruivel": 90, "com_ressalvas": 70},
        "percent_fields": sorted(PERCENT_FIELDS),
        "liquidity_sessions": LIQUIDITY_SESSIONS,
        "formulas": dict(FORMULAS),
        "share_scale_rule": SHARE_SCALE_RULE,
        "non_pit_inputs": [dict(row) for row in NON_PIT_INPUTS],
    }
    return {
        "schema": SCHEMA,
        "issue": ISSUE,
        "tool_commit": tool_commit,
        "parameters": parameters,
        "inputs": {
            "status_invest": sorted((dict(item) for item in status_invest), key=lambda item: str(item["date"])),
            "raw_files": sorted((dict(item) for item in raw_files), key=lambda item: str(item["name"])),
        },
        "dates": dates_out,
        "verdicts": verdicts,
        "dividends": dividends_out,
        "coverage": dict(coverage),
    }


def encode_json(report: Mapping[str, object]) -> bytes:
    """The report as sorted, indented UTF-8 JSON with a final newline (RPT-02)."""
    return (json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def render_markdown(report: Mapping[str, object]) -> str:
    """The Markdown of the report, written from the object alone (RPT-04)."""
    lines: list[str] = ["# Reconstrução PIT dos indicadores de ações", ""]
    lines += _metadata(report)
    lines += _verdict_section(report["verdicts"])
    lines += _parameters_section(report["parameters"])
    lines += _mapping_section(report["dates"])
    lines += _results_section(report["dates"])
    lines += _dividends_section(report["dividends"])
    lines += _coverage_section(report["coverage"])
    lines += _inputs_section(report["inputs"])
    return "\n".join(lines) + "\n"


def _metadata(report: Mapping[str, object]) -> list[str]:
    commit = report["tool_commit"] or "-"
    dates = ", ".join(report["parameters"]["dates"])
    return [
        f"- Esquema: `{report['schema']}` (issue #{report['issue']})",
        f"- Commit da ferramenta: `{commit}`",
        f"- Datas de decisão: {dates}",
        "",
    ]


def _verdict_section(verdicts: Mapping[str, object]) -> list[str]:
    lines = ["## Veredito", ""]
    class_value = f"`{verdicts['class']}`"
    if verdicts["class_differs"]:
        line = (
            f"**Veredito da classe:** {class_value}. Sem os tickers não classificados, "
            f"o veredito seria `{verdicts['class_without_unclassified']}`."
        )
    else:
        line = f"**Veredito da classe:** {class_value}. Os tickers não classificados não mudam o veredito."
    lines += [line, ""]
    lines += [
        f"**Grupo financeiro:** `{verdicts['financial_group']}`. Ele é calculado da mesma forma e não altera a classe.",
        "",
        f"Escopo da classe: {verdicts['class_scope']}.",
        "",
        "### Veredito por indicador",
        "",
    ]
    rows = [
        [_code(field)] + [verdicts["fields"][field][group] for group, _label in GROUPS]
        for field in verdicts["fields"]
    ]
    lines += _table(["indicador"] + [label for _group, label in GROUPS], rows)
    return lines


def _parameters_section(parameters: Mapping[str, object]) -> list[str]:
    elig = parameters["eligibility"]
    tol = parameters["tolerance"]
    thresholds = parameters["thresholds"]
    lines = ["## Parâmetros", ""]
    lines.append(
        f"- Universo elegível: `VALOR DE MERCADO` ≥ {elig['cap_min']} e `LIQUIDEZ MEDIA DIARIA` ≥ "
        f"{elig['liq_min']}. Fontes: {_codes(elig['source'])}."
    )
    lines.append(
        f"- Tolerância: {_num(tol['relative'] * 100, 0)}% relativa; {_num(tol['pp'])} p.p. para os indicadores "
        f"em percentual ({_codes(parameters['percent_fields'])}). A comparação usa o valor arredondado a "
        f"{tol['rounding_decimals']} casas."
    )
    lines.append(
        f"- Veredito por indicador: `reconstruível` com {thresholds['reconstruivel']}% ou mais dentro da "
        f"tolerância; `com ressalvas` com {thresholds['com_ressalvas']}% ou mais; `não reconstruível` abaixo disso."
    )
    lines.append(f"- Liquidez: média de VOLTOT das últimas {parameters['liquidity_sessions']} sessões até d.")
    lines.append(f"- Escala das ações: {parameters['share_scale_rule']}")
    lines += ["", "### Fórmulas", ""]
    lines += _table(["indicador", "fórmula"], [[_code(field), text] for field, text in parameters["formulas"].items()])
    lines += [
        "### Entradas que não são ponto no tempo",
        "",
        "Estas entradas são lidas como foram obtidas, e não como estavam em cada data. O papel de cada uma "
        "limita o que ela pode alterar.",
        "",
    ]
    lines += _table(
        ["entrada", "papel", "uso", "verificação na reconstrução completa"],
        [[row["input"], f"`{row['role']}`", row["use"], row["full_rebuild_check"]] for row in parameters["non_pit_inputs"]],
    )
    return lines


def _mapping_section(dates: Mapping[str, Mapping[str, object]]) -> list[str]:
    lines = ["## Cobertura do mapeamento", ""]
    for iso, entry in dates.items():
        mapping = entry["mapping"]
        fallback = mapping["vm_price_fallback"]
        scale = mapping["share_scale"]
        lines += [
            f"### {iso}",
            "",
            f"- Elegíveis: {entry['eligible']}. Mapeados: {mapping['mapped']}. Não mapeados: {len(mapping['unmapped'])}.",
            f"- Mapeados pela B3 (`mapped_via_b3`): {_list_or_none(mapping['mapped_via_b3'])}.",
            f"- Caíram para o período anterior porque o mais recente chegou depois de d (`refiled_after_d`): "
            f"{_list_or_none(mapping['refiled_after_d'])}.",
            f"- Usaram o preço da outra classe (`vm_price_fallback`): {fallback['count']} empresa(s).",
            f"- Escala de 1000 (`share_scale.x1000`): {_list_or_none(scale['x1000'])}. "
            f"Não verificada (`share_scale.unverified`): {_list_or_none(scale['unverified'])}.",
            "",
            "Tickers não mapeados:",
            "",
        ]
        lines += _table(["ticker", "motivo"], [[_code(row["ticker"]), _code(row["reason"])] for row in mapping["unmapped"]] or [["-", "-"]])
        lines += ["Grupo financeiro, com a regra que classificou cada ticker:", ""]
        lines += _table(["ticker", "regra"], [[_code(row["ticker"]), _code(row["rule"])] for row in mapping["financial"]] or [["-", "-"]])
        lines += ["Tickers não classificados:", ""]
        lines += _table(
            ["ticker", "SETOR do Status Invest", "motivo"],
            [[_code(row["ticker"]), _code(row["si_setor"]), _code(row["reason"])] for row in mapping["unclassified"]] or [["-", "-", "-"]],
        )
        if fallback["companies"]:
            lines += ["Empresas com preço de outra classe:", ""]
            lines += _table(
                ["cd_cvm", "classe sem cotação", "preço de"],
                [[_code(row["cd_cvm"]), _code(row["class_without_quote"]), _code(row["price_from"])] for row in fallback["companies"]],
            )
    return lines


def _results_section(dates: Mapping[str, Mapping[str, object]]) -> list[str]:
    lines = ["## Resultados por data", ""]
    for iso, entry in dates.items():
        rows = []
        for field, groups in entry["fields"].items():
            for group, label in GROUPS:
                stats = groups[group]
                rows.append(
                    [
                        _code(field),
                        label,
                        stats["denominator"],
                        _pct(stats["pct_within"]),
                        _delta(stats["median_diff"], stats["diff_unit"]),
                        _delta(stats["p90_diff"], stats["diff_unit"]),
                        entry["verdicts"][field][group],
                    ]
                )
        lines += [f"### {iso}", ""]
        lines += _table(["indicador", "grupo", "n", "% na tolerância", "mediana", "p90", "veredito"], rows)
        blanks = [
            [_code(field), groups[GROUPS[0][0]]["reference_blank"], groups[GROUPS[0][0]]["reference_blank_with_value"],
             groups[GROUPS[0][0]]["reference_blank_as_zero"]]
            for field, groups in entry["fields"].items()
            if groups[GROUPS[0][0]]["reference_blank"] or groups[GROUPS[0][0]]["reference_blank_as_zero"]
        ]
        if blanks:
            lines += ["Referências em branco no Status Invest (grupo não financeiro):", ""]
            lines += _table(["indicador", "em branco", "com valor reconstruído", "DY tratado como 0,00"], blanks)
        for field, groups in entry["fields"].items():
            for group, label in GROUPS:
                if group in DETAILED_GROUPS:
                    lines += _worst_details(field, label, groups[group])
    return lines


def _worst_details(field: str, label: str, stats: Mapping[str, object]) -> list[str]:
    if not stats["worst"] and not stats["missing"]:
        return []
    title = f"{field}, {label}" + (": 10 piores casos" if stats["worst"] else "")
    lines = ["<details>", f"<summary>{title}</summary>", ""]
    if stats["worst"]:
        rows = [[_code(item["ticker"]), _num(item["reference"]), _num(item["rebuilt"]), _delta(item["diff"], stats["diff_unit"])] for item in stats["worst"]]
        lines += _table(["ticker", "referência", "reconstruído", "diferença"], rows)
    if stats["missing"]:
        missing = ", ".join(f"`{item['ticker']}` ({item['reason']})" for item in stats["missing"])
        lines += [f"Sem valor reconstruído: {missing}.", ""]
    lines += ["</details>", ""]
    return lines


def _dividends_section(dividends: Mapping[str, object]) -> list[str]:
    if dividends["source"] == "secondary":
        source_line = "- Fonte do DY (`source`): `secondary`. Parte dos valores vem do Yahoo Finance."
    else:
        source_line = "- Fonte do DY (`source`): `primary`. Nenhum valor de DY vem do Yahoo Finance."
    lines = ["## Dividendos", "", source_line, f"- Janela: {dividends['window']}.", f"- {dividends['caveat']}", ""]
    lines += ["### Fonte por data", ""]
    lines += _table(
        ["data", "B3", "Yahoo", "sem fonte"],
        [[iso, counts["b3"], counts["yahoo"], counts["none"]] for iso, counts in dividends["by_source"].items()],
    )
    lines += ["### Motivos de troca para o Yahoo", ""]
    lines += _table(
        ["data", "b3_failed", "b3_empty", "b3_incomplete", "b3_short_history"],
        [
            [iso, reasons["b3_failed"], reasons["b3_empty"], reasons["b3_incomplete"], reasons["b3_short_history"]]
            for iso, reasons in dividends["yahoo_reasons"].items()
        ],
    )
    lines += ["Tickers sem dividendos confirmados (`b3_empty_yahoo_failed`):", ""]
    lines += _table(["data", "tickers"], [[iso, count] for iso, count in dividends["b3_empty_yahoo_failed"].items()])
    return lines


def _coverage_section(coverage: Mapping[str, object]) -> list[str]:
    lines = ["## Cobertura histórica e esforço", "", "### Primeiro ano por fonte", ""]
    lines += _table(
        ["fonte", "primeiro ano"],
        [[_code(source), _year_text(info)] for source, info in coverage["earliest_year_by_source"].items()],
    )
    lines += ["### Primeiro ano por indicador", ""]
    lines += _table(
        ["indicador", "primeiro ano"],
        [[_code(field), year] for field, year in coverage["earliest_year_by_field"].items()],
    )
    dy = coverage["dy"]
    lines += ["### Cobertura do DY", ""]
    lines += _table(
        ["fonte", "tickers", "primeira data bruta", "retenção"],
        [
            [_code(source), info["tickers"], info["earliest_raw_date"] or "-", info["retention"]]
            for source, info in dy["by_source"].items()
        ],
    )
    if dy["earliest_year_from"]:
        lines += [f"O DY começa em {dy['earliest_year']}, pela fonte `{dy['earliest_year_from']}`.", ""]
    effort = coverage["effort"]
    lines += ["### Esforço da reconstrução completa", ""]
    if effort["total_projected_seconds"] is None:
        lines.append("- Total projetado: indisponível, porque uma fonte foi medida com 0 bytes.")
    else:
        lines.append(
            f"- Total projetado: {_num(effort['total_projected_seconds'], 0)} s. Carga: "
            f"{_num(effort['load_projected_seconds'], 0)} s. Compute: {_num(effort['compute_projected_seconds'], 0)} s "
            f"({_num(effort['compute_seconds_per_date'])} s por data × {effort['decision_dates']} datas)."
        )
    lines += [f"- Premissa: {effort['assumption']}.", ""]
    lines += _table(
        ["tipo", "segundos", "bytes medidos", "bytes completos", "segundos projetados"],
        [
            [_code(kind), _num(item["seconds"]), item["bytes"], item["full_bytes"], _num_or_none(item["projected_seconds"])]
            for kind, item in effort["load"].items()
        ],
    )
    years = coverage.get("years")
    if years:
        lines += ["### Anos lidos na reconstrução completa", ""]
        lines += _table(["fonte", "primeiro ano", "último ano"], [[_code(source), first, last] for source, (first, last) in years.items()])
    return lines


def _inputs_section(inputs: Mapping[str, object]) -> list[str]:
    lines = ["## Entradas", "", "### Arquivos brutos", ""]
    lines += _table(
        ["arquivo", "URL", "sha256", "bytes"],
        [[item["name"], item["url"], f"`{item['sha256']}`", item["bytes"]] for item in inputs["raw_files"]],
    )
    lines += ["### Status Invest", ""]
    lines += _table(
        ["data", "caminho", "sha256"],
        [[item["date"], item["path"], f"`{item['sha256']}`"] for item in inputs["status_invest"]],
    )
    return lines


def _table(header: Sequence[str], rows: Iterable[Sequence[object]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_cell(item) for item in row) + " |")
    lines.append("")
    return lines


def _code(value: object) -> str:
    return "-" if value is None else f"`{value}`"


def _cell(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "sim" if value else "não"
    if isinstance(value, float):
        return _num(value)
    return str(value).replace("|", "\\|")


def _num(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _num_or_none(value: float | None) -> str:
    return "-" if value is None else _num(value)


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{_num(value)}%"


def _delta(value: float | None, unit: str) -> str:
    if value is None:
        return "-"
    return f"{_num(value)} p.p." if unit == "pp" else f"{_num(value)}%"


def _codes(items: Iterable[str]) -> str:
    return ", ".join(f"`{item}`" for item in items)


def _list_or_none(items: Sequence[str]) -> str:
    return _codes(items) if items else "nenhum"


def _year_text(info: Mapping[str, object]) -> str:
    if info["year"] is None:
        return "-"
    if info["at_probe_floor"]:
        return f"≤ {info['year']} (limite da sondagem)"
    return str(info["year"])
