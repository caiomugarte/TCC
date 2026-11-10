"""The report object of the rebuild and its Markdown (RPT-02, RPT-03, RPT-04, VER-03..VER-08, MAP-02, COV-02, IND-04).

build assembles the object from the per-date statistics and the provenance of the run. The official
verdicts are derived here from the official statistics alone, before any sensitivity is read (VER-08);
the sensitivities (A-22) are stored under "sensitivities" with official: false. render_markdown reads
the object only (RPT-04): a number changed in the object changes the document. encode_json writes the
object as sorted, indented JSON with a final newline.
"""

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict
from datetime import date
from functools import reduce
import json
import math

from . import compare
from .cotahist import LIQUIDITY_SESSIONS, QUOTE_CODBDI
from .fields import ALL_FIELDS, PERCENT_FIELDS, SCORE_FIELDS
from .indicators import FORMULAS
from .status_invest import ELIGIBILITY

SCHEMA = "pit-rebuild-stocks/v2"
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
    "A CVM não informa escala em `composicao_capital`. Para cada arquivamento, o fator f ∈ {10^-2, 10^-1, 1, 10, 10^2, 10^3} "
    "é o que deixa `N_total × f` mais perto da `totalNumberShares` da B3, medido pela distância em log10. "
    "A contagem da B3 só escolhe o fator e nunca entra num valor. Sem registro da B3, ou com contagem zero, `f` é 1 "
    "e a empresa aparece em `share_scale.unverified`. As contagens escalonadas são arredondadas. Limite (√10): se a "
    "contagem mudou menos de √10 (3,16 vezes) entre o arquivamento e a consulta à B3, ela fica como arquivada; se mudou "
    "√10 ou mais, `f` absorve a mudança como a potência de dez mais próxima. Isso é exato só para grupamento de 10:1; "
    "um desdobramento de 1:5 dá ×10, errado por 2 vezes, e erra em `d` quando a mudança é posterior a `d`. "
    "O resíduo (`residual`), `N_total × f / totalNumberShares`, de cada empresa mostra cada caso."
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
    {
        "input": "B3 `stockDividends`",
        "role": "dated_event_current_record",
        "use": "ajusta valores de dividendos por desdobramento, bonificação e grupamento (A-20); só eventos com data-ex ≤ d",
        "full_rebuild_check": "eventos societários como eram conhecidos em cada data",
    },
]
SENSITIVITY_TITLE = "Sensibilidades (não são o veredito oficial)"
SENSITIVITY_NOTE = (
    "Estas linhas não são o veredito oficial. O veredito oficial é o da seção Veredito, e nenhuma sensibilidade entra nele."
)
WITHOUT_REFERENCE_STALE_TEXT = (
    "Denominador do veredito oficial sem os tickers de referência obsoleta: elegíveis sem registro de cotação "
    "em [d − 30 dias, d]. Não é o veredito oficial."
)
ROIC_PEG_FIELDS = ("PEG RATIO", "ROIC")  # D-C: the two indicators whose Status Invest formulas are not public
ROIC_PEG_NOTE = (
    "ROIC e PEG RATIO falham porque as fórmulas do Status Invest para eles não são públicas (A-8 é uma leitura), "
    "não por falta de dados."
)
DIVIDEND_EXTRAS = ("b3_filters", "split_adjust_no_isin", "slash_names")  # carried from the run when given (r5)


def build(
    *,
    dates: Sequence[date],
    tool_commit: str | None,
    status_invest: Sequence[Mapping[str, object]],
    raw_files: Sequence[Mapping[str, object]],
    per_date: Mapping[str, Mapping[str, object]],
    dividends: Mapping[str, object],
    coverage: Mapping[str, object],
    filings_cutoffs: Mapping[object, object] | None = None,
    sensitivities: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """The report object of one run (RPT-02, RPT-03, VER-05..VER-08).

    per_date maps each ISO date to {"eligible": int, "mapping": dict, "fields": {field: {group: FieldStats}}}.
    dividends holds by_source, yahoo_reasons and b3_empty_yahoo_failed, each keyed by ISO date; b3_filters,
    split_adjust_no_isin and slash_names are carried when given. filings_cutoffs maps a decision date to its
    cutoff (--filings-cutoff, A-22 (b)). sensitivities holds the statistics of the non-official sensitivities
    (see _sensitivities); without it the object has no "sensitivities" key.
    """
    iso_dates = [day.isoformat() for day in sorted(dates)]
    if not iso_dates:
        raise ValueError("the report needs at least one decision date")
    if sorted(per_date) != iso_dates:
        raise ValueError("per_date must hold exactly the decision dates of the run")
    cutoffs = _cutoffs(filings_cutoffs or {}, iso_dates)
    dates_out: dict[str, dict[str, object]] = {}
    for iso in iso_dates:
        entry = per_date[iso]
        dates_out[iso] = {
            "eligible": entry["eligible"],
            "mapping": entry["mapping"],
            "fields": {
                field: {group: asdict(entry["fields"][field][group]) for group, _label in GROUPS} for field in ALL_FIELDS
            },
            "verdicts": _field_verdicts(entry["fields"]),
        }
    official = {iso: dates_out[iso]["verdicts"] for iso in iso_dates}
    overall = _overall(official, iso_dates)
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
    dividends_out: dict[str, object] = {
        "source": "secondary" if yahoo_used else "primary",
        "by_source": {iso: dict(dividends["by_source"][iso]) for iso in iso_dates},
        "yahoo_reasons": {iso: dict(dividends["yahoo_reasons"][iso]) for iso in iso_dates},
        "b3_empty_yahoo_failed": {iso: dividends["b3_empty_yahoo_failed"][iso] for iso in iso_dates},
        "window": WINDOW,
        "caveat": CAVEAT,
    }
    for key in DIVIDEND_EXTRAS:
        if key in dividends:
            dividends_out[key] = deepcopy(dividends[key])
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
        "quote_codbdi": list(QUOTE_CODBDI),
        "filings_cutoffs": cutoffs,
        "formulas": dict(FORMULAS),
        "share_scale_rule": SHARE_SCALE_RULE,
        "non_pit_inputs": [dict(row) for row in NON_PIT_INPUTS],
    }
    report_object: dict[str, object] = {
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
    if sensitivities is not None:
        report_object["sensitivities"] = _sensitivities(sensitivities, iso_dates, cutoffs, official, overall)
    return report_object


def _field_verdicts(fields: Mapping[str, Mapping[str, compare.FieldStats]]) -> dict[str, dict[str, str]]:
    """The verdict of every indicator and group from the statistics of one date (VER-01)."""
    return {
        field: {
            group: compare.verdict(fields[field][group].pct_within, fields[field][group].denominator)
            for group, _label in GROUPS
        }
        for field in ALL_FIELDS
    }


def _overall(
    verdicts_by_date: Mapping[str, Mapping[str, Mapping[str, str]]], iso_dates: Sequence[str]
) -> dict[str, dict[str, str]]:
    """The verdict of every indicator and group over the dates of the run: the worse date (VER-02)."""
    return {
        field: {
            group: reduce(compare.overall, [verdicts_by_date[iso][field][group] for iso in iso_dates])
            for group, _label in GROUPS
        }
        for field in ALL_FIELDS
    }


def _iso(value: object) -> str:
    return value.isoformat() if isinstance(value, date) else str(value)


def _cutoffs(given: Mapping[object, object], iso_dates: Sequence[str]) -> dict[str, str]:
    """The filings cutoffs of the run as ISO dates; each names a decision date of the run and is not after it (A-22 (b))."""
    cutoffs: dict[str, str] = {}
    for day, cutoff in given.items():
        iso_day, iso_cutoff = _iso(day), _iso(cutoff)
        if iso_day not in iso_dates:
            raise ValueError(f"the filings cutoff of {iso_day} names a date that is not a decision date")
        if iso_cutoff > iso_day:
            raise ValueError(f"the filings cutoff {iso_cutoff} is after its decision date {iso_day}")
        cutoffs[iso_day] = iso_cutoff
    return dict(sorted(cutoffs.items()))


def _sensitivities(
    given: Mapping[str, object],
    iso_dates: Sequence[str],
    cutoffs: Mapping[str, str],
    official: Mapping[str, Mapping[str, Mapping[str, str]]],
    overall: Mapping[str, Mapping[str, str]],
) -> dict[str, object]:
    """The non-official block (VER-05..VER-08). Its verdicts come from the given statistics, the official ones are never changed.

    given may hold "without_reference_stale": {"excluded": {iso: [ticker]}, "fields": {iso: {field: {group: FieldStats}}}}
    and "filings_cutoff": {iso: {"cutoff": date, "fields": {field: {group: FieldStats}}}}, one entry per cutoff.
    official and overall are the official verdicts, computed before this function runs.
    """
    block: dict[str, object] = {"official": False, "title": SENSITIVITY_TITLE}
    if "without_reference_stale" in given:
        block["without_reference_stale"] = _without_reference_stale(given["without_reference_stale"], iso_dates)
    if "filings_cutoff" in given:
        filings: dict[str, object] = {}
        for iso, item in sorted(given["filings_cutoff"].items()):
            cutoff = _iso(item["cutoff"])
            if iso not in iso_dates or cutoffs.get(iso) != cutoff:
                raise ValueError(f"the filings sensitivity of {iso} does not match the filings cutoffs of the run")
            filings[iso] = _filings_cutoff(iso, cutoff, item["fields"], iso_dates, official)
        block["filings_cutoff"] = filings
    block["class_without_roic_peg"] = _class_without_roic_peg(overall)
    return block


def _without_reference_stale(given: Mapping[str, object], iso_dates: Sequence[str]) -> dict[str, object]:
    """D-A: the class verdict with the reference-stale tickers out of every denominator (A-22 (a))."""
    fields_by_date = given["fields"]
    missing = [iso for iso in iso_dates if iso not in fields_by_date]
    if missing:
        raise ValueError(f"the stale-free sensitivity lacks the statistics of {', '.join(missing)}")
    excluded = given.get("excluded", {})
    verdicts_by_date = {iso: _field_verdicts(fields_by_date[iso]) for iso in iso_dates}
    overall = _overall(verdicts_by_date, iso_dates)
    class_result = compare.class_verdict(overall)
    return {
        "text": WITHOUT_REFERENCE_STALE_TEXT,
        "excluded": {iso: sorted(excluded.get(iso, [])) for iso in iso_dates},
        "dates": {
            iso: {
                "fields": {field: {group: asdict(fields_by_date[iso][field][group]) for group, _label in GROUPS} for field in ALL_FIELDS},
                "verdicts": verdicts_by_date[iso],
            }
            for iso in iso_dates
        },
        "verdicts": {
            "fields": overall,
            "class": class_result["class"],
            "class_without_unclassified": class_result["class_without_unclassified"],
            "financial_group": class_result["financial"],
        },
    }


def _filings_cutoff(
    iso: str,
    cutoff: str,
    fields: Mapping[str, Mapping[str, compare.FieldStats]],
    iso_dates: Sequence[str],
    official: Mapping[str, Mapping[str, Mapping[str, str]]],
) -> dict[str, object]:
    """D-B: one date recomputed with filings received by its cutoff; the other dates keep their official verdicts (A-22 (b))."""
    recomputed = _field_verdicts(fields)
    overall = _overall({other: recomputed if other == iso else official[other] for other in iso_dates}, iso_dates)
    class_result = compare.class_verdict(overall)
    return {
        "text": (
            f"Data {iso} recalculada com filings de DT_RECEB ≤ {cutoff}; cotações, dividendos, elegibilidade e d são os "
            "mesmos. Não é o veredito oficial."
        ),
        "cutoff": cutoff,
        "fields": {field: {group: asdict(fields[field][group]) for group, _label in GROUPS} for field in ALL_FIELDS},
        "date_verdicts": recomputed,
        "verdicts": {
            "fields": overall,
            "class": class_result["class"],
            "class_without_unclassified": class_result["class_without_unclassified"],
            "financial_group": class_result["financial"],
        },
    }


def _class_without_roic_peg(overall: Mapping[str, Mapping[str, str]]) -> dict[str, object]:
    """D-C: the class verdict over the score fields other than ROIC and PEG RATIO, from the official overall verdicts."""
    kept = [field for field in SCORE_FIELDS if field not in ROIC_PEG_FIELDS]
    return {
        "excluded_fields": sorted(ROIC_PEG_FIELDS),
        "class": compare.worst(overall[field]["non_financial"] for field in kept),
        "note": ROIC_PEG_NOTE,
    }


def encode_json(report: Mapping[str, object]) -> bytes:
    """The report as sorted, indented UTF-8 JSON with a final newline (RPT-02)."""
    return (json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def render_markdown(report: Mapping[str, object]) -> str:
    """The Markdown of the report, written from the object alone (RPT-04)."""
    lines: list[str] = ["# Reconstrução PIT dos indicadores de ações", ""]
    lines += _metadata(report)
    lines += _verdict_section(report["verdicts"])
    if "sensitivities" in report:
        lines += _sensitivity_section(report["sensitivities"])
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
        for field in _canonical(verdicts["fields"])
    ]
    lines += _table(["indicador"] + [label for _group, label in GROUPS], rows)
    return lines


def _sensitivity_section(sensitivities: Mapping[str, object]) -> list[str]:
    """The sensitivities below the official verdict; every number is read from the object (RPT-04, VER-08)."""
    lines = [f"## {sensitivities['title']}", "", SENSITIVITY_NOTE, ""]
    rows: list[list[object]] = []
    stale = sensitivities.get("without_reference_stale")
    if stale is not None:
        rows.append(
            [
                "Sem referências obsoletas (D-A)",
                _code(stale["verdicts"]["class"]),
                _code(stale["verdicts"]["financial_group"]),
                "excluídos: " + _excluded_text(stale["excluded"]),
            ]
        )
    for iso, block in sensitivities.get("filings_cutoff", {}).items():
        rows.append(
            [
                f"Data {iso} com filings até {block['cutoff']} (D-B)",
                _code(block["verdicts"]["class"]),
                _code(block["verdicts"]["financial_group"]),
                f"filings de {iso} com DT_RECEB ≤ {block['cutoff']}; as outras datas ficam como no veredito oficial",
            ]
        )
    roic = sensitivities["class_without_roic_peg"]
    rows.append(["Sem ROIC e PEG RATIO (D-C)", _code(roic["class"]), "-", "fora da classe: " + _codes(roic["excluded_fields"])])
    lines += _table(["sensibilidade", "veredito da classe", "veredito do grupo financeiro", "o que muda"], rows)
    lines += [f"Nota (D-C): {roic['note']}", ""]
    return lines


def _excluded_text(excluded: Mapping[str, Sequence[str]]) -> str:
    return "; ".join(f"{iso} {_codes(tickers) if tickers else 'nenhum'}" for iso, tickers in excluded.items())


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
    lines.append(f"- Cotações (`quote_codbdi`, CODBDI): {_codes(parameters['quote_codbdi'])}.")
    cutoffs = parameters["filings_cutoffs"]
    cutoff_text = "; ".join(f"`{day}` com filings recebidos até `{cutoff}`" for day, cutoff in cutoffs.items()) or "nenhum"
    lines.append(f"- Cortes de filings (`filings_cutoffs`): {cutoff_text}.")
    lines.append(f"- Escala das ações: {parameters['share_scale_rule']}")
    lines += ["", "### Fórmulas", ""]
    lines += _table(["indicador", "fórmula"], [[_code(field), parameters["formulas"][field]] for field in _canonical(parameters["formulas"])])
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
        scaled = scale.get("scaled")
        stale_prices = mapping.get("vm_stale_class_price")
        lines += [
            f"### {iso}",
            "",
            f"- Elegíveis: {entry['eligible']}. Mapeados: {mapping['mapped']}. Não mapeados: {len(mapping['unmapped'])}.",
            f"- Mapeados pela B3 (`mapped_via_b3`): {_list_or_none(mapping['mapped_via_b3'])}.",
            f"- Caíram para o período anterior porque o mais recente chegou depois de d (`refiled_after_d`): "
            f"{_list_or_none(mapping['refiled_after_d'])}.",
            f"- Usaram o preço da outra classe (`vm_price_fallback`): {fallback['count']} empresa(s).",
            f"- Sem cotação recente (`reference_stale`, sem registro em [d − 30 dias, d]): "
            f"{_list_or_missing(mapping.get('reference_stale'))}.",
            f"- Preço de classe de pregão antigo (`vm_stale_class_price`): {_count_or_missing(stale_prices, 'classe(s)')}.",
            f"- Escala das contagens (`share_scale.scaled`): {_count_or_missing(scaled, 'empresa(s)')}. "
            f"Não verificada (`share_scale.unverified`): {_list_or_none(scale['unverified'])}.",
            "",
        ]
        if scaled:
            lines += ["Escala das contagens, com o fator e o resíduo de cada empresa:", ""]
            lines += _table(
                ["cd_cvm", "fator", "resíduo"],
                [[_code(row["cd_cvm"]), _power(row["factor"]), _num_or_none(row.get("residual"))] for row in scaled],
            )
        if stale_prices:
            lines += ["Empresas com preço de classe de pregão antigo:", ""]
            lines += _table(
                ["cd_cvm", "classe", "ticker", "último pregão"],
                [[_code(row["cd_cvm"]), row["class"], _code(row["ticker"]), row["session"]] for row in stale_prices],
            )
        lines += ["Tickers não mapeados:", ""]
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
        for field in _canonical(entry["fields"]):
            groups = entry["fields"][field]
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
            for field, groups in ((name, entry["fields"][name]) for name in _canonical(entry["fields"]))
            if groups[GROUPS[0][0]]["reference_blank"] or groups[GROUPS[0][0]]["reference_blank_as_zero"]
        ]
        if blanks:
            lines += ["Referências em branco no Status Invest (grupo não financeiro):", ""]
            lines += _table(["indicador", "em branco", "com valor reconstruído", "DY tratado como 0,00"], blanks)
        for field in _canonical(entry["fields"]):
            groups = entry["fields"][field]
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
    lines += ["### Filtros e ajustes dos registros", ""]
    filters = dividends.get("b3_filters")
    if filters is None:
        lines += ["- Registros da B3 (`dividends.b3_filters`): não informados.", ""]
    else:
        lines += [
            f"- Registros da B3 (`dividends.b3_filters`): {filters['raw_records']} brutos; {filters['duplicates']} "
            f"repetidos, contados uma vez; {filters['capital_returns']} de `REST CAP DIN`, descartados.",
            "",
        ]
    split = dividends.get("split_adjust_no_isin")
    if split is None:
        lines += ["- Classes sem ISIN no ajuste de desdobramento (`dividends.split_adjust_no_isin`): não informadas.", ""]
    else:
        lines += ["Classes sem ISIN no ajuste de desdobramento (`dividends.split_adjust_no_isin`):", ""]
        lines += _table(["data", "classes"], [[iso, count] for iso, count in split.items()])
    slash = dividends.get("slash_names")
    if slash is None:
        lines += ["- Nomes com `/` (`dividends.slash_names`): não informados.", ""]
    elif not slash:
        lines += ["- Nomes com `/` (`dividends.slash_names`): nenhum.", ""]
    else:
        lines += ["Nomes com `/` na B3 (`dividends.slash_names`):", ""]
        lines += _table(
            ["tradingName", "consulta", "registros"],
            [[row["trading_name"], _code(row["queried_as"]), row["total_records"]] for row in slash],
        )
    return lines


def _coverage_section(coverage: Mapping[str, object]) -> list[str]:
    lines = ["## Cobertura histórica e esforço", "", "### Primeiro ano por fonte", ""]
    lines += _table(
        ["fonte", "primeiro ano"],
        [[_code(source), _year_text(info)] for source, info in sorted(coverage["earliest_year_by_source"].items())],
    )
    lines += ["### Primeiro ano por indicador", ""]
    lines += _table(
        ["indicador", "primeiro ano"],
        [[_code(field), coverage["earliest_year_by_field"][field]] for field in _canonical(coverage["earliest_year_by_field"])],
    )
    dy = coverage["dy"]
    lines += ["### Cobertura do DY", ""]
    lines += _table(
        ["fonte", "tickers", "primeira data bruta", "primeiro ano", "mediana do ano por ticker", "retenção"],
        [
            [
                _code(source),
                info["tickers"],
                info["earliest_raw_date"] or "-",
                _year(info["earliest_year"]),
                _year(info["median_ticker_year"]),
                info["retention"],
            ]
            for source, info in sorted(dy["by_source"].items())
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
            for kind, item in sorted(effort["load"].items())
        ],
    )
    years = coverage.get("years")
    if years:
        lines += ["### Anos lidos na reconstrução completa", ""]
        lines += _table(["fonte", "primeiro ano", "último ano"], [[_code(source), first, last] for source, (first, last) in sorted(years.items())])
    return lines


def _inputs_section(inputs: Mapping[str, object]) -> list[str]:
    lines = ["## Entradas", "", "### Arquivos brutos", ""]
    lines += _table(
        ["arquivo", "URL", "sha256", "bytes"],
        [[item["name"], item["url"], f"`{item['sha256']}`", item["bytes"]] for item in inputs["raw_files"]],
    )
    lines += ["### Status Invest", ""]
    lines += _table(
        ["data", "coletado em", "caminho", "sha256"],
        [[item["date"], item.get("collected_on"), item["path"], f"`{item['sha256']}`"] for item in inputs["status_invest"]],
    )
    lines += ["### Data de referência (as-of)", ""]
    for item in inputs["status_invest"]:
        check = item.get("as_of_check")
        if check is None:
            lines += [f"- {item['date']}: checagem não informada.", ""]
            continue
        lines += [
            f"**{item['date']}**: {check['eligible_with_price']} elegíveis com cotação. "
            f"Sessão de melhor correspondência: `{check['best_session']}`.",
            "",
        ]
        lines += _table(["sessão", "tickers com o mesmo preço"], [[session, count] for session, count in check["matches"].items()])
        if check["best_session"] != item["date"]:
            lines += [
                f"**Atenção:** as cotações do CSV de {item['date']} batem melhor com a sessão {check['best_session']}, "
                f"não com {item['date']}.",
                "",
            ]
    return lines


def _canonical(present: Mapping[str, object]) -> list[str]:
    """The indicators held in present, in the order of ALL_FIELDS: the Markdown does not depend on the order of the dict."""
    return [name for name in ALL_FIELDS if name in present]


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


def _list_or_missing(items: Sequence[str] | None) -> str:
    """A list of tickers, or "não informado" when the object does not carry it (not computed, not empty)."""
    return "não informado" if items is None else _list_or_none(items)


def _count_or_missing(rows: Sequence[object] | None, noun: str) -> str:
    return "não informado" if rows is None else f"{len(rows)} {noun}"


def _power(factor: float) -> str:
    """A scale factor as its power of ten: 10^3 for 1000 (A-18)."""
    return f"10^{round(math.log10(factor))}"


def _year(value: float | None) -> str:
    return "-" if value is None else str(int(value))


def _year_text(info: Mapping[str, object]) -> str:
    if info["year"] is None:
        return "-"
    if info["at_probe_floor"]:
        return f"≤ {info['year']} (limite da sondagem)"
    return str(info["year"])
