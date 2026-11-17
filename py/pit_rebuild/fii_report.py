"""Markdown of the FII measurement (issue #60). Every number comes from the report object; nothing is recomputed."""

from .fii_compare import PERCENT_FIELDS, SEGMENT_MAP, MANAGEMENT_MAP
from .fii_status_invest import CATEGORY_FIELDS, FIELDS, NUMERIC_FIELDS

RULES = [
    "Tolerância: diferença relativa de até 5%; nos indicadores em percentual (`DY`, `PERCENTUAL EM CAIXA`, "
    "`CAGR DIVIDENDOS 3 ANOS`, `CAGR VALOR COTA 3 ANOS`), até 0,5 ponto percentual. O valor reconstruído é "
    "arredondado a duas casas antes da comparação. São as regras do relatório de ações.",
    "Denominador: os FIIs elegíveis com valor no Status Invest. Um FII do denominador sem valor reconstruído "
    "conta como fora da tolerância.",
    "Veredito por indicador e data: `reconstruível` com 90% ou mais, `com ressalvas` com 70% ou mais, "
    "`não reconstruível` abaixo de 70%, `sem referência` com denominador zero. Entre as datas vale o pior.",
    "Veredito da classe: o pior dos 13 indicadores.",
    "Critério de liberação (#64): parte 1, todo indicador com pelo menos 70% na tolerância nas duas datas; "
    "parte 2, os 10 piores de cada indicador e data classificados (erro nosso, erro do Status Invest ou diferença de método).",
    "Ponto no tempo: só entram informes cuja versão mais recente foi entregue até `d` (`Data_Entrega`) e pregões até `d`. "
    "Um informe de mês encerrado há mais de 270 dias não dá valor (`stale_informe`).",
    "Proventos: a página de fundos da B3 primeiro; o Yahoo só quando a B3 não alcança o início da janela, e a fonte e o "
    "motivo ficam marcados. O informe mensal da CVM não traz provento por cota.",
]
FORMULAS = [
    ("PRECO", "fechamento do COTAHIST (CODBDI 12) no último pregão até `d`, no máximo 30 dias antes"),
    ("LIQUIDEZ MEDIA DIARIA", "média do `VOLTOT` dos últimos 21 pregões até `d`"),
    ("N COTISTAS", "`Total_Numero_Cotistas` do informe em `d`"),
    ("PATRIMONIO", "`Patrimonio_Liquido` do informe em `d`"),
    ("N COTAS", "`Cotas_Emitidas` do informe em `d`"),
    ("PERCENTUAL EM CAIXA", "`Total_Necessidades_Liquidez` / `Patrimonio_Liquido` × 100"),
    ("P/VP", "`PRECO` / `Valor_Patrimonial_Cotas` do informe em `d`"),
    ("CAGR VALOR COTA 3 ANOS", "`Valor_Patrimonial_Cotas` sobre o do informe 36 meses antes (conhecido em `d`), ^(1/3) − 1"),
    ("DY", "proventos com data-ex em (d − 365 dias, d] / `PRECO` × 100"),
    ("ULTIMO DIVIDENDO", "o último provento com data-ex até `d`"),
    ("CAGR DIVIDENDOS 3 ANOS", "proventos dos 12 meses até `d` sobre os dos 12 meses até `d` − 3 anos, ^(1/3) − 1"),
    ("SETOR", "`Segmento_Atuacao` do informe em `d`, pela tabela de correspondência"),
    ("GESTAO", "`Tipo_Gestao` do informe em `d`, pela tabela de correspondência"),
]
LIMITATIONS = [
    "A página de fundos da B3 traz só os proventos dos últimos meses a partir do dia da coleta. Para as duas datas, a "
    "janela de 12 meses começa antes do primeiro registro dela na maioria dos FIIs, e o DY cai no Yahoo, marcado. A "
    "linha `DY só com a B3` mostra o que a B3 sozinha dá na janela truncada.",
    "O `CAGR DIVIDENDOS 3 ANOS` precisa de quatro anos de proventos, que a B3 não publica: ele vem do Yahoo quando há, marcado.",
    "O campo `Percentual_Dividend_Yield_Mes` do informe mensal não é usado: no informe do XP Malls (CNPJ "
    "28.757.546/0001-00), versão 1 de 2026-01, ele vale −0,058778, o que não é um rendimento distribuído.",
    "Os arquivos anuais da CVM trazem só a versão mais recente de cada informe. Um informe reapresentado depois de `d` "
    "some em `d`, e o fundo usa o informe anterior.",
    "A ligação ticker → fundo é o ISIN do último pregão até `d` (COTAHIST) igual ao `Codigo_ISIN` do informe. Alguns "
    "fundos informam outro ISIN (por exemplo `BR0EI9CTF007`) e só passam ao ISIN negociado num informe posterior a `d`: "
    "esse informe só identifica o fundo, e os valores continuam vindo dos informes entregues até `d` "
    "(`isin_from_later_informe`). Quando mais de um fundo informa o mesmo ISIN, vale o de número de cotas mais perto "
    "do da página da B3, obtida na coleta e não em `d`.",
    "A tabela de correspondência de setor e gestão foi escrita antes da medição. A CVM classifica muitos fundos como "
    "`Multicategoria` ou `Outros`, categorias que o Status Invest divide em várias.",
    "A classificação dos 10 piores é automática: ela procura uma fonte que dê o valor do Status Invest. O caso sem "
    "fonte fica `sem classificação` e precisa de leitura à mão.",
]


def render_markdown(obj: dict) -> str:
    isos = obj["decision_dates"]
    out: list[str] = []
    add = out.append
    add("# Reconstrução PIT dos indicadores de FIIs\n")
    add(f"- Esquema: `{obj['schema']}` (issue #60, spec #75)")
    add(f"- Commit da ferramenta: `{obj['tool_commit']}`")
    add(f"- Datas de decisão: {', '.join(isos)}")
    for item in obj["inputs"]["status_invest"]:
        add(f"- Status Invest de {item['date']} (coletado em {item['collected_on']}): `{item['file']}`, "
            f"{item['rows']} linhas, sha256 `{item['sha256']}`")
    add("")
    add("## Veredito\n")
    add(f"**Veredito da classe:** `{obj['class_verdict']}`.\n")
    part1, part2 = obj["criterion"]["part1"], obj["criterion"]["part2"]
    add(f"Critério de liberação, parte 1 (≥ {part1['threshold_pct']}% nas duas datas): "
        + ("cumprida." if part1["met"] else f"não cumprida ({', '.join(f'`{name}`' for name in part1['failing'])})."))
    add(f"Critério de liberação, parte 2 (piores casos classificados): "
        + ("cumprida." if part2["met"] else f"não cumprida ({part2['open']} de {part2['cases']} casos sem classificação)."))
    add(f"Critério inteiro: {'cumprido' if obj['criterion']['met'] else 'não cumprido'}.\n")
    header = " | ".join(f"{iso}: % na tolerância (n)" for iso in isos)
    add(f"| indicador | {header} | veredito |")
    add("| --- | " + " | ".join("---" for _ in isos) + " | --- |")
    for name in FIELDS:
        block = "fields" if name in NUMERIC_FIELDS else "categories"
        cells = []
        for iso in isos:
            stats = obj["dates"][iso][block][name]
            cells.append(f"{_pct(stats['pct_within'])} ({stats['denominator']})")
        add(f"| `{name}` | " + " | ".join(cells) + f" | {obj['verdicts'][name]['overall']} |")
    add("")
    add("## Regras\n")
    out.extend(f"- {line}" for line in RULES)
    add("")
    add("| indicador | fórmula |")
    add("| --- | --- |")
    out.extend(f"| `{name}` | {text} |" for name, text in FORMULAS)
    add("")
    add("Universo elegível: " + "; ".join(f"`{item}`" for item in obj["eligibility"]["source"])
        + f". Positivos: {', '.join(obj['eligibility']['core_positive'])}; DY ≥ 0; PATRIMONIO ≥ "
        f"{obj['eligibility']['floor']['size_min']:,}".replace(",", ".")
        + f"; LIQUIDEZ MEDIA DIARIA ≥ {obj['eligibility']['floor']['liq_min']:,}".replace(",", ".") + ".\n")
    _universe(out, obj, isos)
    _dividends(out, obj, isos)
    _categories(out, obj, isos)
    _coverage(out, obj)
    _worst(out, obj, isos)
    add("## Limitações\n")
    out.extend(f"- {line}" for line in LIMITATIONS)
    add("")
    return "\n".join(out)


def _universe(out: list[str], obj: dict, isos: list[str]) -> None:
    out.append("## Universo e ligação com os informes\n")
    out.append("| data | elegíveis | com informe em d | ISIN só num informe posterior | sem informe (motivo) | com informe entregue depois de d | sessão que bate com os preços |")
    out.append("| --- | --- | --- | --- | --- | --- | --- |")
    for iso in isos:
        block = obj["dates"][iso]
        mapping = block["mapping"]
        reasons = ", ".join(f"{key}: {value}" for key, value in mapping["without_informe"].items()) or "-"
        check = block["as_of_check"]
        best = check["best"] or "-"
        hits = check["sessions"].get(best, 0) if check["best"] else 0
        later = len(mapping["isin_from_later_informe"])
        out.append(f"| {iso} | {block['eligible']} | {mapping['with_informe']} | {later} | {reasons} | {mapping['delivered_after_d']} | {best} ({hits}) |")
    out.append("")
    for iso in isos:
        check = obj["dates"][iso]["as_of_check"]
        if check["best"] and not check["matches_d"]:
            out.append(f"Aviso: em {iso}, a sessão que mais bate com os preços do CSV é {check['best']}, e não {iso}.\n")


def _dividends(out: list[str], obj: dict, isos: list[str]) -> None:
    out.append("## Fontes de proventos\n")
    out.append("Contagem de FIIs elegíveis por fonte (`b3`, `yahoo` marcado como reserva, `none` sem fonte) e motivo do uso do Yahoo.\n")
    out.append("| data | indicador | fontes | motivos |")
    out.append("| --- | --- | --- | --- |")
    for iso in isos:
        block = obj["dividends"][iso]
        for name in ("DY", "ULTIMO DIVIDENDO", "CAGR DIVIDENDOS 3 ANOS"):
            sources = ", ".join(f"{key}: {value}" for key, value in block[name]["sources"].items())
            reasons = ", ".join(f"{key}: {value}" for key, value in block[name]["reasons"].items()) or "-"
            out.append(f"| {iso} | `{name}` | {sources} | {reasons} |")
    out.append("")
    out.append("`DY só com a B3` (sensibilidade, não entra no veredito): os proventos da B3 com data-ex até `d`, na janela "
               "truncada no primeiro registro dela.\n")
    out.append("| data | % na tolerância (n) |")
    out.append("| --- | --- |")
    for iso in isos:
        item = obj["dividends"][iso]["b3_only_dy"]
        out.append(f"| {iso} | {_pct(item['pct_within'])} ({item['denominator']}) |")
    out.append("")


def _categories(out: list[str], obj: dict, isos: list[str]) -> None:
    out.append("## Setor e gestão\n")
    out.append("Tabela de correspondência, escrita antes da medição (CVM → Status Invest):\n")
    out.append("| campo | CVM | Status Invest |")
    out.append("| --- | --- | --- |")
    for name, table in (("SETOR", SEGMENT_MAP), ("GESTAO", MANAGEMENT_MAP)):
        for cvm_value, targets in table.items():
            out.append(f"| `{name}` | {cvm_value} | {', '.join(sorted(targets)) or 'sem correspondente'} |")
    out.append("")
    for name in CATEGORY_FIELDS:
        for iso in isos:
            stats = obj["dates"][iso]["categories"][name]
            out.append(f"<details>\n<summary>{name}, {iso}: {_pct(stats['pct_within'])} na mesma categoria depois do "
                       f"mapeamento ({stats['within']} de {stats['denominator']}; {stats['reference_blank']} sem valor no Status Invest)</summary>\n")
            out.append("| CVM | Status Invest | FIIs | mesma categoria |")
            out.append("| --- | --- | --- | --- |")
            for row in stats["crosstab"]:
                out.append(f"| {row['cvm']} | {row['status_invest']} | {row['tickers']} | {'sim' if row['same_after_mapping'] else 'não'} |")
            missing = ", ".join(f"`{item['ticker']}` ({item['reason']})" for item in stats["missing"])
            out.append(f"\nSem valor da CVM: {missing or 'nenhum'}.\n\n</details>\n")


def _coverage(out: list[str], obj: dict) -> None:
    coverage = obj["coverage"]
    years = [str(year) for year in coverage["informe_years_listed"]]
    out.append("## Cobertura histórica\n")
    out.append(f"Anos dos informes mensais na listagem da CVM: {', '.join(years)}. Para cada indicador, o primeiro ano em que "
               f"algum informe tem a coluna preenchida e, por ano, o % dos {coverage['matched_funds_last_date']} fundos ligados "
               "na última data que têm a coluna preenchida.\n")
    out.append("| indicador | coluna | ano mais antigo | " + " | ".join(years) + " |")
    out.append("| --- | --- | --- | " + " | ".join("---" for _ in years) + " |")
    for name, item in coverage["informes"].items():
        shares = " | ".join(_pct(item["pct_of_matched_funds_by_year"].get(year)) for year in years)
        out.append(f"| `{name}` | `{item['column']}` | {item['first_year'] or '-'} | {shares} |")
    out.append("")
    b3, yahoo = coverage["b3_dividends"], coverage["yahoo_dividends"]
    out.append(f"- `CAGR VALOR COTA 3 ANOS` precisa de 36 meses: o ano mais antigo é o primeiro ano da coluna mais 3.")
    out.append(f"- Proventos da B3 (`DY`, `ULTIMO DIVIDENDO`): {b3['funds']} fundos com registro; primeiro registro "
               f"(data com) mais antigo {b3['earliest'] or '-'}, mediana {b3['median'] or '-'}.")
    out.append(f"- Proventos do Yahoo (reserva): {yahoo['funds']} tickers com registro; mais antigo {yahoo['earliest'] or '-'}, "
               f"mediana {yahoo['median'] or '-'}.")
    out.append(f"- `PRECO` e `LIQUIDEZ MEDIA DIARIA`: COTAHIST anual; esta medição carregou {', '.join(str(y) for y in coverage['cotahist_years_loaded'])} "
               "e não sondou anos anteriores.\n")


def _worst(out: list[str], obj: dict, isos: list[str]) -> None:
    out.append("## Os 10 piores de cada indicador\n")
    out.append("Classificação pela evidência das fontes (ver Limitações). `usado` é o informe, o pregão ou a fonte do valor "
               "reconstruído; `evidência` é o que sustenta a classificação.\n")
    out.append("| data | indicador | erro nosso | erro do Status Invest | diferença de método | sem classificação |")
    out.append("| --- | --- | --- | --- | --- | --- |")
    for iso in isos:
        for name in FIELDS:
            block = "fields" if name in NUMERIC_FIELDS else "categories"
            counts = {}
            for case in obj["dates"][iso][block][name]["worst"]:
                counts[case["class"]] = counts.get(case["class"], 0) + 1
            out.append(f"| {iso} | `{name}` | {counts.get('erro nosso', 0)} | {counts.get('erro do Status Invest', 0)} | "
                       f"{counts.get('diferença de método', 0)} | {counts.get('sem classificação', 0)} |")
    out.append("")
    for name in NUMERIC_FIELDS:
        unit = "p.p." if name in PERCENT_FIELDS else "%"
        for iso in isos:
            stats = obj["dates"][iso]["fields"][name]
            out.append(f"<details>\n<summary>{name}, {iso}</summary>\n")
            out.append(f"| ticker | referência | reconstruído | diferença ({unit}) | classe | usado | evidência |")
            out.append("| --- | --- | --- | --- | --- | --- | --- |")
            for case in stats["worst"]:
                out.append(f"| `{case['ticker']}` | {_num(case['reference'])} | {_num(case['rebuilt'])} | {_num(case['diff'])} | "
                           f"{case['class']} | {case['used']} | {case['evidence']} |")
            missing = ", ".join(f"`{item['ticker']}` ({item['reason']})" for item in stats["missing"])
            out.append(f"\nSem valor reconstruído: {missing or 'nenhum'}.\n\n</details>\n")
    for name in CATEGORY_FIELDS:
        for iso in isos:
            stats = obj["dates"][iso]["categories"][name]
            if not stats["worst"]:
                continue
            out.append(f"<details>\n<summary>{name}, {iso}: os 10 piores (sem valor da CVM primeiro, depois fora da categoria, por ticker)</summary>\n")
            out.append("| ticker | Status Invest | CVM | classe | evidência |")
            out.append("| --- | --- | --- | --- | --- |")
            for case in stats["worst"]:
                out.append(f"| `{case['ticker']}` | {case['reference'] or '-'} | {case['rebuilt'] or '-'} | {case['class']} | {case['evidence']} |")
            out.append("\n</details>\n")


def _pct(value) -> str:
    return "-" if value is None else f"{value:.2f}%".replace(".", ",")


def _num(value) -> str:
    if value is None:
        return "-"
    return f"{value:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
