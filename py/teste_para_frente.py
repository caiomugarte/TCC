"""Teste para frente de um run Premium contra IBOV, IFIX e CDI (issue #86, spec #82).

A carteira de um run Premium concluído é medida da data de corte do manifesto dele (o início) até a data de corte de
um manifesto posterior publicado (o fim): retorno total com proventos de cada ação e FII, por classe (ponderado pelos
pesos dentro da classe e com peso igual), da carteira inteira (pesos das classes) e dos índices de referência tirados
das séries do manifesto posterior.

Uso (a partir da raiz do repositório):

    python py/teste_para_frente.py --manifest snapshots/<id>.json --run-id <id> [--database-url URL]
    python py/teste_para_frente.py --manifest snapshots/<id>.json --run-json run.json [--saida resultado.json]

Para o ciclo mensal (#87), a mesma conta é ``teste_para_frente(run, manifesto)``, que devolve um dict serializável em
JSON.

Regras:

- Preços: COTAHIST da B3 (ações com CODBDI 02/07/08, FIIs com 12), ``PREULT / FATCOT``. Preço de início: última
  sessão até o início, com no máximo 30 dias. Preço de fim: última sessão até o fim, com no máximo
  ``IDADE_MAXIMA_PRECO_FIM_DIAS`` dias; sem ela o ativo sai marcado ``sem_preco_fim`` (deslistado, incorporado,
  grupamento sem preço) e fica fora das médias da classe, que são renormalizadas sobre os ativos com retorno
  (``peso_coberto`` diz quanto do peso foi medido).
- Proventos: entram os de data-ex em ``início < data-ex <= fim`` (quem comprou no fechamento do corte recebe). Ações:
  histórico de proventos em dinheiro da B3 (``GetListedCashDividends``), data-ex na sessão seguinte ao
  ``lastDatePriorEx``, registros da classe do papel; sem histórico completo da B3 (ou unit sem registros UNT), Yahoo,
  com o motivo. FIIs: página do fundo na B3 (``GetListedSupplementFunds``, RENDIMENTO do ISIN da cota), o mesmo
  caminho do coletor da #60; quando ela não alcança o início (histórico curto) ou falha, Yahoo, com o motivo.
- Desdobramento, bonificação e grupamento de ações (``stockDividends`` da página da empresa na B3) com data-ex no
  intervalo multiplicam a quantidade de papéis: ``retorno = (P_fim * q_fim + soma(provento * q antes da data-ex)) /
  P_início - 1``. Eventos de FIIs não são lidos.
- Pesos: ``sleeve_weight`` das linhas da carteira (as de ``quantity == 0`` não foram compradas e ficam de fora),
  renormalizados para somar 1 na classe. Classes: ``target_weight``. Internacional e cripto entram pelas séries do
  manifesto posterior em BRL (S&P 500 retorno total e BTC vezes PTAX); renda fixa pelo CDI.
- Índices: nível(fim) / nível(início) - 1 nas séries ``ibovespa.csv``, ``ifix.csv`` e ``di.csv`` do manifesto
  posterior; sem o dia do início, a última data anterior, registrada em ``data_inicio``.
- Cache: os arquivos baixados ficam em ``<cache-dir>/<fim>/`` (padrão ``~/.cache/prumo/teste-para-frente``), um
  diretório por data de corte final, porque o cache do ``pit_rebuild`` não renova um arquivo já baixado. Se o
  COTAHIST do cache não tem a sessão do fim, a ferramenta para (código 5) e pede para apagar o diretório.

Exit codes:

    0  resultado gravado (ou impresso)
    2  uso inválido (argumentos, banco não informado, cache dentro de data/, snapshots/ ou baselines/)
    3  run ou manifesto não encontrado
    4  entrada inválida: run não concluído ou não Premium, carteira ilegível, manifesto inválido, não publicado ou
       sem data de corte posterior à do run, séries sem a data de início
    5  fonte indisponível: COTAHIST ausente, desatualizado ou divergente do cache
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
import json
import os
from pathlib import Path
import sys
from typing import Any
import urllib.request

if __package__ in (None, ""):
    _PY = Path(__file__).resolve().parent
    if str(_PY) not in sys.path:
        sys.path.insert(0, str(_PY))

from allocation_data import SnapshotError, read_levels_csv  # noqa: E402
from pit_rebuild import cotahist, dividends, fii_dividends, fii_sources, sources  # noqa: E402
from pit_rebuild.cache import Cache  # noqa: E402
from pit_rebuild.cli import b3_cash_name, b3_company_name, yahoo_name  # noqa: E402
from pit_rebuild.errors import ArgumentError, IntegrityError, SourceError  # noqa: E402
from pit_rebuild.safety import assert_writable_dir  # noqa: E402
from snapshot_manifest import SnapshotManifest, SnapshotManifestError, validate_manifest  # noqa: E402

EXIT_OK = 0
EXIT_USO = 2
EXIT_AUSENTE = 3
EXIT_ENTRADA = 4
EXIT_FONTE = 5

VERSAO = 1
IDADE_MAXIMA_PRECO_FIM_DIAS = 7
DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "prumo" / "teste-para-frente"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLASSE_ACOES = "brazilian_stocks"
CLASSE_FIIS = "fiis"
REFERENCIAS = (("IBOV", "brazilian_stocks"), ("IFIX", "fiis"), ("CDI", "fixed_income"))
CLASSES_POR_SERIE = {  # chave da API -> (rótulo, séries do manifesto multiplicadas para chegar ao nível em BRL)
    "international": ("S&P 500 retorno total em BRL", ("international_equity", "ptax")),
    "fixed_income": ("CDI", ("fixed_income",)),
    "crypto": ("BTC em BRL", ("crypto", "ptax")),
}


class TesteParaFrenteError(Exception):
    exit_code = 1


class UsoInvalido(TesteParaFrenteError):
    exit_code = EXIT_USO


class EntradaAusente(TesteParaFrenteError):
    exit_code = EXIT_AUSENTE


class EntradaInvalida(TesteParaFrenteError):
    exit_code = EXIT_ENTRADA


class FonteIndisponivel(TesteParaFrenteError):
    exit_code = EXIT_FONTE


@dataclass(frozen=True)
class _Linha:
    ticker: str
    peso: float


@dataclass(frozen=True)
class _Run:
    id: str
    manifest_id: str
    inicio: date
    pesos_classes: dict[str, float]
    acoes: list[_Linha]
    fiis: list[_Linha]


def teste_para_frente(
    run: Any,
    manifesto: SnapshotManifest | Path | str,
    *,
    cache_dir: Path | str | None = None,
    offline: bool = False,
    opener: Callable = urllib.request.urlopen,
) -> dict:
    """Teste para frente da carteira de ``run`` até a data de corte de ``manifesto``; um dict serializável em JSON.

    ``run`` é um ``RecommendationRun`` (ou um Mapping com ``id``, ``plan``, ``status``, ``snapshot_id``,
    ``snapshot_cutoff`` e ``result_json``). ``manifesto`` é o manifesto posterior publicado (objeto ou caminho).
    Levanta ``TesteParaFrenteError`` (com ``exit_code``) quando a entrada ou uma fonte obrigatória falha.
    """
    anterior = _ler_run(run)
    posterior = _ler_manifesto(manifesto)
    inicio, fim = anterior.inicio, posterior.cutoff_date
    if fim <= inicio:
        raise EntradaInvalida(
            f"o corte do manifesto posterior ({fim}) não é depois do corte do run ({inicio})"
        )
    referencias = {nome: _retorno_serie(posterior, (chave,), inicio, fim) for nome, chave in REFERENCIAS}

    raiz = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_ROOT
    try:
        diretorio = assert_writable_dir(raiz / fim.isoformat(), "--cache-dir")
    except ArgumentError as exc:
        raise UsoInvalido(str(exc)) from exc
    fontes = _Fontes(Cache(diretorio, offline=offline, opener=opener), inicio, fim)
    try:
        fontes.carregar_cotacoes({linha.ticker for linha in anterior.acoes + anterior.fiis})
        acoes = [fontes.medir(linha, fii=False) for linha in anterior.acoes]
        fiis = [fontes.medir(linha, fii=True) for linha in anterior.fiis]
    except (SourceError, IntegrityError) as exc:
        raise FonteIndisponivel(str(exc)) from exc

    classes = _classes(anterior.pesos_classes, acoes, fiis, referencias, posterior, inicio, fim)
    return {
        "versao": VERSAO,
        "run_id": anterior.id,
        "manifesto_run": anterior.manifest_id,
        "manifesto_posterior": posterior.manifest_id,
        "inicio": inicio.isoformat(),
        "fim": fim.isoformat(),
        "referencias": referencias,
        "carteira": _carteira(classes),
        "classes": classes,
        "ativos": {"acoes": acoes, "fiis": fiis},
        "cache_dir": str(diretorio),
    }


# --- entrada ---------------------------------------------------------------------------------------------------------


def _campo(run: Any, nome: str) -> Any:
    return run.get(nome) if isinstance(run, Mapping) else getattr(run, nome, None)


def _ler_run(run: Any) -> _Run:
    if _campo(run, "plan") != "premium":
        raise EntradaInvalida(f"o run {_campo(run, 'id')} não é Premium (plan={_campo(run, 'plan')!r})")
    if _campo(run, "status") != "completed":
        raise EntradaInvalida(f"o run {_campo(run, 'id')} não está concluído (status={_campo(run, 'status')!r})")
    resultado = _campo(run, "result_json")
    if not isinstance(resultado, Mapping):
        raise EntradaInvalida(f"o run {_campo(run, 'id')} não tem result_json")
    try:
        inicio = date.fromisoformat(str(_campo(run, "snapshot_cutoff"))[:10])
    except ValueError as exc:
        raise EntradaInvalida(f"snapshot_cutoff ilegível: {_campo(run, 'snapshot_cutoff')!r}") from exc
    pesos_classes: dict[str, float] = {}
    for item in resultado.get("classes") or []:
        if not isinstance(item, Mapping) or not isinstance(item.get("key"), str):
            raise EntradaInvalida("result_json.classes ilegível")
        pesos_classes[item["key"]] = _numero(item.get("target_weight"), "classes.target_weight")
    return _Run(
        id=str(_campo(run, "id")),
        manifest_id=str(_campo(run, "snapshot_id")),
        inicio=inicio,
        pesos_classes=pesos_classes,
        acoes=_linhas(resultado, "stocks"),
        fiis=_linhas(resultado, "fiis"),
    )


def _linhas(resultado: Mapping, chave: str) -> list[_Linha]:
    """As linhas compradas da classe (quantity 0 fica de fora), com sleeve_weight renormalizado para somar 1."""
    entradas = resultado.get(chave) or []
    if not isinstance(entradas, list):
        raise EntradaInvalida(f"result_json.{chave} ilegível")
    brutas: list[tuple[str, float]] = []
    for entrada in entradas:
        if not isinstance(entrada, Mapping) or not isinstance(entrada.get("ticker"), str):
            raise EntradaInvalida(f"result_json.{chave} ilegível")
        if entrada.get("quantity") == 0:
            continue
        brutas.append((entrada["ticker"].strip().upper(), _numero(entrada.get("sleeve_weight"), f"{chave}.sleeve_weight")))
    total = sum(peso for _, peso in brutas)
    if brutas and total <= 0:
        raise EntradaInvalida(f"result_json.{chave}: pesos somam {total}")
    return [_Linha(ticker, peso / total) for ticker, peso in brutas]


def _numero(valor: Any, nome: str) -> float:
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        raise EntradaInvalida(f"{nome} não é número: {valor!r}")
    return float(valor)


def _ler_manifesto(manifesto: SnapshotManifest | Path | str) -> SnapshotManifest:
    if not isinstance(manifesto, SnapshotManifest):
        caminho = Path(manifesto)
        if not caminho.is_file():
            raise EntradaAusente(f"manifesto não encontrado: {caminho}")
        manifesto = caminho
    try:
        checado = validate_manifest(manifesto)
    except (SnapshotManifestError, SnapshotError) as exc:
        raise EntradaInvalida(f"manifesto inválido: {exc}") from exc
    if checado.state not in (None, "published"):
        raise EntradaInvalida(f"o manifesto {checado.manifest_id} não está publicado (state={checado.state})")
    return checado


# --- séries do manifesto ---------------------------------------------------------------------------------------------


def _nivel_brl(manifesto: SnapshotManifest, chaves: tuple[str, ...]) -> dict[date, float]:
    resultado: dict[date, float] | None = None
    for chave in chaves:
        try:
            niveis = read_levels_csv(manifesto.source_for(chave).resolved_path())
        except (SnapshotManifestError, SnapshotError) as exc:
            raise EntradaInvalida(f"série {chave} do manifesto posterior ilegível: {exc}") from exc
        serie = niveis["value"] if "value" in niveis else next(iter(niveis.values()))
        resultado = dict(serie) if resultado is None else {
            dia: resultado[dia] * serie[dia] for dia in resultado.keys() & serie.keys()
        }
    return resultado or {}


def _retorno_serie(manifesto: SnapshotManifest, chaves: tuple[str, ...], inicio: date, fim: date) -> dict:
    serie = _nivel_brl(manifesto, chaves)
    antes = [dia for dia in serie if dia <= inicio]
    if not antes or fim not in serie:
        raise EntradaInvalida(f"série {'×'.join(chaves)} do manifesto posterior não cobre {inicio}..{fim}")
    dia_inicio = max(antes)
    return {
        "retorno": serie[fim] / serie[dia_inicio] - 1,
        "data_inicio": dia_inicio.isoformat(),
        "data_fim": fim.isoformat(),
    }


# --- preços e proventos ----------------------------------------------------------------------------------------------


class _Fontes:
    def __init__(self, cache: Cache, inicio: date, fim: date):
        self.cache = cache
        self.inicio = inicio
        self.fim = fim
        self.cotacoes = cotahist.Quotes()

    def carregar_cotacoes(self, tickers: set[str]) -> None:
        anos = range(self.inicio.year, self.fim.year + 1)
        caminhos = [self.cache.get(sources.cotahist_url(ano), f"COTAHIST_A{ano}.ZIP") for ano in anos]
        self.cotacoes = cotahist.load(caminhos, tickers, codbdi=cotahist.QUOTE_CODBDI + cotahist.FII_CODBDI)
        ultima = self.cotacoes.sessions[-1] if self.cotacoes.sessions else None
        if ultima is None or ultima < self.fim:
            raise FonteIndisponivel(
                f"o COTAHIST do cache vai até {ultima}, antes do corte {self.fim}; "
                f"apague {self.cache.root} e rode de novo depois que a B3 publicar o dia"
            )

    def _proxima_sessao(self, dia: date) -> date | None:
        return cotahist.next_session(self.cotacoes, dia)

    def medir(self, linha: _Linha, *, fii: bool) -> dict:
        q, ticker = self.cotacoes, linha.ticker
        inicio = cotahist.quote_at(q, ticker, self.inicio)
        fim = cotahist.quote_at(q, ticker, self.fim, max_age_days=IDADE_MAXIMA_PRECO_FIM_DIAS)
        marcas = []
        if inicio is None:
            marcas.append("sem_preco_inicio")
        if fim is None:
            marcas.append("sem_preco_fim")
        ativo: dict[str, Any] = {
            "ticker": ticker,
            "peso_na_classe": linha.peso,
            "data_preco_inicio": inicio[0].isoformat() if inicio else None,
            "preco_inicio": inicio[1] if inicio else None,
            "data_preco_fim": fim[0].isoformat() if fim else None,
            "preco_fim": fim[1] if fim else None,
        }
        eventos = [] if fii else self._eventos(ticker, marcas)
        ativo["eventos"] = [
            {"tipo": evento.label, "multiplicador": evento.multiplier, "data_ex": data_ex.isoformat()}
            for evento, data_ex in eventos
        ]
        proventos: list[dividends.Dividend] = []
        ativo["fonte_proventos"] = ativo["motivo_fonte_proventos"] = None
        if inicio is not None and fim is not None:
            fonte, motivo, lista = self._proventos_fii(ticker) if fii else self._proventos_acao(ticker)
            ativo["fonte_proventos"], ativo["motivo_fonte_proventos"] = fonte, motivo
            if lista is None:
                marcas.append("sem_fonte_de_proventos")
            proventos = [item for item in lista or [] if item.ex_date and self.inicio < item.ex_date <= self.fim]
        ativo["proventos"] = [{"data_ex": item.ex_date.isoformat(), "valor": item.amount} for item in proventos]
        ativo["retorno_preco"] = ativo["retorno"] = None
        if inicio is not None and fim is not None:
            def quantidade(ate: date, inclusive: bool = True) -> float:
                fator = 1.0
                for evento, data_ex in eventos:
                    if data_ex < ate or (inclusive and data_ex == ate):
                        fator *= evento.multiplier
                return fator

            ativo["retorno_preco"] = fim[1] * quantidade(self.fim) / inicio[1] - 1
            recebido = sum(item.amount * quantidade(item.ex_date, inclusive=False) for item in proventos)
            ativo["retorno"] = (fim[1] * quantidade(self.fim) + recebido) / inicio[1] - 1
        ativo["marcas"] = marcas
        return ativo

    def _eventos(self, ticker: str, marcas: list[str]) -> list[tuple[dividends.StockEvent, date]]:
        empresa = self._empresa(ticker[:4])
        if empresa is None:
            marcas.append("eventos_indisponiveis")
            return []
        q = self.cotacoes
        isins = {cotahist.isin(q, ticker, self.inicio), *cotahist.isins(q, ticker, self.inicio, self.fim)} - {None}
        return dividends.class_events(empresa.stock_events, isins, self.inicio, self.fim, self._proxima_sessao)

    def _empresa(self, raiz: str) -> dividends.B3Company | None:
        caminho = self.cache.get(sources.b3_company_url(raiz), b3_company_name(raiz), required=False)
        return _ler(caminho, dividends.parse_b3_company)

    def _historico_b3(self, nome: str) -> dividends.B3History | None:
        consulta = sources.b3_query_name(nome)
        paginas: list[bytes] = []
        total = pagina = 1
        while pagina <= total:
            caminho = self.cache.get(sources.b3_cash_url(consulta, pagina), b3_cash_name(consulta, pagina), required=False)
            if caminho is None:
                break
            paginas.append(caminho.read_bytes())
            if pagina == 1:
                try:
                    total = int(json.loads(paginas[0])["page"]["totalPages"])
                except (ValueError, KeyError, TypeError):
                    return None
            pagina += 1
        if not paginas:
            return None
        try:
            return dividends.parse_b3_cash(paginas)
        except (ValueError, KeyError, TypeError):
            return None

    def _yahoo(self, ticker: str) -> list[dividends.Dividend] | None:
        periodo = self.fim + timedelta(days=1)
        caminho = self.cache.get(sources.yahoo_url(ticker, periodo), yahoo_name(ticker, periodo), required=False)
        return _ler(caminho, dividends.parse_yahoo)

    def _proventos_acao(self, ticker: str) -> tuple[str, str | None, list[dividends.Dividend] | None]:
        empresa = self._empresa(ticker[:4])
        historico = self._historico_b3(empresa.trading_name) if empresa is not None else None
        escolha = dividends.choose_source(historico, self.fim)
        da_b3 = None
        if escolha[0] == "b3":
            classe = cotahist.share_class(self.cotacoes, ticker, self.inicio)
            registros = dividends.class_records(historico.records, classe or "")
            if classe == "UNT" and not registros:
                escolha = ("yahoo", "unit_sem_registros_unt")
            else:
                da_b3 = dividends.b3_ex_dates(registros, self._proxima_sessao)
        yahoo = self._yahoo(ticker) if escolha[0] != "b3" else None
        return dividends.resolve(escolha, da_b3, yahoo)

    def _proventos_fii(self, ticker: str) -> tuple[str, str | None, list[dividends.Dividend] | None]:
        raiz = ticker[:4]
        caminho = self.cache.get(fii_sources.b3_fund_url(raiz), fii_sources.b3_fund_name(raiz), required=False)
        fundo = _ler(caminho, fii_dividends.parse_b3_fund)
        isin = cotahist.isin(self.cotacoes, ticker, self.inicio)
        da_b3 = None if fundo is None or isin is None else fii_dividends.b3_dividends(fundo, isin, self._proxima_sessao)
        if da_b3 and min(item.ex_date for item in da_b3) <= self.inicio:
            return "b3", None, da_b3
        return fii_dividends.choose(da_b3, self._yahoo(ticker), self.inicio)


def _ler(caminho: Path | None, parser):
    if caminho is None:
        return None
    try:
        return parser(caminho.read_bytes())
    except (ValueError, KeyError, TypeError, IndexError, AttributeError):
        return None  # uma resposta que não é o documento esperado conta como fonte que falhou


# --- agregação -------------------------------------------------------------------------------------------------------


def _agregar(ativos: list[dict]) -> dict:
    medidos = [ativo for ativo in ativos if ativo["retorno"] is not None]
    peso = sum(ativo["peso_na_classe"] for ativo in medidos)
    return {
        "retorno_ponderado": (
            sum(ativo["peso_na_classe"] * ativo["retorno"] for ativo in medidos) / peso if peso > 0 else None
        ),
        "retorno_peso_igual": sum(ativo["retorno"] for ativo in medidos) / len(medidos) if medidos else None,
        "ativos": len(ativos),
        "ativos_com_retorno": len(medidos),
        "peso_coberto": peso,
        "sem_retorno": [ativo["ticker"] for ativo in ativos if ativo["retorno"] is None],
    }


def _classes(
    pesos: Mapping[str, float],
    acoes: list[dict],
    fiis: list[dict],
    referencias: Mapping[str, dict],
    manifesto: SnapshotManifest,
    inicio: date,
    fim: date,
) -> dict:
    classes: dict[str, dict] = {}
    for chave, peso in pesos.items():
        if chave in (CLASSE_ACOES, CLASSE_FIIS):
            item = _agregar(acoes if chave == CLASSE_ACOES else fiis)
            item["referencia"] = "IBOV" if chave == CLASSE_ACOES else "IFIX"
            item["retorno_referencia"] = referencias[item["referencia"]]["retorno"]
        elif chave in CLASSES_POR_SERIE:
            rotulo, series = CLASSES_POR_SERIE[chave]
            retorno = _retorno_serie(manifesto, series, inicio, fim)["retorno"]
            item = {"retorno_ponderado": retorno, "retorno_peso_igual": retorno, "fonte": rotulo}
        else:
            item = {"retorno_ponderado": None, "retorno_peso_igual": None, "fonte": "classe desconhecida"}
        classes[chave] = {"peso_alvo": peso, **item}
    return classes


def _carteira(classes: Mapping[str, dict]) -> dict:
    resultado: dict[str, Any] = {}
    for versao in ("retorno_ponderado", "retorno_peso_igual"):
        medidas = [(item["peso_alvo"], item[versao]) for item in classes.values() if item[versao] is not None]
        peso = sum(peso for peso, _ in medidas)
        resultado[versao] = sum(peso * retorno for peso, retorno in medidas) / peso if peso > 0 else None
        resultado["peso_coberto"] = peso
    resultado["classes_sem_retorno"] = [chave for chave, item in classes.items() if item["retorno_ponderado"] is None]
    return resultado


# --- CLI -------------------------------------------------------------------------------------------------------------


def _ler_run_do_banco(run_id: str, database_url: str):
    api = PROJECT_ROOT / "api"
    if str(api) not in sys.path:
        sys.path.insert(0, str(api))
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from app.db.models import RecommendationRun

    engine = create_engine(database_url)
    try:
        with Session(engine) as session:
            registro = session.get(RecommendationRun, run_id)
            if registro is None:
                raise EntradaAusente(f"run não encontrado: {run_id}")
            return {
                campo: getattr(registro, campo)
                for campo in ("id", "plan", "status", "snapshot_id", "snapshot_cutoff", "result_json")
            }
    finally:
        engine.dispose()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Teste para frente de um run Premium contra IBOV, IFIX e CDI.")
    parser.add_argument("--manifest", required=True, type=Path, help="manifesto posterior publicado")
    origem = parser.add_mutually_exclusive_group(required=True)
    origem.add_argument("--run-id", help="id do run no banco (DATABASE_URL ou --database-url)")
    origem.add_argument("--run-json", type=Path, help="run em JSON (id, plan, status, snapshot_id, snapshot_cutoff, result_json)")
    parser.add_argument("--database-url", default=None, help="URL SQLAlchemy do banco; padrão: $DATABASE_URL")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--offline", action="store_true", help="só o cache, sem rede")
    parser.add_argument("--saida", type=Path, default=None, help="arquivo JSON de saída; padrão: stdout")
    return parser


def main(argv: list[str] | None = None, *, opener: Callable = urllib.request.urlopen) -> int:
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    try:
        if args.run_json is not None:
            if not args.run_json.is_file():
                raise EntradaAusente(f"run não encontrado: {args.run_json}")
            try:
                run = json.loads(args.run_json.read_text(encoding="utf-8"))
            except ValueError as exc:
                raise EntradaInvalida(f"{args.run_json} não é JSON: {exc}") from exc
        else:
            url = args.database_url or os.environ.get("DATABASE_URL")
            if not url:
                raise UsoInvalido("--run-id precisa de --database-url ou DATABASE_URL")
            run = _ler_run_do_banco(args.run_id, url)
        resultado = teste_para_frente(run, args.manifest, cache_dir=args.cache_dir, offline=args.offline, opener=opener)
        texto = json.dumps(resultado, ensure_ascii=False, indent=2) + "\n"
        if args.saida is None:
            sys.stdout.write(texto)
        else:
            try:
                assert_writable_dir(args.saida.parent, "--saida")
            except ArgumentError as exc:
                raise UsoInvalido(str(exc)) from exc
            args.saida.parent.mkdir(parents=True, exist_ok=True)
            args.saida.write_text(texto, encoding="utf-8")
    except TesteParaFrenteError as exc:
        print(f"teste para frente: {exc}", file=sys.stderr)
        return exc.exit_code
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
