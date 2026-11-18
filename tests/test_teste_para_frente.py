"""Teste para frente de um run Premium (issue #86). Sem rede: as fontes vêm de um FakeOpener."""

import contextlib
import io
import json
from datetime import date
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT / "py", PROJECT_ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from allocation_config import ASSET_CLASSES  # noqa: E402
from pit_rebuild import fii_sources, sources  # noqa: E402
from pit_rebuild_support import FakeOpener, FakeResponse  # noqa: E402
from snapshot_manifest import sha256_path  # noqa: E402
import teste_para_frente as tpf  # noqa: E402

INICIO = date(2026, 10, 30)
FIM = date(2026, 11, 27)
SESSOES = [date(2026, 10, 29), INICIO, date(2026, 11, 3), date(2026, 11, 10), date(2026, 11, 26), FIM]


def linha(ticker: str, dia: date, preco: float, *, codbdi: str = "02", especi: str = "ON", isin: str = "BRXXXXACNOR0") -> str:
    """Uma linha do COTAHIST no leiaute de 245 colunas, com PREULT e ISIN escolhidos."""
    texto = [" "] * 245
    for inicio, valor in ((1, "01"), (3, dia.strftime("%Y%m%d")), (11, codbdi), (13, ticker.ljust(12)), (25, "010"),
                          (40, especi.ljust(10)), (109, str(round(preco * 100)).zfill(13)), (171, "100000".zfill(18)),
                          (211, "1".zfill(7)), (231, isin)):
        texto[inicio - 1:inicio - 1 + len(valor)] = list(valor)
    return "".join(texto)


def cotahist_zip(linhas: list[str], ano: int = 2026) -> bytes:
    texto = "\r\n".join([f"00COTAHIST.{ano}BOVESPA {ano}1127".ljust(245), *linhas]) + "\r\n"
    buffer = io.BytesIO()
    import zipfile

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as arquivo:
        arquivo.writestr(f"COTAHIST_A{ano}.TXT", texto.encode("latin-1"))
    return buffer.getvalue()


def json_bytes(valor) -> FakeResponse:
    return FakeResponse(json.dumps(valor).encode("utf-8"))


def empresa(nome: str, eventos: list[dict] | None = None) -> FakeResponse:
    return json_bytes([{"tradingName": nome, "codeCVM": "1", "totalNumberShares": "1.000",
                        "stockDividends": eventos or []}])


def proventos_b3(registros: list[tuple[str, str, str]]) -> FakeResponse:
    """Página única do GetListedCashDividends: (classe, valor, lastDatePriorEx)."""
    resultados = [{"typeStock": classe, "valueCash": valor, "lastDatePriorEx": data, "corporateAction": "DIVIDENDO"}
                  for classe, valor, data in registros]
    return json_bytes({"page": {"pageNumber": 1, "totalRecords": len(resultados), "totalPages": 1},
                       "results": resultados})


def fundo_b3(isin: str, pagamentos: list[tuple[str, str]]) -> FakeResponse:
    """Página do fundo (GetListedSupplementFunds): (valor, lastDatePrior) de RENDIMENTO."""
    return json_bytes({"fund": "FUNDO", "quantity": "1.000", "cashDividends": [
        {"isinCode": isin, "lastDatePrior": data, "rate": valor, "label": "RENDIMENTO"} for valor, data in pagamentos]})


def escrever_niveis(caminho: Path, niveis: dict[date, float]) -> None:
    linhas = ["date,value", *(f"{dia.isoformat()},{valor}" for dia, valor in sorted(niveis.items()))]
    caminho.write_text("\n".join(linhas) + "\n", encoding="utf-8")


def construir_manifesto(raiz: Path, *, estado: str | None = "published", fim: date = FIM,
                        series: dict[str, tuple[float, float]] | None = None) -> Path:
    """Manifesto posterior em disco: cada série vai de um nível em INICIO a outro em `fim`, com um dia antes de INICIO."""
    niveis = {"ibovespa": (100.0, 103.0), "ifix": (200.0, 198.0), "di": (100.0, 101.0),
              "sp500_total_return_usd": (50.0, 55.0), "btc_usd": (1000.0, 900.0), "ptax": (5.0, 5.0)}
    niveis.update(series or {})
    alocacao = raiz / "allocation"
    alocacao.mkdir(parents=True)
    datas = [date(2026, 10, 29), INICIO, fim]
    for nome, (inicio, final) in niveis.items():
        escrever_niveis(alocacao / f"{nome}.csv", {datas[0]: inicio * 0.99, INICIO: inicio, fim: final})
    (raiz / "stocks.csv").write_text("TICKER\nAAA3\n", encoding="utf-8")
    (raiz / "fiis.csv").write_text("TICKER\nAAAA11\n", encoding="utf-8")
    payload = {
        "manifest_id": f"snapshot-{fim.isoformat()}",
        "manifest_version": "1",
        "created_at": f"{fim.isoformat()}T21:00:00+00:00",
        "cutoff_date": fim.isoformat(),
        "common_dates": [dia.isoformat() for dia in datas],
        "supported_classes": list(ASSET_CLASSES),
        "allocation_source": {"path": "allocation", "sha256": sha256_path(alocacao), "provider": "fixture"},
        "stock_source": {"path": "stocks.csv", "sha256": sha256_path(raiz / "stocks.csv"), "provider": "fixture"},
        "fii_source": {"path": "fiis.csv", "sha256": sha256_path(raiz / "fiis.csv"), "provider": "fixture"},
    }
    if estado is not None:
        payload["state"] = estado
    caminho = raiz / "manifest.json"
    caminho.write_text(json.dumps(payload), encoding="utf-8")
    return caminho


CLASSES = [
    {"key": "brazilian_stocks", "target_weight": 0.4},
    {"key": "fiis", "target_weight": 0.2},
    {"key": "international", "target_weight": 0.1},
    {"key": "fixed_income", "target_weight": 0.25},
    {"key": "crypto", "target_weight": 0.05},
]


def run(*, stocks=None, fiis=None, status="completed", plan="premium", cutoff=INICIO, classes=None) -> dict:
    return {
        "id": "run-1",
        "plan": plan,
        "status": status,
        "snapshot_id": f"snapshot-{cutoff.isoformat()}",
        "snapshot_cutoff": cutoff.isoformat(),
        "result_json": {"classes": classes or CLASSES, "stocks": stocks or [], "fiis": fiis or []},
    }


def acao(ticker: str, peso: float, **extra) -> dict:
    return {"ticker": ticker, "sleeve_weight": peso, "portfolio_weight": 0.4 * peso, **extra}


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.manifesto = construir_manifesto(self.tmp / "snapshot")
        self.rotas: dict[str, object] = {}
        self.linhas: list[str] = []
        for dia in SESSOES:  # o calendário do mercado vem de qualquer papel do arquivo
            self.linhas.append(linha("ZZZZ3", dia, 1.0))

    def calcular(self, run_anterior: dict, manifesto: Path | None = None) -> dict:
        self.rotas[sources.cotahist_url(2026)] = FakeResponse(cotahist_zip(self.linhas))
        self.opener = FakeOpener(self.rotas)
        return tpf.teste_para_frente(run_anterior, manifesto or self.manifesto,
                                     cache_dir=self.tmp / "cache", opener=self.opener)


class ReferenciasTest(Base):
    def test_ibov_ifix_e_cdi_vem_das_series_do_manifesto_posterior(self):
        resultado = self.calcular(run())

        self.assertEqual(resultado["inicio"], "2026-10-30")
        self.assertEqual(resultado["fim"], "2026-11-27")
        self.assertAlmostEqual(resultado["referencias"]["IBOV"]["retorno"], 0.03)
        self.assertAlmostEqual(resultado["referencias"]["IFIX"]["retorno"], -0.01)
        self.assertAlmostEqual(resultado["referencias"]["CDI"]["retorno"], 0.01)
        json.dumps(resultado)


ISIN_AAA = "BRAAAAACNOR0"
ISIN_FII = "BRAAAACTF000"


class CarteiraDeFixtureTest(Base):
    """Retorno conhecido: duas ações com preço, uma sem preço no fim, um provento de ação e um de FII no intervalo."""

    def montar(self):
        self.linhas += [
            linha("AAA3", INICIO, 10.0, isin=ISIN_AAA), linha("AAA3", FIM, 11.0, isin=ISIN_AAA),
            linha("BBB3", INICIO, 20.0), linha("BBB3", FIM, 19.0),
            linha("CCC3", INICIO, 5.0), linha("CCC3", date(2026, 11, 3), 4.0),  # deslistado: sem sessão perto do fim
            linha("AAAA11", INICIO, 100.0, codbdi="12", especi="CI", isin=ISIN_FII),
            linha("AAAA11", FIM, 98.0, codbdi="12", especi="CI", isin=ISIN_FII),
            linha("BBBB11", INICIO, 50.0, codbdi="12", especi="CI"), linha("BBBB11", FIM, 51.0, codbdi="12", especi="CI"),
        ]
        self.rotas[sources.b3_company_url("AAA3"[:4])] = empresa("AAA SA")
        self.rotas[sources.b3_cash_url("AAA SA", 1)] = proventos_b3([
            ("ON", "0,30", "20/10/2026"),  # data-ex 29/10: antes do início, fica de fora
            ("ON", "0,50", "03/11/2026"),  # data-ex 10/11: entra
            ("PN", "9,99", "03/11/2026"),  # outra classe
            ("ON", "0,70", "27/11/2026"),  # data-ex depois do fim
        ])
        self.rotas[sources.b3_company_url("BBB3")] = empresa("BBB SA")
        self.rotas[sources.b3_cash_url("BBB SA", 1)] = proventos_b3([("ON", "1,00", "01/10/2026")])
        self.rotas[fii_sources.b3_fund_url("AAAA")] = fundo_b3(ISIN_FII, [("0,90", "02/10/2026"), ("1,00", "03/11/2026")])
        self.rotas[fii_sources.b3_fund_url("BBBB")] = fundo_b3("BRXXXXACNOR0", [("0,40", "01/10/2026")])
        return run(
            stocks=[acao("AAA3", 0.5), acao("BBB3", 0.3), acao("CCC3", 0.2), acao("DDD3", 0.0, quantity=0)],
            fiis=[acao("AAAA11", 0.7), acao("BBBB11", 0.3)],
        )

    def test_retorno_por_ativo_inclui_o_provento_com_data_ex_no_intervalo(self):
        ativos = {item["ticker"]: item for item in self.calcular(self.montar())["ativos"]["acoes"]}

        self.assertAlmostEqual(ativos["AAA3"]["retorno"], 0.15)  # (11 + 0,50) / 10 - 1
        self.assertAlmostEqual(ativos["AAA3"]["retorno_preco"], 0.10)
        self.assertEqual(ativos["AAA3"]["proventos"], [{"data_ex": "2026-11-10", "valor": 0.5}])
        self.assertEqual(ativos["AAA3"]["fonte_proventos"], "b3")
        self.assertAlmostEqual(ativos["BBB3"]["retorno"], -0.05)
        self.assertNotIn("DDD3", ativos)  # quantity 0: não foi comprada

    def test_ticker_sem_preco_no_fim_fica_marcado_e_fora_da_media(self):
        resultado = self.calcular(self.montar())
        ativos = {item["ticker"]: item for item in resultado["ativos"]["acoes"]}

        self.assertIn("sem_preco_fim", ativos["CCC3"]["marcas"])
        self.assertIsNone(ativos["CCC3"]["retorno"])
        acoes = resultado["classes"]["brazilian_stocks"]
        self.assertAlmostEqual(acoes["retorno_ponderado"], 0.075)  # (0,5 × 0,15 + 0,3 × -0,05) / 0,8
        self.assertAlmostEqual(acoes["retorno_peso_igual"], 0.05)
        self.assertAlmostEqual(acoes["peso_coberto"], 0.8)
        self.assertEqual(acoes["sem_retorno"], ["CCC3"])
        self.assertAlmostEqual(acoes["retorno_referencia"], 0.03)

    def test_fiis_usam_proventos_da_pagina_do_fundo_na_b3(self):
        resultado = self.calcular(self.montar())
        fiis = {item["ticker"]: item for item in resultado["ativos"]["fiis"]}

        self.assertAlmostEqual(fiis["AAAA11"]["retorno"], -0.01)  # (98 + 1) / 100 - 1
        self.assertEqual(fiis["AAAA11"]["fonte_proventos"], "b3")
        self.assertAlmostEqual(fiis["BBBB11"]["retorno"], 0.02)
        classe = resultado["classes"]["fiis"]
        self.assertAlmostEqual(classe["retorno_ponderado"], -0.001)
        self.assertAlmostEqual(classe["retorno_peso_igual"], 0.005)
        self.assertAlmostEqual(classe["retorno_referencia"], -0.01)

    def test_carteira_inteira_pondera_as_classes_pelos_pesos_alvo(self):
        resultado = self.calcular(self.montar())

        self.assertAlmostEqual(resultado["classes"]["international"]["retorno_ponderado"], 0.10)
        self.assertAlmostEqual(resultado["classes"]["crypto"]["retorno_ponderado"], -0.10)
        self.assertAlmostEqual(resultado["classes"]["fixed_income"]["retorno_ponderado"], 0.01)
        # 0,4 × 0,075 + 0,2 × -0,001 + 0,1 × 0,10 + 0,25 × 0,01 + 0,05 × -0,10
        self.assertAlmostEqual(resultado["carteira"]["retorno_ponderado"], 0.0373)
        # 0,4 × 0,05 + 0,2 × 0,005 + 0,1 × 0,10 + 0,25 × 0,01 + 0,05 × -0,10
        self.assertAlmostEqual(resultado["carteira"]["retorno_peso_igual"], 0.0285)
        self.assertAlmostEqual(resultado["carteira"]["peso_coberto"], 1.0)
        json.dumps(resultado)


if __name__ == "__main__":
    unittest.main()
