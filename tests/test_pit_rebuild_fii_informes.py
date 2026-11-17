"""The CVM FII monthly reports read point in time (issue #60): which report a fund has at d, the ISIN map, coverage.

Every expectation is worked out by hand from the reports written in each test. No network.
"""

from datetime import date
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from pit_rebuild import fii_informes  # noqa: E402
from pit_rebuild_support import make_zip  # noqa: E402

GERAL = "Tipo_Fundo_Classe;CNPJ_Fundo_Classe;Data_Referencia;Versao;Data_Entrega;Nome_Fundo_Classe;Codigo_ISIN;Segmento_Atuacao;Tipo_Gestao\n"
COMPLEMENTO = "CNPJ_Fundo_Classe;Data_Referencia;Versao;Total_Numero_Cotistas;Valor_Ativo;Patrimonio_Liquido;Cotas_Emitidas;Valor_Patrimonial_Cotas\n"
ATIVO = "CNPJ_Fundo_Classe;Data_Referencia;Versao;Total_Necessidades_Liquidez;Disponibilidades\n"
A = "11.111.111/0001-11"
B = "22.222.222/0001-22"


def year_zip(year: int, geral: str, complemento: str, ativo: str) -> bytes:
    return make_zip({
        f"inf_mensal_fii_geral_{year}.csv": GERAL + geral,
        f"inf_mensal_fii_complemento_{year}.csv": COMPLEMENTO + complemento,
        f"inf_mensal_fii_ativo_passivo_{year}.csv": ATIVO + ativo,
    })


class InformesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        z2023 = root / "inf_mensal_fii_2023.zip"
        z2023.write_bytes(year_zip(
            2023,
            f"Classe;{A};2023-06-01;1;2023-07-14;ALFA FII;BRALFACTF001;Shoppings;Ativa\n",
            f"{A};2023-06-01;1;40000;1010000000;1000000000;10000000;100\n",
            f"{A};2023-06-01;1;10000000;2000000\n",
        ))
        z2026 = root / "inf_mensal_fii_2026.zip"
        z2026.write_bytes(year_zip(
            2026,
            f"Classe;{A};2026-06-01;1;2026-07-15;ALFA FII;BRALFACTF001;Shoppings;Ativa\n"
            f"Classe;{A};2026-07-01;1;2026-08-14;ALFA FII;BRALFACTF001;Shoppings;Ativa\n"
            f"Classe;{B};2026-05-01;1;2026-06-15;DECOY FII;BRALFACTF001;Outros;Definida\n",
            f"{A};2026-06-01;1;50000;1400000000;1331000000;10000000;133,1\n"
            f"{A};2026-07-01;1;51000;1400000000;1340000000;10000000;134\n"
            f"{B};2026-05-01;1;11;100;100;100;1\n",
            f"{A};2026-06-01;1;66550000;1000\n"
            f"{A};2026-07-01;1;67000000;1000\n",
        ))
        self.informes = fii_informes.load([z2023, z2026])

    def test_the_report_at_d_is_the_newest_one_delivered_on_or_before_d(self):
        report = fii_informes.latest(self.informes, A, date(2026, 8, 10))
        self.assertEqual(report.ref, date(2026, 6, 1))
        self.assertEqual(report.delivered, date(2026, 7, 15))
        self.assertEqual(report.holders, 50000)
        self.assertEqual(report.equity, 1331000000)
        self.assertEqual(report.shares, 10000000)
        self.assertAlmostEqual(report.nav_per_share, 133.1)
        self.assertEqual(report.liquidity, 66550000)
        self.assertEqual(report.segment, "Shoppings")
        self.assertEqual(report.management, "Ativa")

    def test_a_report_delivered_after_d_is_not_known_at_d(self):
        self.assertEqual(fii_informes.latest(self.informes, A, date(2026, 8, 14)).ref, date(2026, 7, 1))
        self.assertEqual(fii_informes.delivered_after(self.informes, A, date(2026, 8, 10)), [date(2026, 7, 1)])
        self.assertIsNone(fii_informes.latest(self.informes, A, date(2023, 7, 13)))

    def test_the_report_three_years_back_is_the_same_month_known_at_d(self):
        report = fii_informes.latest(self.informes, A, date(2026, 8, 10))
        old = fii_informes.months_back(self.informes, report, 36, date(2026, 8, 10))
        self.assertEqual(old.ref, date(2023, 6, 1))
        self.assertEqual(old.nav_per_share, 100)

    def test_an_isin_names_every_fund_that_reported_it_up_to_d(self):
        self.assertEqual(fii_informes.funds_of_isin(self.informes, "BRALFACTF001", date(2026, 8, 10)), [A, B])
        self.assertEqual(fii_informes.funds_of_isin(self.informes, "BRALFACTF001", date(2026, 6, 1)), [A])
        self.assertEqual(fii_informes.funds_of_isin(self.informes, "BRZZZZCTF001", date(2026, 8, 10)), [])

    def test_coverage_gives_the_first_year_each_field_has_a_value(self):
        years = fii_informes.first_years(self.informes)
        self.assertEqual(years["Total_Numero_Cotistas"], 2023)
        self.assertEqual(years["Segmento_Atuacao"], 2023)


if __name__ == "__main__":
    unittest.main()
