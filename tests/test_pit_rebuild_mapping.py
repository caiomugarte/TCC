"""Ticker mapping, units, financial group and the quoted universe (MAP-01, MAP-02, MAP-03, IND-04, A-9, A-10, A-17).

The fixtures in tests/fixtures/pit_rebuild/fca/ are two FCA valor mobiliario years (2024 and 2025,
UTF-8, zipped as latin-1 by the test) and a cadastro (latin-1). Companies are fictional. The
2024 year lists BRKM5 alone; the 2025 year lists TAEE11 (1 ON / 2 PN), KLBN11 (1 KLBN3 + 4 KLBN4,
KLBN3 and KLBN4 unlisted), EUCA3 and EUCA4, USIM5 (USIM6 unlisted), OMEG3 (CNPJ not in the
cadastro) and ZETA11 (an unparseable composition).
"""

import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from pit_rebuild import cvm, mapping  # noqa: E402
from pit_rebuild.dividends import B3Company  # noqa: E402
from pit_rebuild.errors import IntegrityError  # noqa: E402
from pit_rebuild_support import make_zip  # noqa: E402

FIXTURES = PROJECT_ROOT / "tests" / "fixtures" / "pit_rebuild" / "fca"
D = date(2026, 8, 7)
EUCATEX = "66.666.001/0001-01"
KLABIN = "77.777.002/0001-02"
USIMINAS = "88.888.003/0001-03"
BRASKEM = "44.444.005/0001-05"
EPSILON = "33.333.004/0001-04"
B3_ROOTS = {"USIM": B3Company("GAMA-SIDER", "1603", None)}


class MappingCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        zips = []
        for year in (2024, 2025):
            member = f"fca_cia_aberta_valor_mobiliario_{year}.csv"
            path = tmp / f"fca_cia_aberta_{year}.zip"
            path.write_bytes(make_zip({member: (FIXTURES / member).read_text(encoding="utf-8")}))
            zips.append(path)
        cls.fca = mapping.load_fca(zips)
        cls.cadastro = cvm.load_cadastro(FIXTURES / "cad_cia_aberta.csv")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def map_one(self, ticker, b3=None):
        return mapping.map_tickers([ticker], self.fca, self.cadastro, B3_ROOTS if b3 is None else b3, D)[ticker]


class MapTickerTests(MappingCase):
    def test_a_ticker_listed_only_in_an_older_fca_year_maps_from_that_year(self):
        self.assertEqual(self.map_one("BRKM5"), mapping.Mapped(BRASKEM, "1605", "fca"))

    def test_a_ticker_absent_from_every_fca_year_maps_through_the_b3_record_of_its_root(self):
        # USIM6 is in no FCA year; the B3 record of root USIM gives codeCVM 1603, the cadastro gives the CNPJ.
        self.assertEqual(self.map_one("USIM6"), mapping.Mapped(USIMINAS, "1603", "b3"))

    def test_reason_not_in_fca_or_b3_when_no_fca_row_and_no_b3_record_for_the_root(self):
        self.assertEqual(self.map_one("NONE3"), mapping.Unmapped("not_in_fca_or_b3"))

    def test_reason_no_cvm_code_when_the_company_is_not_in_the_cadastro(self):
        self.assertEqual(self.map_one("OMEG3"), mapping.Unmapped("no_cvm_code"))

    def test_reason_unit_composition_unparsed_for_a_unit_whose_composition_is_not_read(self):
        self.assertEqual(self.map_one("ZETA11"), mapping.Unmapped("unit_composition_unparsed"))

    def test_a_unit_takes_its_composition_from_the_fca_row(self):
        self.assertEqual(self.map_one("TAEE11"), mapping.Mapped(EPSILON, "1604", "fca", {"ON": 1, "PN": 2}))
        self.assertEqual(
            self.map_one("KLBN11"),
            mapping.Mapped(KLABIN, "1602", "fca", {"KLBN3": 1, "KLBN4": 4}),
        )

    def test_a_unit_with_an_r5_composition_string_maps_with_its_composition(self):
        # ALUP11's FCA composition "1 ON e 2 PN" is read, not reported as unit_composition_unparsed.
        row = mapping.FcaRow(
            ticker="ALUP11", cnpj=EUCATEX, valor_mobiliario=mapping.UNIT,
            composition="1 ON e 2 PN", data_referencia=date(2025, 12, 31), versao=1,
        )
        mapped = mapping.map_tickers(["ALUP11"], mapping.FcaIndex([row]), self.cadastro, {}, D)
        self.assertEqual(mapped["ALUP11"], mapping.Mapped(EUCATEX, self.cadastro[EUCATEX]["cd_cvm"], "fca", {"ON": 1, "PN": 2}))

    def test_the_fca_row_used_has_the_latest_data_referencia_on_or_before_d(self):
        # EUCA4 has rows for 2024-12-31 and 2025-12-31: at 2025-06-30 only the first is on or before the date.
        self.assertEqual(self.fca.latest("EUCA4", date(2025, 6, 30)).data_referencia, date(2024, 12, 31))
        self.assertEqual(self.fca.latest("EUCA4", D).data_referencia, date(2025, 12, 31))


class ParseCompositionTests(unittest.TestCase):
    def test_class_and_ticker_compositions_are_read_and_anything_else_is_none(self):
        self.assertEqual(mapping.parse_composition("1 ON / 2 PN"), {"ON": 1, "PN": 2})
        self.assertEqual(mapping.parse_composition("1 KLBN3 + 4 KLBN4"), {"KLBN3": 1, "KLBN4": 4})
        self.assertEqual(mapping.parse_composition("2 ON e 1 PN"), {"ON": 2, "PN": 1})  # F2: a standalone e splits
        self.assertIsNone(mapping.parse_composition(""))
        self.assertIsNone(mapping.parse_composition(None))

    def test_real_fca_unit_strings_parse(self):
        cases = {
            "1 ON E 2 PN": {"ON": 1, "PN": 2},
            "1 ON e 4 PN": {"ON": 1, "PN": 4},
            "1 ON e 2 PN": {"ON": 1, "PN": 2},
            "2 ações preferenciais e 1 ação ordinária": {"PN": 2, "ON": 1},
            "1 ação ordinária e 4 ações preferenciais": {"ON": 1, "PN": 4},
            "1 ENG3 e 4 ENGI4": {"ENG3": 1, "ENGI4": 4},
            "1 ON+ 2PN": {"ON": 1, "PN": 2},
            "1 ON e 2PNs": {"ON": 1, "PN": 2},  # PNs is PN: no PNs key
            "1 ação ON + 2 ações PN": {"ON": 1, "PN": 2},
            "1 ON E 2 PNA": {"ON": 1, "PNA": 2},
            "1 ON + 1 ON": {"ON": 2},
            "1 ON / 2 PN": {"ON": 1, "PN": 2},
            "1 KLBN3 + 4 KLBN4": {"KLBN3": 1, "KLBN4": 4},
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(mapping.parse_composition(text), expected)

    def test_any_other_part_leaves_the_composition_unparsed(self):
        for text in [
            "1 ON e duas PN",  # a count written in words
            "1 BDR",  # a class the grammar does not name
            "ON + PN",  # a part without a count
            "1 ON - 2 PN",  # a separator the grammar does not split on
            "1 ação especial",  # a share class word the grammar does not name
            "1 ON e",  # an empty part after the standalone e
            "1 ONe 2 PN",  # the e is part of a word, so the part is unparsed
        ]:
            with self.subTest(text=text):
                self.assertIsNone(mapping.parse_composition(text))


class GroupTests(MappingCase):
    def test_classify_by_setor_ativ_for_each_of_the_seven_financial_values(self):
        cnpjs = [
            ("10.000.006/0001-06", "Bancos"),
            ("10.000.007/0001-07", "Emp. Adm. Part. - Bancos"),
            ("10.000.008/0001-08", "Seguradoras e Corretoras"),
            ("10.000.009/0001-09", "Emp. Adm. Part. - Seguradoras e Corretoras"),
            ("10.000.010/0001-10", "Intermediação Financeira"),
            ("10.000.011/0001-11", "Emp. Adm. Part. - Intermediação Financeira"),
            ("10.000.012/0001-12", "Arrendamento Mercantil"),
        ]
        for cnpj, setor in cnpjs:
            with self.subTest(setor=setor):
                self.assertEqual(self.cadastro[cnpj]["setor_ativ"], setor)
                group = mapping.classify(mapping.Mapped(cnpj, "0", "fca"), self.cadastro, None, "Consumo Cíclico")
                self.assertEqual(group, {"group": "financial", "rule": "setor_ativ"})

    def test_classify_by_dre_3_01_when_the_setor_ativ_is_not_financial(self):
        group = mapping.classify(
            mapping.Mapped(EUCATEX, "1601", "fca"),
            self.cadastro,
            "Receitas da Intermediação Financeira",
            "Consumo Cíclico",
        )
        self.assertEqual(group, {"group": "financial", "rule": "dre_3_01"})

    def test_an_unmapped_ticker_outside_financeiro_e_outros_is_non_financial_by_si_setor(self):
        group = mapping.classify(mapping.Unmapped("not_in_fca_or_b3"), self.cadastro, None, "Consumo Cíclico")
        self.assertEqual(group, {"group": "non_financial", "rule": "si_setor"})

    def test_an_unmapped_ticker_with_financeiro_e_outros_is_unclassified(self):
        group = mapping.classify(mapping.Unmapped("not_in_fca_or_b3"), self.cadastro, None, "Financeiro e Outros")
        self.assertEqual(group, {"group": "unclassified", "rule": None})

    def test_a_mapped_ticker_with_financeiro_e_outros_stays_non_financial(self):
        group = mapping.classify(
            mapping.Mapped(EUCATEX, "1601", "fca"),
            self.cadastro,
            "Receita de Venda de Bens e/ou Serviços",
            "Financeiro e Outros",
        )
        self.assertEqual(group, {"group": "non_financial", "rule": None})


class QuotedUniverseTests(MappingCase):
    def test_quote_tickers_takes_each_source_and_leaves_the_unrelated_ticker_out(self):
        eligible = ["EUCA4", "KLBN11", "USIM5"]
        mapped = mapping.map_tickers(eligible, self.fca, self.cadastro, B3_ROOTS, D)
        si_tickers = {"EUCA4", "KLBN11", "USIM5", "USIM6", "PETR4"}  # SI tickers of both dates
        self.assertEqual(
            mapping.quote_tickers(eligible, mapped, self.fca, si_tickers),
            {"EUCA3", "EUCA4", "KLBN3", "KLBN4", "KLBN11", "USIM5", "USIM6"},
        )


class FcaFileTests(unittest.TestCase):
    def test_an_fca_file_without_codigo_negociacao_is_an_integrity_error(self):
        member = "fca_cia_aberta_valor_mobiliario_2025.csv"
        text = (FIXTURES / member).read_text(encoding="utf-8").replace("Codigo_Negociacao", "Codigo_Papel")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fca_cia_aberta_2025.zip"
            path.write_bytes(make_zip({member: text}))
            with self.assertRaises(IntegrityError) as ctx:
                mapping.load_fca([path])
        self.assertEqual(ctx.exception.exit_code, 4)
        self.assertIn("Codigo_Negociacao", ctx.exception.reason)


if __name__ == "__main__":
    unittest.main()
