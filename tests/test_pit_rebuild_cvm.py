"""CVM filings read point-in-time: usability at d, year-to-date rows, refiling, statements, cadastro (PIT-01, PIT-03, A-6, IND-01).

The fixtures live in tests/fixtures/pit_rebuild/cvm/ (UTF-8 text). They are zipped into a
temporary directory with make_zip, which writes them as latin-1 as the CVM files are.
Companies: ALFA (ITR year-to-date rows and a DFP history), BETA (fiscal year starting in June,
values in MIL), EDGE (receipt on d and on d + 1), REFILED (version 2 received after d),
INDIV (individual statements only), plus DELTA, EPSILON and GAMA used by the share tests.
"""

import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from pit_rebuild import cvm  # noqa: E402
from pit_rebuild.errors import IntegrityError  # noqa: E402
from pit_rebuild_support import make_zip  # noqa: E402

FIXTURES = PROJECT_ROOT / "tests" / "fixtures" / "pit_rebuild" / "cvm"
ALFA = "11.111.111/0001-11"
BETA = "22.222.222/0001-22"
GAMA = "33.333.333/0001-33"
DELTA = "44.444.444/0001-44"
EPSILON = "55.555.555/0001-55"
EDGE = "66.666.666/0001-66"
INDIV = "77.777.777/0001-77"
REFILED = "88.888.888/0001-88"
SOLO = "99.999.999/0001-99"
PENULTIMO = "PENÚLTIMO"


def zip_folder(folder: Path, out: Path, index_header_rename: tuple[str, str] | None = None) -> Path:
    members = {path.name: path.read_text(encoding="utf-8") for path in sorted(folder.glob("*.csv"))}
    if index_header_rename is not None:
        old, new = index_header_rename
        index_name = "dfp_cia_aberta_2025.csv"
        members[index_name] = members[index_name].replace(old, new, 1)
    out.write_bytes(make_zip(members))
    return out


def itr_only_stores(tmp: Path) -> tuple[cvm.FilingStore, cvm.FilingStore]:
    """A company with an ITR and no DFP at all, so the fiscal year before its year-to-date is missing."""
    index = (
        "CNPJ_CIA;DT_REFER;VERSAO;DENOM_CIA;CD_CVM;CATEG_DOC;ID_DOC;DT_RECEB;LINK_DOC\n"
        f"{SOLO};2026-03-31;1;SOLO S.A.;1099;ITR;1;2026-05-10 18:00:00;\n"
    )
    dre = (
        "CNPJ_CIA;DT_REFER;VERSAO;DENOM_CIA;CD_CVM;ESCALA_MOEDA;ORDEM_EXERC;DT_INI_EXERC;DT_FIM_EXERC;CD_CONTA;DS_CONTA;VL_CONTA;ST_CONTA_FIXA\n"
        f"{SOLO};2026-03-31;1;SOLO S.A.;1099;UNIDADE;ÚLTIMO;2026-01-01;2026-03-31;3.01;Receita;2800.00;S\n"
        f"{SOLO};2026-03-31;1;SOLO S.A.;1099;UNIDADE;PENÚLTIMO;2025-01-01;2025-03-31;3.01;Receita;2300.00;S\n"
    )
    itr_zip = tmp / "itr_cia_aberta_2026.zip"
    itr_zip.write_bytes(make_zip({"itr_cia_aberta_2026.csv": index, "itr_cia_aberta_DRE_con_2026.csv": dre}))
    dfp_zip = tmp / "dfp_cia_aberta_2025.zip"
    dfp_zip.write_bytes(make_zip({}))
    return cvm.load_filings([dfp_zip], cvm.DFP), cvm.load_filings([itr_zip], cvm.ITR)


class StoreCase(unittest.TestCase):
    """The DFP and ITR stores of the fixtures, built once per test class."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        cls.dfp = cvm.load_filings([zip_folder(FIXTURES / "dfp", tmp / "dfp_cia_aberta_2025.zip")], cvm.DFP)
        cls.itr = cvm.load_filings([zip_folder(FIXTURES / "itr", tmp / "itr_cia_aberta_2026.zip")], cvm.ITR)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()


class FilingStoreTests(StoreCase):
    def test_receipt_on_d_is_usable_and_receipt_on_d_plus_one_is_not(self):
        # EDGE's DFP was received 2026-08-07 12:00; its ITR 2026-08-08 09:00.
        d = date(2026, 8, 7)
        self.assertEqual([f.dt_refer for f in cvm.usable(self.dfp, EDGE, d)], [date(2025, 12, 31)])
        self.assertEqual(cvm.usable(self.dfp, EDGE, date(2026, 8, 6)), [])
        self.assertEqual(cvm.usable(self.itr, EDGE, d), [])
        self.assertEqual([f.dt_refer for f in cvm.usable(self.itr, EDGE, date(2026, 8, 8))], [date(2026, 6, 30)])

    def test_itr_year_to_date_row_is_kept_over_three_month_row_written_last(self):
        # ALFA ITR 2025-06-30: ÚLTIMO YTD 6000 (from 2025-01-01) then ÚLTIMO 3M 3000 (from 2025-04-01), 3M last.
        self.assertEqual(cvm.row_at(self.itr, ALFA, date(2025, 6, 30), "3.01"), (date(2025, 1, 1), 6000.0))
        self.assertEqual(cvm.row_at(self.itr, ALFA, date(2025, 6, 30), "3.01", PENULTIMO), (date(2024, 1, 1), 5000.0))

    def test_year_to_date_row_of_a_fiscal_year_starting_in_june_is_kept(self):
        # BETA ITR 2025-11-30 (fiscal year from 2025-06-01): YTD 22 MIL, three-month 12 MIL written last.
        self.assertEqual(cvm.row_at(self.itr, BETA, date(2025, 11, 30), "3.01"), (date(2025, 6, 1), 22000.0))
        self.assertEqual(cvm.row_at(self.itr, BETA, date(2025, 11, 30), "3.01", PENULTIMO), (date(2024, 6, 1), 20000.0))

    def test_refiled_quarter_falls_back_to_the_previous_period_and_is_flagged(self):
        # REFILED 2026-06-30: version 1 received 2026-08-05, version 2 received 2026-08-20.
        d = date(2026, 8, 7)
        usable_at_d = cvm.usable(self.itr, REFILED, d)
        self.assertEqual([f.dt_refer for f in usable_at_d], [date(2026, 3, 31)])
        self.assertTrue(cvm.refiled_after_d(self.dfp, self.itr, REFILED, d))
        after_refile = cvm.usable(self.itr, REFILED, date(2026, 8, 20))
        self.assertEqual(after_refile[0].dt_refer, date(2026, 6, 30))
        self.assertEqual((after_refile[0].versao, after_refile[0].dt_receb), (2, date(2026, 8, 20)))
        self.assertFalse(cvm.refiled_after_d(self.dfp, self.itr, REFILED, date(2026, 8, 20)))

    def test_company_without_consolidated_income_statement_is_recorded_as_individual(self):
        indiv = cvm.usable(self.dfp, INDIV, date(2026, 3, 25))
        self.assertEqual(indiv[0].statement, "individual")
        self.assertEqual(cvm.row_at(self.dfp, INDIV, date(2025, 12, 31), "3.01"), (date(2025, 1, 1), 4000.0))
        self.assertEqual(cvm.usable(self.dfp, ALFA, date(2026, 8, 7))[0].statement, "consolidated")

    def test_mil_values_are_multiplied_by_1000(self):
        # BETA DFP 2025-05-31 reports 40 in MIL; its ITR 2025-08-31 reports 10 in MIL.
        self.assertEqual(cvm.row_at(self.dfp, BETA, date(2025, 5, 31), "3.01"), (date(2024, 6, 1), 40000.0))
        self.assertEqual(cvm.row_at(self.itr, BETA, date(2025, 8, 31), "3.01"), (date(2025, 6, 1), 10000.0))

    def test_cadastro_yields_cd_cvm_and_setor_ativ(self):
        cadastro = cvm.load_cadastro(FIXTURES / "cad_cia_aberta.csv")
        self.assertEqual(
            cadastro[ALFA],
            {"cd_cvm": "1001", "setor_ativ": "Siderurgia e Metalurgia", "sit": "ATIVO"},
        )
        self.assertEqual(cadastro[BETA]["setor_ativ"], "Energia Elétrica")

    def test_index_without_receipt_date_column_is_an_integrity_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            broken = zip_folder(
                FIXTURES / "dfp",
                Path(tmp) / "dfp_cia_aberta_2025.zip",
                index_header_rename=("DT_RECEB", "DT_RECEBIDO"),
            )
            with self.assertRaises(IntegrityError) as ctx:
                cvm.load_filings([broken], cvm.DFP)
        self.assertEqual(ctx.exception.exit_code, 4)
        self.assertIn("DT_RECEB", ctx.exception.reason)


class AccountFilterTests(StoreCase):
    """load_filings(accounts=...) holds only the rows of those accounts and gives them the same values (IND-01)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        tmp = Path(cls._tmp.name)
        cls.dfp_31 = cvm.load_filings([zip_folder(FIXTURES / "dfp", tmp / "dfp_31.zip")], cvm.DFP, accounts={"3.01"})
        cls.itr_31 = cvm.load_filings([zip_folder(FIXTURES / "itr", tmp / "itr_31.zip")], cvm.ITR, accounts={"3.01"})

    def test_a_kept_account_gives_the_same_ttm_as_the_full_load(self):
        # The ALFA case of test_ttm_after_an_itr: 12000 + 6000 - 5000 = 13000.
        full = cvm.ttm(self.dfp, self.itr, ALFA, date(2025, 8, 14), "3.01")
        filtered = cvm.ttm(self.dfp_31, self.itr_31, ALFA, date(2025, 8, 14), "3.01")
        self.assertEqual(full, cvm.Dated(13000.0, date(2025, 8, 10)))
        self.assertEqual(filtered, full)

    def test_an_account_left_out_is_absent_from_the_filtered_load(self):
        day = date(2025, 8, 14)
        self.assertIsNotNone(cvm.balance(self.dfp, self.itr, ALFA, day, "1"))
        self.assertIsNone(cvm.balance(self.dfp_31, self.itr_31, ALFA, day, "1"))


class AccessorTests(StoreCase):
    def test_ttm_after_an_itr_is_fy_prev_plus_ytd_minus_prior_ytd(self):
        # ALFA at 2025-08-14: ITR 2025-06-30 (received 2025-08-10): 12000 + 6000 - 5000.
        # The three-month rows would give 12000 + 3000 - 2600 = 12400.
        value = cvm.ttm(self.dfp, self.itr, ALFA, date(2025, 8, 14), "3.01")
        self.assertEqual(value, cvm.Dated(13000.0, date(2025, 8, 10)))
        self.assertNotEqual(value.value, 12400.0)

    def test_ttm_of_a_fiscal_year_starting_in_june_uses_the_dfp_ending_2025_05_31(self):
        # BETA at 2026-01-12: ITR 2025-11-30 (fiscal year from 2025-06-01): 40000 (MIL) + 22000 - 20000.
        value = cvm.ttm(self.dfp, self.itr, BETA, date(2026, 1, 12), "3.01")
        self.assertEqual(value, cvm.Dated(42000.0, date(2026, 1, 12)))

    def test_ttm_after_a_dfp_is_its_annual_value(self):
        # At 2025-05-01 the ITR of 2025-03-31 (received 2025-05-12) is not usable; the DFP 2024 is.
        self.assertEqual(cvm.ttm(self.dfp, self.itr, ALFA, date(2025, 5, 1), "3.01"), cvm.Dated(12000.0, date(2025, 3, 20)))

    def test_ttm_is_none_when_the_fiscal_year_before_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty_dfp, solo_itr = itr_only_stores(Path(tmp))
            self.assertIsNone(cvm.ttm(empty_dfp, solo_itr, SOLO, date(2026, 5, 10), "3.01"))

    def test_ttm_year_before_is_the_same_flow_one_year_earlier(self):
        d = date(2026, 8, 7)
        self.assertEqual(cvm.ttm(self.dfp, self.itr, ALFA, d, "3.01"), cvm.Dated(16000.0, date(2026, 8, 6)))
        # 2025-06-30 TTM: 12000 + 6000 - 5000.
        self.assertEqual(cvm.ttm_year_before(self.dfp, self.itr, ALFA, d, "3.01"), cvm.Dated(13000.0, date(2025, 8, 10)))

    def test_annual_reads_the_dfp_five_fiscal_years_back_and_is_none_when_absent(self):
        d = date(2026, 8, 7)
        self.assertEqual(cvm.annual(self.dfp, ALFA, d, "3.01"), cvm.Dated(15000.0, date(2026, 3, 26)))
        self.assertEqual(cvm.annual(self.dfp, ALFA, d, "3.01", years_back=5), cvm.Dated(8000.0, date(2021, 3, 30)))
        self.assertIsNone(cvm.annual(self.dfp, BETA, date(2026, 1, 12), "3.01", years_back=5))

    def test_balance_takes_the_account_of_the_newest_usable_period(self):
        d = date(2026, 8, 7)
        self.assertEqual(cvm.balance(self.dfp, self.itr, ALFA, d, "2.03"), cvm.Dated(31000.0, date(2026, 8, 6)))
        self.assertEqual(cvm.balance(self.dfp, self.itr, ALFA, d, "1"), cvm.Dated(66000.0, date(2026, 8, 6)))
        # ITR 2025-06-30 is the newest usable period at 2026-03-01; the DFP 2024 is at 2025-05-01.
        self.assertEqual(cvm.balance(self.dfp, self.itr, ALFA, date(2026, 3, 1), "2.03").value, 26000.0)
        self.assertEqual(cvm.balance(self.dfp, self.itr, ALFA, date(2025, 5, 1), "2.03").value, 25000.0)

    def test_shares_are_the_issued_counts_of_the_newest_usable_filing_treasury_not_subtracted(self):
        # GAMA: the ITR 2026-03-31 (received 2026-05-14) is newer than the DFP 2025-12-31 (ON 700000000).
        self.assertEqual(
            cvm.shares(self.dfp, self.itr, GAMA, date(2026, 8, 7)),
            cvm.Shares(on=705260684, pref=547818424, total=1253079108, dt_receb=date(2026, 5, 14)),
        )

    def test_shares_are_none_when_the_newest_total_is_zero_even_if_an_older_filing_has_counts(self):
        # DELTA 2025-12-31 files 0;0;0; its 2024-12-31 filing (1000/2000/3000) is not used instead.
        self.assertIsNone(cvm.shares(self.dfp, self.itr, DELTA, date(2026, 8, 7)))

    def test_shares_are_none_when_the_newest_filing_has_no_composition_row(self):
        # EPSILON 2025-12-31 has no composicao_capital row; its 2024-12-31 row (500/500) is not used.
        self.assertIsNone(cvm.shares(self.dfp, self.itr, EPSILON, date(2026, 8, 7)))

    def test_share_scale_picks_the_factor_that_matches_b3_or_is_unverified(self):
        self.assertEqual(cvm.share_scale(31_258 + 61_361, 92_619_256), (1000, True))  # EUCATEX, counts in thousands
        self.assertEqual(cvm.share_scale(1_253_079_108, 1_253_079_108), (1, True))
        self.assertEqual(cvm.share_scale(92_619, None), (1, False))
        self.assertEqual(cvm.share_scale(92_619, 0), (1, False))

    def test_share_scale_refuses_a_zero_count(self):
        with self.assertRaises(ValueError):
            cvm.share_scale(0, 92_619_256)

    def test_dre_description_is_read_from_the_newest_usable_filing(self):
        self.assertEqual(
            cvm.dre_description(self.dfp, self.itr, ALFA, date(2026, 8, 7)),
            "Receita de Venda de Bens e/ou Serviços",
        )
        self.assertEqual(
            cvm.dre_description(self.dfp, self.itr, INDIV, date(2026, 3, 25)),
            "Receitas da Intermediação Financeira",
        )
        self.assertIsNone(cvm.dre_description(self.dfp, self.itr, EDGE, date(2026, 8, 6)))

    def test_every_value_carries_a_dt_receb_not_after_d(self):
        for d in (date(2025, 5, 1), date(2025, 8, 14), date(2026, 3, 1), date(2026, 8, 7)):
            for cnpj in (ALFA, BETA, GAMA, DELTA, EPSILON, INDIV, REFILED):
                results = [
                    cvm.ttm(self.dfp, self.itr, cnpj, d, "3.01"),
                    cvm.ttm_year_before(self.dfp, self.itr, cnpj, d, "3.01"),
                    cvm.annual(self.dfp, cnpj, d, "3.01"),
                    cvm.balance(self.dfp, self.itr, cnpj, d, "2.03"),
                    cvm.shares(self.dfp, self.itr, cnpj, d),
                ]
                for result in results:
                    if result is not None:
                        with self.subTest(cnpj=cnpj, d=d):
                            self.assertLessEqual(result.dt_receb, d)


if __name__ == "__main__":
    unittest.main()
