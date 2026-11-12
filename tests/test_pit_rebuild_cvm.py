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


INDEX_HEADER = "CNPJ_CIA;DT_REFER;VERSAO;DENOM_CIA;CD_CVM;CATEG_DOC;ID_DOC;DT_RECEB;LINK_DOC\n"
DRE_HEADER = "CNPJ_CIA;DT_REFER;VERSAO;DENOM_CIA;CD_CVM;ESCALA_MOEDA;ORDEM_EXERC;DT_INI_EXERC;DT_FIM_EXERC;CD_CONTA;DS_CONTA;VL_CONTA;ST_CONTA_FIXA\n"


def dre_line(cnpj: str, dt_refer: str, ordem: str, dt_ini: str, dt_fim: str, account: str, value: str) -> str:
    return f"{cnpj};{dt_refer};1;CASE S.A.;1;UNIDADE;{ordem};{dt_ini};{dt_fim};{account};Conta {account};{value};S\n"


def inline_store(kind: str, year: int, index_rows: str, statements: dict[str, str], accounts=None) -> cvm.FilingStore:
    """A FilingStore from inline CSV text: the index rows and the DRE rows of each scope (DRE_con, DRE_ind)."""
    prefix = kind.lower()
    members = {f"{prefix}_cia_aberta_{year}.csv": INDEX_HEADER + index_rows}
    for scope, rows in statements.items():
        members[f"{prefix}_cia_aberta_{scope}_{year}.csv"] = DRE_HEADER + rows
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{prefix}_cia_aberta_{year}.zip"
        path.write_bytes(make_zip(members))
        return cvm.load_filings([path], kind, accounts=accounts)


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

    def test_refiled_period_is_flagged_only_between_its_first_and_its_latest_delivery(self):
        # REFILED 2026-06-30: version 1 received 2026-08-05 (first delivery), version 2 received 2026-08-20 (latest).
        filing = self.itr.period(REFILED, date(2026, 6, 30)).filing
        self.assertEqual((filing.first_receb, filing.dt_receb, filing.versao), (date(2026, 8, 5), date(2026, 8, 20), 2))
        self.assertFalse(cvm.refiled_after_d(self.dfp, self.itr, REFILED, date(2026, 8, 4)))  # nothing delivered yet
        self.assertTrue(cvm.refiled_after_d(self.dfp, self.itr, REFILED, date(2026, 8, 5)))  # first delivery on d
        self.assertTrue(cvm.refiled_after_d(self.dfp, self.itr, REFILED, date(2026, 8, 19)))  # latest not yet delivered

    def test_period_whose_first_version_arrives_after_d_is_not_flagged(self):
        # EDGE's ITR 2026-06-30 has one version, received 2026-08-08, after d = 2026-08-07. EDGE's DFP 2025 (received
        # on d) is the newest usable filing, so the ITR is the only candidate for the flag. With no version at d it is
        # not a refiling.
        d = date(2026, 8, 7)
        filing = self.itr.period(EDGE, date(2026, 6, 30)).filing
        self.assertEqual((filing.first_receb, filing.dt_receb), (date(2026, 8, 8), date(2026, 8, 8)))
        self.assertEqual(cvm.usable(self.dfp, EDGE, d)[0].dt_refer, date(2025, 12, 31))
        self.assertFalse(cvm.refiled_after_d(self.dfp, self.itr, EDGE, d))

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


class StatementChoiceTests(unittest.TestCase):
    """F18: consolidated when the ÚLTIMO DRE 3.01 or 3.11 is non-zero; else individual when the filing has individual DRE rows; else consolidated."""

    CNPJ = "12.345.678/0001-90"
    DT_REFER = "2025-12-31"
    DAY = date(2026, 8, 7)

    def store(self, con: list[tuple[str, str]], ind: list[tuple[str, str]] | None = None, accounts=None) -> cvm.FilingStore:
        index = f"{self.CNPJ};{self.DT_REFER};1;CASE S.A.;1;DFP;1;2026-03-20 18:00:00;\n"
        statements = {"DRE_con": "".join(dre_line(self.CNPJ, self.DT_REFER, "ÚLTIMO", "2025-01-01", self.DT_REFER, a, v) for a, v in con)}
        if ind is not None:
            statements["DRE_ind"] = "".join(dre_line(self.CNPJ, self.DT_REFER, "ÚLTIMO", "2025-01-01", self.DT_REFER, a, v) for a, v in ind)
        return inline_store("DFP", 2025, index, statements, accounts=accounts)

    def statement_of(self, store: cvm.FilingStore) -> str:
        return cvm.usable(store, self.CNPJ, self.DAY)[0].statement

    def test_a_consolidated_dre_that_is_all_zero_is_individual_when_the_filing_has_individual_rows(self):
        # TIMS-shaped: DRE_con 3.01 and 3.11 are 0, DRE_ind carries the values.
        store = self.store(con=[("3.01", "0.00"), ("3.11", "0.00")], ind=[("3.01", "5000.00")])
        self.assertEqual(self.statement_of(store), "individual")
        self.assertEqual(cvm.row_at(store, self.CNPJ, date(2025, 12, 31), "3.01"), (date(2025, 1, 1), 5000.0))

    def test_a_non_zero_consolidated_3_01_stays_consolidated_even_with_individual_rows(self):
        store = self.store(con=[("3.01", "1000.00"), ("3.11", "0.00")], ind=[("3.01", "9999.00")])
        self.assertEqual(self.statement_of(store), "consolidated")
        self.assertEqual(cvm.row_at(store, self.CNPJ, date(2025, 12, 31), "3.01"), (date(2025, 1, 1), 1000.0))

    def test_a_non_zero_consolidated_3_11_alone_keeps_the_consolidated_statements(self):
        store = self.store(con=[("3.01", "0.00"), ("3.11", "80.00")], ind=[("3.01", "9999.00")])
        self.assertEqual(self.statement_of(store), "consolidated")

    def test_an_all_zero_consolidated_dre_with_no_individual_rows_stays_consolidated(self):
        store = self.store(con=[("3.01", "0.00"), ("3.11", "0.00")])
        self.assertEqual(self.statement_of(store), "consolidated")
        self.assertEqual(cvm.row_at(store, self.CNPJ, date(2025, 12, 31), "3.01"), (date(2025, 1, 1), 0.0))

    def test_the_statement_choice_does_not_depend_on_an_account_filter(self):
        # The individual DRE holds only 3.02 and the filter keeps none of its rows, yet the filing has individual rows.
        store = self.store(con=[("3.01", "0.00"), ("3.11", "0.00")], ind=[("3.02", "100.00")], accounts={"1"})
        self.assertEqual(self.statement_of(store), "individual")


class NetIncomeTests(unittest.TestCase):
    """F7 (A-21): per DRE column, NET_INCOME is 3.11.01 when its value is non-zero, else 3.11."""

    CNPJ = "23.456.789/0001-01"

    def dfp(self, detail: str, total: str, accounts=None) -> cvm.FilingStore:
        rows = dre_line(self.CNPJ, "2025-12-31", "ÚLTIMO", "2025-01-01", "2025-12-31", "3.11.01", detail)
        rows += dre_line(self.CNPJ, "2025-12-31", "ÚLTIMO", "2025-01-01", "2025-12-31", "3.11", total)
        index = f"{self.CNPJ};2025-12-31;1;CASE S.A.;1;DFP;1;2026-03-20 18:00:00;\n"
        return inline_store("DFP", 2025, index, {"DRE_con": rows}, accounts=accounts)

    def test_a_zero_detail_row_falls_back_to_3_11(self):
        store = self.dfp(detail="0.00", total="-120.00")
        self.assertEqual(cvm.annual(store, self.CNPJ, date(2026, 8, 7), cvm.NET_INCOME), cvm.Dated(-120.0, date(2026, 3, 20)))

    def test_a_non_zero_detail_row_is_the_net_income(self):
        store = self.dfp(detail="80.00", total="100.00")
        self.assertEqual(cvm.annual(store, self.CNPJ, date(2026, 8, 7), cvm.NET_INCOME), cvm.Dated(80.0, date(2026, 3, 20)))

    def test_an_account_filter_naming_only_3_11_still_yields_net_income(self):
        store = self.dfp(detail="80.00", total="100.00", accounts={"3.11"})
        self.assertEqual(cvm.annual(store, self.CNPJ, date(2026, 8, 7), cvm.NET_INCOME), cvm.Dated(80.0, date(2026, 3, 20)))

    def test_ttm_of_an_itr_combines_the_per_row_net_income_of_each_column(self):
        # FY 2025 net income 200 (3.11.01 non-zero). ITR 2026-06-30: YTD 3.11.01 = 50 (non-zero, so 50);
        # its prior-year YTD has 3.11.01 = 0, so the 3.11 row, 30, is used. TTM = 200 + 50 - 30 = 220.
        dfp = self.dfp(detail="200.00", total="210.00")
        itr_rows = (
            dre_line(self.CNPJ, "2026-06-30", "ÚLTIMO", "2026-01-01", "2026-06-30", "3.11.01", "50.00")
            + dre_line(self.CNPJ, "2026-06-30", "ÚLTIMO", "2026-01-01", "2026-06-30", "3.11", "60.00")
            + dre_line(self.CNPJ, "2026-06-30", "PENÚLTIMO", "2025-01-01", "2025-06-30", "3.11.01", "0.00")
            + dre_line(self.CNPJ, "2026-06-30", "PENÚLTIMO", "2025-01-01", "2025-06-30", "3.11", "30.00")
        )
        index = f"{self.CNPJ};2026-06-30;1;CASE S.A.;1;ITR;1;2026-08-06 18:00:00;\n"
        itr = inline_store("ITR", 2026, index, {"DRE_con": itr_rows})
        self.assertEqual(
            cvm.ttm(dfp, itr, self.CNPJ, date(2026, 8, 7), cvm.NET_INCOME),
            cvm.Dated(220.0, date(2026, 8, 6)),
        )


class ShareScaleFactorTests(unittest.TestCase):
    """F13 (A-18): the factor is the power of ten from 10^-2 to 10^3 nearest the B3 count on a log scale."""

    def test_the_factor_is_the_power_of_ten_nearest_the_b3_count(self):
        cases = [
            ((1_000, 10_000), 10),  # ratio 10
            ((10_000, 1_000), 0.1),  # ratio 0.1, an ESPA-shaped count
            ((100_000, 1_000), 0.01),  # ratio 0.01
            ((1, 100), 100),  # ratio 100
            ((31_258, 31_258_000), 1000),  # ratio 1000, EUCATEX: counts in thousands
            ((10, 10), 1),  # ratio 1
            ((10_000, 31_000), 1),  # ratio 3.1: below the √10 boundary, the count stays as filed
            ((10_000, 32_000), 10),  # ratio 3.2: just past the √10 boundary
            ((1_000, 5_000), 10),  # ratio 5, a 1:5 split: the nearest power of ten, off by 2x
            ((1, 50_000), 1000),  # ratio 50,000: 1000, never 10^4
            ((10_000_000, 5_000), 0.01),  # ratio 0.0005: 0.01, never 10^-3
        ]
        for (cvm_total, b3_total), expected in cases:
            with self.subTest(cvm_total=cvm_total, b3_total=b3_total):
                self.assertEqual(cvm.share_scale(cvm_total, b3_total), (expected, True))

    def test_the_factors_are_the_powers_of_ten_from_1e_minus_2_to_1e_3(self):
        self.assertEqual(cvm.SHARE_SCALE_FACTORS, (0.01, 0.1, 1, 10, 100, 1000))


class SubAccountBalanceTests(StoreCase):
    def test_balance_of_a_sub_account_is_its_own_row_of_the_newest_usable_period(self):
        # ALFA's ITR 2026-06-30 (received 2026-08-06) holds 2.03.09 (non-controlling interests) 2500 beside 2.03 31000.
        self.assertEqual(cvm.balance(self.dfp, self.itr, ALFA, date(2026, 8, 7), "2.03.09"), cvm.Dated(2500.0, date(2026, 8, 6)))


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
            cvm.Shares(on=705260684, pref=547818424, total=1253079108, dt_refer=date(2026, 3, 31), dt_receb=date(2026, 5, 14)),
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


class PeriodReceiptBoundaryTests(unittest.TestCase):
    """The helpers that read a period other than the newest one cut it at d by its own DT_RECEB (T33, G1).

    The older filing is received on d + 1 (not used) and on d (used). The newest usable period is covered by
    _newest_usable; these cases pin the cut in _usable_period, the call that annual, ttm_year_before and the
    TTM read of the fiscal year before an ITR all make.
    """

    CNPJ = "10.101.101/0001-01"
    D = date(2026, 8, 7)

    def annual_store(self, receipt_2020: str) -> cvm.FilingStore:
        # DFP 2020-12-31, five years before the base DFP 2025-12-31 (received 2026-03-20), is received at receipt_2020.
        index = (
            f"{self.CNPJ};2020-12-31;1;CASE S.A.;1;DFP;1;{receipt_2020};\n"
            f"{self.CNPJ};2025-12-31;1;CASE S.A.;1;DFP;1;2026-03-20 18:00:00;\n"
        )
        rows = (
            dre_line(self.CNPJ, "2020-12-31", "ÚLTIMO", "2020-01-01", "2020-12-31", "3.01", "800.00")
            + dre_line(self.CNPJ, "2025-12-31", "ÚLTIMO", "2025-01-01", "2025-12-31", "3.01", "1000.00")
        )
        return inline_store("DFP", 2025, index, {"DRE_con": rows})

    def ttm_stores(self, dfp_2025: str = "2026-03-20 18:00:00", itr_2025: str = "2025-08-10 18:00:00"):
        # ITR 2026-06-30 (received 2026-08-06) is the newest usable period at d. Its fiscal year before is DFP 2025-12-31
        # (received dfp_2025). ITR 2025-06-30 (received itr_2025) is the period one year before the newest.
        dfp_index = (
            f"{self.CNPJ};2024-12-31;1;CASE S.A.;1;DFP;1;2025-03-20 18:00:00;\n"
            f"{self.CNPJ};2025-12-31;1;CASE S.A.;1;DFP;1;{dfp_2025};\n"
        )
        dfp_rows = (
            dre_line(self.CNPJ, "2024-12-31", "ÚLTIMO", "2024-01-01", "2024-12-31", "3.01", "800.00")
            + dre_line(self.CNPJ, "2025-12-31", "ÚLTIMO", "2025-01-01", "2025-12-31", "3.01", "1000.00")
        )
        itr_index = (
            f"{self.CNPJ};2025-06-30;1;CASE S.A.;1;ITR;1;{itr_2025};\n"
            f"{self.CNPJ};2026-06-30;1;CASE S.A.;1;ITR;1;2026-08-06 18:00:00;\n"
        )
        itr_rows = (
            dre_line(self.CNPJ, "2025-06-30", "ÚLTIMO", "2025-01-01", "2025-06-30", "3.01", "450.00")
            + dre_line(self.CNPJ, "2025-06-30", "PENÚLTIMO", "2024-01-01", "2024-06-30", "3.01", "400.00")
            + dre_line(self.CNPJ, "2026-06-30", "ÚLTIMO", "2026-01-01", "2026-06-30", "3.01", "600.00")
            + dre_line(self.CNPJ, "2026-06-30", "PENÚLTIMO", "2025-01-01", "2025-06-30", "3.01", "500.00")
        )
        return (
            inline_store("DFP", 2025, dfp_index, {"DRE_con": dfp_rows}),
            inline_store("ITR", 2026, itr_index, {"DRE_con": itr_rows}),
        )

    def test_annual_five_years_back_is_none_when_its_dfp_was_received_after_d(self):
        store = self.annual_store("2026-08-08 09:00:00")
        self.assertEqual(cvm.annual(store, self.CNPJ, self.D, "3.01"), cvm.Dated(1000.0, date(2026, 3, 20)))
        self.assertIsNone(cvm.annual(store, self.CNPJ, self.D, "3.01", years_back=5))

    def test_annual_five_years_back_is_used_when_its_dfp_was_received_on_d(self):
        store = self.annual_store("2026-08-07 12:00:00")
        self.assertEqual(cvm.annual(store, self.CNPJ, self.D, "3.01", years_back=5), cvm.Dated(800.0, self.D))

    def test_ttm_year_before_is_none_when_its_itr_was_received_after_d(self):
        dfp, itr = self.ttm_stores(itr_2025="2026-08-08 09:00:00")
        self.assertIsNone(cvm.ttm_year_before(dfp, itr, self.CNPJ, self.D, "3.01"))

    def test_ttm_year_before_is_used_when_its_itr_was_received_on_d(self):
        dfp, itr = self.ttm_stores(itr_2025="2026-08-07 12:00:00")
        # DFP 2024 (800) + ITR 2025-06-30 YTD (450) - its prior YTD (400)
        self.assertEqual(cvm.ttm_year_before(dfp, itr, self.CNPJ, self.D, "3.01"), cvm.Dated(850.0, self.D))

    def test_ttm_is_none_when_the_dfp_before_its_itr_was_received_after_d(self):
        dfp, itr = self.ttm_stores(dfp_2025="2026-08-08 09:00:00")
        self.assertIsNone(cvm.ttm(dfp, itr, self.CNPJ, self.D, "3.01"))

    def test_ttm_is_used_when_the_dfp_before_its_itr_was_received_on_d(self):
        dfp, itr = self.ttm_stores(dfp_2025="2026-08-07 12:00:00")
        # DFP 2025 (1000) + ITR 2026-06-30 YTD (600) - its prior YTD (500)
        self.assertEqual(cvm.ttm(dfp, itr, self.CNPJ, self.D, "3.01"), cvm.Dated(1100.0, self.D))


class AnnualBaseReceiptTests(unittest.TestCase):
    """annual() takes its base from the DFPs received by d (T34, G2): a later DFP received after d does not move it."""

    CNPJ = "10.101.101/0001-01"
    D = date(2026, 8, 7)

    def store(self, receipt_2025: str) -> cvm.FilingStore:
        # DFPs 2019, 2020 and 2024 are received before d; DFP 2025, the newest, is received at receipt_2025.
        filings = [
            (2019, "2020-03-20 18:00:00", "100.00"),
            (2020, "2021-03-20 18:00:00", "200.00"),
            (2024, "2025-03-20 18:00:00", "800.00"),
            (2025, receipt_2025, "1000.00"),
        ]
        index = "".join(f"{self.CNPJ};{year}-12-31;1;CASE S.A.;1;DFP;1;{receipt};\n" for year, receipt, _ in filings)
        rows = "".join(
            dre_line(self.CNPJ, f"{year}-12-31", "ÚLTIMO", f"{year}-01-01", f"{year}-12-31", "3.01", value)
            for year, _, value in filings
        )
        return inline_store("DFP", 2025, index, {"DRE_con": rows})

    def test_a_dfp_received_after_d_does_not_move_the_base_of_years_back(self):
        store = self.store("2026-08-08 09:00:00")
        # The base is DFP 2024, received 2025-03-20; five years back from it is DFP 2019.
        self.assertEqual(cvm.annual(store, self.CNPJ, self.D, "3.01"), cvm.Dated(800.0, date(2025, 3, 20)))
        self.assertEqual(cvm.annual(store, self.CNPJ, self.D, "3.01", years_back=5), cvm.Dated(100.0, date(2020, 3, 20)))

    def test_a_dfp_received_on_d_is_the_base(self):
        store = self.store("2026-08-07 12:00:00")
        self.assertEqual(cvm.annual(store, self.CNPJ, self.D, "3.01"), cvm.Dated(1000.0, self.D))
        # Five years back from DFP 2025 is DFP 2020, received 2021-03-20.
        self.assertEqual(cvm.annual(store, self.CNPJ, self.D, "3.01", years_back=5), cvm.Dated(200.0, date(2021, 3, 20)))


class YearEarlierItrTests(unittest.TestCase):
    """TTM-01..TTM-03 (B-5): an ITR whose PENÚLTIMO is absent or 0 takes its prior YTD from the ITR one year earlier."""

    CNPJ = "30.303.303/0001-30"
    D = date(2026, 8, 7)

    def stores(self, *, penultimo: str | None = "0.00", earlier_receipt: str | None = "2025-08-10 18:00:00",
               earlier_account: bool = True, dfp_receipt: str = "2026-03-20 18:00:00",
               itr_receipt: str = "2026-08-06 18:00:00") -> tuple[cvm.FilingStore, cvm.FilingStore]:
        # DFP 2025 = 1000; ITR 2026-06-30 YTD 300 with the PENÚLTIMO given; ITR 2025-06-30 YTD 200.
        dfp = inline_store(
            "DFP", 2025,
            f"{self.CNPJ};2025-12-31;1;CASE S.A.;1;DFP;1;{dfp_receipt};\n",
            {"DRE_con": dre_line(self.CNPJ, "2025-12-31", "ÚLTIMO", "2025-01-01", "2025-12-31", "3.01", "1000.00")},
        )
        index = f"{self.CNPJ};2026-06-30;1;CASE S.A.;1;ITR;1;{itr_receipt};\n"
        rows = dre_line(self.CNPJ, "2026-06-30", "ÚLTIMO", "2026-01-01", "2026-06-30", "3.01", "300.00")
        if penultimo is not None:
            rows += dre_line(self.CNPJ, "2026-06-30", "PENÚLTIMO", "2025-01-01", "2025-06-30", "3.01", penultimo)
        if earlier_receipt is not None:
            index = f"{self.CNPJ};2025-06-30;1;CASE S.A.;1;ITR;1;{earlier_receipt};\n" + index
            if earlier_account:
                rows += dre_line(self.CNPJ, "2025-06-30", "ÚLTIMO", "2025-01-01", "2025-06-30", "3.01", "200.00")
        return dfp, inline_store("ITR", 2026, index, {"DRE_con": rows})

    def test_a_zero_comparative_takes_the_prior_ytd_from_last_years_itr(self):
        dfp, itr = self.stores(penultimo="0.00")
        self.assertEqual(
            cvm.ttm(dfp, itr, self.CNPJ, self.D, "3.01"),
            cvm.Dated(1100.0, date(2026, 8, 6), prior_from_itr=True),  # 1000 + 300 - 200
        )

    def test_an_absent_comparative_takes_the_prior_ytd_from_last_years_itr(self):
        dfp, itr = self.stores(penultimo=None)
        self.assertEqual(
            cvm.ttm(dfp, itr, self.CNPJ, self.D, "3.01"),
            cvm.Dated(1100.0, date(2026, 8, 6), prior_from_itr=True),
        )

    def test_the_year_earlier_itr_must_be_usable_at_d(self):
        dfp, itr = self.stores(earlier_receipt="2026-08-08 09:00:00")  # received on d + 1
        self.assertIsNone(cvm.ttm(dfp, itr, self.CNPJ, self.D, "3.01"))
        dfp, itr = self.stores(earlier_receipt="2026-08-07 12:00:00")  # received on d: usable, and the latest receipt
        self.assertEqual(cvm.ttm(dfp, itr, self.CNPJ, self.D, "3.01"), cvm.Dated(1100.0, date(2026, 8, 7), prior_from_itr=True))

    def test_without_the_year_earlier_itr_or_its_account_there_is_no_value(self):
        for label, kwargs in (("absent", {"earlier_receipt": None}), ("without the account", {"earlier_account": False})):
            with self.subTest(case=label):
                dfp, itr = self.stores(**kwargs)
                self.assertIsNone(cvm.ttm(dfp, itr, self.CNPJ, self.D, "3.01"))

    def test_a_non_zero_comparative_is_used_even_when_last_years_itr_exists(self):
        dfp, itr = self.stores(penultimo="150.00")
        self.assertEqual(
            cvm.ttm(dfp, itr, self.CNPJ, self.D, "3.01"),
            cvm.Dated(1150.0, date(2026, 8, 6), prior_from_itr=False),  # 1000 + 300 - 150
        )

    def test_the_date_is_the_latest_receipt_of_the_three_filings_used(self):
        dfp, itr = self.stores(dfp_receipt="2026-08-07 12:00:00")  # the DFP is the latest of the three
        self.assertEqual(cvm.ttm(dfp, itr, self.CNPJ, self.D, "3.01"), cvm.Dated(1100.0, date(2026, 8, 7), prior_from_itr=True))

    def test_a_newest_dfp_gives_its_annual_value_without_the_fallback(self):
        dfp, itr = self.stores(itr_receipt="2026-08-08 09:00:00")  # the ITR is not usable at d: the DFP is the newest
        self.assertEqual(cvm.ttm(dfp, itr, self.CNPJ, self.D, "3.01"), cvm.Dated(1000.0, date(2026, 3, 20), prior_from_itr=False))

    def test_ttm_year_before_takes_the_same_fallback_on_its_itr(self):
        # The period one year before 2026-06-30 is the ITR 2025-06-30 (YTD 200, PENÚLTIMO 0); its prior YTD is the ITR
        # 2024-06-30 (150), and its fiscal year before is the DFP 2024 (800): 800 + 200 - 150 = 850.
        dfp = inline_store(
            "DFP", 2025,
            f"{self.CNPJ};2024-12-31;1;CASE S.A.;1;DFP;1;2025-03-20 18:00:00;\n"
            f"{self.CNPJ};2025-12-31;1;CASE S.A.;1;DFP;1;2026-03-20 18:00:00;\n",
            {"DRE_con": dre_line(self.CNPJ, "2024-12-31", "ÚLTIMO", "2024-01-01", "2024-12-31", "3.01", "800.00")
             + dre_line(self.CNPJ, "2025-12-31", "ÚLTIMO", "2025-01-01", "2025-12-31", "3.01", "1000.00")},
        )
        itr = inline_store(
            "ITR", 2026,
            f"{self.CNPJ};2024-06-30;1;CASE S.A.;1;ITR;1;2024-08-10 18:00:00;\n"
            f"{self.CNPJ};2025-06-30;1;CASE S.A.;1;ITR;1;2025-08-10 18:00:00;\n"
            f"{self.CNPJ};2026-06-30;1;CASE S.A.;1;ITR;1;2026-08-06 18:00:00;\n",
            {"DRE_con": dre_line(self.CNPJ, "2024-06-30", "ÚLTIMO", "2024-01-01", "2024-06-30", "3.01", "150.00")
             + dre_line(self.CNPJ, "2025-06-30", "ÚLTIMO", "2025-01-01", "2025-06-30", "3.01", "200.00")
             + dre_line(self.CNPJ, "2025-06-30", "PENÚLTIMO", "2024-01-01", "2024-06-30", "3.01", "0.00")
             + dre_line(self.CNPJ, "2026-06-30", "ÚLTIMO", "2026-01-01", "2026-06-30", "3.01", "300.00")},
        )
        self.assertEqual(
            cvm.ttm_year_before(dfp, itr, self.CNPJ, self.D, "3.01"),
            cvm.Dated(850.0, date(2025, 8, 10), prior_from_itr=True),
        )


class NewestFilingTests(unittest.TestCase):
    """STL-01 (age input) and ADJ-04 (the filing of a worst row): the newest usable period end and filing at d (B-6)."""

    CNPJ = "40.404.404/0001-40"
    D = date(2026, 8, 7)

    def stores(self, itr_receipt: str = "2026-05-14 18:00:00") -> tuple[cvm.FilingStore, cvm.FilingStore]:
        # DFP 2025-12-31 (received 2026-03-20) and ITR 2026-03-31 (received as given).
        dfp = inline_store(
            "DFP", 2025,
            f"{self.CNPJ};2025-12-31;1;CASE S.A.;1;DFP;1;2026-03-20 18:00:00;\n",
            {"DRE_con": dre_line(self.CNPJ, "2025-12-31", "ÚLTIMO", "2025-01-01", "2025-12-31", "3.01", "1000.00")},
        )
        itr = inline_store(
            "ITR", 2026,
            f"{self.CNPJ};2026-03-31;1;CASE S.A.;1;ITR;1;{itr_receipt};\n",
            {"DRE_con": dre_line(self.CNPJ, "2026-03-31", "ÚLTIMO", "2026-01-01", "2026-03-31", "3.01", "250.00")},
        )
        return dfp, itr

    def test_the_usable_itr_is_the_newest_period_and_its_filing_is_returned(self):
        dfp, itr = self.stores()
        self.assertEqual(cvm.newest_period_end(dfp, itr, self.CNPJ, self.D), date(2026, 3, 31))
        self.assertEqual(cvm.newest_filing(dfp, itr, self.CNPJ, self.D), itr.period(self.CNPJ, date(2026, 3, 31)).filing)

    def test_an_itr_received_on_d_plus_one_leaves_the_dfp_as_the_newest(self):
        dfp, itr = self.stores(itr_receipt="2026-08-08 09:00:00")
        self.assertEqual(cvm.newest_period_end(dfp, itr, self.CNPJ, self.D), date(2025, 12, 31))
        self.assertEqual(cvm.newest_filing(dfp, itr, self.CNPJ, self.D), dfp.period(self.CNPJ, date(2025, 12, 31)).filing)

    def test_no_usable_filing_gives_none_for_both(self):
        dfp, itr = self.stores()
        before_any_receipt = date(2026, 1, 1)
        self.assertIsNone(cvm.newest_period_end(dfp, itr, self.CNPJ, before_any_receipt))
        self.assertIsNone(cvm.newest_filing(dfp, itr, self.CNPJ, before_any_receipt))
        self.assertIsNone(cvm.newest_period_end(dfp, itr, "00.000.000/0001-00", self.D))

    def test_the_stale_filing_limit_is_270_days(self):
        self.assertEqual(cvm.STALE_FILING_DAYS, 270)


COMPOSITION_HEADER = "CNPJ_CIA;DT_REFER;VERSAO;QT_ACAO_ORDIN_CAP_INTEGR;QT_ACAO_PREF_CAP_INTEGR;QT_ACAO_TOTAL_CAP_INTEGR\n"


def inline_composition(kind: str, year: int, index_rows: str, composition_rows: str) -> cvm.FilingStore:
    """A FilingStore from inline CSV text: the index rows and the capital composition rows (composicao_capital)."""
    prefix = kind.lower()
    members = {
        f"{prefix}_cia_aberta_{year}.csv": INDEX_HEADER + index_rows,
        f"{prefix}_cia_aberta_composicao_capital_{year}.csv": COMPOSITION_HEADER + composition_rows,
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{prefix}_cia_aberta_{year}.zip"
        path.write_bytes(make_zip(members))
        return cvm.load_filings([path], kind)


class PreviousSharesTests(unittest.TestCase):
    """SHR-05 (B-4): the newest usable period older than the newest usable one whose composition total is above zero."""

    CNPJ = "50.505.505/0001-50"
    D = date(2025, 8, 7)

    def stores(self, filings: list[tuple[str, str, str, tuple[int, int, int] | None]]):
        """Each filing is (kind, DT_REFER, DT_RECEB, (ON, PREF, TOTAL) or None when it has no composition row)."""
        index = {"DFP": "", "ITR": ""}
        composition = {"DFP": "", "ITR": ""}
        for kind, dt_refer, receipt, counts in filings:
            index[kind] += f"{self.CNPJ};{dt_refer};1;CASE S.A.;1;{kind};1;{receipt};\n"
            if counts is not None:
                composition[kind] += f"{self.CNPJ};{dt_refer};1;{counts[0]};{counts[1]};{counts[2]}\n"
        return (
            inline_composition("DFP", 2024, index["DFP"], composition["DFP"]),
            inline_composition("ITR", 2025, index["ITR"], composition["ITR"]),
        )

    def test_the_middle_period_without_a_composition_is_skipped(self):
        dfp, itr = self.stores([
            ("DFP", "2023-12-31", "2024-03-20 18:00:00", (100, 50, 150)),
            ("ITR", "2024-06-30", "2024-08-10 18:00:00", None),
            ("DFP", "2024-12-31", "2025-03-20 18:00:00", (200, 50, 250)),
        ])
        self.assertEqual(
            cvm.previous_shares(dfp, itr, self.CNPJ, self.D),
            cvm.Shares(on=100, pref=50, total=150, dt_refer=date(2023, 12, 31), dt_receb=date(2024, 3, 20)),
        )

    def test_a_period_whose_composition_total_is_zero_is_skipped(self):
        dfp, itr = self.stores([
            ("DFP", "2023-12-31", "2024-03-20 18:00:00", (100, 50, 150)),
            ("ITR", "2024-06-30", "2024-08-10 18:00:00", (0, 0, 0)),
            ("DFP", "2024-12-31", "2025-03-20 18:00:00", (200, 50, 250)),
        ])
        self.assertEqual(cvm.previous_shares(dfp, itr, self.CNPJ, self.D).dt_refer, date(2023, 12, 31))

    def test_none_when_no_older_usable_period_has_a_composition(self):
        dfp, itr = self.stores([
            ("DFP", "2023-12-31", "2024-03-20 18:00:00", None),
            ("ITR", "2024-06-30", "2024-08-10 18:00:00", None),
            ("DFP", "2024-12-31", "2025-03-20 18:00:00", (200, 50, 250)),
        ])
        self.assertIsNone(cvm.previous_shares(dfp, itr, self.CNPJ, self.D))

    def test_a_previous_period_received_after_d_is_skipped(self):
        # ITR 2025-03-31 is received on d + 1, so the previous count is the DFP 2024, not the newer ITR.
        dfp, itr = self.stores([
            ("DFP", "2024-12-31", "2025-03-20 18:00:00", (200, 50, 250)),
            ("ITR", "2025-03-31", "2025-08-08 09:00:00", (300, 50, 350)),
            ("ITR", "2025-06-30", "2025-08-06 18:00:00", (260, 50, 310)),
        ])
        self.assertEqual(cvm.shares(dfp, itr, self.CNPJ, self.D).dt_refer, date(2025, 6, 30))
        self.assertEqual(
            cvm.previous_shares(dfp, itr, self.CNPJ, self.D),
            cvm.Shares(on=200, pref=50, total=250, dt_refer=date(2024, 12, 31), dt_receb=date(2025, 3, 20)),
        )

    def test_shares_carry_the_period_end_of_the_newest_usable_filing(self):
        dfp, itr = self.stores([
            ("DFP", "2024-12-31", "2025-03-20 18:00:00", (200, 50, 250)),
            ("ITR", "2025-06-30", "2025-08-06 18:00:00", (260, 50, 310)),
        ])
        self.assertEqual(
            cvm.shares(dfp, itr, self.CNPJ, self.D),
            cvm.Shares(on=260, pref=50, total=310, dt_refer=date(2025, 6, 30), dt_receb=date(2025, 8, 6)),
        )


if __name__ == "__main__":
    unittest.main()
