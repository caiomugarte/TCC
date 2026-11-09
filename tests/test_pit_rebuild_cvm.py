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
EDGE = "66.666.666/0001-66"
INDIV = "77.777.777/0001-77"
REFILED = "88.888.888/0001-88"
PENULTIMO = "PENÚLTIMO"


def zip_folder(folder: Path, out: Path, index_header_rename: tuple[str, str] | None = None) -> Path:
    members = {path.name: path.read_text(encoding="utf-8") for path in sorted(folder.glob("*.csv"))}
    if index_header_rename is not None:
        old, new = index_header_rename
        index_name = "dfp_cia_aberta_2025.csv"
        members[index_name] = members[index_name].replace(old, new, 1)
    out.write_bytes(make_zip(members))
    return out


class FilingStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        cls.dfp = cvm.load_filings([zip_folder(FIXTURES / "dfp", tmp / "dfp_cia_aberta_2025.zip")], cvm.DFP)
        cls.itr = cvm.load_filings([zip_folder(FIXTURES / "itr", tmp / "itr_cia_aberta_2026.zip")], cvm.ITR)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

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


if __name__ == "__main__":
    unittest.main()
