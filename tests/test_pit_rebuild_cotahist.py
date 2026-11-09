"""COTAHIST reader: price, liquidity, ISIN, share class, market calendar and class prices at d (PIT-01, IND-01, A-7, A-15, A-16, A-17).

The fixture tests/fixtures/pit_rebuild/cotahist.txt has COTAHIST lines in the real 245-column
layout. The PETR4 line of 2026-11-06 carries the values verified in the spec (45.36 and
BRPETRACNPR6, ESPECI PN); every other price, volume and ISIN is synthetic. The zip is built in
a temporary directory with make_zip, so the tests never read the network or the data trees.
"""

import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from pit_rebuild import cotahist  # noqa: E402
from pit_rebuild.errors import IntegrityError  # noqa: E402
from pit_rebuild_support import make_zip  # noqa: E402

FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "pit_rebuild" / "cotahist.txt"
ALL_TICKERS = {
    "PETR4", "OLDX3", "OLDB3", "OLDB4", "BIGV3", "EUCA3", "EUCA4", "VALE3", "FATX3",
    "USIM3", "USIM5", "USIM6", "USIM11", "TIEA3", "TIEA4", "TIEA5", "BPAC5", "TAEE11",
}


def zip_text(text: str, out: Path) -> Path:
    out.write_bytes(make_zip({"COTAHIST_A2026.TXT": text}))
    return out


class QuotesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.zip_path = zip_text(FIXTURE.read_text(encoding="ascii"), Path(cls._tmp.name) / "COTAHIST_A2026.zip")
        cls.quotes = cotahist.load([cls.zip_path], ALL_TICKERS)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_session_on_d_is_used_and_session_on_d_plus_one_is_not(self):
        # PETR4 closed 40.00 on 2026-08-20 (d) and 41.00 on 2026-08-21 (d + 1).
        self.assertEqual(cotahist.quote_at(self.quotes, "PETR4", date(2026, 8, 20)), (date(2026, 8, 20), 40.0))

    def test_only_standard_lot_cash_market_records_are_kept(self):
        # VALE3 also has a fraction-lot line (CODBDI 96, price 2.00) and a term-market line (TPMERC 020, 1.00).
        self.assertEqual(cotahist.quote_at(self.quotes, "VALE3", date(2026, 8, 21)), (date(2026, 8, 21), 60.0))

    def test_avg_volume_uses_the_last_30_sessions_and_fewer_when_fewer_exist(self):
        # BIGV3 has 31 sessions with VOLTOT (i + 1) * 1000: the last 30 average 16500, all 31 would give 16000.
        self.assertEqual(cotahist.avg_volume(self.quotes, "BIGV3", date(2026, 9, 14)), 16500.0)
        # PETR4 up to 2026-08-24 has 4 sessions: (1e6 + 2e8 + 1.5e8 + 1.2e8) / 4.
        self.assertEqual(cotahist.avg_volume(self.quotes, "PETR4", date(2026, 8, 24)), 117_750_000.0)

    def test_last_session_more_than_30_calendar_days_before_d_is_no_quote(self):
        # OLDX3's last session is 2026-06-01: 30 days before 2026-07-01 is still in time; 44 days before 07-15 is not.
        self.assertEqual(cotahist.quote_at(self.quotes, "OLDX3", date(2026, 7, 1)), (date(2026, 6, 1), 10.0))
        self.assertIsNone(cotahist.quote_at(self.quotes, "OLDX3", date(2026, 7, 15)))

    def test_fatcot_divides_the_price(self):
        # FATX3 is quoted as PREULT 5000.00 with FATCOT 1000: the close is 5.00.
        self.assertEqual(cotahist.quote_at(self.quotes, "FATX3", date(2026, 8, 21)), (date(2026, 8, 21), 5.0))

    def test_share_class_is_the_first_especi_token(self):
        self.assertEqual(cotahist.share_class(self.quotes, "PETR4"), "PN")
        self.assertEqual(cotahist.share_class(self.quotes, "TAEE11"), "UNT")
        self.assertEqual(cotahist.share_class(self.quotes, "BPAC5"), "PNA")

    def test_isin_and_the_2026_11_06_petr4_line_of_the_spec(self):
        self.assertEqual(cotahist.isin(self.quotes, "PETR4"), "BRPETRACNPR6")
        self.assertEqual(cotahist.quote_at(self.quotes, "PETR4", date(2026, 11, 6)), (date(2026, 11, 6), 45.36))

    def test_next_session_of_a_friday_is_the_following_monday_and_none_at_the_edges(self):
        # Friday 2026-08-21 -> Monday 2026-08-24; after the last loaded session and before the first there is none.
        self.assertEqual(cotahist.next_session(self.quotes, date(2026, 8, 21)), date(2026, 8, 24))
        self.assertIsNone(cotahist.next_session(self.quotes, date(2026, 11, 6)))
        self.assertIsNone(cotahist.next_session(self.quotes, date(2020, 3, 2)))

    def test_load_keeps_the_requested_tickers_and_the_calendar_of_all_tickers(self):
        eucatex = cotahist.load([self.zip_path], {"EUCA3", "EUCA4"})
        self.assertEqual(cotahist.share_class(eucatex, "EUCA3"), "ON")
        self.assertEqual(cotahist.share_class(eucatex, "EUCA4"), "PN")
        self.assertNotIn("PETR4", eucatex.rows)
        # PETR4 and TAEE11 are not requested, yet their 2026-08-24 lines are market sessions.
        self.assertEqual(cotahist.next_session(eucatex, date(2026, 8, 21)), date(2026, 8, 24))

    def test_class_prices_pick_the_most_liquid_preferred_ticker_and_skip_units(self):
        # USIM5 (PNA, 500000 volume) beats USIM6 (PNB, 100); USIM11 (UNT) is never a class price.
        self.assertEqual(
            cotahist.class_prices(self.quotes, ["USIM3", "USIM5", "USIM6", "USIM11"], date(2026, 8, 21)),
            {"ON": ("USIM3", 12.0), "PREF": ("USIM5", 7.15)},
        )

    def test_class_prices_break_a_volume_tie_by_ticker(self):
        # TIEA4 and TIEA5 trade the same volume on 2026-08-21: the smaller ticker wins.
        self.assertEqual(
            cotahist.class_prices(self.quotes, ["TIEA3", "TIEA4", "TIEA5"], date(2026, 8, 21)),
            {"ON": ("TIEA3", 5.0), "PREF": ("TIEA4", 6.0)},
        )

    def test_class_prices_give_no_preferred_price_when_its_last_session_is_too_old(self):
        # OLDB4 last traded 2026-06-01, 81 days before 2026-08-21; OLDB3 is too old as well.
        self.assertEqual(
            cotahist.class_prices(self.quotes, ["OLDB3", "OLDB4"], date(2026, 8, 21)),
            {"ON": None, "PREF": None},
        )

    def test_a_trading_line_shorter_than_245_columns_is_an_integrity_error(self):
        short = "01" + "2026-08-21".replace("-", "") + "x" * 50 + "\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = zip_text(short, Path(tmp) / "COTAHIST_A2026.zip")
            with self.assertRaises(IntegrityError) as ctx:
                cotahist.load([path], ALL_TICKERS)
        self.assertEqual(ctx.exception.exit_code, 4)
        self.assertIn("245", ctx.exception.reason)


if __name__ == "__main__":
    unittest.main()
