"""COTAHIST reader: price, liquidity, ISIN, share class, market calendar and class prices at d (PIT-01, IND-01, A-7, A-15, A-16, A-17).

The fixture tests/fixtures/pit_rebuild/cotahist.txt has COTAHIST lines in the real 245-column
layout. The PETR4 line of 2026-11-06 carries the values verified in the spec (45.36 and
BRPETRACNPR6, ESPECI PN); every other price, volume and ISIN is synthetic. The zip is built in
a temporary directory with make_zip, so the tests never read the network or the data trees.
"""

import sys
import tempfile
import unittest
from datetime import date, timedelta
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
    "RECO3", "OITO3", "LOTF3", "LOTL3", "PREFA4", "PREFB4", "PREFC4", "LATE4", "EMAE3",
    "REFA3", "REFB3", "REFC3",
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

    def test_lot_and_recovery_records_are_kept_and_the_lot_wins_the_session(self):
        # VALE3 has a fraction-lot line (CODBDI 96, price 2.00) and a term-market line (TPMERC 020, 1.00): both dropped.
        self.assertEqual(cotahist.quote_at(self.quotes, "VALE3", date(2026, 8, 21)), (date(2026, 8, 21), 60.0))
        # RECO3 (recovery): its CODBDI 07 and 08 lines are quotes.
        self.assertEqual(cotahist.quote_at(self.quotes, "RECO3", date(2026, 7, 2)), (date(2026, 7, 2), 3.5))
        self.assertEqual(cotahist.quote_at(self.quotes, "RECO3", date(2026, 7, 3)), (date(2026, 7, 3), 3.6))
        # Its CODBDI 10, 12 and 58 lines and its TPMERC 020 line of 2026-07-06 are dropped: 07-03 is still its last quote.
        self.assertEqual(cotahist.quote_at(self.quotes, "RECO3", date(2026, 7, 6)), (date(2026, 7, 3), 3.6))
        # OITO3 has only CODBDI 08 lines: its quote comes from them.
        self.assertEqual(cotahist.quote_at(self.quotes, "OITO3", date(2026, 7, 10)), (date(2026, 7, 8), 4.25))
        # On 2026-07-14 LOTF3 has its 08 line first and its 02 line last; LOTL3 the reverse: the 02 close (7.00) wins both.
        self.assertEqual(cotahist.quote_at(self.quotes, "LOTF3", date(2026, 7, 14)), (date(2026, 7, 14), 7.0))
        self.assertEqual(cotahist.quote_at(self.quotes, "LOTL3", date(2026, 7, 14)), (date(2026, 7, 14), 7.0))

    def test_avg_volume_uses_the_last_21_sessions_and_fewer_when_fewer_exist(self):
        self.assertEqual(cotahist.LIQUIDITY_SESSIONS, 21)
        # BIGV3 has 25 sessions up to 2026-09-04 with VOLTOT (i + 1) * 1000: the last 21 average 15000
        # (the last 20 would give 15500 and all 25 give 13000).
        self.assertEqual(cotahist.avg_volume(self.quotes, "BIGV3", date(2026, 9, 4)), 15000.0)
        # Of its 31 sessions the last 21 (VOLTOT 11000 .. 31000) average 21000.
        self.assertEqual(cotahist.avg_volume(self.quotes, "BIGV3", date(2026, 9, 14)), 21000.0)
        # One session in the window (2026-08-03, VOLTOT 1000) is the whole average.
        self.assertEqual(cotahist.avg_volume(self.quotes, "BIGV3", date(2026, 8, 3)), 1000.0)
        # PETR4 up to 2026-08-24 has 4 sessions: (1e6 + 2e8 + 1.5e8 + 1.2e8) / 4.
        self.assertEqual(cotahist.avg_volume(self.quotes, "PETR4", date(2026, 8, 24)), 117_750_000.0)

    def test_last_session_more_than_30_calendar_days_before_d_is_no_quote(self):
        # OLDX3's last session is 2026-06-01: 30 days before 2026-07-01 is still in time; 31 days before 07-02 is not.
        self.assertEqual(cotahist.quote_at(self.quotes, "OLDX3", date(2026, 7, 1)), (date(2026, 6, 1), 10.0))
        self.assertIsNone(cotahist.quote_at(self.quotes, "OLDX3", date(2026, 7, 2)))
        self.assertIsNone(cotahist.quote_at(self.quotes, "OLDX3", date(2026, 7, 15)))

    def test_quote_at_without_an_age_limit_and_last_trade_return_a_session_400_days_old(self):
        d = date(2026, 6, 1) + timedelta(days=400)
        self.assertIsNone(cotahist.quote_at(self.quotes, "OLDX3", d))
        self.assertEqual(cotahist.quote_at(self.quotes, "OLDX3", d, max_age_days=None), (date(2026, 6, 1), 10.0))
        self.assertEqual(cotahist.last_trade(self.quotes, "OLDX3", d), (date(2026, 6, 1), 10.0))

    def test_reference_stale_is_false_only_with_a_session_in_the_30_days_to_d(self):
        d = date(2026, 7, 10)
        self.assertFalse(cotahist.reference_stale(self.quotes, "REFA3", d))  # last session 2026-06-10 = d - 30
        self.assertTrue(cotahist.reference_stale(self.quotes, "REFB3", d))  # last session 2026-06-09 = d - 31
        self.assertTrue(cotahist.reference_stale(self.quotes, "REFC3", d))  # its only session, 2026-07-11, is d + 1
        self.assertTrue(cotahist.reference_stale(self.quotes, "GHOST3", d))  # no session at all

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
            {"ON": ("USIM3", 12.0, date(2026, 8, 21)), "PREF": ("USIM5", 7.15, date(2026, 8, 21))},
        )

    def test_class_prices_break_a_volume_tie_by_ticker(self):
        # TIEA4 and TIEA5 trade the same volume on 2026-08-21: the smaller ticker wins.
        self.assertEqual(
            cotahist.class_prices(self.quotes, ["TIEA3", "TIEA4", "TIEA5"], date(2026, 8, 21)),
            {"ON": ("TIEA3", 5.0, date(2026, 8, 21)), "PREF": ("TIEA4", 6.0, date(2026, 8, 21))},
        )

    def test_a_class_whose_last_session_is_too_old_is_priced_from_it_with_its_session_day(self):
        # OLDB3 (ON) and OLDB4 (PN) last traded 2026-06-01, 81 days before 2026-08-21: no fresh ticker, so each class
        # keeps its last trade at any age, with the session day (F14).
        self.assertEqual(
            cotahist.class_prices(self.quotes, ["OLDB3", "OLDB4"], date(2026, 8, 21)),
            {"ON": ("OLDB3", 19.0, date(2026, 6, 1)), "PREF": ("OLDB4", 20.0, date(2026, 6, 1))},
        )

    def test_an_on_ticker_last_traded_in_2025_prices_its_class_at_that_trade(self):
        # EMAE3 is the only ON ticker and last traded 2025-08-18 at 150.00, more than 30 days before d (F14).
        self.assertEqual(
            cotahist.class_prices(self.quotes, ["EMAE3"], date(2026, 11, 5)),
            {"ON": ("EMAE3", 150.0, date(2025, 8, 18)), "PREF": None},
        )

    def test_a_fresh_preferred_ticker_beats_a_stale_one_with_more_volume(self):
        # PREFA4 traded on 2026-08-19 with 100 volume; PREFB4 last traded 2026-06-02 (80 days before d) with 90000.
        self.assertEqual(
            cotahist.class_prices(self.quotes, ["PREFA4", "PREFB4"], date(2026, 8, 21)),
            {"ON": None, "PREF": ("PREFA4", 9.0, date(2026, 8, 19))},
        )

    def test_a_stale_class_takes_the_ticker_with_the_latest_session(self):
        # Both preferred tickers are stale: PREFC4's session (2026-06-05) is later than PREFB4's, which has more volume.
        self.assertEqual(
            cotahist.class_prices(self.quotes, ["PREFB4", "PREFC4"], date(2026, 8, 21)),
            {"ON": None, "PREF": ("PREFC4", 7.0, date(2026, 6, 5))},
        )

    def test_a_session_after_d_never_prices_a_class(self):
        # LATE4's only session is 2026-08-24, after d = 2026-08-21: it is not used.
        self.assertEqual(
            cotahist.class_prices(self.quotes, ["LATE4"], date(2026, 8, 21)),
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
