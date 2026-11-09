"""The 19 fields of one ticker at d: hand-computed values, units, share counts, reasons and the leak guard (IND-01, IND-03, IND-04, MAP-03, PIT-02, A-8, A-17, A-18).

The synthetic company has 1,000,000 ON and 500,000 PN shares, ON at 10.00 and PN at 8.00, so
VALOR DE MERCADO = 10,000,000 + 4,000,000 = 14,000,000. Its other inputs and every hand-computed
value are listed in SYNTHETIC below. The USIMINAS case uses the counts and prices of 2026-08-07
checked in the spec (A-17): 8,324,781,006.60 with USIM5 at 7.15, not 9,146,508,642.60 with USIM6.
"""

import sys
import unittest
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))

from pit_rebuild import indicators  # noqa: E402
from pit_rebuild.dividends import Dividend  # noqa: E402
from pit_rebuild.errors import PITLeakError  # noqa: E402
from pit_rebuild.fields import ALL_FIELDS  # noqa: E402
from pit_rebuild.indicators import Input, TickerInputs  # noqa: E402

D = date(2026, 8, 7)
SYNTHETIC = {
    "PRECO": 10.0,
    "VALOR DE MERCADO": 14_000_000.0,
    "LIQUIDEZ MEDIA DIARIA": 50_000.0,
    "P/L": 10.0,  # 10 / (1.5e6 / 1.5e6)
    "P/VP": 2.0,  # 10 / (7.5e6 / 1.5e6)
    "PSR": 1.25,  # 10 / (12e6 / 1.5e6)
    "EV/EBIT": 17_000_000 / 3_000_000,  # (14e6 + DL 3e6) / EBIT 3e6
    "DY": 5.0,  # 100 × 0.5 / 10
    "ROE": 20.0,  # 100 × 1.5e6 / 7.5e6
    "ROA": 7.5,  # 100 × 1.5e6 / 20e6
    "ROIC": 100 * 2_400_000 / 11_500_000,  # 100 × (3e6 - 0.6e6) / (7.5e6 + 4e6)
    "MARG. LIQUIDA": 12.5,  # 100 × 1.5e6 / 12e6
    "MARGEM EBIT": 25.0,  # 100 × 3e6 / 12e6
    "LIQ. CORRENTE": 2.0,  # 6e6 / 3e6
    "DIVIDA LIQUIDA / EBIT": 1.0,  # 3e6 / 3e6
    "DIV. LIQ. / PATRI.": 0.4,  # 3e6 / 7.5e6
    "CAGR RECEITAS 5 ANOS": 100 * (2 ** 0.2 - 1),  # 12e6 / 6e6
    "CAGR LUCROS 5 ANOS": 100 * (2 ** 0.2 - 1),  # 1.5e6 / 0.75e6
    "PEG RATIO": 0.5,  # 10 / (100 × (1.5e6 / 1.25e6 - 1))
}


def synthetic(**changes) -> TickerInputs:
    base = TickerInputs(
        ticker="SYNA3",
        price=Input(10.0, D),
        price_on=Input(10.0, D),
        price_pref=Input(8.0, D),
        n_on=1_000_000,
        n_pref=500_000,
        liquidity=Input(50_000.0, D),
        ll=Input(1_500_000.0, D),
        rec=Input(12_000_000.0, D),
        ebit=Input(3_000_000.0, D),
        ir=Input(-600_000.0, D),
        ll_1y=Input(1_250_000.0, D),
        pl=Input(7_500_000.0, D),
        at=Input(20_000_000.0, D),
        ac=Input(6_000_000.0, D),
        pc=Input(3_000_000.0, D),
        debt=Input(4_000_000.0, D),
        cash=Input(1_000_000.0, D),
        rec_fy=Input(12_000_000.0, D),
        rec_fy5=Input(6_000_000.0, D),
        ll_fy=Input(1_500_000.0, D),
        ll_fy5=Input(750_000.0, D),
        dividends=[Dividend(0.5, D - timedelta(days=10))],
    )
    return replace(base, **changes)


class FieldValueTests(unittest.TestCase):
    def test_each_of_the_19_fields_matches_its_hand_computed_value(self):
        result = indicators.compute(synthetic(), D)
        self.assertEqual(list(result), list(ALL_FIELDS))
        for field, expected in SYNTHETIC.items():
            with self.subTest(field=field):
                self.assertIsNone(result[field].reason)
                self.assertAlmostEqual(result[field].value, expected, places=6)

    def test_the_formula_table_names_the_same_19_fields_as_the_config(self):
        self.assertEqual(set(indicators.FORMULAS), set(ALL_FIELDS))
        self.assertEqual(len(indicators.FORMULAS), 19)

    def test_the_pn_ticker_shares_market_value_and_ev_ebit_but_not_p_l_and_psr(self):
        pn = indicators.compute(synthetic(ticker="SYNA4", price=Input(8.0, D)), D)
        self.assertAlmostEqual(pn["VALOR DE MERCADO"].value, 14_000_000.0)
        self.assertAlmostEqual(pn["EV/EBIT"].value, SYNTHETIC["EV/EBIT"], places=6)
        self.assertAlmostEqual(pn["P/L"].value, 8.0)  # 8 / 1.0
        self.assertAlmostEqual(pn["PSR"].value, 1.0)  # 8 / 8.0

    def test_unit_divisor_k_divides_the_price_in_p_l_p_vp_and_psr(self):
        # A unit of 3 shares priced at 30.00 gives the per-share ratios of a 10.00 share.
        unit = indicators.compute(synthetic(ticker="SYNU11", unit_divisor=3, price=Input(30.0, D), dividends=[Dividend(1.5, D - timedelta(days=10))]), D)
        self.assertAlmostEqual(unit["P/L"].value, 10.0)
        self.assertAlmostEqual(unit["P/VP"].value, 2.0)
        self.assertAlmostEqual(unit["PSR"].value, 1.25)
        self.assertAlmostEqual(unit["VALOR DE MERCADO"].value, 14_000_000.0)  # company value, not divided by k
        self.assertAlmostEqual(unit["DY"].value, 5.0)  # per-unit dividends over the unit price


class MarketValueTests(unittest.TestCase):
    def test_usiminas_uses_the_most_liquid_preferred_price_usim5_as_checked_in_the_spec(self):
        # Counts of CVM 2026-08-07 (treasury included), USIM5 at 7.15, the spec's value 8,324,781,006.60.
        inputs = synthetic(n_on=705_260_684, n_pref=547_818_424, price_on=Input(6.25, D), price_pref=Input(7.15, D))
        value = indicators.compute(inputs, D)["VALOR DE MERCADO"].value
        self.assertAlmostEqual(value, 8_324_781_006.60, places=2)
        # The other preferred ticker (USIM6 at 8.65) gives a different value, so the wrong ticker fails.
        wrong = indicators.compute(replace(inputs, price_pref=Input(8.65, D)), D)["VALOR DE MERCADO"].value
        self.assertAlmostEqual(wrong, 9_146_508_642.60, places=2)

    def test_a_missing_preferred_quote_prices_the_preferred_shares_at_the_on_price_and_flags_it(self):
        result = indicators.compute(synthetic(price_pref=None), D)["VALOR DE MERCADO"]
        self.assertAlmostEqual(result.value, 15_000_000.0)  # 1e6 × 10 + 5e5 × 10
        self.assertEqual(result.note, "vm_price_fallback")

    def test_no_class_quote_leaves_market_value_and_ev_ebit_without_value(self):
        result = indicators.compute(synthetic(price_on=None, price_pref=None), D)
        self.assertEqual((result["VALOR DE MERCADO"].value, result["VALOR DE MERCADO"].reason), (None, "no_quote"))
        self.assertEqual(result["EV/EBIT"].reason, "no_quote")


class ShareCountTests(unittest.TestCase):
    def test_without_a_share_count_the_six_share_fields_have_no_value_and_price_roe_and_dy_keep_theirs(self):
        result = indicators.compute(synthetic(n_on=None, n_pref=None), D)
        for field in ("VALOR DE MERCADO", "P/L", "P/VP", "PSR", "EV/EBIT", "PEG RATIO"):
            with self.subTest(field=field):
                self.assertEqual((result[field].value, result[field].reason), (None, "no_share_count"))
        self.assertAlmostEqual(result["PRECO"].value, 10.0)
        self.assertAlmostEqual(result["ROE"].value, 20.0)
        self.assertAlmostEqual(result["DY"].value, 5.0)

    def test_share_scale_1000_multiplies_market_value_and_divides_the_per_share_earnings(self):
        scaled = indicators.compute(synthetic(n_on=1_000_000_000, n_pref=500_000_000), D)
        self.assertAlmostEqual(scaled["VALOR DE MERCADO"].value, 14_000_000_000.0)
        self.assertAlmostEqual(scaled["P/L"].value, 10_000.0)  # per-share earnings 1000 times smaller


class ReasonTests(unittest.TestCase):
    def test_a_negative_net_income_gives_a_negative_p_l(self):
        self.assertAlmostEqual(indicators.compute(synthetic(ll=Input(-1_500_000.0, D)), D)["P/L"].value, -10.0)

    def test_zero_current_liabilities_leave_liq_corrente_empty_with_zero_divisor(self):
        result = indicators.compute(synthetic(pc=Input(0.0, D)), D)["LIQ. CORRENTE"]
        self.assertEqual((result.value, result.reason), (None, "zero_divisor"))

    def test_a_missing_ebit_leaves_ev_ebit_empty_with_the_missing_input_named(self):
        result = indicators.compute(synthetic(ebit=None), D)["EV/EBIT"]
        self.assertEqual((result.value, result.reason), (None, "missing_input: EBIT"))

    def test_cagr_with_a_non_positive_base_is_empty(self):
        result = indicators.compute(synthetic(rec_fy5=Input(-1.0, D), ll_fy5=Input(0.0, D)), D)
        self.assertEqual(result["CAGR RECEITAS 5 ANOS"].reason, "non_positive_base")
        self.assertEqual(result["CAGR LUCROS 5 ANOS"].reason, "non_positive_base")

    def test_cagr_with_no_fiscal_year_five_years_back_is_short_history(self):
        result = indicators.compute(synthetic(rec_fy5=None, ll_fy5=None), D)
        self.assertEqual(result["CAGR RECEITAS 5 ANOS"].reason, "short_history")
        self.assertEqual(result["CAGR LUCROS 5 ANOS"].reason, "short_history")

    def test_peg_needs_the_net_income_one_year_earlier_and_a_growth_that_is_not_zero(self):
        self.assertEqual(indicators.compute(synthetic(ll_1y=None), D)["PEG RATIO"].reason, "short_history")
        self.assertEqual(indicators.compute(synthetic(ll_1y=Input(1_500_000.0, D)), D)["PEG RATIO"].reason, "zero_divisor")

    def test_no_dividend_source_leaves_dy_without_value_and_no_dividends_is_zero(self):
        result = indicators.compute(synthetic(dividends=None), D)["DY"]
        self.assertEqual((result.value, result.reason), (None, "no_dividend_source"))
        self.assertEqual(indicators.compute(synthetic(dividends=[]), D)["DY"].value, 0.0)


class PointInTimeTests(unittest.TestCase):
    def test_an_input_dated_after_d_raises_the_pit_leak_with_exit_code_4(self):
        with self.assertRaises(PITLeakError) as ctx:
            indicators.compute(synthetic(ticker="SYNA3", ll=Input(1_500_000.0, D + timedelta(days=1))), D)
        self.assertEqual(ctx.exception.exit_code, 4)
        self.assertEqual(ctx.exception.offending_date, D + timedelta(days=1))
        self.assertEqual(ctx.exception.ticker, "SYNA3")

    def test_a_dividend_with_ex_date_after_d_raises_instead_of_being_dropped(self):
        with self.assertRaises(PITLeakError) as ctx:
            indicators.compute(synthetic(dividends=[Dividend(0.5, D + timedelta(days=1))]), D)
        self.assertEqual(ctx.exception.offending_date, D + timedelta(days=1))


if __name__ == "__main__":
    unittest.main()
