"""The 19 fields of one ticker at d: hand-computed values, units, share counts, reasons and the leak guard (IND-01, IND-03, IND-04, MAP-03, PIT-02, A-8, A-17, A-18).

The synthetic company has 1,000,000 ON and 500,000 PN shares, ON at 10.00 and PN at 8.00, so
VALOR DE MERCADO = 10,000,000 + 4,000,000 = 14,000,000. Its other inputs and every hand-computed
value are listed in SYNTHETIC below. The USIMINAS case uses the counts and prices of 2026-08-07
checked in the spec (A-17): 8,324,781,006.60 with USIM5 at 7.15, not 9,146,508,642.60 with USIM6.
"""

import sys
import unittest
from dataclasses import fields as dataclass_fields, replace
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))

from pit_rebuild import indicators  # noqa: E402
from pit_rebuild.dividends import Dividend  # noqa: E402
from pit_rebuild.errors import PITLeakError  # noqa: E402
from pit_rebuild.fields import ACCOUNTING_FIELDS, ALL_FIELDS  # noqa: E402
from pit_rebuild.indicators import Input, TickerInputs  # noqa: E402

D = date(2026, 8, 7)
SYNTHETIC = {
    "PRECO": 10.0,
    "VALOR DE MERCADO": 14_000_000.0,
    "LIQUIDEZ MEDIA DIARIA": 50_000.0,
    "P/L": 10.0,  # 10 / (1.5e6 / 1.5e6)
    "P/VP": 2.5,  # 10 / (6e6 / 1.5e6): PL_ctrl 6e6 = PL 7.5e6 less 1.5e6 of non-controlling interests (A-21, F8)
    "PSR": 1.25,  # 10 / (12e6 / 1.5e6)
    "EV/EBIT": 17_000_000 / 3_000_000,  # (14e6 + DL 3e6) / EBIT 3e6
    "DY": 5.0,  # 100 × 0.5 / 10
    "ROE": 25.0,  # 100 × 1.5e6 / abs(PL_ctrl 6e6) (F8, F9); r4 used PL 7.5e6 and gave 20
    "ROA": 7.5,  # 100 × 1.5e6 / 20e6
    "ROIC": 100 * 2_400_000 / 11_500_000,  # 100 × (3e6 - abs(-0.6e6)) / (7.5e6 + 4e6) (F11); IR is negative, so r4 gives the same
    "MARG. LIQUIDA": 12.5,  # 100 × 1.5e6 / 12e6
    "MARGEM EBIT": 25.0,  # 100 × 3e6 / 12e6
    "LIQ. CORRENTE": 2.0,  # 6e6 / 3e6
    "DIVIDA LIQUIDA / EBIT": 1.0,  # 3e6 / 3e6
    "DIV. LIQ. / PATRI.": 0.5,  # 3e6 / 6e6 (PL_ctrl, F8); r4 used PL and gave 0.4
    "CAGR RECEITAS 5 ANOS": 100 * (2 ** 0.2 - 1),  # 12e6 / 6e6
    "CAGR LUCROS 5 ANOS": 100 * (2 ** 0.2 - 1),  # LL_TTM 1.5e6 over LL_FY−5 0.75e6 (F10); the same ratio as r4 gave
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
        pl_ctrl=Input(7_500_000.0, D),  # no non-controlling interests in the base company
        at=Input(20_000_000.0, D),
        ac=Input(6_000_000.0, D),
        pc=Input(3_000_000.0, D),
        debt=Input(4_000_000.0, D),
        cash=Input(1_000_000.0, D),
        rec_fy=Input(12_000_000.0, D),
        rec_fy5=Input(6_000_000.0, D),
        ll_fy5=Input(750_000.0, D),
        dividends=[Dividend(0.5, D - timedelta(days=10))],
    )
    return replace(base, **changes)


class FieldValueTests(unittest.TestCase):
    def test_each_of_the_19_fields_matches_its_hand_computed_value(self):
        # PL_ctrl 6e6 (1.5e6 of non-controlling interests) gives the r5 values of ROE, P/VP and DIV. LIQ. / PATRI.
        result = indicators.compute(synthetic(pl_ctrl=Input(6_000_000.0, D)), D)
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


class ControllersEquityAndCagrTests(unittest.TestCase):
    """The r5 formulas that read PL_ctrl (F8, F9), IR's sign (F11) and LL_TTM in CAGR LUCROS (F10)."""

    def test_roe_divides_by_the_absolute_controllers_equity_so_a_negative_equity_keeps_the_sign_of_the_profit(self):
        # 100 × 120 / |−400| = 30 and 100 × (−120) / |−400| = −30.
        self.assertAlmostEqual(indicators.compute(synthetic(pl_ctrl=Input(-400.0, D), ll=Input(120.0, D)), D)["ROE"].value, 30.0)
        self.assertAlmostEqual(indicators.compute(synthetic(pl_ctrl=Input(-400.0, D), ll=Input(-120.0, D)), D)["ROE"].value, -30.0)

    def test_ir_of_plus_30_and_minus_30_give_the_same_roic_100_times_ebit_minus_30_over_pl_plus_db(self):
        # EBIT 3e6, PL 7.5e6, DB 4e6: 100 × (3e6 − 30) / (7.5e6 + 4e6).
        expected = 100 * (3_000_000 - 30) / (7_500_000 + 4_000_000)
        for ir in (30.0, -30.0):
            with self.subTest(ir=ir):
                self.assertAlmostEqual(indicators.compute(synthetic(ir=Input(ir, D)), D)["ROIC"].value, expected, places=9)

    def test_without_pl_ctrl_roe_pvp_and_div_liq_have_no_value_with_the_input_named_and_roic_keeps_its_value(self):
        result = indicators.compute(synthetic(pl_ctrl=None), D)
        for field in ("ROE", "P/VP", "DIV. LIQ. / PATRI."):
            with self.subTest(field=field):
                self.assertEqual((result[field].value, result[field].reason), (None, "missing_input: PL_ctrl"))
        self.assertAlmostEqual(result["ROIC"].value, SYNTHETIC["ROIC"], places=6)

    def test_cagr_lucros_is_100_times_ll_ttm_over_the_fiscal_year_five_years_back_and_follows_ll_ttm(self):
        # 100 × ((120 / 60)^(1/5) − 1); doubling only LL (TTM) gives 100 × ((240 / 60)^(1/5) − 1).
        first = indicators.compute(synthetic(ll=Input(120.0, D), ll_fy5=Input(60.0, D)), D)["CAGR LUCROS 5 ANOS"].value
        second = indicators.compute(synthetic(ll=Input(240.0, D), ll_fy5=Input(60.0, D)), D)["CAGR LUCROS 5 ANOS"].value
        self.assertAlmostEqual(first, 100 * ((120 / 60) ** (1 / 5) - 1), places=9)
        self.assertAlmostEqual(second, 100 * ((240 / 60) ** (1 / 5) - 1), places=9)

    def test_cagr_lucros_with_a_non_positive_ttm_net_income_is_non_positive_base(self):
        result = indicators.compute(synthetic(ll=Input(-120.0, D), ll_fy5=Input(60.0, D)), D)["CAGR LUCROS 5 ANOS"]
        self.assertEqual((result.value, result.reason), (None, "non_positive_base"))

    def test_the_formula_text_names_pl_ctrl_abs_ir_ll_ttm_and_21_sessions(self):
        self.assertIn("PL_ctrl", indicators.FORMULAS["ROE"])
        self.assertIn("abs(PL_ctrl)", indicators.FORMULAS["ROE"])
        self.assertIn("abs(IR)", indicators.FORMULAS["ROIC"])
        self.assertIn("LL_TTM", indicators.FORMULAS["CAGR LUCROS 5 ANOS"])
        self.assertIn("21 sessões", indicators.FORMULAS["LIQUIDEZ MEDIA DIARIA"])

    def test_tickerinputs_has_pl_ctrl_and_no_ll_fy_field(self):
        names = {item.name for item in dataclass_fields(TickerInputs)}
        self.assertIn("pl_ctrl", names)
        self.assertNotIn("ll_fy", names)


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


class StaleFilingMaskTests(unittest.TestCase):
    """STL-01, STL-02 and B-7: a stale filing blanks the 15 accounting fields with the reason stale_filing; the market fields keep theirs."""

    def test_every_accounting_field_of_a_stale_filing_has_no_value_and_the_stale_reason(self):
        self.assertEqual(indicators.STALE_FILING_REASON, "stale_filing")
        masked = indicators.mask_accounting(indicators.compute(synthetic(), D), indicators.STALE_FILING_REASON)
        for field in ACCOUNTING_FIELDS:
            with self.subTest(field=field):
                self.assertEqual((masked[field].value, masked[field].reason), (None, "stale_filing"))

    def test_the_market_fields_keep_their_value_reason_and_note(self):
        # No preferred quote gives VALOR DE MERCADO its vm_price_fallback note; no dividend source gives DY its reason.
        values = indicators.compute(synthetic(price_pref=None, dividends=None), D)
        masked = indicators.mask_accounting(values, indicators.STALE_FILING_REASON)
        for field in ("PRECO", "VALOR DE MERCADO", "LIQUIDEZ MEDIA DIARIA", "DY"):
            with self.subTest(field=field):
                self.assertEqual(masked[field], values[field])
        self.assertEqual((masked["VALOR DE MERCADO"].value, masked["VALOR DE MERCADO"].note), (15_000_000.0, "vm_price_fallback"))
        self.assertEqual((masked["DY"].value, masked["DY"].reason), (None, "no_dividend_source"))

    def test_masking_keeps_every_key_and_does_not_change_the_input(self):
        values = indicators.compute(synthetic(), D)
        before = dict(values)  # a shallow copy: any in-place change of values shows as a difference
        masked = indicators.mask_accounting(values, indicators.STALE_FILING_REASON)
        self.assertEqual(list(masked), list(values))
        self.assertEqual(values, before)
        self.assertAlmostEqual(values["P/L"].value, 10.0)  # the input still holds its P/L value


class ShareCountRuleTextTests(unittest.TestCase):
    """RPT-06 (AD-033): the formula and the module docstring say the share count takes the B3 events with ex-date in (DT_REFER, d]."""

    def test_the_market_value_formula_names_the_b3_events_of_dt_refer_d_and_keeps_the_a18_scale(self):
        formula = indicators.FORMULAS["VALOR DE MERCADO"]
        self.assertIn("eventos B3 da classe com data-ex em (DT_REFER, d]", formula)
        self.assertIn("(AD-033)", formula)
        self.assertIn("(A-18)", formula)

    def test_the_module_docstring_names_the_b3_events_in_the_share_counts(self):
        self.assertIn("by the A-18 scale and by the B3 events of the class with ex-date in (DT_REFER, d] (AD-033)", indicators.__doc__)


if __name__ == "__main__":
    unittest.main()
