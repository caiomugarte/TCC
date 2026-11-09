from __future__ import annotations

import random
import unittest

from app.services.whole_units import allocate_whole_units


# Run 3289af79 FII selector order and PRECO in cents; targets R$ 100.00 each.
LOT_TEST_1 = (
    ("MXRF11", 864),
    ("KNCR11", 10216),
    ("BTLG11", 9751),
    ("BPML11", 8063),
    ("GGRC11", 835),
    ("ALZR11", 954),
    ("XPML11", 9600),
    ("KNRI11", 14789),
    ("RECT11", 3182),
    ("BRCR11", 3828),
    ("KNUQ11", 9498),
    ("RBRF11", 653),
    ("CPTS11", 700),
)


def allocate(prices: dict[str, int], targets: dict[str, int], class_target: int, **kwargs):
    return allocate_whole_units(list(prices), targets, prices, class_target, **kwargs)


def quantities(allocation) -> tuple[int, ...]:
    return tuple(line.quantity for line in allocation.lines)


def random_case(seed: int):
    rng = random.Random(seed)
    count = rng.randint(1, 8)
    tickers = [f"T{index}" for index in range(count)]
    prices = {ticker: rng.randint(100, 30_000) for ticker in tickers}
    class_target = rng.randint(0, 500_000)
    cuts = sorted(rng.randint(0, class_target) for _ in range(count - 1))
    bounds = [0, *cuts, class_target]
    targets = {ticker: bounds[index + 1] - bounds[index] for index, ticker in enumerate(tickers)}
    return tickers, targets, prices, class_target


class FloorAndRemainderTests(unittest.TestCase):
    def test_floor_when_no_unit_fits_the_remainder(self):
        allocation = allocate({"A": 3000, "B": 6000}, {"A": 10_000, "B": 10_000}, 20_000)

        self.assertEqual(quantities(allocation), (4, 1))
        self.assertEqual(allocation.spent_cents, 18_000)
        self.assertEqual(allocation.cash_residual_cents, 2000)

    def test_remainder_goes_to_the_largest_deficit_that_fits(self):
        allocation = allocate(
            {"A": 700, "B": 800, "C": 5300},
            {"A": 5000, "B": 10_000, "C": 15_000},
            30_000,
        )

        self.assertEqual(quantities(allocation), (10, 15, 2))
        self.assertEqual(allocation.cash_residual_cents, 400)

    def test_a_unit_that_fills_the_class_exactly_is_bought(self):
        allocation = allocate(
            {"A": 1100, "B": 1200, "C": 7000},
            {"A": 10_000, "B": 10_000, "C": 10_000},
            30_000,
        )

        self.assertEqual(quantities(allocation), (10, 10, 1))
        self.assertEqual(allocation.spent_cents, 30_000)
        self.assertEqual(allocation.cash_residual_cents, 0)

    def test_equal_deficits_go_to_the_earlier_ticker(self):
        allocation = allocate({"A": 1000, "B": 1000}, {"A": 1500, "B": 1500}, 3000)

        self.assertEqual(quantities(allocation), (2, 1))

    def test_knri11_kncr11_real_prices(self):
        prices = dict(LOT_TEST_1)
        allocation = allocate(prices, {ticker: 10_000 for ticker in prices}, 130_000)

        self.assertEqual(
            {line.ticker: line.quantity for line in allocation.lines},
            {
                "MXRF11": 12,
                "KNCR11": 1,
                "BTLG11": 1,
                "BPML11": 1,
                "GGRC11": 12,
                "ALZR11": 11,
                "XPML11": 1,
                "KNRI11": 1,
                "RECT11": 3,
                "BRCR11": 2,
                "KNUQ11": 1,
                "RBRF11": 15,
                "CPTS11": 14,
            },
        )
        self.assertEqual([line.ticker for line in allocation.lines], list(prices))
        self.assertEqual(allocation.spent_cents, 129_596)
        self.assertEqual(allocation.cash_residual_cents, 404)
        self.assertEqual(allocation.over_target_cents, 0)

    def test_class_total_never_exceeds_target_without_minimum(self):
        for seed in range(200):
            tickers, targets, prices, class_target = random_case(seed)
            with self.subTest(seed=seed):
                allocation = allocate_whole_units(tickers, targets, prices, class_target)

                spent = sum(line.quantity * line.price_cents for line in allocation.lines)
                self.assertEqual(allocation.spent_cents, spent)
                self.assertLessEqual(spent, class_target)
                self.assertEqual(allocation.cash_residual_cents, class_target - spent)
                self.assertEqual(allocation.over_target_cents, 0)
                self.assertTrue(
                    all(price > allocation.cash_residual_cents for price in prices.values())
                )

    def test_each_invalid_input_raises(self):
        valid = {
            "tickers": ["A", "B"],
            "target_cents": {"A": 500, "B": 500},
            "price_cents": {"A": 100, "B": 200},
            "class_target_cents": 1000,
        }
        cases = {
            "empty tickers": {"tickers": []},
            "duplicate tickers": {"tickers": ["A", "A"]},
            "ticker without target": {"target_cents": {"A": 1000}},
            "ticker without price": {"price_cents": {"A": 100}},
            "zero price": {"price_cents": {"A": 0, "B": 200}},
            "negative price": {"price_cents": {"A": -100, "B": 200}},
            "float price": {"price_cents": {"A": 100.0, "B": 200}},
            "bool price": {"price_cents": {"A": True, "B": 200}},
            "negative target": {"target_cents": {"A": -500, "B": 1500}},
            "float target": {"target_cents": {"A": 500.0, "B": 500}},
            "negative class target": {
                "target_cents": {"A": 0, "B": 0},
                "class_target_cents": -1,
            },
            "float class target": {"class_target_cents": 1000.0},
            "targets do not sum to the class": {"class_target_cents": 1001},
            "negative minimum": {"minimum_count": -1},
            "bool minimum": {"minimum_count": True},
        }
        for label, override in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    allocate_whole_units(**{**valid, **override})


class DropAndMinimumTests(unittest.TestCase):
    def test_ticker_that_never_fits_ends_at_zero_and_its_target_funds_the_others(self):
        allocation = allocate(
            {"AAA3": 1000, "BBB3": 2000, "CCC3": 25_000},
            {"AAA3": 10_000, "BBB3": 10_000, "CCC3": 10_000},
            30_000,
        )

        self.assertEqual(quantities(allocation), (16, 7, 0))
        self.assertEqual([line.ticker for line in allocation.lines], ["AAA3", "BBB3", "CCC3"])
        self.assertEqual(allocation.spent_cents, 30_000)
        self.assertEqual(allocation.cash_residual_cents, 0)
        self.assertFalse(any(line.forced for line in allocation.lines))

    def test_no_renormalisation_after_a_drop(self):
        allocation = allocate(
            {"A": 6000, "B": 7000, "C": 500},
            {"A": 3334, "B": 3333, "C": 3333},
            10_000,
        )

        self.assertEqual(quantities(allocation), (1, 0, 8))
        self.assertEqual(allocation.cash_residual_cents, 0)

    def test_fii_minimum_buys_one_quota_of_the_best_placed_dropped(self):
        allocation = allocate(
            {"F1": 2000, "F2": 3000, "F3": 12_000, "F4": 15_000},
            {"F1": 2500, "F2": 2500, "F3": 2500, "F4": 2500},
            10_000,
            minimum_count=3,
        )

        self.assertEqual(quantities(allocation), (2, 2, 1, 0))
        self.assertEqual([line.forced for line in allocation.lines], [False, False, True, False])
        self.assertEqual(allocation.spent_cents, 22_000)
        self.assertEqual(allocation.over_target_cents, 12_000)
        self.assertEqual(allocation.cash_residual_cents, 0)

    def test_minimum_is_not_applied_when_three_are_bought(self):
        allocation = allocate(
            {"F1": 1000, "F2": 1000, "F3": 1000, "F4": 50_000},
            {"F1": 2500, "F2": 2500, "F3": 2500, "F4": 2500},
            10_000,
            minimum_count=3,
        )

        self.assertEqual(quantities(allocation), (4, 3, 3, 0))
        self.assertFalse(any(line.forced for line in allocation.lines))
        self.assertEqual(allocation.over_target_cents, 0)
        self.assertEqual(allocation.cash_residual_cents, 0)

    def test_minimum_when_no_fii_fits(self):
        allocation = allocate(
            {"F1": 20_000, "F2": 30_000, "F3": 40_000, "F4": 50_000},
            {"F1": 2500, "F2": 2500, "F3": 2500, "F4": 2500},
            10_000,
            minimum_count=3,
        )

        self.assertEqual(quantities(allocation), (1, 1, 1, 0))
        self.assertEqual([line.forced for line in allocation.lines], [True, True, True, False])
        self.assertEqual(allocation.spent_cents, 90_000)
        self.assertEqual(allocation.over_target_cents, 80_000)
        self.assertEqual(allocation.cash_residual_cents, 0)

    def test_minimum_is_capped_by_the_selected_count(self):
        allocation = allocate(
            {"F1": 20_000, "F2": 30_000},
            {"F1": 5000, "F2": 5000},
            10_000,
            minimum_count=3,
        )

        self.assertEqual(quantities(allocation), (1, 1))
        self.assertEqual([line.forced for line in allocation.lines], [True, True])
        self.assertEqual(allocation.over_target_cents, 40_000)

    def test_stocks_get_no_minimum(self):
        allocation = allocate(
            {"AAA3": 20_000, "BBB3": 30_000, "CCC3": 40_000},
            {"AAA3": 4000, "BBB3": 3000, "CCC3": 3000},
            10_000,
        )

        self.assertEqual(quantities(allocation), (0, 0, 0))
        self.assertFalse(any(line.forced for line in allocation.lines))
        self.assertEqual(allocation.spent_cents, 0)
        self.assertEqual(allocation.cash_residual_cents, 10_000)
        self.assertEqual(allocation.over_target_cents, 0)

    def test_total_within_target_except_the_minimum(self):
        for seed in range(200):
            tickers, targets, prices, class_target = random_case(seed)
            with self.subTest(seed=seed):
                plain = allocate_whole_units(tickers, targets, prices, class_target)
                allocation = allocate_whole_units(
                    tickers, targets, prices, class_target, minimum_count=3
                )

                spent = allocation.spent_cents
                self.assertEqual(
                    spent + allocation.cash_residual_cents - allocation.over_target_cents,
                    class_target,
                )
                bought = sum(1 for line in plain.lines if line.quantity > 0)
                missing = max(0, min(3, len(tickers)) - bought)
                zero_lines = [line.ticker for line in plain.lines if line.quantity == 0]
                expected_forced = zero_lines[:missing]
                forced = [line.ticker for line in allocation.lines if line.forced]
                self.assertEqual(forced, expected_forced)
                for before, after in zip(plain.lines, allocation.lines):
                    expected = 1 if after.forced else before.quantity
                    self.assertEqual(after.quantity, expected)
                if forced:
                    self.assertGreater(spent, class_target)
                    self.assertEqual(allocation.over_target_cents, spent - class_target)
                    self.assertEqual(allocation.cash_residual_cents, 0)
                else:
                    self.assertLessEqual(spent, class_target)
                    self.assertEqual(allocation.over_target_cents, 0)


if __name__ == "__main__":
    unittest.main()
