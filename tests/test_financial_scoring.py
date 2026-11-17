"""Issue #79 (spec #75): banks and insurers score with their own indicators, against their own peers.

Every test goes through a CSV in the collector's columns (the Status Invest columns plus SUBSETOR),
read with load_raw_data, as the Premium selector reads the snapshot. No network.

tests/fixtures/financials/common_ranking_before.csv is the ranking of the common universe below
(no SUBSETOR column, no financial) produced by the scoring of commit aea019f, before #79. It is the
source of truth of "a common stock scores exactly as before": regenerate it only from that commit.
"""

import math
from pathlib import Path
import random
import sys
import tempfile
import unittest

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))

from config import PROFILE_WEIGHTS  # noqa: E402
from core.preprocessing import apply_robustness_filter, load_raw_data, preprocess_profile  # noqa: E402
from core.scoring import build_scores  # noqa: E402
from fetch_status_invest import OUTPUT_COLUMNS  # noqa: E402

GOLDEN = PROJECT_ROOT / "tests" / "fixtures" / "financials" / "common_ranking_before.csv"
FILTERS = {"cap_min": 100.0, "liq_min": 100.0}
WEIGHTS = dict(PROFILE_WEIGHTS["caio_new"])
METRICS = [c for c in OUTPUT_COLUMNS if c not in ("TICKER", "PRECO", "SETOR", "VALOR DE MERCADO", "LIQUIDEZ MEDIA DIARIA")]
NOT_APPLICABLE = ("EV/EBIT", "MARGEM EBIT", "ROIC", "DIVIDA LIQUIDA / EBIT", "DIV. LIQ. / PATRI.", "LIQ. CORRENTE", "PSR")
BANKS = "Intermediários Financeiros"
INSURERS = "Previdência e Seguros"
OTHER_FINANCIAL = "Serviços Financeiros Diversos"


def _row(ticker: str, setor: str, seed: int, subsetor: str | None = None) -> dict:
    rng = random.Random(seed)
    row = {column: round(rng.uniform(-5.0, 30.0), 2) for column in METRICS}
    row.update({"TICKER": ticker, "PRECO": 10.0, "VALOR DE MERCADO": 1000.0, "LIQUIDEZ MEDIA DIARIA": 1000.0, "SETOR": setor})
    if subsetor is not None:
        row["SUBSETOR"] = subsetor
    if subsetor in (BANKS, INSURERS):
        row.update(dict.fromkeys(NOT_APPLICABLE, float("nan")))  # as the collector writes a financial
        row["MARGEM BRUTA"] = float("nan")
        row["PATRIMONIO / ATIVOS"] = round(rng.uniform(0.05, 0.25), 4)
    return row


def common_rows() -> list[dict]:
    """Two sectors of common stocks with the gaps the selector has rules for."""
    rows = [_row(f"IND{i}3", "Materiais Básicos", 100 + i) for i in range(1, 7)]
    rows += [_row(f"ENE{i}3", "Utilidade Pública", 200 + i) for i in range(1, 6)]
    rows[1]["DY"] = float("nan")  # no dividend: DY 0
    rows[2]["PEG RATIO"] = float("nan")
    for column in ("LIQ. CORRENTE", "DIVIDA LIQUIDA / EBIT", "DIV. LIQ. / PATRI."):
        rows[3][column] = float("nan")  # whole group empty: leaves by the empty SCORE rule
    for column in METRICS[:8]:
        rows[7][column] = float("nan")  # low coverage: leaves by the robustness filter
    rows[8]["VALOR DE MERCADO"] = 1.0  # below the size filter
    for column in METRICS[8:13]:
        rows[10][column] = float("nan")  # coverage just above 80 %: kept only while SUBSETOR is not a metric
    return rows


def bank_rows(n: int = 4, seed: int = 300) -> list[dict]:
    return [_row(f"BAN{i}3", "Financeiro", seed + i, BANKS) for i in range(1, n + 1)]


def insurer_rows(n: int = 3, seed: int = 400) -> list[dict]:
    return [_row(f"SEG{i}3", "Financeiro", seed + i, INSURERS) for i in range(1, n + 1)]


def with_subsetor(rows: list[dict]) -> list[dict]:
    """The common rows as the collector writes them: with their B3 subsector."""
    return [dict(row, SUBSETOR=f"Subsetor {row['SETOR']}") for row in rows]


def _load(rows: list[dict], directory: str) -> pd.DataFrame:
    columns = [*OUTPUT_COLUMNS, *(["SUBSETOR"] if any("SUBSETOR" in row for row in rows) else [])]
    path = Path(directory) / "stocks.csv"
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return load_raw_data(path)


def clean(rows: list[dict]) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as directory:
        return apply_robustness_filter(preprocess_profile(_load(rows, directory), filters=FILTERS))


def ranked(rows: list[dict]) -> pd.DataFrame:
    return build_scores(clean(rows), factor_weights=WEIGHTS)


def scores(rows: list[dict]) -> dict[str, float]:
    return ranked(rows).set_index("TICKER")["SCORE"].to_dict()


def golden_csv(frame: pd.DataFrame) -> str:
    return frame.to_csv(index=False, lineterminator="\n")


def golden_scores() -> dict[str, float]:
    return pd.read_csv(GOLDEN, float_precision="round_trip").set_index("TICKER")["SCORE"].to_dict()


class CommonStockTests(unittest.TestCase):
    def test_a_common_universe_ranks_byte_for_byte_as_before(self):
        self.assertEqual(golden_csv(ranked(common_rows())), GOLDEN.read_text(encoding="utf-8"))

    def test_common_stocks_score_as_before_next_to_financials_and_a_subsector_column(self):
        before = golden_scores()

        after = scores(with_subsetor(common_rows()) + bank_rows() + insurer_rows())

        self.assertEqual({ticker: after[ticker] for ticker in before}, before)

    def test_the_subsector_is_not_a_metric_with_the_pandas_2_text_dtype(self):
        # pandas 2 reads text as object, which the selector converts to numbers: SUBSETOR would become an
        # all-empty metric, push ENE53 under the 80 % coverage of the robustness filter and lose the
        # subsector that marks a financial.
        before = golden_scores()
        universe = with_subsetor(common_rows()) + bank_rows() + insurer_rows()

        try:
            with pd.option_context("future.infer_string", False):
                after = scores(universe)
        except (KeyError, pd.errors.OptionError):
            self.skipTest("this pandas always reads text as str")

        self.assertIn("ENE53", before)
        self.assertEqual({ticker: after.get(ticker) for ticker in before}, before)
        self.assertEqual(after, scores(universe))



class FinancialSetTests(unittest.TestCase):
    def test_no_bank_or_insurer_leaves_by_the_empty_score_or_the_coverage_rule(self):
        frame = ranked(common_rows() + bank_rows() + insurer_rows())

        financials = {row["TICKER"] for row in bank_rows() + insurer_rows()}
        self.assertEqual(financials - set(frame["TICKER"]), set())
        self.assertTrue(frame.set_index("TICKER").loc[sorted(financials), "SCORE"].map(math.isfinite).all())
        self.assertEqual(financials & set(frame.attrs["exclusion_reasons"]), set())

    def test_an_indicator_that_does_not_apply_does_not_move_a_financial_score(self):
        # A CSV that still carries EV/EBIT, LIQ. CORRENTE... for a bank (e.g. another source) scores the same,
        # for the bank and for the two common stocks of the sector, whose peers are the whole sector.
        filled = [dict(row, **{column: 7.0 + i for i, column in enumerate(NOT_APPLICABLE)}) for row in bank_rows()]
        filled[0].update({"EV/EBIT": -40.0, "LIQ. CORRENTE": 90.0, "ROIC": 80.0})
        others = [_row(f"BOL{i}3", "Financeiro", 500 + i, OTHER_FINANCIAL) for i in range(1, 3)]

        expected = scores(bank_rows() + insurer_rows() + others)
        self.assertEqual(len(expected), 9)
        self.assertEqual(scores(filled + insurer_rows() + others), expected)

    def test_solidity_is_equity_over_assets(self):
        banks = bank_rows()
        solidity = {row["TICKER"]: row["PATRIMONIO / ATIVOS"] for row in banks}

        avg = ranked(banks).set_index("TICKER")["AVG_LIQUIDEZ"]

        self.assertEqual(sorted(avg.index, key=avg.get), sorted(solidity, key=solidity.get))
        top = max(solidity, key=solidity.get)
        banks = [dict(row, **{"PATRIMONIO / ATIVOS": 0.01}) if row["TICKER"] == top else row for row in banks]
        self.assertEqual(ranked(banks).set_index("TICKER")["AVG_LIQUIDEZ"].idxmin(), top)

    def test_a_financial_uses_the_same_profile_weights(self):
        frame = ranked(bank_rows()).set_index("TICKER")
        groups = {"liquidez": "AVG_LIQUIDEZ", "rent": "AVG_RENT", "value": "AVG_VALUE", "growth": "AVG_GROWTH", "div": "AVG_DIV"}

        for ticker, row in frame.iterrows():
            expected = sum(WEIGHTS[group] * row[column] for group, column in groups.items())
            self.assertAlmostEqual(row["SCORE"], expected, places=12, msg=ticker)

    def test_a_common_stock_of_the_financial_sector_keeps_the_common_indicators(self):
        others = [_row(f"BOL{i}3", "Financeiro", 500 + i, OTHER_FINANCIAL) for i in range(1, 4)]
        moved = [dict(others[0], **{"EV/EBIT": others[0]["EV/EBIT"] + 50.0}), *others[1:]]
        peers = bank_rows() + insurer_rows()

        self.assertNotEqual(scores(peers + others)["BOL13"], scores(peers + moved)["BOL13"])


class PeerGroupTests(unittest.TestCase):
    def test_a_bank_score_does_not_change_when_a_company_of_another_sector_joins(self):
        alone = scores(bank_rows())
        self.assertEqual(len(alone), 4)

        joined = scores(bank_rows() + common_rows() + insurer_rows() + [_row("BOL13", "Financeiro", 501, OTHER_FINANCIAL)])

        self.assertEqual({ticker: joined[ticker] for ticker in alone}, alone)

    def test_a_subsector_with_3_eligible_papers_is_its_own_peer_group(self):
        insurers = insurer_rows(3)

        alone = scores(insurers)
        self.assertEqual(len(alone), 3)
        joined = scores(insurers + bank_rows(5))

        self.assertEqual({ticker: joined[ticker] for ticker in alone}, alone)

    def test_a_subsector_with_fewer_than_3_eligible_papers_is_compared_in_the_whole_financial_sector(self):
        insurers = insurer_rows(2)
        # The whole sector is these 6 papers; with every one of them in one subsector the peers are the same 6.
        as_one_subsector = [dict(row, SUBSETOR=BANKS) for row in insurers]

        fallback = scores(insurers + bank_rows(4))
        same_peers = scores(as_one_subsector + bank_rows(4))

        self.assertEqual({t: fallback[t] for t in ("SEG13", "SEG23")}, {t: same_peers[t] for t in ("SEG13", "SEG23")})
        self.assertNotEqual(fallback["SEG13"], scores(insurers + bank_rows(5))["SEG13"])

    def test_only_eligible_papers_count_for_the_3_peers(self):
        insurers = insurer_rows(3)
        insurers[2]["VALOR DE MERCADO"] = 1.0  # fails the size filter: 2 eligible insurers

        with_tiny = scores(insurers + bank_rows(4))
        without = scores(insurers[:2] + bank_rows(4))

        self.assertEqual({t: with_tiny[t] for t in ("SEG13", "SEG23")}, {t: without[t] for t in ("SEG13", "SEG23")})


    def test_the_3_peers_are_3_companies_not_3_share_classes_of_one(self):
        # SEGU3, SEGU4 and SEGU11 are share classes of one insurer: same company-level ratios, so a peer group of
        # only them has zero deviation. They count as one company and are compared in the whole financial sector.
        insurer = _row("SEGU3", "Financeiro", 401, INSURERS)
        classes = [dict(insurer, TICKER=ticker) for ticker in ("SEGU3", "SEGU4", "SEGU11")]

        fallback = scores(classes + bank_rows(4))
        same_peers = scores([dict(row, SUBSETOR=BANKS) for row in classes] + bank_rows(4))

        self.assertEqual({t: fallback.get(t) for t in ("SEGU3", "SEGU4", "SEGU11")},
                         {t: same_peers[t] for t in ("SEGU3", "SEGU4", "SEGU11")})

if __name__ == "__main__":
    unittest.main()
