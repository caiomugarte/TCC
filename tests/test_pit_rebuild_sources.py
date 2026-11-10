"""URLs and the per-date file plan of the rebuild (SRC-01). No network: URLs are only built and parsed."""

import base64
from datetime import date, datetime, timezone
import json
import re
import sys
import unittest
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))

from pit_rebuild import sources  # noqa: E402

B3_COMPANY_PREFIX = "https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/"


def decode_payload(url):
    """Return the JSON object a B3 URL carries in its last path segment."""
    return json.loads(base64.b64decode(unquote(url.rsplit("/", 1)[1])).decode("utf-8"))


def years_in(plan, pattern):
    found = []
    for _url, name in plan:
        match = re.fullmatch(pattern, name)
        if match:
            found.append(int(match.group(1)))
    return sorted(found)


class PlanFileTests(unittest.TestCase):
    def test_plan_for_2026_08_07_and_2026_11_06_holds_each_source_once(self):
        plan = sources.plan_files([date(2026, 8, 7), date(2026, 11, 6)])

        self.assertEqual(years_in(plan, r"dfp_cia_aberta_(\d{4})\.zip"), list(range(2019, 2027)))
        self.assertEqual(years_in(plan, r"itr_cia_aberta_(\d{4})\.zip"), [2024, 2025, 2026])
        self.assertEqual(years_in(plan, r"fca_cia_aberta_(\d{4})\.zip"), list(range(2010, 2027)))
        self.assertEqual(years_in(plan, r"COTAHIST_A(\d{4})\.ZIP"), [2025, 2026])
        self.assertEqual([name for _url, name in plan].count("cad_cia_aberta.csv"), 1)
        self.assertEqual(len({url for url, _name in plan}), len(plan))

    def test_dfp_starts_seven_years_before_the_year_of_the_date(self):
        plan = sources.plan_files([date(2020, 3, 31)])

        self.assertEqual(years_in(plan, r"dfp_cia_aberta_(\d{4})\.zip")[0], 2013)
        self.assertEqual(years_in(plan, r"COTAHIST_A(\d{4})\.ZIP"), [2019, 2020])

    def test_plan_is_sorted_and_each_name_is_the_url_basename(self):
        plan = sources.plan_files([date(2026, 8, 7)])

        self.assertEqual(plan, sorted(plan))
        for url, name in plan:
            self.assertEqual(name, url.rsplit("/", 1)[1])


class B3UrlTests(unittest.TestCase):
    def test_b3_company_url_decodes_to_the_issuing_company_payload(self):
        url = sources.b3_company_url("PETR")

        self.assertTrue(url.startswith(B3_COMPANY_PREFIX))
        self.assertEqual(decode_payload(url), {"issuingCompany": "PETR", "language": "pt-br"})

    def test_b3_cash_url_page_two_decodes_to_the_pagination_payload(self):
        url = sources.b3_cash_url("PETROBRAS", 2)

        self.assertIn("/GetListedCashDividends/", url)
        self.assertEqual(
            decode_payload(url),
            {"language": "pt-br", "pageNumber": 2, "pageSize": 120, "tradingName": "PETROBRAS"},
        )


class B3QueryNameTests(unittest.TestCase):
    def test_b3_query_name_strips_removes_every_slash_and_collapses_inner_whitespace(self):
        self.assertEqual(sources.b3_query_name("AMBEV S/A   "), "AMBEV SA")
        self.assertEqual(sources.b3_query_name("KLABIN S/A"), "KLABIN SA")
        self.assertEqual(sources.b3_query_name("PETROBRAS   "), "PETROBRAS")
        self.assertEqual(sources.b3_query_name("A  /  B"), "A B")

    def test_b3_cash_url_queries_the_normalized_trading_name(self):
        # The name with "/" is sent without it (F15); the page number and size are as before.
        self.assertEqual(
            decode_payload(sources.b3_cash_url("AMBEV S/A", 1)),
            {"language": "pt-br", "pageNumber": 1, "pageSize": 120, "tradingName": "AMBEV SA"},
        )


class YahooUrlTests(unittest.TestCase):
    def test_yahoo_url_has_period1_interval_events_and_no_range(self):
        url = sources.yahoo_url("PETR4", date(2026, 8, 8))
        parts = urlsplit(url)
        query = parse_qs(parts.query)
        period2 = int(datetime(2026, 8, 8, tzinfo=timezone.utc).timestamp())

        self.assertTrue(parts.path.endswith("/PETR4.SA"))
        self.assertEqual(query["period1"], ["946684800"])
        self.assertEqual(query["period2"], [str(period2)])
        self.assertEqual(query["interval"], ["1d"])
        self.assertEqual(query["events"], ["div"])
        self.assertNotIn("range", query)


class HttpsTests(unittest.TestCase):
    def test_every_url_the_module_builds_is_https(self):
        urls = [url for url, _name in sources.plan_files([date(2026, 8, 7)])]
        urls += [
            sources.CAD_URL,
            sources.DFP_LISTING,
            sources.ITR_LISTING,
            sources.FCA_LISTING,
            sources.dfp_url(2026),
            sources.itr_url(2026),
            sources.fca_url(2026),
            sources.cotahist_url(2026),
            sources.b3_company_url("PETR"),
            sources.b3_cash_url("PETROBRAS", 1),
            sources.yahoo_url("PETR4", date(2026, 8, 8)),
        ]

        for url in urls:
            self.assertTrue(url.startswith("https://"), url)


if __name__ == "__main__":
    unittest.main()
