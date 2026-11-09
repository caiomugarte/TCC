"""Status Invest CSV loader and eligible universe of the rebuild (SAFE-02, A-1, MAP-02 input).

The fixture tests/fixtures/pit_rebuild/si_small.csv holds one row per case:
AAAA3 sits exactly on both thresholds; BBBB3 and CCCC3 fall just below one each;
DDDD3 has a blank cap and EEEE3 a non-numeric liquidity; FFFF3 has a blank DY.
The real 2026-08-07 file is read, never written.
"""

import hashlib
import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))

from pit_rebuild.errors import IntegrityError, MissingPathError  # noqa: E402
from pit_rebuild.status_invest import ELIGIBILITY, eligible, load  # noqa: E402

FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "pit_rebuild" / "si_small.csv"
REAL_SI_2026_08_07 = PROJECT_ROOT / "data" / "raw" / "status_invest_fundamentals.csv"
PREMIUM_POLICY = PROJECT_ROOT / "api" / "app" / "services" / "premium_policy.py"


def drop_column(text, column):
    lines = text.splitlines()
    index = lines[0].split(",").index(column)
    kept = [",".join(cell for i, cell in enumerate(line.split(",")) if i != index) for line in lines]
    return "\n".join(kept) + "\n"


class EligibilityTests(unittest.TestCase):
    def test_values_exactly_at_both_thresholds_are_eligible(self):
        si = load(FIXTURE)

        self.assertEqual(si.rows["AAAA3"]["VALOR DE MERCADO"], 200_000_000)
        self.assertEqual(si.rows["AAAA3"]["LIQUIDEZ MEDIA DIARIA"], 50_000)
        self.assertIn("AAAA3", eligible(si))

    def test_values_just_below_either_threshold_are_not_eligible(self):
        si = load(FIXTURE)

        self.assertNotIn("BBBB3", eligible(si))  # cap 199,999,999.99
        self.assertNotIn("CCCC3", eligible(si))  # liquidity 49,999.99

    def test_blank_or_non_numeric_threshold_value_is_none_and_not_eligible(self):
        si = load(FIXTURE)

        self.assertIsNone(si.rows["DDDD3"]["VALOR DE MERCADO"])
        self.assertNotIn("DDDD3", eligible(si))
        self.assertIsNone(si.rows["EEEE3"]["LIQUIDEZ MEDIA DIARIA"])
        self.assertNotIn("EEEE3", eligible(si))

    def test_blank_dy_is_kept_as_none_and_the_eligible_list_is_sorted(self):
        si = load(FIXTURE)

        self.assertIsNone(si.rows["FFFF3"]["DY"])
        self.assertEqual(eligible(si), ["AAAA3", "FFFF3", "GGGG3"])

    def test_setor_is_kept_per_ticker(self):
        si = load(FIXTURE)

        self.assertEqual(si.setor["FFFF3"], "Financeiro e Outros")
        self.assertEqual(si.setor["GGGG3"], "Energia Elétrica")

    def test_eligibility_matches_the_arrojado_anchor_of_premium_policy(self):
        text = PREMIUM_POLICY.read_text(encoding="utf-8")
        stock_rules = text[text.index("STOCK_RULES = {") : text.index("FII_RULES = {")]
        arrojado = stock_rules[stock_rules.index('"arrojado"') :]
        cap = re.search(r'"cap_min":\s*([\d_]+)', arrojado).group(1)
        liq = re.search(r'"liq_min":\s*([\d_]+)', arrojado).group(1)

        self.assertEqual(int(cap.replace("_", "")), ELIGIBILITY["cap_min"])
        self.assertEqual(int(liq.replace("_", "")), ELIGIBILITY["liq_min"])

    def test_real_2026_08_07_file_has_269_eligible_and_66_with_blank_dy(self):
        si = load(REAL_SI_2026_08_07)
        tickers = eligible(si)

        self.assertEqual(len(tickers), 269)
        self.assertEqual(sum(1 for t in tickers if si.rows[t]["DY"] is None), 66)


class LoadIntegrityTests(unittest.TestCase):
    def test_missing_required_column_exits_4_naming_it(self):
        text = FIXTURE.read_text(encoding="utf-8")
        for column in ("TICKER", "SETOR", "VALOR DE MERCADO"):
            with self.subTest(column=column), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "si.csv"
                path.write_text(drop_column(text, column), encoding="utf-8")

                with self.assertRaises(IntegrityError) as ctx:
                    load(path)

                self.assertEqual(ctx.exception.exit_code, 4)
                self.assertEqual(ctx.exception.field_name, column)

    def test_repeated_ticker_exits_4_naming_it(self):
        text = FIXTURE.read_text(encoding="utf-8")
        repeated = next(line for line in text.splitlines() if line.startswith("AAAA3,"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "si.csv"
            path.write_text(text + repeated + "\n", encoding="utf-8")

            with self.assertRaises(IntegrityError) as ctx:
                load(path)

        self.assertEqual(ctx.exception.exit_code, 4)
        self.assertEqual(ctx.exception.field_name, "AAAA3")

    def test_absent_path_exits_3_naming_the_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "absent.csv"

            with self.assertRaises(MissingPathError) as ctx:
                load(missing)

        self.assertEqual(ctx.exception.exit_code, 3)
        self.assertIn(str(missing), str(ctx.exception))

    def test_load_leaves_the_file_bytes_and_mtime_unchanged_and_records_sha256(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "si.csv"
            shutil.copyfile(FIXTURE, path)
            os.utime(path, (1_700_000_000, 1_700_000_000))
            bytes_before = path.read_bytes()
            mtime_before = path.stat().st_mtime_ns

            si = load(path)

            self.assertEqual(path.read_bytes(), bytes_before)
            self.assertEqual(path.stat().st_mtime_ns, mtime_before)
        self.assertEqual(si.sha256, hashlib.sha256(bytes_before).hexdigest())


if __name__ == "__main__":
    unittest.main()
