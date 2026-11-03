"""Checks for `.specs/features/pit-oos-validation/checks.md` (C1-C44).

Every test names the check it proves. Fixtures are synthetic weekday series
whose level on a date depends only on the date, so two retrievals of one
series agree on every date they share unless a test changes a value.
"""

import contextlib
import hashlib
import io
import json
import math
import platform
import re
import shutil
import statistics
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))

import pit_validation as pv  # noqa: E402
from allocation_config import ASSET_CLASSES  # noqa: E402
from allocation_data import load_snapshot_bundle  # noqa: E402
from core.allocation import DailyReturn  # noqa: E402

VALIDATION = PROJECT_ROOT / "docs" / "validation"
PROTOCOL_PATH = VALIDATION / "protocol.json"
REPORT_JSON = VALIDATION / "class-allocation-oos.json"
REPORT_MD = VALIDATION / "class-allocation-oos.md"
SERIES_FILES = {
    "ibovespa": "ibovespa.csv",
    "ifix": "ifix.csv",
    "sp500_total_return_usd": "sp500_total_return_usd.csv",
    "di": "di.csv",
    "btc_usd": "btc_usd.csv",
    "ptax": "ptax.csv",
}
SOURCES = {
    "ibovespa": {"provider": "Yahoo Finance chart API", "symbol": "^BVSP"},
    "ifix": {"provider": "B3 Index Statistics API", "index": "IFIX"},
    "sp500_total_return_usd": {"provider": "Yahoo Finance chart API", "symbol": "^SP500TR"},
    "di": {"provider": "Banco Central do Brasil SGS API", "series": 12},
    "btc_usd": {"provider": "Yahoo Finance chart API", "symbol": "BTC-USD"},
    "ptax": {"provider": "Banco Central do Brasil PTAX OData", "currency": "USD"},
}
SERIES_SHAPE = {
    "ibovespa": (100000.0, 0.00025, 0.05, 37.0),
    "ifix": (2500.0, 0.0002, 0.03, 23.0),
    "sp500_total_return_usd": (4000.0, 0.0003, 0.04, 41.0),
    "di": (100.0, 0.00035, 0.0, 1.0),
    "btc_usd": (700.0, 0.001, 0.2, 17.0),
    "ptax": (3.2, 0.0001, 0.02, 29.0),
}
UNVERIFIABLE_RE = re.compile(r"[0-9]+[.,][0-9]+|%")


def weekdays(start, end):
    days, day = [], start
    while day <= end:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


def level(series, day):
    base, drift, amplitude, period = SERIES_SHAPE[series]
    n = day.toordinal() - 736000
    return base * math.exp(drift * n + amplitude * math.sin(n / period))


def write_retrieval(directory, dates, retrieved_at, cutoff=None, sources=None, tweak=None):
    """Six level CSVs plus metadata; `tweak(series, day, value)` alters one level."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for series, filename in SERIES_FILES.items():
        lines = ["date,value"]
        for day in dates:
            value = level(series, day)
            if tweak is not None:
                value = tweak(series, day, value)
            lines.append(f"{day.isoformat()},{value!r}")
        (directory / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")
    metadata = {
        "source": "synthetic fixture",
        "retrieved_at": retrieved_at,
        "cutoff_date": (cutoff or dates[-1]).isoformat(),
        "sources": sources or SOURCES,
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return directory


def committed_protocol():
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def stub_strategy(rows, *, allocation_config):
    """Fast stand-in: risky share from the profile's volatility cap."""

    share = min(1.0, float(allocation_config["volatility_cap"]) * 4)
    weights = {name: share / 4 for name in ASSET_CLASSES}
    weights["fixed_income"] = 1.0 - share
    first, last = rows[0].date, rows[-1].date
    return {
        "weights": weights,
        "provenance": {
            "window": {"years_used": round((last - first).days / 365.25, 4), "full_history": True}
        },
    }


class Fixture:
    """A snapshot and a second retrieval in a temp dir, with a protocol for both."""

    def __init__(
        self,
        start=date(2016, 7, 21),
        end=date(2021, 7, 21),
        other_start=None,
        other_end=None,
        other_retrieved_at="2026-03-02T10:00:00-03:00",
        sensitivity=True,
    ):
        self.tmp = Path(tempfile.mkdtemp(prefix="pit-"))
        self.snap = self.tmp / "snap"
        self.other = self.tmp / "other"
        self.out = self.tmp / "out"
        self.out.mkdir()
        self.dates = weekdays(start, end)
        write_retrieval(self.snap, self.dates, "2026-01-01T10:00:00-03:00")
        other_dates = weekdays(other_start or start, other_end or end + timedelta(days=14))
        write_retrieval(self.other, other_dates, other_retrieved_at)
        protocol = committed_protocol()
        protocol["snapshot_dir"] = str(self.snap)
        protocol["retrievals"] = [str(self.snap), str(self.other)]
        if not sensitivity:
            protocol["cost"].pop("sensitivity_bps")
        self.write_protocol(protocol)

    @property
    def protocol_path(self):
        return self.tmp / "protocol.json"

    @property
    def audit_path(self):
        return self.tmp / "pit-audit.json"

    def protocol(self):
        return json.loads(self.protocol_path.read_text(encoding="utf-8"))

    def write_protocol(self, protocol):
        self.protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    def run(self, *argv):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = pv.main([str(item) for item in argv])
        return code, stderr.getvalue()

    def audit(self):
        return self.run("audit", "--protocol", self.protocol_path, "--out", self.audit_path)

    def audit_run_to(self, out):
        return self.run("audit", "--protocol", self.protocol_path, "--out", out)

    def report(self, strategy=stub_strategy):
        with mock.patch.object(pv, "default_strategy", strategy):
            return self.run(
                "report",
                "--protocol",
                self.protocol_path,
                "--audit",
                self.audit_path,
                "--out-dir",
                self.out,
            )

    def load_report(self):
        return json.loads((self.out / pv.REPORT_JSON).read_text(encoding="utf-8"))

    def compute(self, strategy=stub_strategy):
        """run_report on this fixture without writing (after audit())."""

        protocol_raw = self.protocol_path.read_bytes()
        audit_raw = self.audit_path.read_bytes()
        return pv.run_report(
            json.loads(protocol_raw), protocol_raw, json.loads(audit_raw), audit_raw, strategy
        )

    def cleanup(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class FixtureCase(unittest.TestCase):
    def fixture(self, **kwargs):
        fixture = Fixture(**kwargs)
        self.addCleanup(fixture.cleanup)
        return fixture

    def snapshot_of(self, directory):
        return {path.name: path.read_bytes() for path in sorted(Path(directory).iterdir())}


def rows_for(dates, returns=None):
    returns = returns or {}
    return tuple(
        DailyReturn(day, {name: returns.get((day, name), 0.0) for name in ASSET_CLASSES})
        for day in dates
    )


def real_audit():
    protocol = committed_protocol()
    return pv.build_audit(protocol, PROTOCOL_PATH.read_bytes())


def real_trimmed_rows():
    audit = real_audit()
    start, end = pv.run_period_from_audit(audit)
    rows, dropped = pv.trim_rows(load_snapshot_bundle(PROJECT_ROOT / "data" / "allocation").rows, start, end)
    return rows, dropped


# --------------------------------------------------------------------------- S1


class ProtocolFileTests(unittest.TestCase):
    def test_committed_protocol_values(self):  # C1
        protocol = committed_protocol()
        self.assertEqual(protocol["protocol_id"], "pit-walk-forward-v1")
        self.assertEqual(protocol["snapshot_dir"], "data/allocation")
        self.assertEqual(
            protocol["retrievals"],
            [
                "data/allocation",
                "data/allocation_caio_new",
                "data/allocation_caio_last",
                "data/allocation_retrieval_2026-10-29",
            ],
        )
        self.assertEqual(protocol["series"], SERIES_FILES)
        self.assertEqual(
            protocol["series_kind"],
            {
                "ibovespa": "exchange_index_close",
                "ifix": "exchange_index_close",
                "sp500_total_return_usd": "exchange_index_close",
                "di": "official_rate",
                "ptax": "official_fixing",
                "btc_usd": "spot_close",
            },
        )
        self.assertEqual(
            protocol["decision"],
            {
                "data_rule": "rows_dated_on_or_before_decision_date",
                "execution_lag_common_dates": 1,
                "min_lookback_years": 3,
                "max_lookback_years": 10,
                "test_months": 12,
                "offset_months": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                "rebalance": "reset_to_target_at_execution_date_drift_between",
            },
        )
        self.assertEqual(protocol["profiles"], ["conservador", "moderado", "arrojado"])
        self.assertEqual(protocol["benchmarks"], ["cdi", "ibov", "naive_equal_weight"])
        self.assertEqual(
            protocol["cost"],
            {
                "cost_bps": 10,
                "sensitivity_bps": [0, 10, 50],
                "basis": "one_way_on_traded_value_all_classes",
                "taxes": "excluded",
            },
        )
        self.assertEqual(protocol["method"]["id"], "equal-weight-risky-v1")
        self.assertEqual(protocol["method"]["adopted_on"], "2026-11-01")
        self.assertEqual(
            protocol["method"]["adr"],
            "docs/adr/0003-equal-weight-risky-classes-with-fixed-income.md",
        )
        self.assertEqual(set(protocol["layers"]), {"stock_selection", "fii_selection"})


class ReadmeTests(unittest.TestCase):
    def test_readme_states_protocol(self):  # C2
        raw = (VALIDATION / "README.md").read_text(encoding="utf-8")
        text = " ".join(raw.split())
        required = [
            "usa só linhas datadas `<= t`",
            "próxima data comum",
            "ainda é dos pesos anteriores",
            "volta ao alvo só em cada data de execução",
            "a cada 12 meses",
            "12 deslocamentos mensais",
            "a partir de 3 anos",
            "até a última execução em ou antes de `run_period.end`",
            "`cdi`",
            "`ibov`",
            "`naive_equal_weight`",
            "`cost_bps` 10 por lado (one-way) sobre o valor negociado",
            "Impostos e taxas de fundos estão excluídos",
            "`^BVSP`, espelho do índice oficial",
            "`^SP500TR`, espelho do índice oficial",
            "BTC-USD, cotação composta",
            "00:00 UTC",
            "pelo menos 30 dias",
            "pelo menos 250 datas",
            "`1e-9`",
            "Corte para as datas verificadas",
            "não revisão observada entre duas captações (`days_apart` dias); nenhum vintage as-of",
            "`VALIDATED` só quando as seis séries são `PIT_QUALIFIED`",
            "python3 py/pit_validation.py audit --protocol docs/validation/protocol.json "
            "--out docs/validation/pit-audit.json",
            "python3 py/pit_validation.py report --protocol docs/validation/protocol.json "
            "--audit docs/validation/pit-audit.json --out-dir docs/validation",
        ]
        for phrase in required:
            with self.subTest(phrase=phrase):
                self.assertIn(" ".join(phrase.split()), text)
        for series, kind in committed_protocol()["series_kind"].items():
            with self.subTest(series=series):
                self.assertRegex(raw, rf"\| `{series}` \| `{kind}` \|")


class ProtocolValidationTests(FixtureCase):
    def _cases(self):
        def without(key):
            return lambda p: p["benchmarks"].remove(key)

        def set_cost(key, value):
            return lambda p: p["cost"].__setitem__(key, value)

        return [
            ("no-ibov", without("ibov"), "ibov"),
            ("sp500-added", lambda p: p["benchmarks"].append("sp500"), "sp500"),
            ("cost-removed", lambda p: p.pop("cost"), "cost_bps"),
            ("cost-string", set_cost("cost_bps", "10"), "cost_bps"),
            ("cost-negative", set_cost("cost_bps", -1), "cost_bps"),
            ("cost-nan", set_cost("cost_bps", float("nan")), "cost_bps"),
            ("cost-null", set_cost("cost_bps", None), "cost_bps"),
            ("basis-removed", lambda p: p["cost"].pop("basis"), "basis"),
            ("taxes-empty", set_cost("taxes", ""), "taxes"),
            ("kind-unknown", lambda p: p["series_kind"].__setitem__("btc_usd", "close"), "btc_usd"),
        ]

    def test_invalid_protocol_exits_2(self):  # C3
        fixture = self.fixture()
        self.assertEqual(fixture.audit()[0], 0)
        (fixture.out / pv.REPORT_JSON).write_text("previous json\n", encoding="utf-8")
        (fixture.out / pv.REPORT_MD).write_text("previous md\n", encoding="utf-8")
        before = self.snapshot_of(fixture.out)
        base = fixture.protocol()
        for name, change, needle in self._cases():
            with self.subTest(case=name):
                protocol = json.loads(json.dumps(base))
                change(protocol)
                fixture.write_protocol(protocol)
                code, stderr = fixture.report()
                self.assertEqual(code, 2)
                lines = stderr.strip().splitlines()
                self.assertEqual(len(lines), 1)
                self.assertRegex(lines[0], r"^pit_validation: [^:]+: .+")
                self.assertIn(needle, lines[0])
                self.assertEqual(self.snapshot_of(fixture.out), before)

    def test_zero_cost_is_accepted(self):  # C3
        fixture = self.fixture()
        self.assertEqual(fixture.audit()[0], 0)
        protocol = fixture.protocol()
        protocol["cost"]["cost_bps"] = 0
        fixture.write_protocol(protocol)
        code, stderr = fixture.report()
        self.assertEqual(code, 0, stderr)


class CliPathTests(FixtureCase):
    def test_missing_paths_exit_3(self):  # C4
        fixture = self.fixture()
        self.assertEqual(fixture.audit()[0], 0)
        missing = fixture.tmp / "does-not-exist"
        cases = []
        protocol = fixture.protocol()

        bad_snapshot = json.loads(json.dumps(protocol))
        bad_snapshot["snapshot_dir"] = str(missing)
        bad_snapshot["retrievals"][0] = str(missing)
        bad_retrieval = json.loads(json.dumps(protocol))
        bad_retrieval["retrievals"].append(str(missing))
        cases = [("snapshot_dir", bad_snapshot), ("retrieval", bad_retrieval)]

        for command in ("audit", "report"):
            out = fixture.tmp / "new-audit.json"
            with self.subTest(command=command, path="--protocol"):
                if command == "audit":
                    code, stderr = fixture.run("audit", "--protocol", missing, "--out", out)
                else:
                    code, stderr = fixture.run(
                        "report", "--protocol", missing, "--audit", fixture.audit_path,
                        "--out-dir", fixture.out,
                    )
                self.assertEqual(code, 3)
                self.assertIn(str(missing), stderr)
                self.assertFalse(out.exists())
                self.assertEqual(list(fixture.out.iterdir()), [])
            for label, changed in cases:
                with self.subTest(command=command, path=label):
                    fixture.write_protocol(changed)
                    if command == "audit":
                        code, stderr = fixture.audit_run_to(out)
                    else:
                        code, stderr = fixture.report()
                    self.assertEqual(code, 3)
                    self.assertIn(str(missing), stderr)
                    self.assertFalse(out.exists())
                    self.assertEqual(list(fixture.out.iterdir()), [])
            fixture.write_protocol(protocol)
        with self.subTest(command="report", path="--audit"):
            code, stderr = fixture.run(
                "report", "--protocol", fixture.protocol_path, "--audit", missing,
                "--out-dir", fixture.out,
            )
            self.assertEqual(code, 3)
            self.assertIn(str(missing), stderr)
            self.assertEqual(list(fixture.out.iterdir()), [])


# --------------------------------------------------------------------------- S2


def schedule_for(dates, offset, start=None, end=None):
    start = start or dates[0]
    end = end or dates[-1]
    return pv.build_schedule(dates, start, end, 3, offset, 12)


class ScheduleTests(unittest.TestCase):
    def test_weekday_fixture_offsets(self):  # C5
        dates = weekdays(date(2016, 7, 21), date(2026, 7, 21))
        zero = schedule_for(dates, 0)
        self.assertEqual(zero.decisions[0], date(2019, 7, 19))
        self.assertEqual(zero.executions[0], date(2019, 7, 22))
        self.assertEqual(len(zero.windows), 6)
        self.assertEqual(zero.windows[0], (date(2019, 7, 22), date(2020, 7, 22)))
        self.assertEqual(zero.windows[-1], (date(2024, 7, 22), date(2025, 7, 22)))
        self.assertNotIn(date(2026, 7, 21), zero.decisions)
        self.assertEqual(zero.period_end, date(2025, 7, 22))
        eleven = schedule_for(dates, 11)
        self.assertEqual(eleven.decisions[0], date(2020, 6, 19))
        self.assertEqual(len(eleven.windows), 6)
        self.assertEqual(eleven.windows[-1], (date(2025, 6, 23), date(2026, 6, 22)))

    def test_real_snapshot_schedule(self):  # C6
        rows, _ = real_trimmed_rows()
        dates = [row.date for row in rows]
        zero = schedule_for(dates, 0)
        self.assertEqual(zero.decisions[0], date(2019, 10, 31))
        self.assertEqual(zero.executions[0], date(2019, 11, 1))
        self.assertEqual(len(zero.windows), 6)
        self.assertEqual(pv.add_months(date(2016, 10, 31), 40), date(2020, 2, 29))
        four = schedule_for(dates, 4)
        self.assertEqual(four.decisions[0], date(2020, 2, 28))
        self.assertEqual(four.executions[0], date(2020, 3, 2))
        self.assertEqual(four.decisions[5], date(2025, 2, 28))
        self.assertEqual(four.executions[5], date(2025, 3, 5))
        nine = schedule_for(dates, 9)
        self.assertEqual(len(nine.windows), 5)
        self.assertEqual(nine.windows[-1], (date(2024, 8, 1), date(2025, 8, 1)))

    def test_period_end_and_union(self):  # C7
        rows, _ = real_trimmed_rows()
        dates = [row.date for row in rows]
        for offset in range(12):
            schedule = schedule_for(dates, offset)
            with self.subTest(offset=offset):
                covered = []
                for j, (lower, upper) in enumerate(schedule.windows):
                    if j:
                        self.assertEqual(lower, schedule.windows[j - 1][1])
                    covered.extend(d for d in dates if lower < d <= upper)
                self.assertEqual(len(covered), len(set(covered)))
                self.assertEqual(
                    covered,
                    [d for d in dates if schedule.period_start < d <= schedule.period_end],
                )
                self.assertEqual(schedule.period_start, schedule.executions[0])
                self.assertEqual(schedule.period_end, schedule.windows[-1][1])
        self.assertEqual(schedule_for(dates, 8).period_end, date(2026, 7, 1))
        self.assertEqual(schedule_for(dates, 9).period_end, date(2025, 8, 1))


class IntegrityTests(FixtureCase):
    def _broken(self, change):
        original = pv.build_schedule

        def builder(dates, start, end, lookback, offset, step):
            schedule = original(dates, start, end, lookback, offset, step)
            return change(schedule, dates)

        return builder

    def test_overlap_gap_dropped_exit_4(self):  # C8
        fixture = self.fixture(end=date(2023, 7, 21))
        self.assertEqual(fixture.audit()[0], 0)

        def shift_window_2(step):
            def change(schedule, dates):
                if schedule.offset != 0:
                    return schedule
                windows = list(schedule.windows)
                lower, upper = windows[2]
                windows[2] = (dates[dates.index(lower) + step], upper)
                return pv.Schedule(
                    schedule.offset, schedule.decisions, schedule.executions, tuple(windows)
                )

            return change

        def drop_last(schedule, dates):
            if schedule.offset != 8:
                return schedule
            return pv.Schedule(
                schedule.offset,
                schedule.decisions[:-1],
                schedule.executions[:-1],
                schedule.windows[:-1],
            )

        cases = [
            ("overlap", shift_window_2(-1), "offset_0"),
            ("gap", shift_window_2(+1), "offset_0"),
            ("dropped", drop_last, "offset_8"),
        ]
        for name, change, needle in cases:
            with self.subTest(case=name):
                with mock.patch.object(pv, "build_schedule", self._broken(change)):
                    code, stderr = fixture.report()
                self.assertEqual(code, 4, stderr)
                self.assertIn(needle, stderr)
                self.assertRegex(stderr, r"\d{4}-\d{2}-\d{2}")
                self.assertEqual(list(fixture.out.iterdir()), [])
                with mock.patch.object(pv, "build_schedule", self._broken(change)):
                    with self.assertRaises(pv.IntegrityError):
                        fixture.compute()

    def test_audit_protocol_mismatch_exits_4(self):  # C42
        fixture = self.fixture()
        self.assertEqual(fixture.audit()[0], 0)
        audited = fixture.protocol()

        def revisable_btc(protocol):
            protocol["series_kind"]["btc_usd"] = "vendor_composite_close"
            return "series_kind.btc_usd"

        def snapshot_only(protocol):
            protocol["retrievals"] = [protocol["snapshot_dir"]]
            return "retrievals"

        for change in (revisable_btc, snapshot_only):
            with self.subTest(case=change.__name__):
                protocol = json.loads(json.dumps(audited))
                field = change(protocol)
                fixture.write_protocol(protocol)
                code, stderr = fixture.report()
                self.assertEqual(code, 4, stderr)
                self.assertIn(field, stderr)
                self.assertEqual(list(fixture.out.iterdir()), [])
        fixture.write_protocol(audited)
        self.assertEqual(fixture.report()[0], 0)

    def test_audit_layer_mismatch_exits_4(self):  # C44
        fixture = self.fixture()
        self.assertEqual(fixture.audit()[0], 0)
        audited = fixture.protocol()

        def stock_missing_added(protocol):
            protocol["layers"]["stock_selection"]["missing"].append("one more missing input")
            return "layers.stock_selection.missing"

        def stock_lookahead_changed(protocol):
            protocol["layers"]["stock_selection"]["excluded_lookahead"]["caio_stocks"] = "picked in 2016"
            return "layers.stock_selection.excluded_lookahead"

        def fii_lookahead_removed(protocol):
            protocol["layers"]["fii_selection"].pop("excluded_lookahead")
            return "layers.fii_selection.excluded_lookahead"

        for change in (stock_missing_added, stock_lookahead_changed, fii_lookahead_removed):
            with self.subTest(case=change.__name__):
                protocol = json.loads(json.dumps(audited))
                field = change(protocol)
                fixture.write_protocol(protocol)
                code, stderr = fixture.report()
                self.assertEqual(code, 4, stderr)
                self.assertIn(field, stderr)
                self.assertEqual(list(fixture.out.iterdir()), [])
        fixture.write_protocol(audited)
        self.assertEqual(fixture.report()[0], 0)

    def test_audit_hash_mismatch_exits_4(self):  # C34
        fixture = self.fixture()
        self.assertEqual(fixture.audit()[0], 0)
        path = fixture.snap / "ifix.csv"
        lines = path.read_text(encoding="utf-8").splitlines()
        day, value = lines[-1].split(",")
        lines[-1] = f"{day},{float(value) * 1.001!r}"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        code, stderr = fixture.report()
        self.assertEqual(code, 4)
        self.assertIn("ifix", stderr)
        self.assertEqual(list(fixture.out.iterdir()), [])


# --------------------------------------------------------------------------- S3


class LeakTests(FixtureCase):
    def test_spy_sees_rows_up_to_decision(self):  # C10
        fixture = self.fixture(end=date(2026, 7, 21), sensitivity=False)
        self.assertEqual(fixture.audit()[0], 0)
        calls = []

        def spy(rows, *, allocation_config):
            calls.append((allocation_config["name"], max(row.date for row in rows), rows[-1].date))
            return stub_strategy(rows, allocation_config=allocation_config)

        report = fixture.compute(spy)
        self.assertEqual(report["verdicts"]["class_allocation"], "VALIDATED")
        expected = []
        for profile in ("conservador", "moderado", "arrojado"):
            paths = report["profiles"][profile]["paths"]
            self.assertEqual(len(paths), 12)
            for path in paths:
                self.assertEqual(len(path["decision_dates"]), 6)
                expected.extend((profile, date.fromisoformat(d)) for d in path["decision_dates"])
        self.assertEqual(len(expected), 6 * 12 * 3)
        self.assertEqual([(name, seen) for name, seen, _ in calls], expected)
        self.assertTrue(all(seen == last for _, seen, last in calls))

    def _rows(self):
        dates = weekdays(date(2018, 1, 1), date(2023, 12, 29))
        rng = __import__("random").Random(11)
        return tuple(
            DailyReturn(
                day,
                {
                    name: (0.0002 if name == "fixed_income" else rng.gauss(0.0004, 0.004))
                    for name in ASSET_CLASSES
                },
            )
            for day in dates
        )

    def test_future_perturbation_is_inert(self):  # C11
        rows = self._rows()
        t = date(2022, 6, 15)
        config = pv.profile_config("moderado")
        base = pv.decide(rows, t, pv.default_strategy, config)
        future = tuple(
            row if row.date <= t else DailyReturn(row.date, {name: 0.5 for name in ASSET_CLASSES})
            for row in rows
        )
        self.assertEqual(pv.decide(future, t, pv.default_strategy, config)["weights"], base["weights"])
        crash = tuple(
            row
            if row.date != t
            else DailyReturn(
                row.date,
                {name: (0.0 if name == "fixed_income" else -0.6) for name in ASSET_CLASSES},
            )
            for row in rows
        )
        moved = pv.decide(crash, t, pv.default_strategy, config)["weights"]
        self.assertNotEqual(moved["fixed_income"], base["weights"]["fixed_income"])

    def test_leaky_slice_raises_and_exits_4(self):  # C12
        dates = weekdays(date(2020, 1, 1), date(2020, 3, 31))
        rows = rows_for(dates)
        t = date(2020, 2, 14)
        execution = date(2020, 2, 17)

        def through_exec(all_rows, decision):
            return tuple(row for row in all_rows if row.date <= execution)

        def one_short(all_rows, decision):
            return tuple(row for row in all_rows if row.date < decision)

        with mock.patch.object(pv, "rows_for_decision", through_exec):
            with self.assertRaises(pv.PITLeakError) as caught:
                pv.decide(rows, t, stub_strategy, pv.profile_config("moderado"))
        self.assertEqual(caught.exception.decision_date, t)
        self.assertEqual(caught.exception.offending_date, execution)
        self.assertIn("2020-02-14", str(caught.exception))
        self.assertIn("2020-02-17", str(caught.exception))
        with mock.patch.object(pv, "rows_for_decision", one_short):
            with self.assertRaises(pv.IntegrityError):
                pv.decide(rows, t, stub_strategy, pv.profile_config("moderado"))

        fixture = self.fixture()
        self.assertEqual(fixture.audit()[0], 0)

        def next_row(all_rows, decision):
            later = [row.date for row in all_rows if row.date > decision]
            stop = later[0] if later else decision
            return tuple(row for row in all_rows if row.date <= stop)

        with mock.patch.object(pv, "rows_for_decision", next_row):
            code, stderr = fixture.report()
        self.assertEqual(code, 4)
        self.assertIn("received a row dated", stderr)
        self.assertEqual(list(fixture.out.iterdir()), [])


class CutoffTests(FixtureCase):
    def test_cutoff_mismatch_exits_4(self):  # C13
        fixture = self.fixture()
        self.assertEqual(fixture.audit()[0], 0)
        last = fixture.dates[-1]
        metadata_path = fixture.snap / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["cutoff_date"], last.isoformat())
        self.assertEqual(fixture.report()[0], 0)
        for out in fixture.out.iterdir():
            out.unlink()
        for cutoff in (last + timedelta(days=4), last - timedelta(days=3)):
            with self.subTest(cutoff=cutoff):
                metadata["cutoff_date"] = cutoff.isoformat()
                metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
                code, stderr = fixture.report()
                self.assertEqual(code, 4)
                self.assertIn(cutoff.isoformat(), stderr)
                self.assertIn(last.isoformat(), stderr)
                self.assertEqual(list(fixture.out.iterdir()), [])


class TrimTests(FixtureCase):
    def test_run_uses_verified_range_only(self):  # C14
        fixture = self.fixture(other_start=date(2016, 8, 1), sensitivity=False)
        self.assertEqual(fixture.audit()[0], 0)
        seen = {"strategy": [], "simulate": []}

        def spy(rows, *, allocation_config):
            seen["strategy"].append(rows[0])
            return stub_strategy(rows, allocation_config=allocation_config)

        original = pv.simulate_path

        def simulate_spy(rows, schedule, weights, cost_bps):
            seen["simulate"].append(rows[0])
            return original(rows, schedule, weights, cost_bps)

        with mock.patch.object(pv, "simulate_path", simulate_spy):
            report = fixture.compute(spy)
        self.assertEqual(report["verdicts"]["class_allocation"], "VALIDATED")
        self.assertEqual(report["run_period"]["start"], "2016-08-01")
        self.assertEqual(report["run_period"]["unverified_common_rows"], 7)
        self.assertTrue(seen["strategy"] and seen["simulate"])
        for first in seen["strategy"] + seen["simulate"]:
            self.assertEqual(first.date, date(2016, 8, 1))
            self.assertEqual(dict(first.returns), {name: 0.0 for name in ASSET_CLASSES})

    def test_run_stops_at_verified_end(self):  # C41
        fixture = self.fixture(end=date(2022, 7, 21), sensitivity=False)
        path = fixture.other / "btc_usd.csv"
        lines = path.read_text(encoding="utf-8").splitlines()
        kept = [lines[0]] + [line for line in lines[1:] if line[:10] <= "2022-06-30"]
        path.write_text("\n".join(kept) + "\n", encoding="utf-8")
        self.assertEqual(fixture.audit()[0], 0)
        audit = json.loads(fixture.audit_path.read_text(encoding="utf-8"))
        self.assertEqual(audit["series"]["btc_usd"]["verified_range"]["end"], "2022-06-30")
        self.assertEqual(audit["series"]["ibovespa"]["verified_range"]["end"], "2022-07-21")
        last_seen = {"strategy": [], "simulate": []}

        def spy(rows, *, allocation_config):
            last_seen["strategy"].append(rows[-1].date)
            return stub_strategy(rows, allocation_config=allocation_config)

        original = pv.simulate_path

        def simulate_spy(rows, schedule, weights, cost_bps):
            last_seen["simulate"].append(rows[-1].date)
            return original(rows, schedule, weights, cost_bps)

        with mock.patch.object(pv, "simulate_path", simulate_spy):
            report = fixture.compute(spy)
        self.assertEqual(report["verdicts"]["class_allocation"], "VALIDATED")
        self.assertEqual(report["run_period"]["start"], "2016-07-21")
        self.assertEqual(report["run_period"]["end"], "2022-06-30")
        self.assertEqual(report["run_period"]["unverified_common_rows"], 15)
        # one simulate call per path for the strategy and each of the three benchmarks
        paths = len(report["profiles"]["conservador"]["paths"])
        self.assertEqual(len(last_seen["simulate"]), 3 * paths * 4)
        self.assertTrue(last_seen["strategy"])
        for day in last_seen["strategy"] + last_seen["simulate"]:
            self.assertLessEqual(day, date(2022, 6, 30))
        self.assertIn(date(2022, 6, 30), last_seen["simulate"])
        for profile in report["profiles"].values():
            for record in profile["paths"]:
                self.assertLessEqual(record["period_end"], "2022-06-30")

    def test_real_run_period(self):  # C15
        audit = real_audit()
        self.assertEqual(pv.run_period_from_audit(audit), (date(2016, 10, 31), date(2026, 7, 21)))
        rows, dropped = real_trimmed_rows()
        self.assertEqual(rows[0].date, date(2016, 10, 31))
        self.assertEqual(rows[-1].date, date(2026, 7, 21))
        self.assertEqual(dropped, 69)

    def test_committed_report_run_period(self):  # C16
        report = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
        self.assertEqual(
            report["run_period"],
            {"start": "2016-10-31", "end": "2026-07-21", "unverified_common_rows": 69},
        )
        for profile in (report.get("profiles") or {}).values():
            for path in profile["paths"]:
                self.assertGreater(path["period_start"], "2016-10-31")
                self.assertLessEqual(path["period_end"], "2026-07-21")


class SimulationTests(unittest.TestCase):
    FI = {name: (1.0 if name == "fixed_income" else 0.0) for name in ASSET_CLASSES}
    CRYPTO = {name: (1.0 if name == "crypto" else 0.0) for name in ASSET_CLASSES}

    def _lag_case(self, cost_bps):
        e0, thu, fri, mon, tue = (
            date(2020, 1, 8), date(2020, 1, 9), date(2020, 1, 10), date(2020, 1, 13), date(2020, 1, 14),
        )
        returns = {
            (e0, "fixed_income"): 0.03,
            (e0, "crypto"): 0.4,
            (mon, "crypto"): 0.10,
            (tue, "crypto"): 0.05,
        }
        rows = rows_for([e0, thu, fri, mon, tue], returns)
        schedule = pv.Schedule(0, (date(2020, 1, 7), fri), (e0, mon, tue), ((e0, mon), (mon, tue)))
        path = pv.simulate_path(rows, schedule, [self.FI, self.CRYPTO], cost_bps)
        return dict(zip(path.dates, path.values)), (e0, fri, mon, tue)

    def test_execution_lag(self):  # C17
        values, (e0, fri, mon, tue) = self._lag_case(0)
        self.assertEqual(values[e0], 1.0)
        self.assertEqual(values[fri], 1.0)
        self.assertEqual(values[mon], 1.0)
        self.assertAlmostEqual(values[tue] / values[mon] - 1.0, 0.05, places=12)
        values, (e0, fri, mon, tue) = self._lag_case(10)
        self.assertAlmostEqual(values[e0], 0.999, places=12)
        self.assertAlmostEqual(values[mon], 0.999 * (1 - 0.001 * 2.0), places=12)
        self.assertAlmostEqual(values[tue] / values[mon] - 1.0, 0.05, places=12)

    def test_cost_factor_examples(self):  # C18
        zero = {name: 0.0 for name in ASSET_CLASSES}
        half = dict(zero, brazilian_stocks=0.5, fixed_income=0.5)
        sixty = dict(zero, brazilian_stocks=0.6, fixed_income=0.4)
        self.assertAlmostEqual(pv.execution_factor(10, {}, half), 0.999, places=15)
        self.assertAlmostEqual(pv.execution_factor(10, zero, half), 0.999, places=15)
        self.assertEqual(pv.execution_factor(10, half, half), 1.0)
        self.assertAlmostEqual(pv.execution_factor(10, sixty, half), 0.9998, places=15)
        for held in ({}, half, sixty):
            self.assertEqual(pv.execution_factor(0, held, half), 1.0)

    def test_benchmarks(self):  # C19
        self.assertEqual(pv.BENCHMARK_WEIGHTS["cdi"], self.FI)
        self.assertEqual(
            pv.BENCHMARK_WEIGHTS["ibov"],
            {name: (1.0 if name == "brazilian_stocks" else 0.0) for name in ASSET_CLASSES},
        )
        self.assertEqual(pv.BENCHMARK_WEIGHTS["naive_equal_weight"], {name: 0.2 for name in ASSET_CLASSES})
        dates = weekdays(date(2020, 1, 1), date(2022, 12, 30))
        schedule = pv.Schedule(
            0,
            (date(2019, 12, 31), date(2020, 12, 31), date(2021, 12, 31)),
            (dates[0], date(2021, 1, 1), date(2022, 1, 3), date(2022, 12, 30)),
            (
                (dates[0], date(2021, 1, 1)),
                (date(2021, 1, 1), date(2022, 1, 3)),
                (date(2022, 1, 3), date(2022, 12, 30)),
            ),
        )
        cdi = pv.simulate_path(rows_for(dates), schedule, [self.FI] * 3, 10)
        self.assertAlmostEqual(cdi.values[-1], 0.999, places=15)
        doubled = rows_for(dates, {(date(2020, 6, 1), "crypto"): 1.0})
        naive = pv.simulate_path(doubled, schedule, [pv.BENCHMARK_WEIGHTS["naive_equal_weight"]] * 3, 10)
        at = dict(zip(naive.dates, naive.values))
        before = at[date(2020, 12, 31)]
        turnover = abs(0.2 - 1 / 3) + 4 * abs(0.2 - 1 / 6)
        self.assertAlmostEqual(turnover, 0.2667, places=4)
        self.assertAlmostEqual(at[date(2021, 1, 1)], before * (1 - 0.001 * turnover), places=12)

        schedules = []
        original = pv.simulate_path

        def spy(rows, sched, weights, cost_bps):
            schedules.append((sched, cost_bps))
            return original(rows, sched, weights, cost_bps)

        with mock.patch.object(pv, "simulate_path", spy):
            metrics = pv.series_paths_metrics(rows_for(dates), schedule, [self.CRYPTO] * 3, 10)
        self.assertEqual(set(metrics), {"strategy", "cdi", "ibov", "naive_equal_weight"})
        self.assertEqual(schedules, [(schedule, 10)] * 4)


# --------------------------------------------------------------------------- S5


def stats_of(values):
    p25, median, p75 = statistics.quantiles(values, n=4, method="inclusive")
    return {"paths": len(values), "min": min(values), "p25": p25, "median": median, "p75": p75, "max": max(values)}


class DistributionTests(unittest.TestCase):
    def test_quantiles_example(self):  # C20
        values = [round(0.01 * i, 2) for i in range(12, 0, -1)]
        result = pv.summarize(values)
        self.assertEqual(result["paths"], 12)
        self.assertAlmostEqual(result["min"], 0.01)
        self.assertAlmostEqual(result["p25"], 0.0375)
        self.assertAlmostEqual(result["median"], 0.065)
        self.assertAlmostEqual(result["p75"], 0.0925)
        self.assertAlmostEqual(result["max"], 0.12)

    def test_share_positive(self):  # C21
        self.assertEqual(pv.summarize_excess([-0.01, 0.0, 0.02, 0.03])["share_positive"], 0.5)


class ReportTests(FixtureCase):
    def test_paths_allow_recompute(self):  # C22
        fixture = self.fixture(end=date(2022, 7, 21))
        self.assertEqual(fixture.audit()[0], 0)
        report = fixture.compute()
        self.assertEqual(report["verdicts"]["class_allocation"], "VALIDATED")
        rows, _ = pv.trim_rows(load_snapshot_bundle(fixture.snap).rows, fixture.dates[0], fixture.dates[-1])
        for profile in ("conservador", "moderado", "arrojado"):
            data = report["profiles"][profile]
            paths = data["paths"]
            self.assertGreaterEqual(len(paths), 2)
            for path in paths:
                windows = path["windows"]
                self.assertEqual(len(path["decision_dates"]), len(windows))
                self.assertEqual(len(path["execution_dates"]), len(windows) + 1)
                self.assertEqual(path["execution_dates"][-1], path["period_end"])
                self.assertEqual(path["execution_dates"][0], path["period_start"])
                self.assertIn(path["offset"], range(12))
                for window in windows:
                    self.assertEqual(set(window["weights"]), set(ASSET_CLASSES))
                    self.assertIsInstance(window["years_used"], float)
                    self.assertIs(window["full_history"], True)
                schedule = pv.Schedule(
                    path["offset"],
                    tuple(date.fromisoformat(d) for d in path["decision_dates"]),
                    tuple(date.fromisoformat(d) for d in path["execution_dates"]),
                    tuple(
                        (date.fromisoformat(w["execution_date"]), date.fromisoformat(w["end_date"]))
                        for w in windows
                    ),
                )
                recomputed = pv.series_paths_metrics(rows, schedule, [w["weights"] for w in windows], 10)
                for series in ("strategy", "cdi", "ibov", "naive_equal_weight"):
                    for metric in ("annualized_return", "annualized_volatility", "max_drawdown"):
                        self.assertEqual(path["metrics"][series][metric], recomputed[series][metric])
            dist = data["distribution"]
            for series in ("strategy", "cdi", "ibov", "naive_equal_weight"):
                for metric in ("annualized_return", "annualized_volatility", "max_drawdown"):
                    with self.subTest(profile=profile, series=series, metric=metric):
                        expected = stats_of([p["metrics"][series][metric] for p in paths])
                        for key, value in expected.items():
                            self.assertAlmostEqual(dist["series"][series][metric][key], value, places=15)
            for benchmark in ("cdi", "ibov", "naive_equal_weight"):
                excess = [
                    p["metrics"]["strategy"]["annualized_return"] - p["metrics"][benchmark]["annualized_return"]
                    for p in paths
                ]
                summary = dist["excess_annualized_return"][benchmark]
                for key, value in stats_of(excess).items():
                    self.assertAlmostEqual(summary[key], value, places=15)
                self.assertEqual(summary["share_positive"], sum(e > 0 for e in excess) / len(excess))

    def _check_unverifiable(self, fixture, overlap, status):
        self.assertEqual(fixture.audit()[0], 0)
        code, stderr = fixture.report()
        self.assertEqual(code, 0, stderr)
        report = fixture.load_report()
        self.assertEqual(report["verdicts"]["class_allocation"], "UNVERIFIABLE")
        self.assertNotIn("profiles", report)
        text = (fixture.out / pv.REPORT_MD).read_text(encoding="utf-8")
        self.assertTrue(text.startswith("# "))
        period = report["run_period"]
        needles = [
            report["protocol_id"],
            report["protocol_sha256"],
            report["audit_sha256"],
            f"`run_period`: {period['start'] or '-'} a {period['end'] or '-'}",
            "`non_revision`",
            f"`selection_overlap`: `{overlap}`",
            " 30 dias",
            " 250 datas",
            "`1e-9`",
            "foram escolhidos em 2026-11-01",
            "Os 12 deslocamentos se sobrepõem",
            "janela de 10 anos da produção",
            "nenhum vintage as-of",
            "retorno total (metodologia B3)",
            "impostos excluídos",
            "python3 py/pit_validation.py audit --protocol docs/validation/protocol.json",
            "python3 py/pit_validation.py report --protocol docs/validation/protocol.json",
        ]
        needles += [item for items in report["missing"].values() for item in items]
        for needle in needles:
            with self.subTest(needle=needle):
                self.assertIn(needle, text)
        for layer in pv.LAYERS:
            self.assertIn(f"| `{layer}` | `{report['verdicts'][layer]}` |", text)
        for name in SERIES_FILES:
            self.assertRegex(text, rf"\| `{name}` \| `[a-z_]+` \| `{status}` \|")
            self.assertIn(report["snapshot"]["files"][SERIES_FILES[name]], text)
        offending = [line for line in text.splitlines() if UNVERIFIABLE_RE.search(line)]
        self.assertEqual(offending, [])
        self.assertNotIn(platform.python_version(), text)
        return report

    def test_unverifiable_outputs(self):  # C33
        short = self.fixture(end=date(2020, 7, 31))
        report = self._check_unverifiable(short, "full", "PIT_QUALIFIED")
        self.assertEqual(report["missing"]["class_allocation"], ["insufficient_history"])
        unpaired = self.fixture(other_retrieved_at="2026-01-21T10:00:00-03:00")
        report = self._check_unverifiable(unpaired, "none", "PIT_MISSING")
        self.assertIn("ifix: no_retrieval_pair_30d", report["missing"]["class_allocation"])

    def test_committed_environment_matches(self):  # C35
        committed = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
        self.assertRegex(committed["environment"]["python"], r"^\d+\.\d+\.\d+$")
        fixture = self.fixture()
        self.assertEqual(fixture.audit()[0], 0)
        self.assertEqual(fixture.compute()["environment"], {"python": platform.python_version()})


class MarkdownTests(FixtureCase):
    def test_markdown_statements(self):  # C24
        fixture = self.fixture(end=date(2022, 7, 21))
        self.assertEqual(fixture.audit()[0], 0)
        report = fixture.compute()
        protocol = fixture.protocol()
        audit = json.loads(fixture.audit_path.read_text(encoding="utf-8"))
        text = pv.render_markdown(report, protocol, audit)
        for profile in protocol["profiles"]:
            self.assertIn(f"### {profile}", text)
        self.assertIn("| Série | min | p25 | mediana | p75 | max | caminhos |", text)
        for needle in (
            "O método (ADR 0003), seus tetos (`ALLOCATION_PROFILE_ANCHORS`) e sua janela de 10 anos "
            "(`WINDOW_YEARS`) foram escolhidos em 2026-11-01 depois de ver o período 2019-2026.",
            "`selection_overlap`: `full`.",
            "Toda janela avaliada precede essa data",
            "Os 12 deslocamentos se sobrepõem no calendário: mostram a sensibilidade à data de "
            "início e não são amostras independentes.",
            "Nenhuma decisão vê a janela de 10 anos da produção",
            "`full_history` verdadeiro em todas as decisões",
            "`full_history` verdadeiro significa que o histórico disponível na decisão era menor "
            "que a janela de 10 anos, então a estratégia usou todo ele",
            "a ADR 0003 rejeita janelas de 3 anos por não conterem nenhuma crise",
            "não revisão observada entre duas captações (`days_apart` dias); nenhum vintage as-of",
            "`ibovespa` 60",
            "Yahoo `^BVSP`",
            "retorno total (metodologia B3)",
            "\"price index; dividends not included\"",
            "a ADR 0003 trata o Ibovespa como retorno total",
            "`cost_bps` 10 por lado (one-way)",
            "impostos excluídos",
        ):
            with self.subTest(needle=needle):
                self.assertIn(needle, text)
        for overlap in ("partial", "none"):
            changed = json.loads(json.dumps(report))
            changed["selection_overlap"]["class_allocation"] = overlap
            other = pv.render_markdown(changed, protocol, audit)
            with self.subTest(overlap=overlap):
                self.assertIn(f"`selection_overlap`: `{overlap}`.", other)
                self.assertNotIn("Toda janela avaliada precede essa data", other)
                self.assertIn("Os 12 deslocamentos se sobrepõem no calendário", other)


class SensitivityTests(FixtureCase):
    def test_sensitivity_rows(self):  # C25
        fixture = self.fixture(end=date(2022, 7, 21))
        self.assertEqual(fixture.audit()[0], 0)
        calls = []

        def counting(rows, *, allocation_config):
            calls.append(rows[-1].date)
            return stub_strategy(rows, allocation_config=allocation_config)

        report = fixture.compute(counting)
        decisions = sum(
            len(path["decision_dates"]) for data in report["profiles"].values() for path in data["paths"]
        )
        self.assertEqual(len(calls), decisions)
        rows, _ = pv.trim_rows(load_snapshot_bundle(fixture.snap).rows, fixture.dates[0], fixture.dates[-1])
        for profile, data in report["profiles"].items():
            sensitivity = data["cost_sensitivity"]
            self.assertEqual([row["cost_bps"] for row in sensitivity], [0, 10, 50])
            headline = data["distribution"]
            ten = sensitivity[1]
            for series in ("strategy", "cdi", "ibov", "naive_equal_weight"):
                self.assertEqual(ten["annualized_return"][series], headline["series"][series]["annualized_return"])
            for benchmark in ("cdi", "ibov", "naive_equal_weight"):
                self.assertEqual(ten["excess_annualized_return"][benchmark], headline["excess_annualized_return"][benchmark])
            per_bps = {0: [], 10: [], 50: []}
            for path in data["paths"]:
                schedule = pv.Schedule(
                    path["offset"],
                    tuple(date.fromisoformat(d) for d in path["decision_dates"]),
                    tuple(date.fromisoformat(d) for d in path["execution_dates"]),
                    tuple(
                        (date.fromisoformat(w["execution_date"]), date.fromisoformat(w["end_date"]))
                        for w in path["windows"]
                    ),
                )
                weights = [w["weights"] for w in path["windows"]]
                returns = {}
                for bps in (0, 10, 50):
                    metrics = pv.series_paths_metrics(rows, schedule, weights, bps)
                    per_bps[bps].append(metrics)
                    returns[bps] = metrics["strategy"]["annualized_return"]
                with self.subTest(profile=profile, offset=path["offset"]):
                    self.assertGreater(returns[0], returns[10])
                    self.assertGreater(returns[10], returns[50])
            for row in sensitivity:
                for series in ("strategy", "cdi", "ibov", "naive_equal_weight"):
                    expected = stats_of([m[series]["annualized_return"] for m in per_bps[row["cost_bps"]]])
                    for key, value in expected.items():
                        self.assertAlmostEqual(row["annualized_return"][series][key], value, places=15)
        without = self.fixture(end=date(2022, 7, 21), sensitivity=False)
        self.assertEqual(without.audit()[0], 0)
        for data in without.compute()["profiles"].values():
            self.assertNotIn("cost_sensitivity", data)

    def test_invalid_sensitivity_exits_2(self):  # C26
        fixture = self.fixture()
        self.assertEqual(fixture.audit()[0], 0)
        base = fixture.protocol()
        for value in ([0, "x"], [], [-1]):
            with self.subTest(value=value):
                protocol = json.loads(json.dumps(base))
                protocol["cost"]["sensitivity_bps"] = value
                fixture.write_protocol(protocol)
                code, stderr = fixture.report()
                self.assertEqual(code, 2)
                self.assertIn("sensitivity_bps", stderr)
                self.assertEqual(list(fixture.out.iterdir()), [])


# --------------------------------------------------------------------------- S6


def series_levels(dates, series="ibovespa", scale=1.0):
    return {day: level(series, day) * scale for day in dates}


def rfile(name, retrieved, levels, sha=None, provider="Yahoo Finance chart API", identifier=None):
    return pv.RetrievalFile(
        name,
        sha if sha is not None else (hashlib.sha256(name.encode()).hexdigest() if levels is not None else None),
        provider,
        identifier if identifier is not None else {"symbol": "^BVSP"},
        retrieved,
        levels,
    )


class AuditTests(unittest.TestCase):
    DATES = weekdays(date(2017, 6, 1), date(2018, 7, 31))  # more than 300 weekdays, holds 2018-03-14

    def classify(self, files, kind="exchange_index_close"):
        return pv.audit_series("ibovespa", kind, "A", files)

    def test_rule_examples(self):  # C27
        dates = self.DATES
        self.assertGreater(len(dates), 300)
        snap = series_levels(dates)
        a = rfile("A", "2026-01-01T09:00:00-03:00", snap)

        with self.subTest(case="qualified"):
            record = self.classify([a, rfile("B", "2026-03-01T09:00:00-03:00", dict(snap))])
            self.assertEqual(record["status"], "PIT_QUALIFIED")
            self.assertEqual(record["evidence"], "non_revision")
            self.assertEqual(record["days_apart"], 59)
            self.assertEqual(record["reasons"], [])
            self.assertEqual(record["verified_range"], {"start": dates[0].isoformat(), "end": dates[-1].isoformat()})
            self.assertEqual(record["unverified_dates"], {"count": 0, "first": None, "last": None})

        with self.subTest(case="pairs only with the snapshot group"):
            files = [
                rfile("B", "2026-01-01T09:00:00-03:00", dict(snap)),
                rfile("A", "2026-01-21T09:00:00-03:00", snap),
                rfile("C", "2026-02-10T09:00:00-03:00", dict(snap)),
            ]
            record = self.classify(files)
            self.assertEqual(sorted(p["other_group"][0] for p in record["pairs"]), ["B", "C"])
            self.assertEqual([p["days_apart"] for p in record["pairs"]], [20, 20])
            self.assertEqual(record["status"], "PIT_MISSING")
            self.assertEqual(record["reasons"], ["no_retrieval_pair_30d"])

        with self.subTest(case="values_revised"):
            other = dict(snap)
            other[dates[100]] *= 1 + 1e-6
            record = self.classify([a, rfile("B", "2026-03-01T09:00:00-03:00", other)])
            self.assertEqual((record["status"], record["reasons"]), ("PIT_MISSING", ["values_revised"]))

        with self.subTest(case="rebased di"):
            di = series_levels(dates, "di")
            files = [
                rfile("A", "2026-01-01T09:00:00-03:00", di, identifier={"series": 12}),
                rfile(
                    "B",
                    "2026-03-01T09:00:00-03:00",
                    {d: v * 1.7 for d, v in di.items()},
                    identifier={"series": 12},
                ),
            ]
            record = pv.audit_series("di", "official_rate", "A", files)
            self.assertEqual(record["status"], "PIT_QUALIFIED")

        with self.subTest(case="provider_mismatch"):
            other = rfile("B", "2026-03-01T09:00:00-03:00", dict(snap), identifier={"symbol": "^IBOV"})
            record = self.classify([a, other])
            self.assertEqual((record["status"], record["reasons"]), ("PIT_MISSING", ["provider_mismatch"]))

        with self.subTest(case="overlap_below_250"):
            short = {d: snap[d] for d in dates[-249:]}
            record = self.classify([a, rfile("B", "2026-03-01T09:00:00-03:00", short)])
            self.assertEqual((record["status"], record["reasons"]), ("PIT_MISSING", ["overlap_below_250"]))

        with self.subTest(case="overlap_incomplete"):
            holed = dict(snap)
            del holed[date(2018, 3, 14)]
            record = self.classify([a, rfile("B", "2026-03-01T09:00:00-03:00", holed)])
            self.assertEqual((record["status"], record["reasons"]), ("PIT_MISSING", ["overlap_incomplete"]))

        with self.subTest(case="series_absent"):
            files = [a, rfile("B", "2026-03-01T09:00:00-03:00", dict(snap)), rfile("C", "2026-04-01", None)]
            record = self.classify(files)
            self.assertEqual(record["status"], "PIT_QUALIFIED")
            absent = [r for r in record["retrievals"] if r["retrieval"] == "C"][0]
            self.assertEqual(absent["reasons"], ["series_absent"])
            record = self.classify([rfile("A", "2026-01-01", None), rfile("B", "2026-03-01", dict(snap))])
            self.assertEqual((record["status"], record["reasons"]), ("PIT_MISSING", ["series_absent"]))

        for kind in ("vendor_composite_close", "vendor_adjusted_close"):
            with self.subTest(case=kind):
                record = self.classify([a, rfile("B", "2026-03-01T09:00:00-03:00", dict(snap))], kind)
                self.assertEqual((record["status"], record["reasons"]), ("PIT_MISSING", ["kind_revisable"]))

        with self.subTest(case="most shared dates wins"):
            fewer = {d: snap[d] for d in dates[20:]}
            files = [
                a,
                rfile("B", "2026-02-10T09:00:00-03:00", dict(snap)),
                rfile("C", "2026-04-01T09:00:00-03:00", fewer),
            ]
            record = self.classify(files)
            self.assertEqual(record["days_apart"], 40)
            self.assertEqual(record["verified_range"]["start"], dates[0].isoformat())
            tie = [a, rfile("B", "2026-02-10T09:00:00-03:00", dict(snap)), rfile("C", "2026-04-01T09:00:00-03:00", dict(snap))]
            self.assertEqual(self.classify(tie)["days_apart"], 90)

        with self.subTest(case="trim to a later paired start"):
            later = {d: snap[d] for d in dates if d >= date(2017, 7, 3)}
            record = self.classify([a, rfile("B", "2026-03-01T09:00:00-03:00", later)])
            self.assertEqual(record["status"], "PIT_QUALIFIED")
            self.assertEqual(record["verified_range"]["start"], "2017-07-03")
            unverified = [d for d in dates if d < date(2017, 7, 3)]
            self.assertEqual(
                record["unverified_dates"],
                {"count": len(unverified), "first": unverified[0].isoformat(), "last": unverified[-1].isoformat()},
            )

    def test_non_finite_level_is_revision(self):  # C43
        dates = self.DATES
        snap = series_levels(dates)
        a = rfile("A", "2026-01-01T09:00:00-03:00", snap)
        other = dict(snap)
        other[dates[100]] = float("nan")
        record = self.classify([a, rfile("B", "2026-03-01T09:00:00-03:00", other)])
        self.assertEqual((record["status"], record["reasons"]), ("PIT_MISSING", ["values_revised"]))
        self.assertIsNone(record["pairs"][0]["max_rel_diff_daily_factor"])
        record = self.classify([rfile("A", "2026-01-01T09:00:00-03:00", other), rfile("B", "2026-03-01T09:00:00-03:00", dict(snap))])
        self.assertEqual((record["status"], record["reasons"]), ("PIT_MISSING", ["values_revised"]))

    def test_real_audit(self):  # C28
        audit = real_audit()
        ibov = audit["series"]["ibovespa"]
        by_name = {r["retrieval"]: r for r in ibov["retrievals"]}
        repo = ["data/allocation", "data/allocation_caio_new", "data/allocation_caio_last"]
        for name in repo:
            self.assertEqual(sorted(by_name[name]["same_bytes_as"]), sorted(set(repo) - {name}))
        group = [g for g in ibov["groups"] if "data/allocation" in g["members"]][0]
        self.assertEqual(sorted(group["members"]), sorted(repo))
        self.assertEqual(group["retrieved_at"][:10], "2026-07-24")
        self.assertEqual(len(ibov["pairs"]), 1)
        pair = ibov["pairs"][0]
        self.assertEqual(pair["other_group"], ["data/allocation_retrieval_2026-10-29"])
        self.assertEqual(pair["days_apart"], 97)
        self.assertEqual(pair["shared_dates"], 2413)
        expected = {
            "ibovespa": ("exchange_index_close", 70),
            "ifix": ("exchange_index_close", 70),
            "sp500_total_return_usd": ("exchange_index_close", 71),
            "di": ("official_rate", 70),
            "btc_usd": ("spot_close", 100),
            "ptax": ("official_fixing", 70),
        }
        for name, (kind, unverified) in expected.items():
            with self.subTest(series=name):
                record = audit["series"][name]
                self.assertEqual(record["series_kind"], kind)
                self.assertEqual(record["status"], "PIT_QUALIFIED")
                self.assertEqual(record["evidence"], "non_revision")
                self.assertEqual(record["days_apart"], 97)
                self.assertEqual(
                    record["unverified_dates"],
                    {"count": unverified, "first": "2016-07-21", "last": "2016-10-28"},
                )

    def test_audit_record_fields(self):  # C29
        audit = json.loads((VALIDATION / "pit-audit.json").read_text(encoding="utf-8"))
        self.assertEqual(set(audit["series"]), set(SERIES_FILES))
        for name, record in audit["series"].items():
            with self.subTest(series=name):
                for key in ("status", "reasons", "evidence", "days_apart", "verified_range", "unverified_dates"):
                    self.assertIn(key, record)
                self.assertEqual(len(record["retrievals"]), 4)
                for entry in record["retrievals"]:
                    for key in ("provider", "identifier", "retrieved_at", "sha256", "same_bytes_as", "first_date", "last_date", "rows"):
                        self.assertIn(key, entry)
                    self.assertTrue(entry["identifier"])
                for pair in record["pairs"]:
                    for key in ("snapshot_retrieved_at", "other_retrieved_at", "days_apart", "shared_dates", "max_rel_diff_daily_factor"):
                        self.assertIn(key, pair)
                self.assertEqual(set(record["verified_range"]), {"start", "end"})
                self.assertEqual(set(record["unverified_dates"]), {"count", "first", "last"})

    def test_selection_layers_recorded(self):  # C30
        audit = json.loads((VALIDATION / "pit-audit.json").read_text(encoding="utf-8"))
        stock = audit["layers"]["stock_selection"]
        fii = audit["layers"]["fii_selection"]
        self.assertEqual(stock["status"], "PIT_MISSING")
        self.assertEqual(fii["status"], "PIT_MISSING")
        joined = " | ".join(stock["missing"] + fii["missing"])
        for item in (
            "Status Invest fundamentals as published on each decision date",
            "historical listed universe including delisted tickers",
            "as-of (not retroactively adjusted) total-return prices per ticker",
            "as-of FII eligibility data (liquidity, segment, listing)",
        ):
            self.assertIn(item, joined)
        self.assertEqual(stock["excluded_lookahead"], {"caio_stocks": "picked in 2025-2026"})
        self.assertEqual(fii["excluded_lookahead"], {"caio_fiis": "picked in 2025-2026"})
        protocol = committed_protocol()
        self.assertEqual(stock["missing"], protocol["layers"]["stock_selection"]["missing"])
        self.assertEqual(fii["missing"], protocol["layers"]["fii_selection"]["missing"])


# --------------------------------------------------------------------------- S7


def qualified_audit():
    return {
        "series": {
            name: {"status": "PIT_QUALIFIED", "reasons": []} for name in SERIES_FILES
        }
    }


class VerdictTests(unittest.TestCase):
    def test_verdict_rule(self):  # C31
        audit = qualified_audit()
        status, missing = pv.class_verdict(audit, {"conservador": 12, "moderado": 12, "arrojado": 12})
        self.assertEqual((status, missing), ("VALIDATED", []))
        verdicts = pv.verdicts_for(status)
        self.assertEqual(
            verdicts,
            {
                "class_allocation": "VALIDATED",
                "basic_recommendation": "VALIDATED",
                "stock_selection": "UNVERIFIABLE",
                "fii_selection": "UNVERIFIABLE",
                "premium_recommendation": "UNVERIFIABLE",
            },
        )
        audit["series"]["ifix"] = {"status": "PIT_MISSING", "reasons": ["no_retrieval_pair_30d"]}
        status, missing = pv.class_verdict(audit, {"conservador": 12, "moderado": 12, "arrojado": 12})
        self.assertEqual(status, "UNVERIFIABLE")
        self.assertIn("ifix: no_retrieval_pair_30d", missing)
        verdicts = pv.verdicts_for(status)
        self.assertEqual(verdicts["basic_recommendation"], "UNVERIFIABLE")
        self.assertEqual(verdicts["premium_recommendation"], "UNVERIFIABLE")

        fixture = Fixture(end=date(2020, 9, 30))
        self.addCleanup(fixture.cleanup)
        self.assertEqual(fixture.audit()[0], 0)
        report = fixture.compute()
        self.assertEqual(report["verdicts"]["class_allocation"], "VALIDATED")
        self.assertEqual(report["evidence"], {"class_allocation": "non_revision"})
        self.assertEqual(report["verdicts"]["premium_recommendation"], "UNVERIFIABLE")

    def test_insufficient_history_examples(self):  # C9
        cases = [
            (date(2020, 8, 31), 2, "VALIDATED"),
            (date(2020, 7, 31), 1, "UNVERIFIABLE"),
            (date(2020, 7, 21), 0, "UNVERIFIABLE"),
        ]
        for end, paths, expected in cases:
            with self.subTest(end=end):
                fixture = Fixture(end=end, sensitivity=False)
                self.addCleanup(fixture.cleanup)
                self.assertEqual(fixture.audit()[0], 0)
                report = fixture.compute()
                self.assertEqual(report["run_period"]["end"], end.isoformat())
                self.assertEqual(report["verdicts"]["class_allocation"], expected)
                missing = report["missing"]["class_allocation"]
                self.assertEqual(missing, [] if expected == "VALIDATED" else ["insufficient_history"])
                if expected == "VALIDATED":
                    for profile in report["profiles"].values():
                        self.assertEqual(len(profile["paths"]), paths)
                        self.assertEqual(
                            [(p["offset"], p["execution_dates"][1]) for p in profile["paths"]],
                            [(0, "2020-07-22"), (1, "2020-08-24")],
                        )
                else:
                    self.assertNotIn("profiles", report)
        dates = weekdays(date(2016, 7, 21), date(2020, 7, 31))
        self.assertEqual(sum(1 for k in range(12) if schedule_for(dates, k).windows), 1)
        dates = weekdays(date(2016, 7, 21), date(2020, 7, 21))
        self.assertEqual(sum(1 for k in range(12) if schedule_for(dates, k).windows), 0)

    def test_selection_overlap_examples(self):  # C32
        end = date(2026, 7, 21)
        first = [date(2019, 11, 1), date(2019, 12, 2)]
        self.assertEqual(pv.selection_overlap(end, date(2026, 11, 1), first), "full")
        self.assertEqual(pv.selection_overlap(end, date(2026, 1, 1), first), "partial")
        self.assertEqual(pv.selection_overlap(end, date(2019, 1, 1), first), "none")
        self.assertEqual(pv.selection_overlap(end, date(2019, 11, 1), first), "none")
        short = weekdays(date(2024, 1, 2), date(2026, 12, 31))
        executions = [s.executions[0] for s in (schedule_for(short, k) for k in range(12)) if s.executions]
        self.assertEqual(executions, [])
        self.assertEqual(pv.selection_overlap(short[-1], date(2026, 11, 1), executions), "none")


class CliTests(FixtureCase):
    def test_failed_write_leaves_previous_outputs(self):  # C36
        fixture = self.fixture()
        self.assertEqual(fixture.audit()[0], 0)
        (fixture.out / pv.REPORT_JSON).write_text("previous json\n", encoding="utf-8")
        (fixture.out / pv.REPORT_MD).write_text("previous md\n", encoding="utf-8")
        before = self.snapshot_of(fixture.out)
        original = tempfile.NamedTemporaryFile
        calls = []

        def failing(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise OSError("disk full")
            return original(*args, **kwargs)

        with mock.patch.object(pv.tempfile, "NamedTemporaryFile", failing):
            code, stderr = fixture.report()
        self.assertNotEqual(code, 0)
        self.assertEqual(len(calls), 2)
        self.assertIn("disk full", stderr)
        self.assertEqual(self.snapshot_of(fixture.out), before)

        audit_before = fixture.audit_path.read_bytes()
        calls.clear()

        def failing_write(*args, **kwargs):
            handle = original(*args, **kwargs)
            handle.write = mock.Mock(side_effect=OSError("disk full"))
            return handle

        with mock.patch.object(pv.tempfile, "NamedTemporaryFile", failing_write):
            code, _ = fixture.audit()
        self.assertNotEqual(code, 0)
        self.assertEqual(fixture.audit_path.read_bytes(), audit_before)
        self.assertEqual(sorted(p.name for p in fixture.tmp.iterdir()), sorted(["snap", "other", "out", "protocol.json", "pit-audit.json"]))


# --------------------------------------------------------------------------- S5 real data


class RealDataTests(unittest.TestCase):
    def test_years_used_real(self):  # C23
        rows, _ = real_trimmed_rows()
        config = pv.profile_config("moderado")
        first = pv.decide(rows, date(2019, 10, 31), pv.default_strategy, config)
        self.assertEqual(first["years_used"], 2.9979)
        self.assertIs(first["full_history"], True)
        last = pv.decide(rows, date(2025, 6, 30), pv.default_strategy, config)
        self.assertEqual(last["years_used"], 8.6626)
        self.assertIs(last["full_history"], True)
        report = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
        for profile in (report.get("profiles") or {}).values():
            for path in profile["paths"]:
                for window in path["windows"]:
                    self.assertIs(window["full_history"], True)
        offset8 = [p for p in report["profiles"]["moderado"]["paths"] if p["offset"] == 8][0]
        self.assertEqual(offset8["decision_dates"][-1], "2025-06-30")
        self.assertEqual(offset8["windows"][-1]["years_used"], 8.6626)


# --------------------------------------------------------------------------- S8


class NoticeTests(unittest.TestCase):
    DOCS = {
        "docs/adr/0003-equal-weight-risky-classes-with-fixed-income.md": "../validation/class-allocation-oos.md",
        "docs/research/class-allocation-methods-2026-11.md": "../validation/class-allocation-oos.md",
        "docs/research/class-allocation-2026-11/README.md": "../../validation/class-allocation-oos.md",
    }

    def test_notices_match_report(self):  # C37
        report = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
        verdict = report["verdicts"]["class_allocation"]
        overlap = report["selection_overlap"]["class_allocation"]
        for relative, link in self.DOCS.items():
            with self.subTest(doc=relative):
                path = PROJECT_ROOT / relative
                lines = path.read_text(encoding="utf-8").splitlines()
                self.assertTrue(lines[0].startswith("# "))
                first_body = next(line for line in lines[1:] if line.strip())
                expected = (
                    "> **Validação point-in-time:** os números históricos deste documento não têm "
                    "fonte point-in-time auditada. Veredito da alocação entre classes: "
                    f"{verdict}; sobreposição com o período da escolha do método: {overlap} "
                    f"(ver [docs/validation/class-allocation-oos.md]({link}))."
                )
                self.assertEqual(first_body, expected)
                self.assertTrue((path.parent / link).resolve() == REPORT_MD.resolve())


# --------------------------------------------------------------------------- door 2


class DataCopyTests(unittest.TestCase):
    SOURCE_SHA256 = {
        "ibovespa.csv": "35e9ae574c100afe5dff56e8b8e7f9b02567a6d4a0438b6606189db8def37645",
        "ifix.csv": "1bf7c8cf2091c145bebf851efa7bc2af7420e9f34d8ed625940a0af97e3e7cc7",
        "sp500_total_return_usd.csv": "bb0af247e2e170cb5188be0a4b792e78c1d8196385bf842c8c07a77b36d3b706",
        "di.csv": "650457a93cda97b52ca22643083d06414b5079807e710282b7c9c17733852877",
        "btc_usd.csv": "092a6a33f1a01ee45589788472841b849335dafc843f9f2b920ed0d7af255f3c",
        "ptax.csv": "ac8cd03f4c97140af74b44dcca3317a5ead4295e341eee2a65557c292b579212",
        "metadata.json": "b23ded728b9c1c737f172139c6b2208d51ae0ed0b437b730d820e3697a8242bc",
    }

    def test_retrieval_copy_hashes(self):  # C40
        directory = PROJECT_ROOT / "data" / "allocation_retrieval_2026-10-29"
        names = sorted(path.name for path in directory.iterdir())
        self.assertEqual(names, sorted(self.SOURCE_SHA256))
        self.assertFalse([name for name in names if name.startswith("caio_")])
        for name, expected in self.SOURCE_SHA256.items():
            with self.subTest(file=name):
                self.assertEqual(hashlib.sha256((directory / name).read_bytes()).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
