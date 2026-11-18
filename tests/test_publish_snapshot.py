"""Issue #85: um comando gera, valida, checa e publica o snapshot do mês, ou não publica nada."""

import contextlib
from datetime import date, datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT / "py", PROJECT_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from fetch_status_invest import OUTPUT_COLUMNS as STOCK_COLUMNS  # noqa: E402
from fetch_status_invest_fii import OUTPUT_COLUMNS as FII_COLUMNS  # noqa: E402
import publish_snapshot  # noqa: E402
from publish_snapshot import PublishSnapshotError, SanityLimits, publish_snapshot as publish  # noqa: E402
from snapshot_manifest import SnapshotManifest, latest_compatible_manifest  # noqa: E402

SERIES = ("ibovespa", "ifix", "sp500_total_return_usd", "di", "btc_usd", "ptax")
LIMITS = SanityLimits(min_scored_stocks=10, min_eligible_fiis=5, max_universe_drop=0.20, max_cutoff_lag_days=5)


def business_days(start: date, end: date) -> list:
    days, day = [], start
    while day <= end:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


def stocks_csv(count: int) -> str:
    """Ações com números variados em três setores; capitalização e liquidez passam em todo perfil."""
    sectors = ("Consumo", "Energia", "Saúde")
    rows = [",".join(STOCK_COLUMNS)]
    for index in range(count):
        values = {column: f"{1 + ((index * 7 + position * 3) % 11) * 0.5:.2f}" for position, column in enumerate(STOCK_COLUMNS)}
        values |= {"TICKER": f"A{index:03d}3", "SETOR": sectors[index % 3],
                   "VALOR DE MERCADO": "10000000000", "LIQUIDEZ MEDIA DIARIA": "5000000"}
        rows.append(",".join(values[column] for column in STOCK_COLUMNS))
    return "\n".join(rows) + "\n"


def fiis_csv(count: int, liquidity: str = "3000000") -> str:
    """FIIs com os seis números centrais positivos; patrimônio e liquidez passam em todo perfil."""
    sectors = ("Papéis", "Logística", "Shoppings")
    rows = [",".join(FII_COLUMNS)]
    for index in range(count):
        values = {column: f"{1 + ((index * 5 + position * 3) % 7) * 0.5:.2f}" for position, column in enumerate(FII_COLUMNS)}
        values |= {"TICKER": f"F{index:03d}11", "SETOR": sectors[index % 3], "GESTAO": "Ativa",
                   "PATRIMONIO": "5000000000", "LIQUIDEZ MEDIA DIARIA": liquidity}
        rows.append(",".join(values[column] for column in FII_COLUMNS))
    return "\n".join(rows) + "\n"


class FakeStages:
    """As etapas de rede trocadas por fixtures: séries até ``last`` (por série) e seleção com N ações e M FIIs."""

    def __init__(self, *, stocks=30, fiis=20, last=None, fii_liquidity="3000000"):
        self.stocks, self.fiis, self.last, self.fii_liquidity = stocks, fiis, dict(last or {}), fii_liquidity
        self.fetch_calls, self.selector_calls = [], []

    def fetch_series(self, snapshot_dir: Path, start: date, end: date) -> None:
        self.fetch_calls.append((snapshot_dir, start, end))
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        for name in SERIES:
            days = business_days(end - timedelta(days=20), self.last.get(name, end))
            lines = ["date,value", *(f"{day.isoformat()},{100 + index}" for index, day in enumerate(days))]
            (snapshot_dir / f"{name}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (snapshot_dir / "metadata.json").write_text("{}\n", encoding="utf-8")

    def create_selectors(self, out_dir: Path, day: date, cache_dir: Path) -> None:
        self.selector_calls.append((out_dir, day, cache_dir))
        out_dir.mkdir(parents=True, exist_ok=True)
        stocks = stocks_csv(self.stocks).encode("utf-8")
        fiis = fiis_csv(self.fiis, self.fii_liquidity).encode("utf-8")
        (out_dir / "stocks.csv").write_bytes(stocks)
        (out_dir / "fiis.csv").write_bytes(fiis)
        (out_dir / "stocks.provenance.json").write_text(json.dumps({
            "provider": "cvm-b3", "as_of": day.isoformat(), "csv_sha256": hashlib.sha256(stocks).hexdigest(),
            "raw_files": {},
        }), encoding="utf-8")
        (out_dir / "fiis.provenance.json").write_text(json.dumps({
            "provider": "statusinvest", "retrieved_at": "2026-11-17T12:00:00Z",
            "csv_sha256": hashlib.sha256(fiis).hexdigest(),
            "recent_trade_filter": {"collected_on": day.isoformat(), "window_days": 30, "cotahist": []},
        }), encoding="utf-8")


class Workspace:
    def __init__(self, root: Path):
        self.registry = root / "snapshots"
        self.registry.mkdir()
        self.cache_root = root / "cache"

    def publish(self, day: date, stages: FakeStages, limits: SanityLimits = LIMITS, **kwargs):
        clock = datetime(day.year, day.month, day.day, 12, tzinfo=timezone.utc)
        return publish(
            day, registry=self.registry, cache_root=self.cache_root, limits=limits, now=lambda: clock,
            fetch_series=stages.fetch_series, create_selectors=stages.create_selectors, **kwargs,
        )

    def latest(self) -> SnapshotManifest:
        return latest_compatible_manifest(self.registry)

    def states(self) -> dict:
        return {path.stem: json.loads(path.read_text(encoding="utf-8"))["state"]
                for path in sorted(self.registry.glob("*.json"))}


class PublishSnapshotTests(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.space = Workspace(Path(self._directory.name))

    def test_publishes_the_month_snapshot_when_every_stage_and_check_passes(self):
        result = self.space.publish(date(2026, 11, 17), FakeStages())

        latest = self.space.latest()
        self.assertEqual(latest.manifest_id, "mensal-2026-11-17")
        self.assertEqual(latest.state, "published")
        self.assertEqual(latest.cutoff_date, date(2026, 11, 17))
        self.assertEqual(result.manifest.manifest_id, "mensal-2026-11-17")
        self.assertFalse(result.reused)
        self.assertEqual(self.space.states(), {"mensal-2026-11-17": "published"})

    def test_a_second_call_with_the_same_date_is_a_no_op_returning_the_published_manifest(self):
        first = self.space.publish(date(2026, 11, 17), FakeStages())
        published_bytes = first.manifest_path.read_bytes()
        again = FakeStages()

        result = self.space.publish(date(2026, 11, 17), again)

        self.assertTrue(result.reused)
        self.assertEqual(result.manifest.manifest_id, "mensal-2026-11-17")
        self.assertEqual(again.fetch_calls, [])
        self.assertEqual(again.selector_calls, [])
        self.assertEqual(first.manifest_path.read_bytes(), published_bytes)

    def test_a_published_manifest_whose_cutoff_is_the_date_is_returned_without_fetching(self):
        # O corte sai das séries: publicar --date 2026-11-17 (terça) com IBOV parado na segunda dá corte 11-16;
        # pedir depois --date 2026-11-16 é o mesmo ciclo.
        self.space.publish(date(2026, 11, 17), FakeStages(last={"ibovespa": date(2026, 11, 16)}))
        again = FakeStages()

        result = self.space.publish(date(2026, 11, 16), again)

        self.assertTrue(result.reused)
        self.assertEqual(result.manifest.manifest_id, "mensal-2026-11-17")
        self.assertEqual(again.fetch_calls, [])
        self.assertEqual(self.space.states(), {"mensal-2026-11-17": "published"})

    def test_a_new_date_whose_effective_cutoff_is_already_published_returns_that_manifest(self):
        # Publicado na sexta 11-13; pedir o sábado 11-14 dá o mesmo corte (sexta) depois de buscar as séries.
        self.space.publish(date(2026, 11, 13), FakeStages())
        again = FakeStages()

        result = self.space.publish(date(2026, 11, 14), again)

        self.assertTrue(result.reused)
        self.assertEqual(result.manifest.manifest_id, "mensal-2026-11-13")
        self.assertEqual(len(again.fetch_calls), 1)
        self.assertEqual(self.space.states(), {"mensal-2026-11-13": "published"})
        self.assertFalse((self.space.registry / "data" / "mensal-2026-11-14").exists())

    def test_a_failing_stage_stops_with_its_name_and_keeps_the_previous_snapshot(self):
        self.space.publish(date(2026, 10, 15), FakeStages())

        def offline_fetch(snapshot_dir, start, end):
            raise ConnectionError("Yahoo fora do ar")

        stages = FakeStages()
        stages.fetch_series = offline_fetch
        with self.assertRaises(PublishSnapshotError) as caught:
            self.space.publish(date(2026, 11, 17), stages)

        self.assertEqual(caught.exception.stage, "series")
        self.assertEqual(caught.exception.exit_code, 4)
        self.assertIn("Yahoo fora do ar", str(caught.exception))
        self.assertEqual(stages.selector_calls, [])
        self.assertEqual(self.space.latest().manifest_id, "mensal-2026-10-15")

    def test_a_retry_after_a_failed_attempt_replaces_its_leftovers_and_publishes(self):
        with self.assertRaises(PublishSnapshotError) as caught:
            self.space.publish(date(2026, 11, 17), FakeStages(stocks=3))
        self.assertEqual(caught.exception.stage, "sanidade")
        self.assertEqual(self.space.states(), {"mensal-2026-11-17": "validated"})

        result = self.space.publish(date(2026, 11, 17), FakeStages())

        self.assertFalse(result.reused)
        self.assertEqual(self.space.states(), {"mensal-2026-11-17": "published"})
        self.assertEqual(self.space.latest().manifest_id, "mensal-2026-11-17")

    def test_a_retired_manifest_with_the_same_id_is_never_replaced(self):
        result = self.space.publish(date(2026, 11, 17), FakeStages())
        raw = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        raw["state"] = "retired"
        result.manifest_path.write_text(json.dumps(raw), encoding="utf-8")
        stages = FakeStages()

        with self.assertRaises(PublishSnapshotError) as caught:
            self.space.publish(date(2026, 11, 17), stages)

        self.assertEqual((caught.exception.stage, caught.exception.exit_code), ("registro", 3))
        self.assertEqual(stages.fetch_calls, [])
        self.assertEqual(self.space.states(), {"mensal-2026-11-17": "retired"})


if __name__ == "__main__":
    unittest.main()
