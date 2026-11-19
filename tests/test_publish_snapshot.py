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
import zipfile

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
        leftover = self.space.registry / "data" / "mensal-2026-11-17" / "series" / "sobra.csv"
        leftover.parent.mkdir(parents=True, exist_ok=True)
        leftover.write_text("date,value\n", encoding="utf-8")

        result = self.space.publish(date(2026, 11, 17), FakeStages())

        self.assertFalse(leftover.exists())
        self.assertFalse(result.reused)
        self.assertEqual(self.space.states(), {"mensal-2026-11-17": "published"})
        self.assertEqual(self.space.latest().manifest_id, "mensal-2026-11-17")

    def assert_check_blocks_publish(self, check: str, stages: FakeStages, limits: SanityLimits = LIMITS):
        self.space.publish(date(2026, 10, 15), FakeStages())

        with self.assertRaises(PublishSnapshotError) as caught:
            self.space.publish(date(2026, 11, 17), stages, limits)

        self.assertEqual((caught.exception.stage, caught.exception.check), ("sanidade", check))
        self.assertEqual(caught.exception.exit_code, 7)
        self.assertEqual(self.space.latest().manifest_id, "mensal-2026-10-15")
        self.assertEqual(self.space.states(), {"mensal-2026-10-15": "published", "mensal-2026-11-17": "validated"})

    def test_too_few_scored_stocks_blocks_the_publish(self):
        self.assert_check_blocks_publish("acoes_com_score", FakeStages(), LIMITS.__class__(
            min_scored_stocks=31, min_eligible_fiis=5, max_universe_drop=0.20, max_cutoff_lag_days=5))

    def test_too_few_eligible_fiis_blocks_the_publish(self):
        # Liquidez de 1 milhão passa no moderado e no arrojado, não no conservador (2 milhões).
        self.assert_check_blocks_publish("fiis_elegiveis", FakeStages(fii_liquidity="1000000"))

    def test_an_index_series_ending_before_the_date_blocks_the_publish(self):
        for series, name in (("ibovespa", "IBOV"), ("ifix", "IFIX"), ("di", "CDI")):
            with self.subTest(series=series):
                self.setUp()
                self.assert_check_blocks_publish(
                    "cobertura_indices", FakeStages(last={series: date(2026, 11, 6)}))

    def test_a_stale_cutoff_from_another_class_series_blocks_the_publish(self):
        self.assert_check_blocks_publish("defasagem_corte", FakeStages(last={"btc_usd": date(2026, 11, 6)}))

    def test_a_stock_universe_drop_beyond_the_limit_blocks_the_publish(self):
        self.assert_check_blocks_publish("queda_universo_acoes", FakeStages(stocks=23))

    def test_a_fii_universe_drop_beyond_the_limit_blocks_the_publish(self):
        self.assert_check_blocks_publish("queda_universo_fiis", FakeStages(fiis=15))

    def test_a_drop_within_the_limit_publishes(self):
        self.space.publish(date(2026, 10, 15), FakeStages())

        result = self.space.publish(date(2026, 11, 17), FakeStages(stocks=24, fiis=16))

        self.assertEqual(self.space.latest().manifest_id, "mensal-2026-11-17")
        self.assertEqual({check.name: check.passed for check in result.checks}, {
            "acoes_com_score": True, "fiis_elegiveis": True, "cobertura_indices": True, "defasagem_corte": True,
            "queda_universo_acoes": True, "queda_universo_fiis": True,
        })

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


def raw_cache(root: Path, names: dict) -> Path:
    """Um cache do pit_rebuild com ``names`` (nome -> bytes) e o manifest.json com o hash de cada um."""
    root.mkdir(parents=True)
    files = {}
    for name, data in names.items():
        (root / name).write_bytes(data)
        files[name] = {"bytes": len(data), "name": name, "retrieved_at": "2026-11-17T12:00:00Z",
                       "sha256": hashlib.sha256(data).hexdigest(), "url": f"https://example.invalid/{name}"}
    sizes = {f"https://example.invalid/{name}": len(data) for name, data in names.items()}
    manifest = {"files": files, "sizes": sizes, "probes": {"https://example.invalid/COTAHIST_A2005.ZIP": False}}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


SEED_FILES = {
    "COTAHIST_A2024.ZIP": b"cotahist 2024", "COTAHIST_A2025.ZIP": b"cotahist 2025", "COTAHIST_A2026.ZIP": b"cotahist 2026",
    "dfp_cia_aberta_2019.zip": b"dfp 2019", "dfp_cia_aberta_2025.zip": b"dfp 2025", "dfp_cia_aberta_2026.zip": b"dfp 2026",
    "itr_cia_aberta_2024.zip": b"itr 2024", "itr_cia_aberta_2026.zip": b"itr 2026",
    "fca_cia_aberta_2010.zip": b"fca 2010", "fca_cia_aberta_2026.zip": b"fca 2026",
    "inf_mensal_fii_2023.zip": b"inf 2023", "inf_mensal_fii_2026.zip": b"inf 2026",
    "cad_cia_aberta.csv": b"cadastro", "listing_dfp.html": b"listing",
    "b3_cash_AMBEV_SA_p1.json": b"{}", "b3_company_ABEV.json": b"{}", "b3_detail_1234.json": b"{}",
    "yahoo_ABCP11_2026-11-07.json": b"{}",
}
FROZEN_2026 = ["COTAHIST_A2024.ZIP", "dfp_cia_aberta_2019.zip", "fca_cia_aberta_2010.zip",
               "inf_mensal_fii_2023.zip", "itr_cia_aberta_2024.zip"]


class CycleCacheTests(unittest.TestCase):
    """O cache de brutos de cada --date: só os anuais congelados vêm do ciclo anterior; o resto é baixado de novo."""

    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)

    def cached_names(self, cache: Path) -> list:
        return sorted(path.name for path in cache.iterdir() if path.name != "manifest.json")

    def test_a_new_date_is_seeded_only_with_frozen_yearly_files_of_the_seed_cache(self):
        seed = raw_cache(self.root / "seed", SEED_FILES)

        cache = publish_snapshot.prepare_cycle_cache(self.root / "ciclos", date(2026, 12, 15), seed_cache=seed)

        self.assertEqual(cache, self.root / "ciclos" / "2026-12-15")
        self.assertEqual(self.cached_names(cache), FROZEN_2026)
        manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(sorted(manifest["files"]), FROZEN_2026)
        self.assertEqual(sorted(manifest["sizes"]), [f"https://example.invalid/{name}" for name in FROZEN_2026])
        self.assertEqual(manifest["probes"], {"https://example.invalid/COTAHIST_A2005.ZIP": False})

    def test_the_seeded_files_are_served_by_the_cache_without_a_download_and_the_rest_is_downloaded(self):
        from pit_rebuild.cache import Cache

        seed = raw_cache(self.root / "seed", SEED_FILES)
        cache_dir = publish_snapshot.prepare_cycle_cache(self.root / "ciclos", date(2026, 12, 15), seed_cache=seed)
        requested = []

        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        december = io.BytesIO()
        with zipfile.ZipFile(december, "w") as archive:
            archive.writestr("COTAHIST_A2026.TXT", "dezembro")

        def opener(request, timeout):
            requested.append(request.full_url)
            return Response(december.getvalue())

        cache = Cache(cache_dir, offline=False, opener=opener)
        cache.verify()
        frozen = cache.get("https://example.invalid/COTAHIST_A2024.ZIP", "COTAHIST_A2024.ZIP")
        fresh = cache.get("https://example.invalid/COTAHIST_A2026.ZIP", "COTAHIST_A2026.ZIP")

        self.assertEqual(frozen.read_bytes(), b"cotahist 2024")
        self.assertEqual(fresh.read_bytes(), december.getvalue())
        self.assertEqual(requested, ["https://example.invalid/COTAHIST_A2026.ZIP"])
        self.assertEqual((seed / "COTAHIST_A2026.ZIP").read_bytes(), b"cotahist 2026")

    def test_the_latest_earlier_cycle_cache_is_the_seed(self):
        ciclos = self.root / "ciclos"
        raw_cache(ciclos / "2026-10-15", {"dfp_cia_aberta_2019.zip": b"outubro"})
        raw_cache(ciclos / "2026-11-17", {"dfp_cia_aberta_2020.zip": b"novembro"})
        raw_cache(ciclos / "2027-01-15", {"dfp_cia_aberta_2021.zip": b"janeiro"})
        seed = raw_cache(self.root / "seed", {"dfp_cia_aberta_2018.zip": b"semente"})

        cache = publish_snapshot.prepare_cycle_cache(ciclos, date(2026, 12, 15), seed_cache=seed)

        self.assertEqual(self.cached_names(cache), ["dfp_cia_aberta_2020.zip"])

    def test_an_existing_cache_of_the_same_date_is_reused_as_is(self):
        ciclos = self.root / "ciclos"
        existing = raw_cache(ciclos / "2026-12-15", {"COTAHIST_A2026.ZIP": b"baixado em dezembro"})
        seed = raw_cache(self.root / "seed", SEED_FILES)

        cache = publish_snapshot.prepare_cycle_cache(ciclos, date(2026, 12, 15), seed_cache=seed)

        self.assertEqual(cache, existing)
        self.assertEqual(self.cached_names(cache), ["COTAHIST_A2026.ZIP"])

    def test_without_any_seed_the_cycle_cache_starts_empty(self):
        cache = publish_snapshot.prepare_cycle_cache(self.root / "ciclos", date(2026, 12, 15))

        self.assertEqual(self.cached_names(cache), [])

    def test_publish_hands_the_selection_the_cache_of_its_date(self):
        space = Workspace(self.root)
        stages = FakeStages()

        space.publish(date(2026, 11, 17), stages)

        self.assertEqual(stages.selector_calls[0][2], space.cache_root / "2026-11-17")
        self.assertTrue((space.cache_root / "2026-11-17" / "manifest.json").is_file())


class CommandLineTests(unittest.TestCase):
    """Os códigos de saída da docstring de scripts/publish_snapshot.py, pelo main real."""

    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)
        self.space = Workspace(self.root)
        self.limits = self.root / "limits.json"
        self.limits.write_text(json.dumps({"min_scored_stocks": 10, "min_eligible_fiis": 5}), encoding="utf-8")

    def run_main(self, stages: FakeStages, *extra, day="2026-11-17"):
        clock = datetime(2026, 11, 17, 12, tzinfo=timezone.utc)
        out, err = io.StringIO(), io.StringIO()
        argv = ["--date", day, "--registry", str(self.space.registry), "--cache-root", str(self.space.cache_root),
                "--limits", str(self.limits), *extra]
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = publish_snapshot.main(argv, now=lambda: clock, fetch_series=stages.fetch_series,
                                         create_selectors=stages.create_selectors)
        return code, out.getvalue(), err.getvalue()

    def test_exit_0_publishes_and_exit_0_again_is_the_no_op(self):
        code, out, _ = self.run_main(FakeStages())
        self.assertEqual(code, 0)
        self.assertIn("publicado", out)
        again = FakeStages()
        code, out, _ = self.run_main(again)
        self.assertEqual(code, 0)
        self.assertIn("já publicado: mensal-2026-11-17", out)
        self.assertEqual(again.fetch_calls, [])

    def test_exit_1_when_the_cycle_cache_cannot_be_prepared(self):
        self.space.cache_root.write_text("não é um diretório", encoding="utf-8")
        code, _, err = self.run_main(FakeStages())
        self.assertEqual(code, 1)
        self.assertIn("etapa cache", err)

    def test_exit_2_on_invalid_arguments(self):
        for extra, day in (((), "2026-13-01"), (("--bogus",), "2026-11-17")):
            with self.subTest(day=day, extra=extra):
                with contextlib.redirect_stderr(io.StringIO()):
                    code, _, _ = self.run_main(FakeStages(), *extra, day=day)
                self.assertEqual(code, 2)
        self.limits.write_text(json.dumps({"min_score": 1}), encoding="utf-8")
        code, _, err = self.run_main(FakeStages())
        self.assertEqual(code, 2)
        self.assertIn("min_score", err)

    def test_exit_3_when_the_registry_does_not_exist(self):
        self.space.registry.rmdir()
        code, _, err = self.run_main(FakeStages())
        self.assertEqual(code, 3)
        self.assertIn("etapa registro", err)

    def test_exit_4_when_the_series_fail(self):
        stages = FakeStages()
        stages.fetch_series = lambda *args: (_ for _ in ()).throw(ConnectionError("sem rede"))
        code, _, err = self.run_main(stages)
        self.assertEqual(code, 4)
        self.assertIn("etapa series", err)

    def test_exit_5_when_the_selection_fails(self):
        stages = FakeStages()
        stages.create_selectors = lambda *args: (_ for _ in ()).throw(RuntimeError("Status Invest fora"))
        code, _, err = self.run_main(stages)
        self.assertEqual(code, 5)
        self.assertIn("etapa selecao", err)

    def test_exit_6_when_the_manifest_cannot_be_built(self):
        # Sem datas comuns entre as séries das classes, build_manifest recusa.
        code, _, err = self.run_main(FakeStages(last={"ptax": date(2026, 10, 1)}))
        self.assertEqual(code, 6)
        self.assertIn("etapa build", err)
        self.assertEqual(self.space.states(), {})

    def test_exit_7_when_a_sanity_check_fails(self):
        code, _, err = self.run_main(FakeStages(stocks=3))
        self.assertEqual(code, 7)
        self.assertIn("acoes_com_score", err)
        self.assertEqual(self.space.states(), {"mensal-2026-11-17": "validated"})


if __name__ == "__main__":
    unittest.main()
