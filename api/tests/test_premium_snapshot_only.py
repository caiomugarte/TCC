"""Issue #17: the Premium executor reads only the registered snapshot."""

import hashlib
import json
import random
import socket
import tempfile
import unittest
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.adapters.premium_optimization import run_premium_optimization
from app.db.base import Base
from app.db.models import Account, ProfileRecord
from app.repositories.recommendations import RecommendationRepository
from app.services.premium_executor import PremiumExecutor

from allocation_config import ASSET_CLASSES
from fetch_status_invest import OUTPUT_COLUMNS as STOCK_COLUMNS
from fetch_status_invest_fii import OUTPUT_COLUMNS as FII_COLUMNS
from snapshot_manifest import SnapshotManifest, sha256_path

from tests.test_premium_integration import policy as fixture_policy


DATES = ["2026-10-29", "2026-10-30"]
STOCKS = ["AAA3", "BBB3", "CCC3", "DDD3"]
FIIS = ["AAA11", "BBB11", "CCC11", "DDD11"]


def _write_levels(path: Path) -> None:
    lines = ["date,value", *(f"{day},{100 + index}" for index, day in enumerate(DATES))]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_selector(path: Path, columns, tickers, sector: str) -> None:
    rows = [",".join(columns)]
    for ticker in tickers:
        values = {"TICKER": ticker, "PRECO": "10.00", "SETOR": sector}
        rows.append(",".join(values.get(column, "1") for column in columns))
    path.write_bytes(("\n".join(rows) + "\n").encode("utf-8"))


def build_snapshot(root: Path) -> Path:
    """A registered snapshot on disk: allocation series plus stock/FII selector CSVs."""

    allocation = root / "allocation"
    allocation.mkdir(parents=True)
    for name in ("ibovespa", "ifix", "sp500_total_return_usd", "di", "btc_usd", "ptax"):
        _write_levels(allocation / f"{name}.csv")
    _write_selector(root / "stocks.csv", STOCK_COLUMNS, STOCKS, "Bens Industriais")
    _write_selector(root / "fiis.csv", FII_COLUMNS, FIIS, "Papéis")
    payload = {
        "manifest_id": "snapshot-2026-10-30",
        "manifest_version": "1",
        "created_at": "2026-10-30T21:00:00+00:00",
        "cutoff_date": DATES[-1],
        "common_dates": DATES,
        "supported_classes": list(ASSET_CLASSES),
        "allocation_source": {
            "path": "allocation",
            "sha256": sha256_path(allocation),
            "provider": "snapshot-build:allocation",
        },
        "stock_source": {
            "path": "stocks.csv",
            "sha256": sha256_path(root / "stocks.csv"),
            "provider": "cvm-b3",
        },
        "fii_source": {
            "path": "fiis.csv",
            "sha256": sha256_path(root / "fiis.csv"),
            "provider": "statusinvest",
        },
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest_path


def _pick(input_path, seed, universe_size):
    """A deterministic stand-in selector: picks two tickers from the CSV with the seed."""

    lines = Path(input_path).read_text(encoding="utf-8").splitlines()[1:]
    tickers = sorted(line.split(",")[0] for line in lines)
    assert len(tickers) == universe_size
    chosen = random.Random(seed).sample(tickers, 2)
    return {"selected_tickers": chosen, "sleeve_weights": {ticker: 0.5 for ticker in chosen}}


ALLOCATION = {
    "current_target": {
        "selected": {
            "profile_winner": {
                "weights": {
                    "brazilian_stocks": 0.4,
                    "fiis": 0.2,
                    "international_equity": 0.1,
                    "fixed_income": 0.2,
                    "crypto": 0.1,
                }
            }
        }
    }
}


class PremiumSnapshotOnlyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.engine = create_engine(f"sqlite:///{self.root / 'snapshot-only.sqlite'}")
        Base.metadata.create_all(self.engine)
        self.sessions = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)
        self.session = self.sessions()
        self.account = Account(email="snapshot-only@example.com")
        self.session.add(self.account)
        self.session.flush()
        self.profile = ProfileRecord(
            account_id=self.account.id,
            version=1,
            answers={},
            dimensions={},
            suitability_score=0.5,
            generic_profile="moderado",
            investable_capital_brl=100_000,
            consented_at=datetime.now(timezone.utc),
        )
        self.session.add(self.profile)
        self.session.commit()
        self.snapshot_root = self.root / "registry"
        self.snapshot_root.mkdir()
        self.manifest_path = build_snapshot(self.snapshot_root)
        self.engine_calls = []

    def tearDown(self):
        self.session.close()
        self.engine.dispose()
        self.directory.cleanup()

    def queue_run(self, *, recorded_hashes=None, seed=1234):
        manifest = SnapshotManifest.load(self.manifest_path)
        hashes = {key: source.sha256 for key, source in manifest.sources.items()}
        if recorded_hashes:
            hashes.update(recorded_hashes)
        policy = fixture_policy(
            self.profile,
            rules_version="premium-policy-fixture",
            source_snapshot_ids=(manifest.manifest_id,),
            cutoff_date=DATES[-1],
            random_seed=seed,
        )
        policy["plan"] = "premium"
        policy["provenance"]["source_snapshot_hashes"] = hashes
        repository = RecommendationRepository(self.session)
        run = repository.create_queued(
            account_id=self.account.id,
            profile_id=self.profile.id,
            plan="premium",
            policy=policy,
            provenance={
                **policy["provenance"],
                "manifest_id": manifest.manifest_id,
                "manifest_path": str(self.manifest_path),
                "manifest": manifest.as_dict(),
            },
        )
        self.session.commit()
        return run

    def runner(self):
        def engine(name, size):
            def call(**kwargs):
                self.engine_calls.append(name)
                return _pick(kwargs["input_path"], kwargs["seed"], size)

            return call

        def allocation(*_args, **_kwargs):
            self.engine_calls.append("allocation")
            return ALLOCATION

        return partial(
            run_premium_optimization,
            snapshot_loader=lambda _manifest: SimpleNamespace(rows=(), metadata={}),
            allocation_engine=allocation,
            stock_engine=engine("stocks", len(STOCKS)),
            fii_engine=engine("fiis", len(FIIS)),
        )

    def execute(self, run):
        executor = PremiumExecutor(
            session_factory=self.sessions,
            optimization_runner=self.runner(),
            workspace_root=self.root / "workspaces",
        )
        try:
            executor.execute_run(run.id)
        finally:
            executor.shutdown()
        with self.sessions() as check:
            return check.get(type(run), run.id)

    def assert_failed_before_calculation(self, stored, code):
        self.assertEqual(stored.status, "failed")
        self.assertEqual(stored.failure_code, code)
        self.assertIsNone(stored.result_json)
        self.assertEqual(self.engine_calls, [])

    def test_premium_run_completes_with_the_network_blocked(self):
        run = self.queue_run()

        def no_network(*_args, **_kwargs):
            raise AssertionError("Premium run opened a network connection")

        with mock.patch.object(socket.socket, "connect", no_network), mock.patch.object(
            socket, "create_connection", no_network
        ), mock.patch.object(socket, "getaddrinfo", no_network):
            stored = self.execute(run)

        self.assertEqual(stored.status, "completed", stored.failure_code)
        self.assertEqual(self.engine_calls, ["allocation", "stocks", "fiis"])

    def test_tampered_selector_csv_fails_with_snapshot_changed_before_any_calculation(self):
        run = self.queue_run()
        stocks = self.snapshot_root / "stocks.csv"
        stocks.write_bytes(stocks.read_bytes().replace(b"AAA3", b"ZZZ3"))

        self.assert_failed_before_calculation(self.execute(run), "snapshot_changed")

    def reregister(self, name: str, data: bytes) -> None:
        """Replace a selector CSV and record its new hash, as a snapshot build would."""

        (self.snapshot_root / name).write_bytes(data)
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        key = {"stocks.csv": "stock_source", "fiis.csv": "fii_source"}[name]
        payload[key]["sha256"] = hashlib.sha256(data).hexdigest()
        self.manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    def test_registered_but_malformed_selector_csv_fails_with_snapshot_changed(self):
        cases = {
            "missing column": b"TICKER,PRECO\nAAA3,10.00\n",
            "duplicate ticker": (self.snapshot_root / "stocks.csv").read_bytes()
            + (self.snapshot_root / "stocks.csv").read_bytes().splitlines(keepends=True)[1],
            "ragged row": (self.snapshot_root / "stocks.csv").read_bytes() + b"EEE3,1\n",
        }
        original_stocks = (self.snapshot_root / "stocks.csv").read_bytes()
        for label, data in cases.items():
            with self.subTest(case=label):
                self.engine_calls.clear()
                self.reregister("stocks.csv", data)
                run = self.queue_run()
                self.assert_failed_before_calculation(self.execute(run), "snapshot_changed")
        self.reregister("stocks.csv", original_stocks)

    def test_hash_differing_from_the_one_recorded_in_the_policy_fails_with_snapshot_changed(self):
        for key in ("stock", "fii", "allocation"):
            with self.subTest(source=key):
                run = self.queue_run(recorded_hashes={key: "0" * 64})
                self.assert_failed_before_calculation(self.execute(run), "snapshot_changed")

    def test_invalid_manifest_sha256_fails_with_snapshot_changed(self):
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        payload["manifest_sha256"] = "f" * 64
        self.manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        run = self.queue_run()

        self.assert_failed_before_calculation(self.execute(run), "snapshot_changed")

    def test_missing_selector_file_fails_with_snapshot_unavailable(self):
        for name in ("stocks.csv", "fiis.csv"):
            with self.subTest(file=name):
                run = self.queue_run()
                data = (self.snapshot_root / name).read_bytes()
                (self.snapshot_root / name).unlink()
                try:
                    self.assert_failed_before_calculation(self.execute(run), "snapshot_unavailable")
                finally:
                    (self.snapshot_root / name).write_bytes(data)

    def test_missing_manifest_fails_with_snapshot_unavailable(self):
        run = self.queue_run()
        with self.sessions() as edit:
            stored = edit.get(type(run), run.id)
            stored.provenance_json = {
                key: value for key, value in stored.provenance_json.items() if key != "manifest"
            }
            edit.commit()
        self.manifest_path.unlink()

        self.assert_failed_before_calculation(self.execute(run), "snapshot_unavailable")

    def test_run_without_any_persisted_manifest_fails_with_snapshot_unavailable(self):
        run = self.queue_run()
        with self.sessions() as edit:
            stored = edit.get(type(run), run.id)
            stored.provenance_json = {"manifest_id": "snapshot-2026-10-30"}
            edit.commit()

        self.assert_failed_before_calculation(self.execute(run), "snapshot_unavailable")

    def test_live_collection_is_never_called_on_the_premium_path(self):
        import fetch_status_invest
        import fetch_status_invest_fii
        from app.services import b3_trading
        from pit_rebuild import collector as pit_collector

        live = AssertionError("live collection called on the Premium path")
        run = self.queue_run()
        with mock.patch.object(fetch_status_invest, "refresh", side_effect=live) as stocks, \
                mock.patch.object(fetch_status_invest_fii, "refresh", side_effect=live) as fiis, \
                mock.patch.object(pit_collector, "collect", side_effect=live) as cvm_b3, \
                mock.patch.object(b3_trading, "filter_recent_trades", side_effect=live) as b3:
            stored = self.execute(run)

        self.assertEqual(stored.status, "completed", stored.failure_code)
        for collector in (stocks, fiis, cvm_b3, b3):
            collector.assert_not_called()

    def test_provenance_records_the_verified_hashes_in_todays_shape(self):
        stock_sha = hashlib.sha256((self.snapshot_root / "stocks.csv").read_bytes()).hexdigest()
        fii_sha = hashlib.sha256((self.snapshot_root / "fiis.csv").read_bytes()).hexdigest()
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        run = self.queue_run()

        stored = self.execute(run)

        self.assertEqual(stored.status, "completed", stored.failure_code)
        provenance = stored.result_json["provenance"]
        self.assertEqual(
            provenance["selector_sources"],
            {
                "stocks": {
                    "provider": "cvm-b3",
                    "retrieved_at": "2026-10-30T21:00:00+00:00",
                    "sha256": stock_sha,
                },
                "fiis": {
                    "provider": "statusinvest",
                    "retrieved_at": "2026-10-30T21:00:00+00:00",
                    "sha256": fii_sha,
                },
            },
        )
        self.assertEqual(
            provenance["allocation_history"]["source_snapshot_hashes"],
            {
                "allocation": manifest["allocation_source"]["sha256"],
                "stock": stock_sha,
                "fii": fii_sha,
            },
        )

    def test_same_seed_on_the_same_snapshot_gives_the_same_portfolio(self):
        first = self.execute(self.queue_run(seed=97))
        second = self.execute(self.queue_run(seed=97))

        self.assertEqual(first.status, "completed", first.failure_code)
        self.assertEqual(second.status, "completed", second.failure_code)
        self.assertNotEqual(first.id, second.id)
        for key in ("classes", "stocks", "fiis"):
            self.assertEqual(first.result_json[key], second.result_json[key])


if __name__ == "__main__":
    unittest.main()
