import copy
import hashlib
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.adapters.premium_optimization import run_premium_optimization
from app.db.base import Base
from app.db.models import Account, ProfileRecord, RecommendationRun
from app.services.premium_executor import PremiumExecutor
from app.services.premium_recommendation import PremiumRecommendationService
from app.services.status_invest_inputs import SourceSnapshot, StatusInvestInputs
from app.repositories.recommendations import RecommendationRepository
from app.routers.recommendations import read_recommendation
from app.schemas.recommendation import PremiumRecommendationRequest
from core.protection_zone import select_with_fallback


class Manifest:
    manifest_id = "manifest-integration"
    manifest_path = None

    def source_for(self, key):
        return SimpleNamespace(resolved_path=lambda: Path(f"/tmp/{key}.csv"))

    def as_dict(self):
        return {
            "manifest_id": self.manifest_id,
            "manifest_version": "1",
            "created_at": "2026-07-21T00:00:00+00:00",
            "cutoff_date": "2026-07-21",
            "common_dates": ["2026-07-20", "2026-07-21"],
            "supported_classes": [
                "brazilian_stocks",
                "fiis",
                "international_equity",
                "fixed_income",
                "crypto",
            ],
            "sources": {},
        }


def _default_selector_inputs():
    return StatusInvestInputs(
        stocks=SourceSnapshot(
            Path("/tmp/run/status-invest/stocks.csv"),
            "statusinvest",
            "2026-07-21T00:00:00Z",
            "stock-hash",
        ),
        fiis=SourceSnapshot(
            Path("/tmp/run/status-invest/fiis.csv"),
            "statusinvest",
            "2026-07-21T00:00:00Z",
            "fii-hash",
        ),
    )


CHEAP_PRICES = {
    "stocks": {**{f"S{index:02d}": "1.00" for index in range(20)}, "AAA3": "1.00", "BBB3": "1.00"},
    "fiis": {**{f"F{index:02d}": "1.00" for index in range(20)}, "AAA11": "1.00", "BBB11": "1.00"},
}


def priced_selector_inputs(workspace, prices=CHEAP_PRICES):
    """Selector CSVs with a PRECO per ticker; the hashes come from the bytes written."""

    source_root = Path(workspace) / "status-invest"
    source_root.mkdir(parents=True, exist_ok=True)
    snapshots = {}
    for role in ("stocks", "fiis"):
        path = source_root / f"{role}.csv"
        rows = "".join(f"{ticker},{price}\n" for ticker, price in prices[role].items())
        path.write_text(f"TICKER,PRECO\n{rows}", encoding="utf-8")
        snapshots[role] = SourceSnapshot(
            path=path.resolve(),
            provider="statusinvest",
            retrieved_at="2026-07-21T00:00:00Z",
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    return StatusInvestInputs(stocks=snapshots["stocks"], fiis=snapshots["fiis"])


def policy(profile, **kwargs):
    return {
        "profile": {"profile_revision": profile.version},
        "allocation": {
            "volatility_cap": 0.2,
            "drawdown_cap": 0.3,
            "crypto_risk_contribution_cap": 0.5,
            "hhi_penalty": 0.1,
            "risk_adjusted_weights": {"return": 0.7},
            "class_constraints": {
                "minimum_class_weight": 0.0,
                "minimum_weights": {},
                "maximum_weights": {},
                "hhi_max": None,
            },
        },
        "stocks": {
            "selection_preset": "fixture",
            "n_assets": 2,
            "factor_weights": {"liquidez": 1.0},
            "liquidity_and_size_filters": {},
            "lambda_hhi": 0.2,
            "system_ga_config": {"population": 4, "generations": 2, "run_count": 1},
        },
        "fiis": {
            "selection_preset": "fixture",
            "n_assets": 2,
            "factor_weights": {"liquidity": 1.0},
            "liquidity_and_size_filters": {},
            "lambda_hhi": 0.2,
            "system_ga_config": {"population": 4, "generations": 2, "run_count": 1},
        },
        "provenance": {
            "policy_version": kwargs["rules_version"],
            "model_versions": {"allocation": "fixture", "stocks": "fixture", "fiis": "fixture"},
            "source_snapshot_ids": list(kwargs["source_snapshot_ids"]),
            "source_snapshot_hashes": {},
            "cutoff_date": kwargs["cutoff_date"],
            "random_seed": kwargs["random_seed"],
        },
    }


class PremiumIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.engine = create_engine(f"sqlite:///{Path(self.temp.name) / 'integration.sqlite'}")
        Base.metadata.create_all(self.engine)
        self.sessions = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)
        self.session = self.sessions()
        self.account = Account(email="integration@example.com")
        self.session.add(self.account)
        self.session.flush()
        self.profile = ProfileRecord(
            account_id=self.account.id,
            version=1,
            answers={"restricoes": ["nenhuma"]},
            dimensions={"apetite": 0.5},
            suitability_score=0.5,
            generic_profile="moderado",
            investable_capital_brl=100_000,
            consented_at=datetime.now(timezone.utc),
        )
        self.session.add(self.profile)
        self.session.commit()

    def tearDown(self):
        self.session.close()
        self.engine.dispose()
        self.temp.cleanup()

    def runner(self, policy_value, manifest, workspace, capital, *, selector_inputs=None):
        def allocation(*_args, **_kwargs):
            return {
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

        selector = lambda **_kwargs: {
            "selected_tickers": ["AAA3", "BBB3"],
            "sleeve_weights": {"AAA3": 0.5, "BBB3": 0.5},
        }
        fii = lambda **_kwargs: {
            "selected_tickers": ["AAA11", "BBB11"],
            "sleeve_weights": {"AAA11": 0.5, "BBB11": 0.5},
        }
        return run_premium_optimization(
            policy_value,
            manifest,
            workspace,
            capital,
            selector_inputs=selector_inputs or _default_selector_inputs(),
            snapshot_loader=lambda _manifest: SimpleNamespace(rows=(), metadata={}),
            allocation_engine=allocation,
            stock_engine=selector,
            fii_engine=fii,
        )

    def test_service_executor_and_replay_store_complete_deterministic_payload(self):
        manifest = Manifest()
        executor = PremiumExecutor(
            session_factory=self.sessions,
            manifest_loader=lambda _provenance: manifest,
            manifest_validator=lambda value: value,
            status_invest_inputs_loader=lambda _workspace: _default_selector_inputs(),
            optimization_runner=self.runner,
            workspace_root=Path(self.temp.name) / "workspaces",
        )
        service = PremiumRecommendationService(
            manifest_loader=lambda: manifest,
            manifest_validator=lambda value: value,
            policy_resolver=policy,
            executor=executor,
        )
        try:
            run = service.create_run(
                self.account,
                PremiumRecommendationRequest(),
                self.session,
            )
            executor.futures[run.id].result(timeout=10)
            with self.sessions() as check:
                stored = check.get(type(run), run.id)
                self.assertEqual(stored.status, "completed")
                self.assertEqual(sum(item["target_amount_brl"] for item in stored.result_json["classes"]), 100_000)
                self.assertEqual(stored.result_json["provenance"]["random_seed"], run.provenance_json["random_seed"])
                first_result = stored.result_json
                stored_policy = stored.policy_json
            replay = self.runner(
                stored_policy,
                manifest,
                Path(self.temp.name) / "replay",
                100_000,
            )
            self.assertEqual(first_result["classes"], replay["classes"])
            self.assertEqual(first_result["stocks"], replay["stocks"])
            self.assertEqual(first_result["fiis"], replay["fiis"])
        finally:
            executor.shutdown()

    def test_active_run_is_reused_and_force_creates_after_terminal(self):
        manifest = Manifest()

        class Executor:
            def __init__(self):
                self.submitted = []

            def submit(self, run_id):
                self.submitted.append(run_id)

        executor = Executor()
        service = PremiumRecommendationService(
            manifest_loader=lambda: manifest,
            manifest_validator=lambda value: value,
            policy_resolver=policy,
            executor=executor,
        )

        first = service.create_run(
            self.account,
            PremiumRecommendationRequest(),
            self.session,
        )
        reused = service.create_run(
            self.account,
            PremiumRecommendationRequest(),
            self.session,
        )
        self.assertEqual(reused.id, first.id)
        self.assertEqual(executor.submitted, [first.id])

        RecommendationRepository(self.session).mark_failed(
            first.id,
            "fixture_failure",
            "fixture terminal state",
        )
        self.session.commit()

        forced = service.create_run(
            self.account,
            PremiumRecommendationRequest(force=True),
            self.session,
        )
        self.assertNotEqual(forced.id, first.id)
        self.assertEqual(executor.submitted, [first.id, forced.id])


def zone_selector(prefix, frequencies_for):
    """Fake selector: applies the real core zone when it receives incumbents."""

    def engine(**kwargs):
        config = kwargs.get("config", kwargs.get("selection_config"))
        n_assets = int(config["n_assets"])
        frequencies = frequencies_for(prefix, n_assets, bool(kwargs.get("incumbents")))
        selection, record = select_with_fallback(
            kwargs.get("incumbents", []),
            frequencies,
            {ticker: 0.5 for ticker in frequencies},
            n_assets,
            config.get("protection_zone") or {"K": 0, "delta": 0.0},
            eligible=set(frequencies),
            dedup_base_company=False,
            hhi_of=lambda _tickers: 0.0,
            hhi_max=None,
        )
        tickers = list(selection.tickers)
        result = {
            "selected_tickers": tickers,
            "sleeve_weights": {ticker: 1 / len(tickers) for ticker in tickers},
        }
        if kwargs.get("incumbents"):
            result["protection_zone"] = record
        return result

    return engine


class ProtectionZoneIntegrationTests(unittest.TestCase):
    """PZ-24, PZ-28: a second v3 Premium run treats the first as incumbent."""

    tearDown = PremiumIntegrationTests.tearDown

    def setUp(self):
        PremiumIntegrationTests.setUp(self)
        self.profile.dimensions = {
            "apetite": 0.5,
            "capacidade": 0.5,
            "liquidez": 0.5,
            "conhecimento": 0.5,
        }
        self.profile.restrictions_json = ["nenhuma"]
        self.session.commit()

    @staticmethod
    def frequencies_for(prefix, n_assets, second_snapshot):
        # n_assets + 3 candidates; the second snapshot reverses the ranking.
        tickers = [f"{prefix}{index:02d}" for index in range(n_assets + 3)]
        if second_snapshot:
            tickers.reverse()
        return {ticker: round(0.95 - 0.05 * index, 2) for index, ticker in enumerate(tickers)}

    def zone_runner(
        self, policy_value, manifest, workspace, capital, *, selector_inputs=None, incumbents=None, whole_units=False
    ):
        kwargs = {} if incumbents is None else {"incumbents": incumbents}
        return run_premium_optimization(
            policy_value,
            manifest,
            workspace,
            capital,
            selector_inputs=selector_inputs or _default_selector_inputs(),
            snapshot_loader=lambda _manifest: SimpleNamespace(rows=(), metadata={}),
            allocation_engine=lambda *_args, **_kwargs: {
                "weights": {
                    "brazilian_stocks": 0.4,
                    "fiis": 0.2,
                    "international_equity": 0.1,
                    "fixed_income": 0.2,
                    "crypto": 0.1,
                }
            },
            stock_engine=zone_selector("S", self.frequencies_for),
            fii_engine=zone_selector("F", self.frequencies_for),
            whole_units=whole_units,
            **kwargs,
        )

    def services(self):
        manifest = Manifest()
        executor = PremiumExecutor(
            session_factory=self.sessions,
            manifest_loader=lambda _provenance: manifest,
            manifest_validator=lambda value: value,
            status_invest_inputs_loader=priced_selector_inputs,
            optimization_runner=self.zone_runner,
            workspace_root=Path(self.temp.name) / "workspaces",
        )
        service = PremiumRecommendationService(
            manifest_loader=lambda: manifest,
            manifest_validator=lambda value: value,
            executor=executor,
        )
        return executor, service

    def create_and_wait(self, executor, service, *, plan="premium", force=False):
        run = service.create_run(
            self.account,
            PremiumRecommendationRequest(force=force),
            self.session,
            plan=plan,
        )
        executor.futures[run.id].result(timeout=10)
        with self.sessions() as check:
            stored = check.get(type(run), run.id)
            self.assertEqual(stored.status, "completed")
            return stored

    def test_second_premium_run_records_the_first_as_incumbent(self):
        executor, service = self.services()
        try:
            first = self.create_and_wait(executor, service)
            second = self.create_and_wait(executor, service, force=True)
        finally:
            executor.shutdown()

        self.assertNotEqual(second.id, first.id)
        self.assertEqual(first.policy_version, "premium-policy-v4")
        self.assertEqual(second.policy_version, "premium-policy-v4")
        first_zone = first.result_json["provenance"]["protection_zone"]
        self.assertIsNone(first_zone["incumbent_run_id"])
        self.assertEqual(first_zone["stocks"]["status"], "no_incumbents")
        self.assertEqual(first_zone["fiis"]["status"], "no_incumbents")

        zone = second.result_json["provenance"]["protection_zone"]
        self.assertEqual(zone["incumbent_run_id"], first.id)
        for key in ("stocks", "fiis"):
            record = zone[key]
            first_tickers = sorted(item["ticker"] for item in first.result_json[key])
            second_tickers = sorted(item["ticker"] for item in second.result_json[key])
            self.assertEqual(record["status"], "applied")
            self.assertEqual(record["incumbents"], first_tickers)
            self.assertIsInstance(record["turnover"], int)
            self.assertIsInstance(record["exits"], int)
            self.assertEqual(sorted(record["kept"] + record["entered"]), second_tickers)
            self.assertEqual(record["turnover"], len(set(second_tickers) - set(first_tickers)))
            self.assertGreater(record["turnover"], 0)
            swapped_in = {item["in"]: item["out"] for item in record["replaced"]}
            reasons = {item["ticker"]: item["reasons"] for item in second.result_json[key]}
            for ticker in second_tickers:
                if ticker in swapped_in:
                    expected = f"substituiu {swapped_in[ticker]}"
                else:
                    expected = "mantido" if ticker in first_tickers else "novo"
                self.assertEqual(reasons[ticker], [expected])

    def test_basic_run_after_a_premium_run_has_no_protection_zone(self):
        executor, service = self.services()
        try:
            self.create_and_wait(executor, service)
            basic = self.create_and_wait(executor, service, plan="basic")
        finally:
            executor.shutdown()

        self.assertEqual(basic.policy_version, "basic-policy-v2")
        self.assertNotIn("protection_zone", basic.result_json["provenance"])
        self.assertNotIn("protection_zone", basic.result_json["policy"]["stocks"])
        self.assertNotIn("protection_zone", basic.result_json["policy"]["fiis"])
        self.assertNotIn("protection_zone", basic.policy_json["stocks"])
        self.assertTrue(
            all(item["reasons"] == [] for item in basic.result_json["stocks"] + basic.result_json["fiis"])
        )



PRE_CHANGE_SELECTOR_CONFIG_KEYS = {
    "selection_preset",
    "n_assets",
    "factor_weights",
    "liquidity_and_size_filters",
    "lambda_hhi",
    "system_ga_config",
    "protection_zone",
}


class SegmentCapIntegrationTests(unittest.TestCase):
    """SEG-32, SEG-33: a v4 run persists and forwards the FII segment cap; a
    run whose stored policy lacks it (queued under v3) executes uncapped."""

    setUp = ProtectionZoneIntegrationTests.setUp
    tearDown = PremiumIntegrationTests.tearDown
    frequencies_for = staticmethod(ProtectionZoneIntegrationTests.frequencies_for)

    def recording_runner(
        self, policy_value, manifest, workspace, capital, *, selector_inputs=None, incumbents=None, whole_units=False
    ):
        def recording(key, engine):
            def wrapped(**kwargs):
                self.engine_calls.setdefault(key, []).append(kwargs)
                return engine(**kwargs)

            return wrapped

        kwargs = {} if incumbents is None else {"incumbents": incumbents}
        return run_premium_optimization(
            policy_value,
            manifest,
            workspace,
            capital,
            selector_inputs=selector_inputs or _default_selector_inputs(),
            snapshot_loader=lambda _manifest: SimpleNamespace(rows=(), metadata={}),
            allocation_engine=lambda *_args, **_kwargs: {
                "weights": {
                    "brazilian_stocks": 0.4,
                    "fiis": 0.2,
                    "international_equity": 0.1,
                    "fixed_income": 0.2,
                    "crypto": 0.1,
                }
            },
            stock_engine=recording("stocks", zone_selector("S", self.frequencies_for)),
            fii_engine=recording("fiis", zone_selector("F", self.frequencies_for)),
            whole_units=whole_units,
            **kwargs,
        )

    def executor(self, manifest):
        self.engine_calls = {}
        return PremiumExecutor(
            session_factory=self.sessions,
            manifest_loader=lambda _provenance: manifest,
            manifest_validator=lambda value: value,
            status_invest_inputs_loader=priced_selector_inputs,
            optimization_runner=self.recording_runner,
            workspace_root=Path(self.temp.name) / "workspaces",
        )

    def stored(self, run_id):
        with self.sessions() as check:
            return check.get(RecommendationRun, run_id)

    def test_v4_run_records_and_forwards_the_cap(self):
        manifest = Manifest()
        executor = self.executor(manifest)
        service = PremiumRecommendationService(
            manifest_loader=lambda: manifest,
            manifest_validator=lambda value: value,
            executor=executor,
        )
        try:
            run = service.create_run(self.account, PremiumRecommendationRequest(), self.session)
            executor.futures[run.id].result(timeout=10)
        finally:
            executor.shutdown()

        stored = self.stored(run.id)
        self.assertEqual(stored.status, "completed")
        self.assertEqual(stored.policy_version, "premium-policy-v4")
        self.assertEqual(stored.policy_json["provenance"]["policy_version"], "premium-policy-v4")
        self.assertEqual(stored.policy_json["fiis"]["max_per_segment"], 2)
        self.assertNotIn("max_per_segment", stored.policy_json["stocks"])
        self.assertEqual(len(self.engine_calls["fiis"]), 1)
        self.assertEqual(self.engine_calls["fiis"][0]["selection_config"]["max_per_segment"], 2)
        self.assertNotIn("max_per_segment", self.engine_calls["stocks"][0]["config"])

    def test_stored_v3_policy_runs_without_the_cap(self):
        manifest = Manifest()

        class QueueOnly:
            def __init__(self):
                self.submitted = []

            def submit(self, run_id):
                self.submitted.append(run_id)

        queue = QueueOnly()
        service = PremiumRecommendationService(
            manifest_loader=lambda: manifest,
            manifest_validator=lambda value: value,
            executor=queue,
        )
        run = service.create_run(self.account, PremiumRecommendationRequest(), self.session)
        self.assertEqual(queue.submitted, [run.id])

        # A run queued before the deploy: its stored policy is v3, without the cap.
        stored_policy = copy.deepcopy(run.policy_json)
        stored_policy["fiis"].pop("max_per_segment")
        stored_policy["provenance"]["policy_version"] = "premium-policy-v3"
        run.policy_json = stored_policy
        run.policy_version = "premium-policy-v3"
        self.session.commit()

        executor = self.executor(manifest)
        try:
            executor.execute_run(run.id)
        finally:
            executor.shutdown()

        stored = self.stored(run.id)
        self.assertEqual(stored.status, "completed")
        self.assertEqual(stored.policy_version, "premium-policy-v3")
        self.assertNotIn("max_per_segment", stored.policy_json["fiis"])
        fii_config = self.engine_calls["fiis"][0]["selection_config"]
        self.assertNotIn("max_per_segment", fii_config)
        self.assertEqual(set(fii_config), PRE_CHANGE_SELECTOR_CONFIG_KEYS)


WHOLE_UNIT_FIIS = ["WUA11", "WUB11", "WUC11", "WUD11"]
WHOLE_UNIT_PRICES = {
    "stocks": {"AAA3": "1.00", "BBB3": "1.00"},
    # WUD11 cannot be bought once the other three FIIs take their share of the class.
    "fiis": {"WUA11": "10.00", "WUB11": "10.00", "WUC11": "10.00", "WUD11": "6000.00"},
}


def fixed_selector(tickers):
    """Fake engine: selects ``tickers`` in this order and applies the core zone to incumbents."""

    frequencies = {ticker: round(0.95 - 0.05 * index, 2) for index, ticker in enumerate(tickers)}

    def engine(**kwargs):
        config = kwargs.get("config", kwargs.get("selection_config"))
        selection, record = select_with_fallback(
            kwargs.get("incumbents", []),
            frequencies,
            {ticker: 0.5 for ticker in tickers},
            len(tickers),
            config.get("protection_zone") or {"K": 0, "delta": 0.0},
            eligible=set(tickers),
            dedup_base_company=False,
            hhi_of=lambda _tickers: 0.0,
            hhi_max=None,
        )
        selected = list(selection.tickers)
        result = {
            "selected_tickers": selected,
            "sleeve_weights": {ticker: 1 / len(selected) for ticker in selected},
        }
        if kwargs.get("incumbents"):
            result["protection_zone"] = record
        return result

    return engine


class WholeUnitIntegrationTests(unittest.TestCase):
    """LOT-18, LOT-20, LOT-21: whole-unit runs through the executor, the store and the route."""

    setUp = ProtectionZoneIntegrationTests.setUp
    tearDown = PremiumIntegrationTests.tearDown
    create_and_wait = ProtectionZoneIntegrationTests.create_and_wait

    def runner(
        self, policy_value, manifest, workspace, capital, *, selector_inputs=None, incumbents=None, whole_units=False
    ):
        def recording(key, engine):
            def wrapped(**kwargs):
                self.engine_calls.setdefault(key, []).append(kwargs)
                return engine(**kwargs)

            return wrapped

        kwargs = {} if incumbents is None else {"incumbents": incumbents}
        return run_premium_optimization(
            policy_value,
            manifest,
            workspace,
            capital,
            selector_inputs=selector_inputs,
            snapshot_loader=lambda _manifest: SimpleNamespace(rows=(), metadata={}),
            allocation_engine=lambda *_args, **_kwargs: {
                "weights": {
                    "brazilian_stocks": 0.4,
                    "fiis": 0.2,
                    "international_equity": 0.1,
                    "fixed_income": 0.2,
                    "crypto": 0.1,
                }
            },
            stock_engine=recording("stocks", fixed_selector(["AAA3", "BBB3"])),
            fii_engine=recording("fiis", fixed_selector(WHOLE_UNIT_FIIS)),
            whole_units=whole_units,
            **kwargs,
        )

    def services(self):
        self.engine_calls = {}
        manifest = Manifest()
        executor = PremiumExecutor(
            session_factory=self.sessions,
            manifest_loader=lambda _provenance: manifest,
            manifest_validator=lambda value: value,
            status_invest_inputs_loader=lambda workspace: priced_selector_inputs(workspace, WHOLE_UNIT_PRICES),
            optimization_runner=self.runner,
            workspace_root=Path(self.temp.name) / "workspaces",
        )
        service = PremiumRecommendationService(
            manifest_loader=lambda: manifest,
            manifest_validator=lambda value: value,
            executor=executor,
        )
        return executor, service

    def test_ticker_dropped_in_run_one_is_not_an_incumbent_in_run_two(self):
        executor, service = self.services()
        try:
            first = self.create_and_wait(executor, service)
            second = self.create_and_wait(executor, service, force=True)
        finally:
            executor.shutdown()

        self.assertNotEqual(second.id, first.id)
        self.assertNotIn("incumbents", self.engine_calls["fiis"][0])
        self.assertEqual(self.engine_calls["fiis"][1]["incumbents"], ["WUA11", "WUB11", "WUC11"])

        self.session.expire_all()
        response = read_recommendation(first.id, self.account, self.session)
        self.assertEqual(
            {item.ticker: item.quantity for item in response.fiis},
            {"WUA11": 667, "WUB11": 667, "WUC11": 666},
        )
        self.assertEqual([item.price_brl for item in response.fiis], [10.0, 10.0, 10.0])
        self.assertEqual([item.ticker for item in response.fiis_dropped], ["WUD11"])
        self.assertEqual(response.fiis_dropped[0].reasons[-1], "não coube no capital")
        fii_class = next(item for item in first.result_json["classes"] if item["key"] == "fiis")
        self.assertEqual(fii_class["cash_residual_brl"], 0.0)
        self.assertEqual(fii_class["over_target_brl"], 0.0)

    def test_stored_pre_change_result_reads_and_feeds_incumbents(self):
        class QueueOnly:
            def submit(self, _run_id):
                return None

        manifest = Manifest()
        queued = PremiumRecommendationService(
            manifest_loader=lambda: manifest,
            manifest_validator=lambda value: value,
            executor=QueueOnly(),
        )
        run = queued.create_run(self.account, PremiumRecommendationRequest(), self.session)
        # Queued before the deploy: the stored policy does not declare whole units.
        stored_policy = copy.deepcopy(run.policy_json)
        del stored_policy["provenance"]["model_versions"]["whole_units"]
        run.policy_json = stored_policy
        self.session.commit()

        executor, service = self.services()
        try:
            executor.execute_run(run.id)
            self.create_and_wait(executor, service, force=True)
        finally:
            executor.shutdown()

        with self.sessions() as check:
            first = check.get(RecommendationRun, run.id)
            self.assertEqual(first.status, "completed")
            self.assertNotIn("quantity", first.result_json["fiis"][0])
            self.assertNotIn("fiis_dropped", first.result_json)
            stored_tickers = sorted(item["ticker"] for item in first.result_json["fiis"])

        self.assertEqual(stored_tickers, sorted(WHOLE_UNIT_FIIS))
        self.assertEqual(sorted(self.engine_calls["fiis"][1]["incumbents"]), stored_tickers)
        self.session.expire_all()
        response = read_recommendation(run.id, self.account, self.session)
        self.assertIsNone(response.fiis[0].quantity)
        self.assertEqual(response.fiis_dropped, [])


if __name__ == "__main__":
    unittest.main()
