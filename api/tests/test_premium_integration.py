import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.adapters.premium_optimization import run_premium_optimization
from app.db.base import Base
from app.db.models import Account, ProfileRecord
from app.services.premium_executor import PremiumExecutor
from app.services.premium_recommendation import PremiumRecommendationService
from app.schemas.recommendation import PremiumRecommendationRequest


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

    def runner(self, policy_value, manifest, workspace, capital):
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


if __name__ == "__main__":
    unittest.main()
