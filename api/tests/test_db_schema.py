import unittest
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session

from app.db.base import Base
from app.db.models import Account, Entitlement, ProfileRecord


class DatabaseSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)

    def tearDown(self) -> None:
        Base.metadata.drop_all(self.engine)
        self.engine.dispose()

    def test_product_tables_exist_with_account_foreign_keys(self):
        table_names = set(inspect(self.engine).get_table_names())
        self.assertEqual(
            table_names,
            {"accounts", "profiles", "recommendation_runs", "portfolio_snapshots", "entitlements"},
        )
        for table_name in table_names - {"accounts"}:
            foreign_keys = inspect(self.engine).get_foreign_keys(table_name)
            self.assertTrue(any(fk["referred_table"] == "accounts" for fk in foreign_keys))

    def test_profile_and_entitlement_are_account_owned(self):
        with Session(self.engine) as session:
            account = Account(email="user@example.com")
            session.add(account)
            session.flush()

            profile = ProfileRecord(
                account_id=account.id,
                version=1,
                answers={"objetivo": "crescimento"},
                dimensions={"apetite": 0.8},
                suitability_score=Decimal("0.80000"),
                generic_profile="arrojado",
                investable_capital_brl=Decimal("1000.00"),
                consented_at=datetime.now(timezone.utc),
            )
            entitlement = Entitlement(account_id=account.id)
            session.add_all([profile, entitlement])
            session.commit()

            self.assertEqual(profile.account_id, account.id)
            self.assertEqual(entitlement.account_id, account.id)
            self.assertEqual(entitlement.plan, "basic")
            self.assertEqual(entitlement.status, "inactive")


if __name__ == "__main__":
    unittest.main()
