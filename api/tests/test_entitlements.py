import unittest

from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.base import Base
from app.db.models import Account, Entitlement
from app.config import AppSettings
from app.entitlements.dependencies import require_premium, require_premium_pilot
from app.routers.premium import read_premium_access


class EntitlementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.session = Session(self.engine)
        self.account = Account(email="user@example.com")
        self.session.add(self.account)
        self.session.flush()

    def tearDown(self) -> None:
        self.session.close()
        Base.metadata.drop_all(self.engine)
        self.engine.dispose()

    def test_basic_account_cannot_access_premium_endpoint(self) -> None:
        with self.assertRaises(HTTPException) as error:
            require_premium(self.account, self.session)

        self.assertEqual(error.exception.status_code, 403)
        self.assertEqual(error.exception.detail["code"], "premium_required")

    def test_active_premium_entitlement_grants_access(self) -> None:
        entitlement = Entitlement(
            account_id=self.account.id,
            plan="premium",
            status="active",
        )
        self.session.add(entitlement)
        self.session.commit()

        granted = require_premium(self.account, self.session)
        response = read_premium_access(granted)

        self.assertEqual(response.access, "granted")
        self.assertEqual(response.entitlement_status, "active")

    def test_grace_period_keeps_premium_access(self) -> None:
        entitlement = Entitlement(
            account_id=self.account.id,
            plan="premium",
            status="grace_period",
        )
        self.session.add(entitlement)
        self.session.commit()

        self.assertEqual(require_premium(self.account, self.session).status, "grace_period")

    def test_premium_account_outside_pilot_allowlist_is_rejected(self) -> None:
        self.session.add(
            Entitlement(account_id=self.account.id, plan="premium", status="active")
        )
        self.session.commit()

        with self.assertRaises(HTTPException) as error:
            require_premium_pilot(
                self.account,
                self.session,
                AppSettings("development", frozenset()),
            )

        self.assertEqual(error.exception.status_code, 403)
        self.assertEqual(error.exception.detail["code"], "premium_pilot_required")

    def test_premium_allowlist_grants_active_and_grace_access(self) -> None:
        entitlement = Entitlement(
            account_id=self.account.id,
            plan="premium",
            status="grace_period",
        )
        self.session.add(entitlement)
        self.session.commit()

        granted = require_premium_pilot(
            self.account,
            self.session,
            AppSettings("production", frozenset({self.account.id})),
        )

        self.assertEqual(granted.status, "grace_period")

    def test_empty_production_allowlist_fails_closed_before_pilot_grant(self) -> None:
        self.session.add(
            Entitlement(account_id=self.account.id, plan="premium", status="active")
        )
        self.session.commit()

        with self.assertRaises(HTTPException) as error:
            require_premium_pilot(
                self.account,
                self.session,
                AppSettings("production", frozenset()),
            )

        self.assertEqual(error.exception.detail["code"], "premium_pilot_required")


if __name__ == "__main__":
    unittest.main()
