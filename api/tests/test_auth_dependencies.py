import unittest
from unittest.mock import patch

from clerk_backend_api.security.types import AuthStatus, RequestState
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from starlette.requests import Request

from app.auth.dependencies import ClerkIdentity, get_current_account, get_current_identity
from app.db.base import Base


class AuthDependencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.session = Session(self.engine)

    def tearDown(self) -> None:
        self.session.close()
        Base.metadata.drop_all(self.engine)
        self.engine.dispose()

    @staticmethod
    def request() -> Request:
        return Request({"type": "http", "method": "GET", "path": "/", "headers": []})

    @patch(
        "app.auth.dependencies.authenticate_request",
        return_value=RequestState(
            AuthStatus.SIGNED_IN,
            payload={"sub": "user_clerk_123", "email": "user@example.com"},
        ),
    )
    def test_identity_comes_from_verified_clerk_state(self, _authenticate) -> None:
        identity = get_current_identity(self.request())

        self.assertEqual(identity, ClerkIdentity("user_clerk_123", "user@example.com"))

    @patch(
        "app.auth.dependencies.authenticate_request",
        return_value=RequestState(AuthStatus.SIGNED_OUT),
    )
    def test_signed_out_request_is_rejected(self, _authenticate) -> None:
        with self.assertRaises(HTTPException) as error:
            get_current_identity(self.request())

        self.assertEqual(error.exception.status_code, 401)

    def test_first_authenticated_request_provisions_local_account(self) -> None:
        identity = ClerkIdentity("user_clerk_123", "user@example.com")

        account = get_current_account(identity, self.session)
        loaded = get_current_account(identity, self.session)

        self.assertEqual(account.id, loaded.id)
        self.assertEqual(account.auth_provider, "clerk")
        self.assertEqual(account.auth_subject, identity.subject)

    def test_email_is_optional_for_account_provisioning(self) -> None:
        account = get_current_account(ClerkIdentity("user_clerk_123", None), self.session)

        self.assertIsNone(account.email)


if __name__ == "__main__":
    unittest.main()
