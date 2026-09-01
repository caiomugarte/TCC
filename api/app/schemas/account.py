from __future__ import annotations

from typing import Literal

from app.schemas.profile import ProfileSchema


class AccountResponse(ProfileSchema):
    id: str
    email: str | None
    plan: Literal["basic", "premium"]
    entitlement_status: Literal["active", "inactive", "grace_period"]
