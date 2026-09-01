from typing import Literal

from pydantic import BaseModel


class PremiumAccessResponse(BaseModel):
    feature: Literal["premium_access"]
    access: Literal["granted"]
    plan: Literal["premium"]
    entitlement_status: Literal["active", "grace_period"]
