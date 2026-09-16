from __future__ import annotations

import os
from dataclasses import dataclass


def _account_ids(value: str | None) -> frozenset[str]:
    return frozenset(
        item.strip()
        for item in (value or "").split(",")
        if item.strip()
    )


@dataclass(frozen=True)
class AppSettings:
    environment: str
    premium_pilot_account_ids: frozenset[str]

    @classmethod
    def from_env(cls) -> AppSettings:
        return cls(
            environment=os.getenv("APP_ENV", "development").strip().lower(),
            premium_pilot_account_ids=_account_ids(
                os.getenv("PREMIUM_PILOT_ACCOUNT_IDS")
            ),
        )


Settings = AppSettings


def get_settings() -> AppSettings:
    """Load settings per dependency resolution so tests do not leak env state."""

    return AppSettings.from_env()
