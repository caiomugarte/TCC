"""Dev-only: grant an active Premium entitlement and pilot access to an account.

Usage (from api/):
    ../.venv/bin/python scripts/grant_premium.py --list
    ../.venv/bin/python scripts/grant_premium.py --latest
    ../.venv/bin/python scripts/grant_premium.py --account-id <uuid>

The pilot list is read from api/.env when the API starts, so restart the API
after granting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_ROOT))

from sqlalchemy import desc, select  # noqa: E402

from app.db.models import Account, Entitlement  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402

ENV_FILE = API_ROOT / ".env"
PILOT_KEY = "PREMIUM_PILOT_ACCOUNT_IDS"


def _add_to_pilot(account_id: str) -> bool:
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines() if ENV_FILE.exists() else []
    for index, line in enumerate(lines):
        if line.startswith(f"{PILOT_KEY}="):
            ids = [item.strip() for item in line.split("=", 1)[1].split(",") if item.strip()]
            if account_id in ids:
                return False
            lines[index] = f"{PILOT_KEY}={','.join([*ids, account_id])}"
            break
    else:
        lines.append(f"{PILOT_KEY}={account_id}")
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--account-id")
    target.add_argument("--latest", action="store_true", help="most recently created account")
    target.add_argument("--list", action="store_true")
    args = parser.parse_args()

    with SessionLocal() as session:
        if args.list:
            for account in session.scalars(select(Account).order_by(Account.created_at)):
                entitlement = session.scalar(
                    select(Entitlement).where(Entitlement.account_id == account.id)
                )
                state = f"{entitlement.plan}/{entitlement.status}" if entitlement else "none"
                print(f"{account.id}  {account.created_at:%Y-%m-%d %H:%M}  {account.auth_subject}  {state}")
            return 0

        if args.latest:
            account = session.scalar(select(Account).order_by(desc(Account.created_at)).limit(1))
        else:
            account = session.get(Account, args.account_id)
        if account is None:
            print("account not found (log in once through the web app first)", file=sys.stderr)
            return 1

        entitlement = session.scalar(select(Entitlement).where(Entitlement.account_id == account.id))
        if entitlement is None:
            entitlement = Entitlement(account_id=account.id)
            session.add(entitlement)
        entitlement.plan = "premium"
        entitlement.status = "active"
        entitlement.provider = "manual_dev"
        session.commit()

    added = _add_to_pilot(account.id)
    print(f"premium/active granted to {account.id} ({account.auth_subject})")
    print(f"{PILOT_KEY} {'updated' if added else 'already had it'} in {ENV_FILE}; restart the API")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
