# Prumo API

FastAPI boundary for the SaaS shell. Clerk verifies browser sessions; the API
maps verified Clerk `sub` claims to local account records.

From repository root:

```bash
cp api/.env.example api/.env
PYTHONPATH=api uvicorn app.main:app --reload --env-file api/.env
```

Health check: `GET http://127.0.0.1:8000/health`.

Set `CLERK_SECRET_KEY` (or `CLERK_JWT_KEY`) and
`CLERK_AUTHORIZED_PARTIES` in `api/.env` before calling protected endpoints.
