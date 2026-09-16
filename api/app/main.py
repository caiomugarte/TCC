import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.account import router as account_router
from app.routers.health import router as health_router
from app.routers.portfolio import router as portfolio_router
from app.routers.premium import router as premium_router
from app.routers.profile import router as profile_router
from app.routers.recommendations import router as recommendations_router
from app.routers.review import router as review_router
from app.services.premium_executor import get_default_executor

app = FastAPI(title="Prumo API", version="0.1.0")
cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)
app.include_router(health_router)
app.include_router(account_router)
app.include_router(profile_router)
app.include_router(recommendations_router)
app.include_router(portfolio_router)
app.include_router(premium_router)
app.include_router(review_router)


@app.on_event("startup")
def recover_premium_runs() -> None:
    get_default_executor().recover_stale()


@app.on_event("shutdown")
def stop_premium_executor() -> None:
    get_default_executor().shutdown()
