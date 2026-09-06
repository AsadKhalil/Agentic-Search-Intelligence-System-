"""ASGI entrypoint: uvicorn app.main:app --reload"""
from __future__ import annotations

from fastapi import FastAPI

from app.api import router
from app.config import get_settings
from app.db import init_db
from app.observability.logging import configure_logging

settings = get_settings()
configure_logging(settings.log_level, settings.log_format)
init_db()

app = FastAPI(
    title="Agentic Search Intelligence System",
    version="1.0.0",
    description="LangGraph DAG over DataForSEO for search-visibility research.",
)
app.include_router(router)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "mock_dataforseo": settings.mock_dataforseo,
        "serp_transport": "serpapi" if settings.serpapi_api_key else (
            "mock fixtures" if settings.mock_dataforseo else "dataforseo"),
        "volume_and_chatgpt_transport": (
            "mock fixtures" if settings.mock_dataforseo else "dataforseo"),
        "llm_mode": "openai" if settings.openai_api_key else "scripted",
    }
