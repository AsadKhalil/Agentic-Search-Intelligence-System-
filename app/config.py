"""Central configuration. Everything tunable lives here and is documented in .env.example."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- DataForSEO -------------------------------------------------------
    # When true (the default) every DataForSEO call is served from local
    # fixtures instead of the network. See README "Which mode am I running in".
    mock_dataforseo: bool = True
    dataforseo_login: str = ""
    dataforseo_password: str = ""
    dataforseo_base_url: str = "https://api.dataforseo.com"

    # Probability that the mock transport injects a transient 5xx, so the
    # retry/fallback paths can be exercised without touching the network.
    mock_failure_rate: float = 0.0
    mock_latency_ms: int = 0

    # --- LLM --------------------------------------------------------------
    # Empty api key => deterministic FakeLLM, so the graph runs offline.
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_timeout_seconds: float = 45.0

    # --- Resilience -------------------------------------------------------
    http_timeout_seconds: float = 20.0
    retry_max_attempts: int = 4
    retry_base_delay_seconds: float = 0.5
    retry_max_delay_seconds: float = 8.0
    retry_jitter: bool = True

    circuit_breaker_enabled: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_reset_seconds: float = 30.0

    # --- App --------------------------------------------------------------
    database_url: str = "sqlite:///./agentic_search.db"
    log_level: str = "INFO"
    log_format: str = "json"  # "json" | "console"
    max_planned_queries: int = 8


@lru_cache
def get_settings() -> Settings:
    return Settings()
