"""Test setup. Everything runs offline: mock DataForSEO transport + ScriptedToolCallingLLM.

Environment is set before any app import because app.db builds its engine at import time.
"""
import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="agentic-search-tests-")
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP}/test.db"
os.environ["MOCK_DATAFORSEO"] = "true"
os.environ["OPENAI_API_KEY"] = ""
os.environ["RETRY_BASE_DELAY_SECONDS"] = "0.001"
os.environ["RETRY_MAX_DELAY_SECONDS"] = "0.004"
os.environ["RETRY_MAX_ATTEMPTS"] = "3"
os.environ["LOG_LEVEL"] = "CRITICAL"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.db import engine  # noqa: E402
from app.graph.build import build_graph  # noqa: E402
from app.graph.state import initial_state  # noqa: E402
from app.llm import ScriptedToolCallingLLM  # noqa: E402
from app.models import Base  # noqa: E402
from app.schemas import ProfileSnapshot  # noqa: E402
from app.tools.dataforseo import DataForSEOClient  # noqa: E402
from app.tools.mock import MockBackend  # noqa: E402

DOMAIN = "acme.io"
QUESTION = "What are the best project management tools for remote teams?"


class PlannerFailsLLM(ScriptedToolCallingLLM):
    """Raises only when tools are bound, so the planner fails while the analyzer works."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.bound_tools:
            raise TimeoutError("planner request timed out")
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


@pytest.fixture
def settings():
    get_settings.cache_clear()
    return get_settings()


@pytest.fixture
def db():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)


@pytest.fixture
def backend():
    return MockBackend(domain_hint=DOMAIN)


@pytest.fixture
def profile():
    return ProfileSnapshot(uuid="profile-1", name="Acme", domain=DOMAIN,
                           industry="project management", competitors=["asana.com"])


@pytest.fixture
def make_run(settings, profile):
    """Build a graph over a given backend/LLM and run it; returns the final state."""

    def _run(backend=None, llm=None, question=QUESTION, prof=None):
        prof = prof or profile
        graph = build_graph(
            llm=llm or ScriptedToolCallingLLM(),
            client=DataForSEOClient(settings, backend=backend or MockBackend(domain_hint=DOMAIN)),
            settings=settings,
        )
        return graph.invoke(initial_state(profile=prof, question=question,
                                          correlation_id="test-correlation"))

    return _run


@pytest.fixture
def api(db, settings, backend):
    from app.api import Runner, get_runner
    from app.main import app

    app.dependency_overrides[get_runner] = lambda: Runner(
        llm=ScriptedToolCallingLLM(),
        client=DataForSEOClient(settings, backend=backend),
        settings=settings,
    )
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


@pytest.fixture
def created_profile(api):
    response = api.post("/api/v1/profiles", json={
        "name": "Acme", "domain": DOMAIN, "industry": "project management",
        "description": "Remote-first project management.", "competitors": ["asana.com"],
    })
    assert response.status_code == 201
    return response.json()
