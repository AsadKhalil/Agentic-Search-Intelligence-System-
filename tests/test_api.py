"""HTTP contract (PLAN §8, §9.10)."""
import json

from app.observability.logging import JsonFormatter, redact

from tests.conftest import QUESTION


def test_create_and_read_profile(api, created_profile):
    assert created_profile["uuid"]
    read = api.get(f"/api/v1/profiles/{created_profile['uuid']}")
    assert read.status_code == 200
    body = read.json()
    assert body["domain"] == "acme.io"
    assert body["total_runs"] == 0
    assert body["last_run_status"] is None
    assert body["average_opportunity_score"] is None


def test_profile_summary_after_a_run(api, created_profile):
    api.post(f"/api/v1/profiles/{created_profile['uuid']}/run",
             json={"question": QUESTION}).raise_for_status()
    body = api.get(f"/api/v1/profiles/{created_profile['uuid']}").json()
    assert body["total_runs"] == 1
    assert body["last_run_status"] == "completed"
    assert 0 < body["average_opportunity_score"] <= 1


def test_create_profile_rejects_bad_body(api):
    assert api.post("/api/v1/profiles", json={"name": "x"}).status_code == 422
    assert api.post("/api/v1/profiles",
                    json={"name": "x", "domain": "a.io", "nope": 1}).status_code == 422


def test_unknown_profile_is_404(api):
    assert api.get("/api/v1/profiles/does-not-exist").status_code == 404
    assert api.post("/api/v1/profiles/does-not-exist/run",
                    json={"question": QUESTION}).status_code == 404


def test_run_requires_a_question(api, created_profile):
    url = f"/api/v1/profiles/{created_profile['uuid']}/run"
    assert api.post(url).status_code == 422
    assert api.post(url, json={}).status_code == 422
    assert api.post(url, json={"question": ""}).status_code == 422
    assert api.post(url, json={"question": QUESTION}).status_code == 200


def test_run_response_shape(api, created_profile):
    body = api.post(f"/api/v1/profiles/{created_profile['uuid']}/run",
                    json={"question": QUESTION}).json()
    for field in ("run_uuid", "status", "planned_call_count", "extracted_record_count",
                  "insights", "recommendations", "report", "metrics", "correlation_id",
                  "tokens_used", "errors"):
        assert field in body, field
    assert body["status"] == "completed"
    assert body["planned_call_count"] > 0
    assert body["extracted_record_count"] > 0
    assert body["metrics"]["node_sequence"][-1] == "report"
    assert body["report"]["summary"]
    assert body["insights"][0]["opportunity_score"] >= body["insights"][-1][
        "opportunity_score"], "insights are ranked"


def test_query_filters_and_pagination(api, created_profile):
    uuid = created_profile["uuid"]
    api.post(f"/api/v1/profiles/{uuid}/run", json={"question": QUESTION}).raise_for_status()

    every = api.get(f"/api/v1/profiles/{uuid}/queries").json()
    assert every["total"] >= 3
    scores = [q["opportunity_score"] for q in every["items"]]
    assert scores == sorted(scores, reverse=True)

    page = api.get(f"/api/v1/profiles/{uuid}/queries?page=1&per_page=2").json()
    assert len(page["items"]) == 2 and page["total"] == every["total"]
    page_two = api.get(f"/api/v1/profiles/{uuid}/queries?page=2&per_page=2").json()
    assert {q["uuid"] for q in page["items"]} & {q["uuid"] for q in page_two["items"]} == set()

    floor = max(scores) - 0.0001
    filtered = api.get(f"/api/v1/profiles/{uuid}/queries?min_score={floor}").json()
    assert all(q["opportunity_score"] >= floor for q in filtered["items"])

    for status, check in (("visible", lambda q: q["domain_visible"] is True),
                          ("not_visible", lambda q: q["domain_visible"] is False),
                          ("unknown", lambda q: q["domain_visible"] is None)):
        rows = api.get(f"/api/v1/profiles/{uuid}/queries?status={status}").json()["items"]
        assert all(check(q) for q in rows), status

    assert api.get(f"/api/v1/profiles/{uuid}/queries?status=sideways").status_code == 422
    assert api.get(f"/api/v1/profiles/{uuid}/queries?min_score=5").status_code == 422


def test_reads_are_empty_before_any_run(api, created_profile):
    uuid = created_profile["uuid"]
    assert api.get(f"/api/v1/profiles/{uuid}/queries").json()["total"] == 0
    assert api.get(f"/api/v1/profiles/{uuid}/recommendations").json()["total"] == 0


def test_recommendations_point_at_real_queries(api, created_profile):
    uuid = created_profile["uuid"]
    api.post(f"/api/v1/profiles/{uuid}/run", json={"question": QUESTION}).raise_for_status()
    queries = api.get(f"/api/v1/profiles/{uuid}/queries").json()["items"]
    recs = api.get(f"/api/v1/profiles/{uuid}/recommendations").json()["items"]

    assert recs
    known = {q["uuid"] for q in queries}
    assert all(r["target_query_uuid"] in known for r in recs)
    assert all(r["priority"] in {"high", "medium", "low"} for r in recs)


def test_recheck_of_unknown_query_is_404(api):
    assert api.post("/api/v1/queries/nope/recheck").status_code == 404


def test_health(api):
    body = api.get("/health").json()
    assert body["status"] == "ok" and body["mock_dataforseo"] is True


def test_logs_are_json_and_redact_secrets(caplog):
    """Observability contract: structured lines, correlation id, no secrets."""
    assert redact({"password": "hunter2"}) == {"password": "***redacted***"}
    assert redact({"nested": [{"api_key": "sk-live"}]}) == {
        "nested": [{"api_key": "***redacted***"}]}
    assert redact({"authorization": "Basic abc"})["authorization"] == "***redacted***"

    import logging
    record = logging.LogRecord("app.test", logging.INFO, __file__, 1, "tool.call.ok",
                               None, None)
    record.tool = "google_serp"
    record.dataforseo_password = "secret"
    line = json.loads(JsonFormatter().format(record))
    assert line["event"] == "tool.call.ok"
    assert line["tool"] == "google_serp"
    assert line["dataforseo_password"] == "***redacted***"
