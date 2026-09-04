"""Query identity and the recheck contract (PLAN §6, §9.9)."""
from app.db import SessionLocal
from app.models import PipelineRun, Query, Recommendation

from tests.conftest import QUESTION


def _run(api, profile_uuid, question=QUESTION):
    response = api.post(f"/api/v1/profiles/{profile_uuid}/run", json={"question": question})
    assert response.status_code == 200, response.text
    return response.json()


def test_one_row_per_logical_query_not_per_api_call(api, created_profile):
    body = _run(api, created_profile["uuid"])
    with SessionLocal() as session:
        rows = session.query(Query).filter(Query.run_uuid == body["run_uuid"]).all()

    keys = [r.query_key for r in rows]
    assert len(keys) == len(set(keys)), "several tool calls about one query = one row"
    # the plan issues more calls than it investigates distinct queries
    assert body["planned_call_count"] >= len(keys)
    assert all(r.query_key == " ".join(r.query_text.lower().split()) for r in rows)


def test_sources_merge_into_one_row(api, created_profile):
    body = _run(api, created_profile["uuid"])
    with SessionLocal() as session:
        rows = session.query(Query).filter(Query.run_uuid == body["run_uuid"]).all()

    primary = max(rows, key=lambda r: (r.estimated_search_volume is not None,
                                       r.domain_visible is not None))
    assert primary.estimated_search_volume is not None      # keyword_metrics
    assert primary.domain_visible is not None               # organic
    assert primary.retrieval_status == "ok"


def test_domain_visible_is_organic_only_ai_mentions_are_separate(api, created_profile):
    body = _run(api, created_profile["uuid"])
    with SessionLocal() as session:
        rows = session.query(Query).filter(Query.run_uuid == body["run_uuid"]).all()

    serp_rows = [r for r in rows if r.domain_visible is not None]
    assert serp_rows, "at least one query was checked organically"
    for row in serp_rows:
        # three separate surfaces, never collapsed into one boolean
        assert "organic" in row.evidence
        assert row.ai_overview_mentioned is None or isinstance(row.ai_overview_mentioned, bool)
    assert any(r.chatgpt_mentioned is not None for r in rows)


def test_failed_query_still_gets_a_row_and_can_be_rechecked(api, backend, created_profile):
    """The query most worth rechecking is the one whose retrieval failed."""
    backend.fail_first_n = {"google_serp": 99, "keyword_metrics": 99,
                            "chatgpt_response": 99}
    body = _run(api, created_profile["uuid"])
    assert body["status"] == "partial"

    with SessionLocal() as session:
        rows = session.query(Query).filter(Query.run_uuid == body["run_uuid"]).all()
    assert rows, "a failed run still records what it tried to look at"
    assert all(r.retrieval_status == "failed" for r in rows)
    assert all(r.estimated_search_volume is None and r.domain_visible is None
               for r in rows)
    assert all(r.opportunity_score > 0 for r in rows), "nullable metrics still score"

    # the same rows are visible through the API, so they can be rechecked
    listed = api.get(f"/api/v1/profiles/{created_profile['uuid']}/queries").json()
    assert listed["total"] == len(rows)

    backend.fail_first_n.clear()
    target = listed["items"][0]
    recheck = api.post(f"/api/v1/queries/{target['uuid']}/recheck")
    assert recheck.status_code == 200, recheck.text
    assert recheck.json()["kind"] == "recheck"

    with SessionLocal() as session:
        updated = session.get(Query, target["uuid"])
        assert updated.retrieval_status == "ok"
        assert updated.estimated_search_volume is not None
        assert updated.domain_visible is not None
        assert updated.opportunity_score != target["opportunity_score"]


def test_recheck_updates_in_place_and_keeps_recommendations_visible(api, created_profile):
    body = _run(api, created_profile["uuid"])
    listed = api.get(f"/api/v1/profiles/{created_profile['uuid']}/queries").json()
    before = api.get(f"/api/v1/profiles/{created_profile['uuid']}/recommendations").json()
    assert before["total"] > 0

    target = listed["items"][0]
    api.post(f"/api/v1/queries/{target['uuid']}/recheck").raise_for_status()

    after_queries = api.get(f"/api/v1/profiles/{created_profile['uuid']}/queries").json()
    assert after_queries["total"] == listed["total"], "recheck must not add a duplicate row"
    assert {q["uuid"] for q in after_queries["items"]} == {q["uuid"] for q in listed["items"]}

    # written under the recheck run's uuid, but still reachable through the full run's rows
    after = api.get(f"/api/v1/profiles/{created_profile['uuid']}/recommendations").json()
    assert after["total"] > 0, "joining on recommendation.run_uuid would have lost these"
    with SessionLocal() as session:
        recheck_run = session.query(PipelineRun).filter(
            PipelineRun.kind == "recheck").one()
        moved = session.query(Recommendation).filter(
            Recommendation.run_uuid == recheck_run.uuid).all()
    assert moved, "the rechecked query's recommendations were rewritten"


def test_latest_full_run_wins_and_recheck_does_not_shadow_it(api, created_profile):
    first = _run(api, created_profile["uuid"])
    second = _run(api, created_profile["uuid"], "Where do we rank for enterprise crm?")
    listed = api.get(f"/api/v1/profiles/{created_profile['uuid']}/queries").json()
    keys = {q["run_uuid"] for q in listed["items"]}
    assert keys == {second["run_uuid"]}
    assert first["run_uuid"] not in keys

    api.post(f"/api/v1/queries/{listed['items'][0]['uuid']}/recheck").raise_for_status()
    after = api.get(f"/api/v1/profiles/{created_profile['uuid']}/queries").json()
    assert {q["run_uuid"] for q in after["items"]} == {second["run_uuid"]}


def test_run_row_records_metrics_and_report(api, created_profile):
    body = _run(api, created_profile["uuid"])
    with SessionLocal() as session:
        run = session.get(PipelineRun, body["run_uuid"])
    assert run.status == "completed" and run.kind == "full"
    assert run.duration_ms > 0
    assert run.metrics["node_sequence"][0] == "plan_queries"
    assert run.metrics["nodes"]["retrieve"]["api_calls"] >= 3
    assert run.report["summary"]
    assert run.correlation_id
