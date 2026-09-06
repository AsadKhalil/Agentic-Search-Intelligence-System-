"""SerpApi -> DataForSEO envelope translation (no network).

The adapter is the only place where another provider's shape becomes this pipeline's
shape. If it drifts, every downstream number is quietly wrong rather than absent, so it
is tested against real SerpApi response fragments.
"""
from app.graph.nodes import _extract_serp
from app.tools.serpapi import SerpApiBackend, domain_of, to_envelope


def test_domain_extraction_matches_what_the_pipeline_compares_on():
    assert domain_of("https://www.asana.com/uses/agile") == "asana.com"
    assert domain_of("http://acme.io") == "acme.io"
    assert domain_of("https://blog.acme.io:8443/post") == "blog.acme.io"
    assert domain_of(None) is None
    assert domain_of("") is None


def test_organic_results_become_rankable_items():
    payload = {
        "organic_results": [
            {"position": 1, "title": "Asana", "link": "https://asana.com/agile"},
            {"position": 2, "title": "Acme", "link": "https://www.acme.io/agile-planning"},
        ],
        "search_information": {"total_results": 12_300_000},
    }
    envelope = to_envelope(payload, "agile planning tools")
    assert envelope["status_code"] == 20000

    # the real extractor, unchanged, must find the domain at the right position
    records = _extract_serp(envelope, ["agile planning tools"], "acme.io")
    organic = [r for r in records if r.source == "organic"]
    assert len(organic) == 1
    assert organic[0].domain_visible is True
    assert organic[0].visibility_position == 2
    assert organic[0].evidence["top_domains"] == ["asana.com", "acme.io"]


def test_absent_domain_reads_as_not_visible_not_as_missing_data():
    payload = {"organic_results": [
        {"position": 1, "link": "https://monday.com/agile"},
    ]}
    records = _extract_serp(to_envelope(payload, "agile planning tools"),
                            ["agile planning tools"], "acme.io")
    assert records[0].domain_visible is False
    assert records[0].visibility_position is None


def test_inline_ai_overview_references_are_carried_over():
    payload = {
        "organic_results": [{"position": 1, "link": "https://acme.io/x"}],
        "ai_overview": {"references": [
            {"link": "https://acme.io/guide", "source": "Acme"},
            {"link": "https://asana.com/guide", "source": "Asana"},
        ]},
    }
    records = _extract_serp(to_envelope(payload, "agile planning tools"),
                            ["agile planning tools"], "acme.io")
    overview = [r for r in records if r.source == "ai_overview"]
    assert len(overview) == 1
    assert overview[0].ai_overview_mentioned is True


def test_a_deferred_ai_overview_emits_no_record_rather_than_a_false_negative():
    """SerpApi often returns only a page_token. Recording 'not mentioned' from that would
    be an assertion the response never made."""
    payload = {
        "organic_results": [{"position": 3, "link": "https://acme.io/x"}],
        "ai_overview": {"page_token": "abc123"},
    }
    records = _extract_serp(to_envelope(payload, "agile planning tools"),
                            ["agile planning tools"], "acme.io")
    assert [r.source for r in records] == ["organic"]


def test_errors_map_onto_the_status_codes_the_pipeline_classifies_on():
    empty = to_envelope({"error": "Google hasn't returned any results for this query."}, "x")
    assert empty["status_code"] == 40102          # usable: no results is an answer

    bad_key = to_envelope({"error": "Invalid API key."}, "x")
    assert bad_key["status_code"] == 40100        # terminal: retrying cannot help
    assert "Invalid API key" in bad_key["status_message"]


def test_non_serp_tools_fall_through_to_the_configured_backend(settings):
    class Recording:
        def __init__(self):
            self.seen = []

        def post(self, path, body, timeout, tool):
            self.seen.append(tool)
            return 200, {"status_code": 20000, "tasks": []}

    fallback = Recording()
    backend = SerpApiBackend(settings, fallback=fallback)
    backend.post("/x", [{"keywords": ["a"]}], 5.0, "keyword_metrics")
    backend.post("/y", [{"query_text": "a"}], 5.0, "chatgpt_response")
    assert fallback.seen == ["keyword_metrics", "chatgpt_response"]
