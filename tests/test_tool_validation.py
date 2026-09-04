"""The validation gate between the model's tool call and any paid HTTP call (PLAN §9.5)."""
import pytest

from app.llm import ScriptedToolCallingLLM
from app.resilience import ToolArgumentError
from app.schemas import ChatGptResponseArgs, GoogleSerpArgs
from app.tools.dataforseo import TOOLS, validate_args


def test_tools_are_bindable_and_named_for_the_registry():
    assert [t.name for t in TOOLS] == ["google_serp", "keyword_metrics", "chatgpt_response"]
    assert TOOLS[0].args_schema is GoogleSerpArgs


def test_missing_required_field():
    with pytest.raises(ToolArgumentError, match="keyword"):
        validate_args("google_serp", {"depth": 10})


def test_wrong_type():
    with pytest.raises(ToolArgumentError, match="depth"):
        validate_args("google_serp", {"keyword": "crm", "depth": "ten"})


def test_out_of_range_depth():
    with pytest.raises(ToolArgumentError, match="less than or equal to 200"):
        validate_args("google_serp", {"keyword": "crm", "depth": 500})


def test_hallucinated_field_is_rejected():
    with pytest.raises(ToolArgumentError, match="sort_by"):
        validate_args("google_serp", {"keyword": "crm", "sort_by": "relevance"})


def test_user_prompt_over_500_chars():
    with pytest.raises(ToolArgumentError, match="at most 500"):
        validate_args("chatgpt_response", {
            "query_text": "crm", "user_prompt": "x" * 600, "model_name": "gpt-4o-mini",
        })


def test_keyword_over_ten_words():
    with pytest.raises(ToolArgumentError, match="10 words"):
        validate_args("keyword_metrics", {"keywords": ["one two three four five six "
                                                       "seven eight nine ten eleven"]})


def test_keyword_over_eighty_chars():
    with pytest.raises(ToolArgumentError, match="80 characters"):
        validate_args("keyword_metrics", {"keywords": ["crm " + "x" * 90]})


def test_too_many_keywords():
    with pytest.raises(ToolArgumentError, match="at most 700"):
        validate_args("keyword_metrics", {"keywords": [f"kw{i}" for i in range(701)]})


def test_reasoning_model_token_floor():
    with pytest.raises(ToolArgumentError, match="1024"):
        validate_args("chatgpt_response", {
            "query_text": "crm", "user_prompt": "hi", "model_name": "o3-mini",
            "max_output_tokens": 512,
        })
    # a non-reasoning model at the same budget is fine
    assert validate_args("chatgpt_response", {
        "query_text": "crm", "user_prompt": "hi", "model_name": "gpt-4o-mini",
        "max_output_tokens": 512,
    }).max_output_tokens == 512


def test_max_output_tokens_bounds():
    for value in (8, 5000):
        with pytest.raises(ToolArgumentError, match="max_output_tokens"):
            validate_args("chatgpt_response", {
                "query_text": "crm", "user_prompt": "hi",
                "model_name": "gpt-4o-mini", "max_output_tokens": value,
            })


def test_unknown_tool():
    with pytest.raises(ToolArgumentError, match="unknown tool"):
        validate_args("google_maps", {})


def test_query_text_is_not_sent_to_the_provider():
    from app.tools.dataforseo import request_body

    body = request_body("chatgpt_response", ChatGptResponseArgs(
        query_text="best crm", user_prompt="hi", model_name="gpt-4o-mini"))
    assert "query_text" not in body[0]
    assert body[0]["user_prompt"] == "hi"


def test_bad_tool_call_is_recorded_and_routed_not_raised(make_run):
    """A malformed tool call must land in errors[] and let the run continue."""
    script = [[
        {"name": "google_serp", "args": {"depth": 10}, "id": "bad", "type": "tool_call"},
        {"name": "keyword_metrics", "args": {"keywords": ["best crm software"]},
         "id": "good", "type": "tool_call"},
    ]]
    state = make_run(llm=ScriptedToolCallingLLM(tool_call_script=script))

    assert any(e.kind == "tool_argument" and e.tool == "google_serp"
               for e in state["errors"])
    assert state["report_document"]["status"] == "partial"
    assert state["normalized"], "the valid call still ran"
