"""
Empty / blocked / refused response contract (offline, no API keys).

The rule (see "Empty, Blocked, and Refused Responses" in USE.md):
Djinnite raises when the response cannot be used as the method promises,
and returns when the model produced its own answer, even an empty one or
a refusal.

    uv run pytest tests/test_empty_response.py -v
"""

from types import SimpleNamespace as NS

import pytest

from djinnite.config_loader import ModelInfo, ModelCosting
from djinnite.ai_providers.base_provider import (
    AIResponse,
    AIEmptyResponseError,
    AIOutputTruncatedError,
)
from djinnite.ai_providers.gemini_provider import GeminiProvider
from djinnite.ai_providers.claude_provider import ClaudeProvider
from djinnite.ai_providers.openai_provider import OpenAIProvider
from djinnite.ai_providers.grok_provider import GrokProvider


SCHEMA = {"type": "object", "properties": {"v": {"type": "integer"}}, "required": ["v"]}

_INFO = ModelInfo(
    id="test-model", name="Test", context_window=100_000, max_output_tokens=1000,
    costing=ModelCosting(input_per_1m=1.0, output_per_1m=2.0, source="published"),
)


def _bare(cls, client):
    """Provider instance with a stub client, skipping SDK construction."""
    p = cls.__new__(cls)
    p.api_key = "x"
    p.model = "test-model"
    p._model_info = _INFO
    p._require_pricing = True
    p._client = client
    return p


# ------------------------------------------------------------------
# Gemini
# ------------------------------------------------------------------

def _gemini(response):
    client = NS(models=NS(generate_content=lambda **kw: response))
    return _bare(GeminiProvider, client)


def _g_usage():
    return NS(prompt_token_count=100, candidates_token_count=0,
              total_token_count=100, thoughts_token_count=None)


def _g_candidate(finish, parts, ratings=None):
    return NS(finish_reason=finish, finish_message=None, safety_ratings=ratings,
              content=NS(parts=parts) if parts is not None else None,
              grounding_metadata=None)


def _g_response(candidates, block_reason=None):
    return NS(candidates=candidates, text=None, usage_metadata=_g_usage(),
              prompt_feedback=NS(block_reason=block_reason, block_reason_message=None,
                                 safety_ratings=None))


def _g_call(provider, json_mode):
    if json_mode:
        return provider.generate_json("q", schema=SCHEMA, force=True)
    return provider.generate("q")


@pytest.mark.parametrize("json_mode", [False, True])
def test_gemini_prompt_blocked_raises(json_mode):
    p = _gemini(_g_response([], block_reason="SAFETY"))
    with pytest.raises(AIEmptyResponseError) as ei:
        _g_call(p, json_mode)
    e = ei.value
    assert "SAFETY" in str(e)
    assert e.reason == "SAFETY"
    assert e.partial_response.content == ""
    assert e.partial_response.block_reason == "SAFETY"
    assert e.partial_response.usage["total_cost"] > 0  # billed tokens still reported


@pytest.mark.parametrize("json_mode", [False, True])
def test_gemini_no_candidates_no_block_raises(json_mode):
    p = _gemini(_g_response([]))
    with pytest.raises(AIEmptyResponseError) as ei:
        _g_call(p, json_mode)
    assert ei.value.reason == "empty"


@pytest.mark.parametrize("json_mode", [False, True])
def test_gemini_candidate_filtered_raises_with_ratings(json_mode):
    ratings = [NS(category="HARM_CATEGORY_DANGEROUS_CONTENT", probability="HIGH", blocked=True)]
    p = _gemini(_g_response([_g_candidate("SAFETY", None, ratings)]))
    with pytest.raises(AIEmptyResponseError) as ei:
        _g_call(p, json_mode)
    msg = str(ei.value)
    assert "SAFETY" in msg
    assert "HARM_CATEGORY_DANGEROUS_CONTENT=HIGH[blocked]" in msg


def test_gemini_empty_stop_json_raises():
    p = _gemini(_g_response([_g_candidate("STOP", [NS(text="", inline_data=None)])]))
    with pytest.raises(AIEmptyResponseError):
        _g_call(p, json_mode=True)


def test_gemini_empty_stop_generate_returns_honest_empty():
    p = _gemini(_g_response([_g_candidate("STOP", [NS(text="", inline_data=None)])]))
    r = _g_call(p, json_mode=False)
    assert r.content == ""
    assert "STOP" in r.finish_reason
    assert r.block_reason is None


@pytest.mark.parametrize("json_mode", [False, True])
def test_gemini_success_unchanged(json_mode):
    text = '{"v": 1}'
    p = _gemini(_g_response([_g_candidate("STOP", [NS(text=text, inline_data=None)])]))
    r = _g_call(p, json_mode)
    assert isinstance(r.content, str) and r.content == text


def test_gemini_max_tokens_still_truncation():
    p = _gemini(_g_response([_g_candidate("MAX_TOKENS", [NS(text='{"v"', inline_data=None)])]))
    with pytest.raises(AIOutputTruncatedError):
        _g_call(p, json_mode=True)


# ------------------------------------------------------------------
# Claude
# ------------------------------------------------------------------

def _claude(stop_reason, blocks):
    p = _bare(ClaudeProvider, client=None)
    response = NS(content=blocks, stop_reason=stop_reason, usage=None)
    acc = {"input_tokens": 100, "output_tokens": 5, "thinking_tokens": 0,
           "server_tool_use_input_tokens": 0, "search_units": 0}
    p._run_with_continuation = lambda kwargs: (response, acc)
    return p


def test_claude_refusal_json_raises():
    p = _claude("refusal", [NS(type="text", text="I can't help with that.")])
    with pytest.raises(AIEmptyResponseError) as ei:
        p.generate_json("q", schema=SCHEMA, force=True)
    assert ei.value.reason == "refusal"
    assert ei.value.partial_response.usage["total_cost"] > 0


def test_claude_refusal_generate_returns():
    p = _claude("refusal", [NS(type="text", text="I can't help with that.")])
    r = p.generate("q")
    assert r.finish_reason == "refusal"
    assert r.content == "I can't help with that."


def test_claude_no_text_json_raises():
    p = _claude("end_turn", [NS(type="server_tool_use")])
    with pytest.raises(AIEmptyResponseError) as ei:
        p.generate_json("q", schema=SCHEMA, force=True)
    assert ei.value.reason == "empty"


# ------------------------------------------------------------------
# OpenAI / Grok (Responses API)
# ------------------------------------------------------------------

def _responses_api(cls, *, status="completed", reason=None, blocks=()):
    response = NS(
        output=[NS(type="message", content=list(blocks))],
        output_text="",
        status=status,
        incomplete_details=NS(reason=reason) if reason else None,
        usage=NS(input_tokens=100, output_tokens=5, total_tokens=105,
                 output_tokens_details=None),
    )
    client = NS(responses=NS(create=lambda **kw: response))
    return _bare(cls, client)


@pytest.mark.parametrize("cls", [OpenAIProvider, GrokProvider])
@pytest.mark.parametrize("json_mode", [False, True])
def test_responses_content_filter_is_block_not_truncation(cls, json_mode):
    p = _responses_api(cls, status="incomplete", reason="content_filter",
                       blocks=[NS(type="output_text", text="partial")])
    with pytest.raises(AIEmptyResponseError) as ei:
        if json_mode:
            p.generate_json("q", schema=SCHEMA, force=True)
        else:
            p.generate("q")
    assert ei.value.reason == "content_filter"
    assert not isinstance(ei.value, AIOutputTruncatedError)
    assert ei.value.partial_response.block_reason == "content_filter"


@pytest.mark.parametrize("cls", [OpenAIProvider, GrokProvider])
def test_responses_max_output_tokens_still_truncation(cls):
    p = _responses_api(cls, status="incomplete", reason="max_output_tokens",
                       blocks=[NS(type="output_text", text='{"v"')])
    with pytest.raises(AIOutputTruncatedError):
        p.generate_json("q", schema=SCHEMA, force=True)


@pytest.mark.parametrize("cls", [OpenAIProvider, GrokProvider])
def test_responses_refusal_generate_returns(cls):
    p = _responses_api(cls, blocks=[NS(type="refusal", refusal="I can't help.")])
    r = p.generate("q")
    assert r.content == "I can't help."
    assert r.finish_reason == "refusal"


@pytest.mark.parametrize("cls", [OpenAIProvider, GrokProvider])
def test_responses_refusal_json_raises(cls):
    p = _responses_api(cls, blocks=[NS(type="refusal", refusal="I can't help.")])
    with pytest.raises(AIEmptyResponseError) as ei:
        p.generate_json("q", schema=SCHEMA, force=True)
    assert ei.value.reason == "refusal"
    assert "I can't help." in str(ei.value)


@pytest.mark.parametrize("cls", [OpenAIProvider, GrokProvider])
def test_responses_empty_json_raises(cls):
    p = _responses_api(cls, blocks=[])
    with pytest.raises(AIEmptyResponseError) as ei:
        p.generate_json("q", schema=SCHEMA, force=True)
    assert ei.value.reason == "empty"


# ------------------------------------------------------------------
# AIResponse invariant
# ------------------------------------------------------------------

def test_airesponse_content_never_none():
    r = AIResponse(content=None, model="m", provider="p")
    assert r.content == ""
