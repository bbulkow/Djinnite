"""
Claude 5.x thinking (offline).

Facts these tests encode (Anthropic docs, 2026-10):

* Omitting ``thinking`` runs ADAPTIVE thinking on every 5.x model, so
  ``thinking=None`` means "provider default", and ``thinking=False`` must send
  an explicit ``{"type": "disabled"}``.
* Sonnet 5.5 and Opus 5.5 (and the Fable models) reject ``disabled``;
  Sonnet 5.5's lowest setting is ``{"type": "between_tools"}`` (no other
  field, effort high or below), accepted by no other model.
* ``{"type": "enabled", "budget_tokens": N}`` is a 400 on Opus 4.7+ and 5.x,
  so a combination probe that sends it records bogus incompatibilities.

    uv run pytest tests/test_claude_thinking_5x.py -v
"""

import anthropic
import pytest

from djinnite.ai_providers.base_provider import AIProviderError
from djinnite.ai_providers.claude_provider import ClaudeProvider
from djinnite.ai_providers.gemini_provider import GeminiProvider
from djinnite.ai_providers.openai_provider import OpenAIProvider
from djinnite.ai_providers.grok_provider import GrokProvider
from djinnite.tests._stubs import (
    bare, info, StubAnthropic, claude_response, StubGenai, StubResponses, anthropic_error,
)

SCHEMA = {"type": "object", "properties": {"v": {"type": "integer"}}, "required": ["v"]}

SONNET_55 = dict(  # what a corrected catalog says about claude-sonnet-5-5
    thinking=["on"], thinking_style=["adaptive", "effort", "between_tools"],
    effort_levels=["low", "medium", "high", "xhigh", "max"], structured_json=["on", "off"],
)
OPUS_55 = dict(
    thinking=["on"], thinking_style=["adaptive", "effort"],
    effort_levels=["low", "medium", "high", "xhigh", "max"], structured_json=["on", "off"],
)
SONNET_5 = dict(thinking=["on", "off"], thinking_style=["adaptive", "effort"])


def _claude(caps, n=1):
    stub = StubAnthropic([claude_response() for _ in range(n)])
    return bare(ClaudeProvider, stub, model_info=info(**caps)), stub


# ----------------------------------------------------------- None / False

def test_none_sends_no_thinking_block():
    p, stub = _claude(SONNET_55)
    p.generate("q")
    assert "thinking" not in stub.calls[0]
    assert "output_config" not in stub.calls[0]


@pytest.mark.parametrize("json_mode", [False, True])
def test_false_sends_explicit_disabled(json_mode):
    p, stub = _claude(SONNET_5)
    if json_mode:
        p.generate_json("q", schema=SCHEMA, force=True, thinking=False)
    else:
        p.generate("q", thinking=False)
    assert stub.calls[0]["thinking"] == {"type": "disabled"}


@pytest.mark.parametrize("caps", [SONNET_55, OPUS_55])
def test_false_refused_locally_where_thinking_cannot_be_off(caps):
    p, stub = _claude(caps)
    with pytest.raises(AIProviderError, match="does not support disabling thinking"):
        p.generate_json("q", schema=SCHEMA, thinking=False)
    assert stub.calls == []


# --------------------------------------------------------- between_tools

@pytest.mark.parametrize("json_mode", [False, True])
def test_between_tools_request_shape(json_mode):
    p, stub = _claude(SONNET_55)
    if json_mode:
        p.generate_json("q", schema=SCHEMA, thinking="between_tools")
    else:
        p.generate("q", thinking="between_tools")
    req = stub.calls[0]
    # No other field inside thinking, and no effort: the model default
    # (high on Sonnet 5.5) is the highest level between_tools accepts.
    assert req["thinking"] == {"type": "between_tools"}
    assert "effort" not in req.get("output_config", {})


def test_between_tools_counts_as_thinking_off_for_incompatibilities():
    caps = dict(SONNET_55, incompatible=[{"thinking": "on", "structured_json": "on"}])
    p, stub = _claude(caps, n=1)
    p.generate_json("q", schema=SCHEMA, thinking="between_tools")  # allowed
    # The pre-flight ValueError surfaces wrapped, as it always has.
    with pytest.raises(AIProviderError, match="incompatible"):
        p.generate_json("q", schema=SCHEMA, thinking="high")


def test_between_tools_resolution_and_builders():
    p, _ = _claude(SONNET_55)
    assert p._resolve_thinking("between_tools") == "between_tools"
    assert p._resolve_thinking("BETWEEN_TOOLS") == "between_tools"
    assert p._build_claude_thinking("between_tools", 8192) == {"type": "between_tools"}
    assert p._build_claude_effort("between_tools") is None
    assert p._thinking_active("between_tools") is False
    assert p._thinking_active("high") is True


@pytest.mark.parametrize("caps", [OPUS_55, SONNET_5])
def test_between_tools_rejected_where_not_listed(caps):
    p, stub = _claude(caps)
    with pytest.raises(ValueError, match="not supported by model") as ei:
        p.generate("q", thinking="between_tools")
    assert "thinking=True" in str(ei.value)  # names a form the model accepts
    assert stub.calls == []


def test_between_tools_offered_as_an_alternative():
    p, _ = _claude(dict(thinking=["on"], thinking_style=["between_tools", "adaptive"]))
    with pytest.raises(ValueError) as ei:
        p._resolve_thinking(2048)
    assert "thinking='between_tools'" in str(ei.value)


@pytest.mark.parametrize("cls, client", [
    (GeminiProvider, StubGenai()), (OpenAIProvider, StubResponses()), (GrokProvider, StubResponses()),
])
def test_between_tools_is_claude_only(cls, client):
    p = bare(cls, client, model_info=info(thinking=["on", "off"],
                                          thinking_style=["adaptive", "effort", "between_tools"]))
    with pytest.raises(ValueError, match="not supported by provider"):
        p.generate("q", thinking="between_tools")
    # ...even without a catalog entry, where pre-flight is otherwise skipped.
    p._model_info = None
    with pytest.raises(ValueError, match="not supported by provider"):
        p._resolve_thinking("between_tools")


def test_effort_levels_still_work_on_55():
    p, stub = _claude(SONNET_55)
    p.generate_json("q", schema=SCHEMA, thinking="low")
    assert stub.calls[0]["output_config"]["effort"] == "low"
    assert "thinking" not in stub.calls[0]


# ----------------------------------------------------------------- probes

@pytest.mark.parametrize("styles, expected", [
    (["adaptive", "effort"], {"type": "adaptive"}),
    (["adaptive", "budget", "effort"], {"type": "adaptive"}),
    (["budget"], {"type": "enabled", "budget_tokens": 1024}),
    (None, {"type": "adaptive"}),
])
def test_combination_probe_sends_a_shape_the_model_accepts(styles, expected):
    p = bare(ClaudeProvider, StubAnthropic(), model_info=None)
    p._probe_supported_states = {"thinking_style": styles}
    kwargs = p._build_combination_probe_request({"thinking": "on", "structured_json": "on"})
    assert kwargs["thinking"] == expected
    assert "format" in kwargs["output_config"]


def test_combination_probe_falls_back_to_catalog_styles():
    p = bare(ClaudeProvider, StubAnthropic(), model_info=info(thinking_style=["budget"]))
    kwargs = p._build_combination_probe_request({"thinking": "on"})
    assert kwargs["thinking"]["type"] == "enabled"


def test_orchestrator_hands_styles_to_the_builder():
    """A budget-rejecting model no longer yields thinking+json incompatibility."""
    def create(**kw):
        if kw.get("thinking", {}).get("type") == "enabled":
            raise anthropic_error(anthropic.BadRequestError, 400, "budget_tokens not supported")
        return claude_response()
    stub = StubAnthropic()
    stub.messages.create = create
    p = bare(ClaudeProvider, stub, model_info=None)
    found = p.probe_incompatible_combinations({
        "thinking": ["on"], "structured_json": ["on", "off"],
        "thinking_style": ["adaptive", "effort"],
    })
    assert found == []


@pytest.mark.parametrize("error, expected", [
    (None, True),
    (anthropic_error(anthropic.BadRequestError, 400, "thinking.type.disabled is not supported"), False),
    (anthropic_error(anthropic.RateLimitError, 429, "slow down"), None),
])
def test_probe_thinking_disable_actually_probes(error, expected):
    stub = StubAnthropic(create_error=error)
    p = bare(ClaudeProvider, stub, model_info=None)
    assert p.probe_thinking_disable() is expected
    assert stub.create_calls[0]["thinking"] == {"type": "disabled"}


def test_probe_thinking_style_detects_between_tools():
    accepted = {"adaptive", "between_tools"}  # Sonnet 5.5

    def create(**kw):
        if kw["thinking"]["type"] not in accepted:
            raise anthropic_error(anthropic.BadRequestError, 400, "not supported for this model")
        return claude_response()
    stub = StubAnthropic()
    stub.messages.create = create
    p = bare(ClaudeProvider, stub, model_info=None)
    assert p.probe_thinking_style() == ["adaptive", "between_tools"]
    assert p.probe_thinking() is True


def test_between_tools_alone_is_not_evidence_of_thinking():
    stub = StubAnthropic()
    p = bare(ClaudeProvider, stub, model_info=None)
    p.probe_thinking_style = lambda: ["between_tools"]
    assert p.probe_thinking() is False
