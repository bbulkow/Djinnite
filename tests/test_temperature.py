"""
Temperature: Djinnite never supplies one unless configured, and logs and
drops one the model cannot take.

Until 2026-11-03 the Gemini, OpenAI and Grok providers defaulted to 0.7
(``generate``) and 0.3 (``generate_json``), and every provider silently
dropped a temperature on a model whose ``capabilities.temperature`` lacks
``"any"``, but kept it on a model that records temperature as
incompatible with thinking, where the invented default tripped the
pre-flight on the gpt-5.1 to gpt-5.4 family whenever the caller asked for
thinking. Now both cases are logged and dropped, and the request goes
ahead (Brian, 2026-11-03).

Claude never carries a temperature at all (0.7.0): API version 2026-10-21,
the default on anthropic>=2.0 for the direct and Vertex clients, returns
400 for any request that sends temperature/top_p/top_k, and nothing
replaces them. The drop is provider-level, so even a catalog that still
says ``["any", "default"]`` cannot put a 400 on the wire.

    uv run pytest tests/test_temperature.py -v
"""

import pytest

from djinnite.ai_providers.claude_provider import ClaudeProvider
from djinnite.ai_providers.gemini_provider import GeminiProvider
from djinnite.ai_providers.grok_provider import GrokProvider
from djinnite.ai_providers.openai_provider import OpenAIProvider
from djinnite.tests._stubs import (
    bare, info, StubAnthropic, claude_response, StubGenai, StubResponses,
)

SCHEMA = {"type": "object", "properties": {"v": {"type": "integer"}}, "required": ["v"]}
PTYPES = ["claude", "gemini", "openai", "grok"]


def _make(ptype, model_info):
    if ptype == "claude":
        stub = StubAnthropic([claude_response(), claude_response()])
        return bare(ClaudeProvider, stub, model_info=model_info), stub
    if ptype == "gemini":
        stub = StubGenai()
        return bare(GeminiProvider, stub, model_info=model_info), stub
    stub = StubResponses()
    cls = OpenAIProvider if ptype == "openai" else GrokProvider
    return bare(cls, stub, model_info=model_info), stub


def _call(p, json_mode, **kw):
    if json_mode:
        return p.generate_json("q", schema=SCHEMA, force=True, **kw)
    return p.generate("q", **kw)


def _temperature_in(request):
    """The temperature a recorded SDK request carries, wherever it sits."""
    if isinstance(request, dict):
        if request.get("temperature") is not None:
            return request["temperature"]
        for value in request.values():
            found = _temperature_in(value)
            if found is not None:
                return found
        return None
    value = getattr(request, "temperature", None)
    return value if isinstance(value, (int, float)) else None


@pytest.mark.parametrize("ptype", PTYPES)
@pytest.mark.parametrize("json_mode", [False, True])
def test_no_temperature_is_sent_unless_the_caller_gives_one(ptype, json_mode):
    p, stub = _make(ptype, info(temperature=["any", "default"]))
    _call(p, json_mode)
    assert _temperature_in(stub.calls[-1]) is None


@pytest.mark.parametrize("ptype", [t for t in PTYPES if t != "claude"])
@pytest.mark.parametrize("json_mode", [False, True])
def test_a_requested_temperature_is_sent(ptype, json_mode):
    p, stub = _make(ptype, info(temperature=["any", "default"]))
    _call(p, json_mode, temperature=0.4)
    assert _temperature_in(stub.calls[-1]) == 0.4


@pytest.mark.parametrize("json_mode", [False, True])
def test_claude_never_sends_a_temperature_even_when_the_catalog_says_any(json_mode, caplog):
    """A stale catalog entry must not put a 400-rejected parameter on the wire."""
    p, stub = _make("claude", info(temperature=["any", "default"]))
    with caplog.at_level("INFO", logger="djinnite"):
        _call(p, json_mode, temperature=0.4)
    request = stub.calls[-1]
    assert _temperature_in(request) is None
    assert "extra_body" not in request
    assert "temperature=0.4 dropped" in caplog.text
    assert "does not accept a temperature" in caplog.text


def test_claude_temperature_probe_sends_no_request():
    """There is nothing to discover: every model rejects it under 2026-10-21."""
    p, stub = _make("claude", info(temperature=["any", "default"]))
    assert p.probe_temperature() is False
    assert stub.create_calls == []
    assert stub.calls == []


def test_claude_combination_probe_never_adds_a_temperature():
    p, _ = _make("claude", info(temperature=["any", "default"], thinking=["on", "off"],
                                thinking_style=["adaptive"]))
    request = p._build_combination_probe_request({"temperature": "any", "thinking": "on"})
    assert "extra_body" not in request
    assert _temperature_in(request) is None
    assert request["thinking"] == {"type": "adaptive"}


@pytest.mark.parametrize("ptype", PTYPES)
@pytest.mark.parametrize("json_mode", [False, True])
def test_a_temperature_the_model_cannot_take_is_logged_and_dropped(ptype, json_mode, caplog):
    """The request goes ahead without it, and the log says so."""
    p, stub = _make(ptype, info(temperature=["default"]))
    with caplog.at_level("INFO", logger="djinnite"):
        _call(p, json_mode, temperature=0.4)
    assert _temperature_in(stub.calls[-1]) is None
    assert "temperature=0.4 dropped" in caplog.text
    assert "does not accept a temperature" in caplog.text


GPT_54 = dict(temperature=["any", "default"], thinking=["on", "off"],
              thinking_style=["effort"],
              incompatible=[{"temperature": "any", "thinking": "on"}])


@pytest.mark.parametrize("ptype", ["openai", "grok"])
def test_thinking_without_a_temperature_passes_the_incompatibility_check(ptype):
    p, stub = _make(ptype, info(**GPT_54))
    p.generate("q", thinking=True)
    assert _temperature_in(stub.calls[-1]) is None


@pytest.mark.parametrize("ptype", ["openai", "grok"])
def test_thinking_with_a_temperature_drops_the_temperature_where_incompatible(ptype, caplog):
    p, stub = _make(ptype, info(**GPT_54))
    with caplog.at_level("INFO", logger="djinnite"):
        p.generate("q", thinking=True, temperature=0.5)
    request = stub.calls[-1]
    assert _temperature_in(request) is None
    assert request["reasoning"] == {"effort": "medium"}  # thinking kept
    assert "temperature=0.5 dropped" in caplog.text
    assert "incompatible" in caplog.text


@pytest.mark.parametrize("ptype", ["openai", "grok"])
def test_thinking_off_keeps_the_temperature(ptype):
    """The incompatibility is with thinking on; off, the temperature is sent."""
    p, stub = _make(ptype, info(**GPT_54))
    p.generate("q", thinking=False, temperature=0.5)
    assert _temperature_in(stub.calls[-1]) == 0.5
