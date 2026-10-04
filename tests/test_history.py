"""
Multi-turn input: ``history=[{role, content}, ...]`` on generate / generate_json.

``history`` holds the earlier turns, oldest first; ``prompt`` is always the
final user turn. Djinnite keeps no session state. Without ``history`` every
request must be byte-identical to the single-turn request.

All offline: stub SDK clients record the native request.

    uv run pytest tests/test_history.py -v
"""

import copy

import pytest

from djinnite.ai_providers.base_provider import AIProviderError, BaseAIProvider, AIResponse
from djinnite.ai_providers.claude_provider import ClaudeProvider
from djinnite.ai_providers.gemini_provider import GeminiProvider
from djinnite.ai_providers.openai_provider import OpenAIProvider
from djinnite.ai_providers.grok_provider import GrokProvider
from djinnite.config_loader import ModelInfo, ModelCosting, VisionLimits
from djinnite.tests._stubs import (
    bare, StubAnthropic, claude_response, StubGenai, StubResponses,
)


SCHEMA = {"type": "object", "properties": {"v": {"type": "integer"}}, "required": ["v"]}

HISTORY = [
    {"role": "user", "content": "first question"},
    {"role": "assistant", "content": '{"v": 1}'},
    {"role": "user", "content": [{"type": "text", "text": "second question"}]},
    {"role": "assistant", "content": "second answer"},
]


def _claude():
    stub = StubAnthropic([claude_response(), claude_response()])
    return bare(ClaudeProvider, stub), stub


def _gemini():
    stub = StubGenai()
    return bare(GeminiProvider, stub), stub


def _openai():
    stub = StubResponses()
    return bare(OpenAIProvider, stub), stub


def _grok():
    stub = StubResponses()
    return bare(GrokProvider, stub), stub


def _call(p, json_mode, **kw):
    if json_mode:
        return p.generate_json("final question", schema=SCHEMA, force=True, **kw)
    return p.generate("final question", **kw)


# ------------------------------------------------------------- unchanged

@pytest.mark.parametrize("make", [_claude, _gemini, _openai, _grok])
@pytest.mark.parametrize("json_mode", [False, True])
def test_no_history_request_is_unchanged(make, json_mode):
    """history=None and history=[] send exactly the single-turn request."""
    p1, s1 = make()
    _call(p1, json_mode)
    p2, s2 = make()
    _call(p2, json_mode, history=None)
    p3, s3 = make()
    _call(p3, json_mode, history=[])
    assert repr(s1.calls[0]) == repr(s2.calls[0]) == repr(s3.calls[0])


def test_single_turn_shapes_are_todays():
    p, s = _claude()
    p.generate("q")
    assert s.calls[0]["messages"] == [{"role": "user", "content": [{"type": "text", "text": "q"}]}]

    p, s = _gemini()
    p.generate("q")
    contents = s.calls[0]["contents"]
    assert isinstance(contents, list) and len(contents) == 1
    assert not hasattr(contents[0], "role")  # a flat Part, not a Content

    for make in (_openai, _grok):
        p, s = make()
        p.generate("q")
        assert s.calls[0]["input"] == "q"


# ------------------------------------------------------------ with history

@pytest.mark.parametrize("json_mode", [False, True])
@pytest.mark.parametrize("system_prompt", [None, "You are terse."])
def test_claude_messages_carry_turns_in_order(json_mode, system_prompt):
    p, s = _claude()
    _call(p, json_mode, history=HISTORY, system_prompt=system_prompt)
    req = s.calls[0]
    msgs = req["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant", "user"]
    assert msgs[0]["content"] == [{"type": "text", "text": "first question"}]
    # Assistant turns -- including replayed JSON -- go out as plain text.
    assert msgs[1]["content"] == '{"v": 1}'
    assert msgs[3]["content"] == "second answer"
    assert msgs[-1]["content"] == [{"type": "text", "text": "final question"}]
    if system_prompt:
        assert req["system"] == system_prompt
    else:
        assert "system" not in req
    if json_mode:
        # Constraint decoding is request-level: it constrains only the new turn.
        assert req["output_config"]["format"]["type"] == "json_schema"
        assert all("output_config" not in m for m in msgs)


@pytest.mark.parametrize("json_mode", [False, True])
@pytest.mark.parametrize("system_prompt", [None, "You are terse."])
def test_gemini_contents_carry_roles_user_and_model(json_mode, system_prompt):
    p, s = _gemini()
    _call(p, json_mode, history=HISTORY, system_prompt=system_prompt)
    req = s.calls[0]
    contents = req["contents"]
    assert [c.role for c in contents] == ["user", "model", "user", "model", "user"]
    assert contents[0].parts[0].text == "first question"
    assert contents[1].parts[0].text == '{"v": 1}'
    assert contents[-1].parts[0].text == "final question"
    if system_prompt:
        assert req["config"]["system_instruction"] == system_prompt
    else:
        assert "system_instruction" not in req["config"]
    if json_mode:
        assert req["config"]["response_schema"]


@pytest.mark.parametrize("make", [_openai, _grok])
@pytest.mark.parametrize("json_mode", [False, True])
@pytest.mark.parametrize("system_prompt", [None, "You are terse."])
def test_responses_input_carries_turns(make, json_mode, system_prompt):
    p, s = make()
    _call(p, json_mode, history=HISTORY, system_prompt=system_prompt)
    req = s.calls[0]
    items = req["input"]
    assert [i["role"] for i in items] == ["user", "assistant", "user", "assistant", "user"]
    # The Responses API rejects input_text parts in the assistant role.
    assert items[1]["content"] == '{"v": 1}'
    assert items[0]["content"] == [{"type": "input_text", "text": "first question"}]
    assert items[-1]["content"] == [{"type": "input_text", "text": "final question"}]
    if system_prompt:
        assert req["instructions"] == system_prompt
    else:
        assert "instructions" not in req
    if json_mode:
        assert req["text"]["format"]["type"] == "json_schema"


def test_history_is_not_mutated():
    for make in (_claude, _gemini, _openai, _grok):
        hist = copy.deepcopy(HISTORY)
        p, _ = make()
        p.generate("q", history=hist)
        p.generate_json("q", schema=SCHEMA, force=True, history=hist)
        assert hist == HISTORY


def test_claude_continuation_keeps_history():
    """pause_turn continuation appends to the turns, it does not replace them."""
    stub = StubAnthropic([claude_response(stop="pause_turn"), claude_response()])
    p = bare(ClaudeProvider, stub)
    p.generate("final question", history=HISTORY)
    first, second = stub.calls
    assert len(second["messages"]) == len(first["messages"]) + 2
    assert second["messages"][:5] == first["messages"]


# --------------------------------------------------------------- validation

@pytest.mark.parametrize("bad, match", [
    ("not a list", "list of turns"),
    ([{"role": "system", "content": "x"}], "'user' or 'assistant'"),
    ([{"role": "assistant", "content": "x"}], "start with a 'user' turn"),
    ([{"role": "user"}], "'role' and 'content'"),
    ([{"role": "user", "content": "a"},
      {"role": "assistant", "content": [{"type": "image", "image_data": b"x"}]}], "text only"),
    ([{"role": "user", "content": "a"}, {"role": "assistant", "content": "   "}], "empty"),
])
def test_bad_history_raises_value_error(bad, match):
    p, s = _claude()
    with pytest.raises(ValueError, match=match):
        p.generate("q", history=bad)
    assert s.calls == []  # rejected before any request


def test_vision_limits_count_images_across_turns():
    """The image cap is per request, so history images count too."""
    vision_info = ModelInfo(
        id="test-model", name="Test", context_window=100_000, max_output_tokens=1000,
        costing=ModelCosting(input_per_1m=1.0, output_per_1m=2.0, source="published"),
        vision_limits=VisionLimits(max_images_per_request=1),
    )
    img = {"type": "image", "image_data": b"\x89PNG\r\n\x1a\n" + b"\x00" * 30}
    hist = [{"role": "user", "content": [img]}, {"role": "assistant", "content": "ok"}]
    stub = StubAnthropic([claude_response()])
    p = bare(ClaudeProvider, stub, model_info=vision_info)
    with pytest.raises(AIProviderError, match="Too many images"):
        p.generate([img], history=hist)
    assert stub.calls == []


# ------------------------------------------------------------ base fallback

class _LegacyProvider(BaseAIProvider):
    """A subclass whose generate() predates ``history``."""
    PROVIDER_NAME = "legacy"

    def _initialize_client(self):
        self._client = None

    def generate(self, prompt, system_prompt=None, temperature=0.7,
                 max_output_tokens=None, web_search=False, thinking=None):
        return AIResponse(content='{"v": 1}', model=self.model, provider="legacy")

    def is_available(self):
        return True

    def list_models(self):
        return []


class _HistoryProvider(_LegacyProvider):
    seen = None

    def generate(self, prompt, system_prompt=None, temperature=0.7,
                 max_output_tokens=None, web_search=False, thinking=None, *, history=None):
        _HistoryProvider.seen = history
        return AIResponse(content='{"v": 1}', model=self.model, provider="legacy")


def test_base_fallback_keeps_legacy_subclasses_working():
    p = _LegacyProvider(api_key="k", model="m")
    assert p.generate_json("q", schema=SCHEMA).content == '{"v": 1}'


def test_base_fallback_passes_history_through():
    p = _HistoryProvider(api_key="k", model="m")
    p.generate_json("q", schema=SCHEMA, history=HISTORY)
    assert _HistoryProvider.seen == HISTORY
