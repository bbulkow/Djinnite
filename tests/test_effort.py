"""
Djinnite effort levels (offline): ``effort=`` and ``config/request_policy.json``.

Rules (THINKING_EFFORT_DESIGN.md section 2, Brian 2026-11-03):

* six ordered levels, ``minimal`` to ``max``; the vendor word comes from the
  policy file's per-provider vocabulary, then steps to the nearest level in
  the model's catalog ``effort_levels`` (a tie goes lower);
* an omitted effort preserves native defaults unless an explicit level is
  required to honor thinking=True; an inapplicable effort is logged and
  disregarded, never turned into a token budget;
* a level string in ``thinking=`` is a deprecated alias for ``effort=``;
* a vendor rejecting the level Djinnite chose is a ``DjinniteEffortMappingError``.

    uv run pytest tests/test_effort.py -v
"""

import json

import pytest

from djinnite.ai_providers.base_provider import AIProviderError, DjinniteEffortMappingError
from djinnite.ai_providers.claude_provider import ClaudeProvider
from djinnite.ai_providers.gemini_provider import GeminiProvider
from djinnite.ai_providers.grok_provider import GrokProvider
from djinnite.ai_providers.openai_provider import OpenAIProvider
from djinnite.config_loader import EFFORT_LEVEL_VALUES, RequestPolicy, load_request_policy, load_model_catalog
from djinnite.tests._stubs import (
    bare, info, StubAnthropic, claude_response, StubGenai, StubResponses,
)

SCHEMA = {"type": "object", "properties": {"v": {"type": "integer"}}, "required": ["v"]}

EFFORT_ONLY = dict(thinking=["on", "off"], thinking_style=["effort"],
                   effort_levels=["low", "medium", "high", "xhigh"])
PRO = dict(thinking=["on"], thinking_style=["effort"], effort_levels=["high"])
NO_LEVELS = dict(thinking=["on", "off"], thinking_style=["effort"])
BUDGET_ONLY = dict(thinking=["on", "off"], thinking_style=["budget"])
NO_THINKING = dict(thinking=["off"], thinking_style=[])
CLAUDE_55 = dict(thinking=["on"], thinking_style=["adaptive", "effort"],
                 effort_levels=["low", "medium", "high", "xhigh", "max"])
CLAUDE_45 = dict(thinking=["on", "off"], thinking_style=["budget", "effort"],
                 effort_levels=["low", "medium", "high"])


def _openai(caps):
    stub = StubResponses()
    return bare(OpenAIProvider, stub, model_info=info(**caps)), stub


def _grok(caps):
    stub = StubResponses()
    return bare(GrokProvider, stub, model_info=info(**caps)), stub


def _gemini(caps):
    stub = StubGenai()
    return bare(GeminiProvider, stub, model_info=info(**caps)), stub


def _claude(caps, n=1):
    stub = StubAnthropic([claude_response() for _ in range(n)])
    return bare(ClaudeProvider, stub, model_info=info(**caps)), stub


# ----------------------------------------------------------------- policy

def test_policy_ships_medium_and_no_temperature():
    policy = load_request_policy()
    assert policy.default("chatgpt", "effort") == "medium"
    assert policy.default("claude", "temperature") is None
    for ptype in ("chatgpt", "claude", "gemini", "grok"):
        assert set(policy.vocabulary(ptype)) == set(EFFORT_LEVEL_VALUES)


@pytest.mark.parametrize("bad, match", [
    ({"defaults": {"effort": "extreme"}}, "not one of"),
    ({"defaults": {"temperature": "warm"}}, "number or null"),
    ({"defaults": {"top_p": 1}}, "unknown defaults"),
    ({"providers": {"llama": {}}}, "not a provider type"),
    ({"providers": {"chatgpt": {"effort_vocabulary": {"low": "low"}}}}, "exactly"),
    ({"providers": {"chatgpt": {"effort_vocabulary": dict(
        zip(EFFORT_LEVEL_VALUES, ["none"] + list(EFFORT_LEVEL_VALUES[1:])))}}}, "not one of"),
    ({"providers": {"chatgpt": {"effort_vocabulary": dict(
        zip(EFFORT_LEVEL_VALUES, reversed(EFFORT_LEVEL_VALUES)))}}}, "backwards"),
])
def test_policy_file_is_validated(tmp_path, bad, match):
    path = tmp_path / "request_policy.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_request_policy(path)


# ---------------------------------------------------------------- mapping

@pytest.mark.parametrize("level, sent", [
    ("minimal", "low"), ("low", "low"), ("medium", "medium"),
    ("high", "high"), ("xhigh", "xhigh"), ("max", "xhigh"),
])
def test_levels_step_to_the_models_nearest_accepted_level(level, sent):
    p, stub = _openai(EFFORT_ONLY)
    p.generate("q", effort=level)
    assert stub.calls[0]["reasoning"] == {"effort": sent}


def test_tie_goes_to_the_lower_level():
    p, _ = _openai(dict(EFFORT_ONLY, effort_levels=["low", "high"]))
    assert p._map_effort("medium") == "low"


def test_single_level_model_gets_its_only_level():
    p, stub = _openai(PRO)
    p.generate("q", effort="minimal")
    assert stub.calls[0]["reasoning"] == {"effort": "high"}


def test_vocabulary_applies_when_the_catalog_has_no_levels():
    p, stub = _gemini(NO_LEVELS)
    p.generate("q", effort="max")  # Gemini's vocabulary stops at high
    assert stub.calls[0]["config"]["thinking_config"]["thinking_level"].name == "HIGH"


def test_invalid_level_is_rejected_locally():
    p, stub = _openai(EFFORT_ONLY)
    with pytest.raises(ValueError, match="effort must be one of"):
        p.generate("q", effort="extreme")
    assert stub.calls == []


# --------------------------------------------------------------- defaults

@pytest.mark.parametrize("make", [_openai, _grok])
def test_default_effort_is_only_used_when_needed_for_thinking_true(make):
    p, stub = make(EFFORT_ONLY)
    p.generate("q")                      # thinking=None
    p.generate("q", thinking=True)
    assert "reasoning" not in stub.calls[0]
    assert stub.calls[1]["reasoning"] == {"effort": "medium"}


def test_explicit_effort_is_stepped_on_always_on_models():
    p, stub = _openai(PRO)
    p.generate("q", thinking=True, effort="medium")
    assert stub.calls[0]["reasoning"] == {"effort": "high"}


def test_openai_thinking_off_is_none_and_ignores_effort(caplog):
    p, stub = _openai(EFFORT_ONLY)
    with caplog.at_level("INFO", logger="djinnite"):
        p.generate("q", thinking=False, effort="high")
    assert stub.calls[0]["reasoning"] == {"effort": "none"}
    assert "effort='high' disregarded" in caplog.text


def test_no_catalog_entry_still_sends_the_default_for_thinking_true():
    p, stub = _openai(EFFORT_ONLY)
    p._model_info = None
    p._require_pricing = False  # no catalog entry, so no price either
    p.generate("q", thinking=True)
    assert stub.calls[0]["reasoning"] == {"effort": "medium"}
    p.generate("q")
    assert "reasoning" not in stub.calls[1]


# ------------------------------------------------------------- disregard

def test_budget_only_model_disregards_effort_and_never_fakes_a_budget(caplog):
    p, stub = _gemini(BUDGET_ONLY)
    with caplog.at_level("INFO", logger="djinnite"):
        p.generate("q", thinking=True, effort="high")
    assert stub.calls[0]["config"]["thinking_config"] == {"thinking_budget": -1}
    assert "effort='high' disregarded" in caplog.text


def test_non_thinking_model_disregards_effort_and_succeeds():
    p, stub = _openai(NO_THINKING)
    p.generate("q", effort="high")
    assert "reasoning" not in stub.calls[0]


def test_gemini_effort_becomes_thinking_level_and_int_stays_a_budget():
    p, stub = _gemini(dict(thinking=["on", "off"], thinking_style=["budget", "effort"]))
    p.generate("q", effort="low")
    assert stub.calls[0]["config"]["thinking_config"]["thinking_level"].name == "LOW"
    p.generate("q", thinking=2048, effort="low")  # Gemini: alternatives; budget wins
    assert stub.calls[1]["config"]["thinking_config"] == {"thinking_budget": 2048}


# ----------------------------------------------------------------- Claude

def test_claude_carries_effort_beside_disabled_and_beside_a_budget():
    p, stub = _claude(CLAUDE_45, n=2)
    p.generate("q", thinking=False, effort="xhigh")
    assert stub.calls[0]["thinking"] == {"type": "disabled"}
    assert stub.calls[0]["output_config"] == {"effort": "high"}  # stepped
    p.generate("q", thinking=2048, effort="low", max_output_tokens=8192)
    assert stub.calls[1]["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert stub.calls[1]["output_config"] == {"effort": "low"}


def test_claude_effort_never_adds_a_thinking_block():
    p, stub = _claude(CLAUDE_55)
    p.generate_json("q", schema=SCHEMA, force=True, effort="max")
    assert "thinking" not in stub.calls[0]
    assert stub.calls[0]["output_config"]["effort"] == "max"
    assert "format" in stub.calls[0]["output_config"]


# ------------------------------------------------------------------ alias

@pytest.mark.parametrize("make, key", [(_openai, "reasoning"), (_grok, "reasoning")])
def test_thinking_level_string_is_a_deprecated_alias(make, key):
    p, stub = make(EFFORT_ONLY)
    with pytest.warns(DeprecationWarning, match="pass effort='high'"):
        p.generate("q", thinking="High")
    assert stub.calls[0][key] == {"effort": "high"}


def test_alias_and_effort_together_is_an_error():
    p, stub = _openai(EFFORT_ONLY)
    with pytest.raises(ValueError, match="pass only effort="):
        p.generate("q", thinking="low", effort="high")
    assert stub.calls == []


# ---------------------------------------------------------- mapping error

class _Rejecting:
    def __init__(self):
        self.responses = type("R", (), {"create": staticmethod(self._create)})()

    @staticmethod
    def _create(**kw):
        err = Exception("Unsupported value: 'low' is not supported with this model.")
        err.status_code = 400
        err.body = {"message": str(err), "param": "reasoning.effort",
                    "code": "unsupported_value"}
        raise err


def test_vendor_rejecting_the_mapped_level_is_a_mapping_error():
    p = bare(OpenAIProvider, _Rejecting(), model_info=info(**EFFORT_ONLY))
    with pytest.raises(DjinniteEffortMappingError) as ei:
        p.generate("q", effort="minimal")
    assert ei.value.requested == "minimal"
    assert ei.value.sent == "low"
    assert "request_policy.json" in str(ei.value)


def _generate(p, json_mode, **kwargs):
    if json_mode:
        return p.generate_json("q", schema=SCHEMA, force=True, **kwargs)
    return p.generate("q", **kwargs)


@pytest.mark.parametrize("make", [_openai, _grok, _gemini, _claude])
@pytest.mark.parametrize("json_mode", [False, True])
def test_ordinary_calls_preserve_defaults_even_with_configured_effort(make, json_mode):
    p, stub = make(EFFORT_ONLY)
    p._policy_cache = RequestPolicy(defaults={"effort": "max"})
    _generate(p, json_mode)
    request = stub.calls[-1]
    assert "reasoning" not in request
    assert "effort" not in request.get("output_config", {})
    assert "thinking_config" not in request.get("config", {})


@pytest.mark.parametrize("make", [_openai, _grok])
@pytest.mark.parametrize("json_mode", [False, True])
@pytest.mark.parametrize("default", [None, "max"])
def test_thinking_true_survives_null_default_and_maps_configured_default(make, json_mode, default):
    p, stub = make(EFFORT_ONLY)
    p._policy_cache = RequestPolicy(defaults={"effort": default})
    _generate(p, json_mode, thinking=True)
    assert stub.calls[-1]["reasoning"] == {"effort": "medium" if default is None else "xhigh"}


@pytest.mark.parametrize("make", [_openai, _grok])
@pytest.mark.parametrize("json_mode", [False, True])
def test_always_on_models_need_no_effort_to_enable_thinking(make, json_mode):
    p, stub = make(PRO)
    _generate(p, json_mode, thinking=True)
    assert "reasoning" not in stub.calls[-1]


@pytest.mark.parametrize("json_mode", [False, True])
@pytest.mark.parametrize("thinking", [True, False, 2048, "between_tools"])
def test_claude_thinking_controls_do_not_inject_effort(json_mode, thinking):
    caps = dict(CLAUDE_45, thinking_style=["budget", "adaptive", "effort", "between_tools"])
    p, stub = _claude(caps)
    _generate(p, json_mode, thinking=thinking, max_output_tokens=8192)
    assert "effort" not in stub.calls[-1].get("output_config", {})


@pytest.mark.parametrize("json_mode", [False, True])
@pytest.mark.parametrize("styles", [["budget"], ["budget", "effort"]])
def test_gemini_thinking_true_uses_native_thinking_without_inventing_effort(json_mode, styles):
    p, stub = _gemini(dict(thinking=["on", "off"], thinking_style=styles))
    _generate(p, json_mode, thinking=True)
    config = stub.calls[-1]["config"]
    if "effort" in styles:
        assert "thinking_config" not in config  # Gemini level-based models default on.
    else:
        assert config["thinking_config"] == {"thinking_budget": -1}


@pytest.mark.parametrize("json_mode", [False, True])
@pytest.mark.parametrize("backend", ["gemini", "vertexai"])
@pytest.mark.parametrize("model, requested, sent", [
    ("gemini-3.8-flash", "minimal", "LOW"),
    ("gemini-3.7-flash", "minimal", "LOW"),
    ("gemini-4-argon", "minimal", "LOW"),
    ("gemini-3.1-pro-preview", "minimal", "LOW"),
    ("gemini-3.6-flash", "minimal", "MINIMAL"),
    ("gemini-3.8-flash", "max", "HIGH"),
])
def test_gemini_maps_real_catalog_levels_on_both_access_paths(json_mode, backend, model, requested, sent):
    model_info = load_model_catalog().get_model("gemini", model)
    stub = StubGenai()
    p = bare(GeminiProvider, stub, model_info=model_info, model=model, backend=backend)
    _generate(p, json_mode, effort=requested)
    assert stub.calls[-1]["config"]["thinking_config"]["thinking_level"].name == sent


@pytest.mark.parametrize("json_mode", [False, True])
@pytest.mark.parametrize("model", ["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5-2025-08-07"])
def test_gpt5_search_selects_nearest_compatible_effort_without_dropping_search(json_mode, model):
    caps = dict(EFFORT_ONLY, effort_levels=["minimal", "low", "medium", "high"])
    p, stub = _openai(caps)
    p.model = model
    _generate(p, json_mode, effort="minimal", web_search=True)
    assert stub.calls[-1]["reasoning"] == {"effort": "low"}
    assert stub.calls[-1]["tools"] == [{"type": "web_search_preview"}]
    assert len(stub.calls) == 1


@pytest.mark.parametrize("web_search, model", [(False, "gpt-5"), (True, "gpt-future")])
def test_gpt5_search_restriction_is_scoped_to_the_known_combination(web_search, model):
    p, stub = _openai(dict(EFFORT_ONLY, effort_levels=["minimal", "low", "medium", "high"]))
    p.model = model
    p.generate("q", effort="minimal", web_search=web_search)
    assert stub.calls[-1]["reasoning"] == {"effort": "minimal"}


@pytest.mark.parametrize("json_mode", [False, True])
@pytest.mark.parametrize("backend", [None, "vertexai"])
@pytest.mark.parametrize("level", ["xhigh", "max"])
def test_between_tools_preserved_while_effort_steps_down(json_mode, backend, level):
    caps = dict(CLAUDE_55, thinking_style=["adaptive", "effort", "between_tools"])
    p, stub = _claude(caps)
    p.backend = backend
    _generate(p, json_mode, thinking="between_tools", effort=level)
    assert stub.calls[-1]["thinking"] == {"type": "between_tools"}
    assert stub.calls[-1]["output_config"]["effort"] == "high"
    assert len(stub.calls) == 1


def test_unsatisfiable_effort_combination_fails_before_sdk_without_removing_search():
    p, stub = _openai(dict(EFFORT_ONLY, effort_levels=["minimal"]))
    p.model = "gpt-5"
    with pytest.raises(AIProviderError, match="no supported effort level compatible"):
        p.generate("q", effort="minimal", web_search=True)
    assert stub.calls == []


@pytest.mark.parametrize("make", [_openai, _grok, _gemini, _claude])
@pytest.mark.parametrize("json_mode", [False, True])
def test_omitting_effort_does_not_hide_an_always_on_incompatibility(make, json_mode):
    p, stub = make(dict(PRO, incompatible=[{"thinking": "on", "web_search": "on"}]))
    with pytest.raises(AIProviderError, match="incompatible"):
        _generate(p, json_mode, web_search=True)
    assert stub.calls == []
