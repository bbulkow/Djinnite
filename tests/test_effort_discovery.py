"""
Reasoning-effort discovery on the Responses-API providers (offline).

Three defects found 2026-11-03 (THINKING_EFFORT_DESIGN.md section 1):

* update_models counted only adaptive/budget as evidence of thinking, so an
  effort-only model was written ``thinking: ["off"]``.
* The OpenAI/Grok probes sent ``effort: "low"`` and ``"none"``; models that
  reject both (the ``-pro`` family) were recorded as non-reasoning, and any
  other error (404, 5xx) was recorded as "unsupported" too.
* A confirmed-empty style list was stored as ``None`` ("not probed"), so
  every non-reasoning model was re-probed and billed on every refresh.

The error bodies below are OpenAI's, captured 2026-11-03.

    uv run pytest tests/test_effort_discovery.py -v
"""

import pytest

from djinnite.ai_providers.grok_provider import GrokProvider
from djinnite.ai_providers.openai_provider import OpenAIProvider
from djinnite.config_loader import AIConfig
from djinnite.scripts.update_models import merge_model_data

from djinnite.tests._stubs import bare

PROBE_CLASSES = [OpenAIProvider, GrokProvider]


class _VendorError(Exception):
    """Duck-types the openai SDK's APIStatusError: ``status_code`` and ``body``."""

    def __init__(self, status_code, body):
        super().__init__(str(body))
        self.status_code = status_code
        self.body = body


def _unsupported_value(model, level, supported):
    listed = ", ".join(f"'{s}'" for s in supported[:-1]) + f", and '{supported[-1]}'"
    return _VendorError(400, {
        "message": f"Unsupported value: '{level}' is not supported with the "
                   f"'{model}' model. Supported values are: {listed}.",
        "type": "invalid_request_error", "param": "reasoning.effort",
        "code": "unsupported_value"})


NO_REASONING = _VendorError(400, {
    "message": "Unsupported parameter: 'reasoning.effort' is not supported with this model.",
    "type": "invalid_request_error", "param": "reasoning.effort",
    "code": "unsupported_parameter"})

SHUT_DOWN = _VendorError(404, {
    "message": "The model `gpt-test` has been deprecated.",
    "type": "invalid_request_error", "param": None, "code": "model_not_found"})


class _Responses:
    def __init__(self, accepted, error_for):
        self.accepted, self.error_for, self.sent = accepted, error_for, []

    def create(self, **kw):
        level = kw["reasoning"]["effort"]
        self.sent.append(level)
        if level in self.accepted:
            return object()
        raise self.error_for(level)


class _Client:
    def __init__(self, accepted=(), error_for=None):
        self.responses = _Responses(set(accepted), error_for)


def _ladder_model(supported):
    """A model accepting exactly ``supported`` (OpenAI's own spelling)."""
    return _Client(accepted=supported,
                   error_for=lambda lvl: _unsupported_value("gpt-test", lvl, supported))


@pytest.mark.parametrize("cls", PROBE_CLASSES)
def test_pro_model_is_recorded_as_reasoning_with_its_levels(cls):
    """gpt-5.5-pro: rejects minimal, low and none; accepts medium/high/xhigh."""
    p = bare(cls, _ladder_model(["medium", "high", "xhigh"]))
    assert p.probe_thinking_style() == ["effort"]
    assert p.probed_effort_levels == ["medium", "high", "xhigh"]
    assert p.probed_effort_level == "medium"
    assert p.probe_thinking_disable() is False  # "none" rejected: always on
    request = p._build_combination_probe_request({"thinking": "on"})
    assert request["reasoning"] == {"effort": "medium"}


@pytest.mark.parametrize("cls", PROBE_CLASSES)
def test_none_is_not_an_effort_level(cls):
    """gpt-5.5 lists 'none' among its values; it is the off switch."""
    p = bare(cls, _ladder_model(["none", "low", "medium", "high", "xhigh"]))
    assert p.probe_thinking_style() == ["effort"]
    assert p.probed_effort_levels == ["low", "medium", "high", "xhigh"]
    assert p.probe_thinking_disable() is True


@pytest.mark.parametrize("cls", PROBE_CLASSES)
def test_levels_come_from_the_max_rejection_when_minimal_is_accepted(cls):
    """gpt-5 accepts minimal and rejects max; no third request is needed."""
    client = _ladder_model(["minimal", "low", "medium", "high"])
    p = bare(cls, client)
    assert p.probe_thinking_style() == ["effort"]
    assert p.probed_effort_levels == ["minimal", "low", "medium", "high"]
    assert client.responses.sent == ["minimal", "max"]


@pytest.mark.parametrize("cls", PROBE_CLASSES)
def test_combination_probe_avoids_the_ends_of_the_ladder(cls):
    """OpenAI rejects web search at 'minimal'. Probing combinations there
    recorded {thinking: on, web_search: on} as incompatible on gpt-5."""
    p = bare(cls, _ladder_model(["minimal", "low", "medium", "high"]))
    p.probe_thinking_style()
    request = p._build_combination_probe_request({"thinking": "on", "web_search": "on"})
    assert request["reasoning"] == {"effort": "medium"}


@pytest.mark.parametrize("cls", PROBE_CLASSES)
def test_single_level_model_combines_at_its_only_level(cls):
    """gpt-5-pro accepts only 'high'."""
    client = _ladder_model(["high"])
    p = bare(cls, client)
    assert p.probe_thinking_style() == ["effort"]
    assert p.probed_effort_levels == ["high"]
    assert p.probed_effort_level == "high"
    assert client.responses.sent == ["minimal", "high"]


@pytest.mark.parametrize("cls", PROBE_CLASSES)
def test_model_accepting_both_ends_has_unknown_levels(cls):
    p = bare(cls, _Client(accepted=["minimal", "max", "medium"]))
    assert p.probe_thinking_style() == ["effort"]
    assert p.probed_effort_levels is None
    assert p.probed_effort_level == "medium"


@pytest.mark.parametrize("cls", PROBE_CLASSES)
def test_only_the_vendor_saying_no_reasoning_records_none(cls):
    p = bare(cls, _Client(error_for=lambda lvl: NO_REASONING))
    assert p.probe_thinking_style() == []
    assert p.probe_thinking_disable() is False


@pytest.mark.parametrize("cls", PROBE_CLASSES)
def test_any_other_error_is_inconclusive(cls):
    """A 404 (shut-down model) used to be recorded as 'no reasoning'."""
    p = bare(cls, _Client(error_for=lambda lvl: SHUT_DOWN))
    assert p.probe_thinking_style() is None
    assert p.probe_thinking_disable() is None


# ------------------------------------------------------------ update_models

class _FakeResponsesModel:
    """Probe answers of an OpenAI model; class attributes set per test."""
    PROVIDER_NAME = "chatgpt"
    styles = ["effort"]
    disable = True
    levels = None

    def __init__(self, api_key=None, model=None, **kw):
        self.model = model
        self.probed_effort_levels = None

    def discover_modalities(self, model_id):
        return {"input": ["text"], "output": ["text"]}

    def probe_structured_json(self):
        return True

    def probe_temperature(self):
        return True

    def probe_json_with_search(self):
        return True

    def probe_web_search(self):
        return True

    def probe_thinking_style(self):
        self.probed_effort_levels = type(self).levels
        return type(self).styles

    def probe_thinking_disable(self):
        return type(self).disable

    def probe_incompatible_combinations(self, supported_states):
        return []


def _merge(fake, caps, reprobe=True):
    model_id = "gpt-test"
    new = [{"id": model_id, "name": model_id, "context_window": 1_050_000,
            "max_output_tokens": 128_000}]
    existing = [{"id": model_id, "name": model_id, "context_window": 1_050_000,
                 "max_output_tokens": 128_000,
                 "modalities": {"input": ["text"], "output": ["text"]},
                 "costing": {"input_per_1m": 5.0, "output_per_1m": 30.0,
                             "source": "published"},
                 "capabilities": caps}]
    merged = merge_model_data(new, existing, fake(), fake, "k", AIConfig(),
                              reprobe={model_id} if reprobe else None)
    return merged[0]["capabilities"]


def test_effort_only_model_is_recorded_as_thinking():
    class Fake(_FakeResponsesModel):
        styles, disable = ["effort"], True
    caps = _merge(Fake, {"thinking": ["on", "off"], "thinking_style": ["effort"]})
    assert caps["thinking"] == ["on", "off"]
    assert caps["thinking_style"] == ["effort"]


def test_probed_levels_reach_the_catalog():
    class Fake(_FakeResponsesModel):
        styles, disable, levels = ["effort"], False, ["medium", "high", "xhigh"]
    caps = _merge(Fake, {})
    assert caps["thinking"] == ["on"]
    assert caps["effort_levels"] == ["medium", "high", "xhigh"]


def test_confirmed_no_thinking_is_stored_and_not_reprobed():
    class NoThinking(_FakeResponsesModel):
        styles, disable = [], False
    caps = _merge(NoThinking, {})
    assert caps["thinking"] == ["off"]
    assert caps["thinking_style"] == []

    class MustNotProbe(_FakeResponsesModel):
        def probe_thinking_style(self):
            raise AssertionError("a confirmed-empty style list was re-probed")
    again = _merge(MustNotProbe, dict(caps), reprobe=False)
    assert again["thinking_style"] == []
