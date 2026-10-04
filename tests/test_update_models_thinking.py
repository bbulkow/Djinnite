"""
update_models: thinking discovery fixes and the platform block (offline).

* ``effort`` joins ``thinking_style`` whenever the provider enumerates
  ``effort_levels`` -- AFTER the probe merge, which replaces thinking_style
  wholesale (that is how sonnet-5-5 / opus-5-5 / fable-5-1 lost it).
* ``thinking_style`` reaches the combination probe, so it can send a thinking
  shape the model accepts.
* ``between_tools`` is a setting, not evidence that the model thinks.
* A refresh carries each model's ``platforms`` block forward untouched.

    uv run pytest tests/test_update_models_thinking.py -v
"""

import copy

from djinnite.config_loader import AIConfig
from djinnite.scripts.update_models import _ensure_effort_style, merge_model_data


def test_ensure_effort_style():
    caps = {"thinking_style": ["adaptive"], "effort_levels": ["low", "high"]}
    _ensure_effort_style(caps)
    assert caps["thinking_style"] == ["adaptive", "effort"]
    _ensure_effort_style(caps)  # idempotent
    assert caps["thinking_style"] == ["adaptive", "effort"]


def test_ensure_effort_style_leaves_unknown_and_effortless_alone():
    unknown = {"thinking_style": None, "effort_levels": ["low"]}
    _ensure_effort_style(unknown)
    assert unknown["thinking_style"] is None
    no_effort = {"thinking_style": ["budget"], "effort_levels": None}
    _ensure_effort_style(no_effort)
    assert no_effort["thinking_style"] == ["budget"]
    _ensure_effort_style(None)  # tolerated


class _FakeClaude:
    """A provider whose probes answer like Sonnet 5.5."""
    PROVIDER_NAME = "claude"
    seen_states = None

    def __init__(self, api_key=None, model=None, **kw):
        self.model = model

    def discover_modalities(self, model_id):
        return {"input": ["text"], "output": ["text"]}

    def probe_structured_json(self):
        return True

    def probe_temperature(self):
        return False

    def probe_json_with_search(self):
        return True

    def probe_web_search(self):
        return True

    def probe_thinking_style(self):
        return ["adaptive", "between_tools"]

    def probe_thinking_disable(self):
        return False  # disabled is a 400

    def probe_incompatible_combinations(self, supported_states):
        _FakeClaude.seen_states = dict(supported_states)
        return []


PLATFORMS = {"vertexai": {"locations": {"global": "available", "us": "no_quota"},
                          "probed": "2026-10-04"}}


def _run():
    model_id = "claude-test-55"
    new = [{"id": model_id, "name": "Test 5.5", "context_window": 1_000_000,
            "max_output_tokens": 128_000, "effort_levels": ["low", "medium", "high"]}]
    existing = [{
        "id": model_id, "name": "Test 5.5", "context_window": 1_000_000,
        "max_output_tokens": 128_000,
        "modalities": {"input": ["text"], "output": ["text"]},
        "costing": {"input_per_1m": 2.0, "output_per_1m": 10.0, "source": "published"},
        "capabilities": {"thinking": ["on", "off"], "thinking_style": ["adaptive"],
                         "incompatible": [{"thinking": "on", "structured_json": "on"}]},
        "platforms": copy.deepcopy(PLATFORMS),
    }]
    merged = merge_model_data(new, existing, _FakeClaude(), _FakeClaude, "k", AIConfig(),
                              reprobe={model_id})
    return merged[0]


def test_reprobe_corrects_55_style_thinking():
    model = _run()
    caps = model["capabilities"]
    assert caps["thinking"] == ["on"]  # cannot be turned off
    assert caps["thinking_style"] == ["adaptive", "between_tools", "effort"]
    assert caps["incompatible"] == []
    assert _FakeClaude.seen_states["thinking_style"] == ["adaptive", "between_tools"]


def test_refresh_preserves_the_platforms_block():
    assert _run()["platforms"] == PLATFORMS
