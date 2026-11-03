"""Focused vendor acceptance tests for effort normalization; opt in with --live.

Uses existing direct-mode entries. Each case makes one small request with no
invented token cap. The offline suite checks exact outgoing request settings;
these tests check that the vendor accepts them and returns usable content.
"""

import json
from dataclasses import replace

import pytest

from djinnite.config_loader import load_request_policy


@pytest.mark.live
@pytest.mark.parametrize("ptype, model, kwargs, json_mode, null_default", [
    ("gemini", "gemini-3.8-flash", {"effort": "minimal"}, True, False),
    ("claude", "claude-sonnet-5-5", {"thinking": "between_tools", "effort": "max"}, True, False),
    ("chatgpt", "gpt-5", {"effort": "minimal", "web_search": True}, False, False),
    ("chatgpt", "gpt-5.4", {"thinking": True}, False, True),
])
def test_effort_request_is_accepted(ai_config, ptype, model, kwargs, json_mode, null_default):
    entry = ai_config.direct_entry(ptype)
    if entry is None:
        pytest.skip(f"no direct-mode entry for {ptype}")
    p = ai_config.build_provider(entry, model)
    if null_default:
        policy = load_request_policy()
        p._policy_cache = replace(policy, defaults=dict(policy.defaults, effort=None),
                                  provider_defaults={})
    if json_mode:
        response = p.generate_json(
            'Return the JSON object {"v": 1}.',
            schema={"type": "object", "properties": {"v": {"type": "integer"}},
                    "required": ["v"]},
            **kwargs,
        )
    else:
        response = p.generate("Reply with the single word: ready", **kwargs)
    print(f"[OK] {ptype}/{model} input={response.input_tokens} "
          f"output={response.output_tokens} thinking={response.thinking_tokens} "
          f"catalog_cost=${response.total_cost:.6f}")
    assert response.content.strip()
    if json_mode:
        assert json.loads(response.content)["v"] == 1
    assert response.input_tokens > 0
    assert response.output_tokens > 0
