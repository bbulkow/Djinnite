"""
Claude 5.x thinking semantics against the Anthropic API (opt-in: ``--live``).

These are properties of the *model*, so they are verified in direct mode,
cheaply, with the Anthropic key -- no Vertex quota needed. The platform tier
(``test_e2e_vertexai_claude.py``) verifies Djinnite delivers them through
Vertex. Rejected requests (the ``disabled`` 400s) are not billed.

Providers here are built WITHOUT a catalog entry, so each request reaches the
API whatever the catalog currently says: the API is what is under test.

    uv run pytest tests/test_live_claude_thinking.py --live -rA
"""

import pytest

from djinnite.ai_providers.base_provider import AIProviderError
from djinnite.ai_providers.claude_provider import ClaudeProvider
from djinnite.tests import _contract as contract

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def claude_key(ai_config):
    if not ai_config.is_usable("claude") or ai_config.providers["claude"].mode != "direct":
        pytest.skip("no direct-mode Claude API key in ai_config.json")
    return ai_config.providers["claude"].api_key


def _claude(key, model):
    return ClaudeProvider(api_key=key, model=model, require_pricing=False)


@pytest.mark.parametrize("model", ["claude-sonnet-5-5", "claude-opus-5-5"])
def test_disabled_thinking_is_rejected(claude_key, model):
    with pytest.raises(AIProviderError) as ei:
        _claude(claude_key, model).generate("hi", max_output_tokens=64, thinking=False)
    assert "disabled" in str(ei.value).lower(), str(ei.value)


def test_between_tools_is_accepted_on_sonnet_55(claude_key):
    r = contract.check_generate_json(_claude(claude_key, "claude-sonnet-5-5"),
                                     thinking="between_tools")
    assert r.content


def test_between_tools_is_rejected_on_opus_55(claude_key):
    with pytest.raises(AIProviderError) as ei:
        _claude(claude_key, "claude-opus-5-5").generate(
            "hi", max_output_tokens=64, thinking="between_tools")
    assert "between_tools" in str(ei.value), str(ei.value)


def test_thinking_tokens_reported_by_the_api(claude_key):
    """output_tokens_details.thinking_tokens arrives in direct mode."""
    r = _claude(claude_key, "claude-sonnet-5-5").generate_json(
        "What is 17 * 23? Show no working; JSON only.",
        {"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"]},
        max_output_tokens=contract.MAX_OUT)
    assert r.thinking_tokens is not None, r.usage
    assert r.thinking_tokens <= r.output_tokens  # a subset, never added on top


def test_disabled_really_turns_thinking_off_on_sonnet_5(claude_key):
    r = contract.check_generate_json(_claude(claude_key, "claude-sonnet-5"), thinking=False)
    assert r.thinking_tokens == 0, r.usage
