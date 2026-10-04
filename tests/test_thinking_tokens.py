"""
Thinking-token accounting (offline).

* Claude reports ``usage.output_tokens_details.thinking_tokens``, a SUBSET of
  ``output_tokens`` (the inclusive billing total). Djinnite reports it as
  ``thinking_tokens`` and bills it once, inside ``output_tokens``. A response
  that does not report it gives ``None`` -- unknown, never 0.
* Gemini reports ``thoughts_token_count`` SEPARATELY from
  ``candidates_token_count``; it is billed at the output rate on top.

    uv run pytest tests/test_thinking_tokens.py -v
"""

import pytest

from djinnite.ai_providers.claude_provider import ClaudeProvider
from djinnite.ai_providers.gemini_provider import GeminiProvider
from djinnite.tests._stubs import bare, StubAnthropic, claude_response, StubGenai, gemini_response

SCHEMA = {"type": "object", "properties": {"v": {"type": "integer"}}, "required": ["v"]}


def _claude(*responses):
    return bare(ClaudeProvider, StubAnthropic(list(responses)))


@pytest.mark.parametrize("json_mode", [False, True])
def test_claude_reports_thinking_from_output_tokens_details(json_mode):
    p = _claude(claude_response(input_tokens=1000, output_tokens=500, thinking_tokens=300))
    r = p.generate_json("q", schema=SCHEMA, force=True) if json_mode else p.generate("q")
    assert r.thinking_tokens == 300
    assert r.output_tokens == 500          # inclusive of the 300 thinking tokens
    assert r.total_tokens == 1500
    assert r.usage["_thinking_billed_separately"] is False
    # $1 / $2 per 1M: thinking billed once, inside output_tokens.
    assert r.usage["token_cost"] == pytest.approx(1000 * 1e-6 + 500 * 2e-6)


def test_claude_unreported_thinking_is_none_not_zero():
    p = _claude(claude_response(output_tokens=40, report_details=False))
    assert p.generate("q").thinking_tokens is None


def test_claude_reported_zero_stays_zero():
    p = _claude(claude_response(thinking_tokens=0))
    assert p.generate("q").thinking_tokens == 0


def test_claude_thinking_sums_over_continuation_turns():
    p = _claude(claude_response(stop="pause_turn", thinking_tokens=10),
                claude_response(thinking_tokens=5))
    r = p.generate("q")
    assert r.thinking_tokens == 15


def test_claude_any_unreported_turn_makes_the_total_unknown():
    p = _claude(claude_response(stop="pause_turn", thinking_tokens=10),
                claude_response(report_details=False))
    assert p.generate("q").thinking_tokens is None


@pytest.mark.parametrize("json_mode", [False, True])
def test_gemini_bills_thoughts_on_top_of_candidates(json_mode):
    p = bare(GeminiProvider, StubGenai(gemini_response(prompt=11, candidates=7, thoughts=50)))
    r = p.generate_json("q", schema=SCHEMA, force=True) if json_mode else p.generate("q")
    assert r.output_tokens == 7 and r.thinking_tokens == 50
    assert r.total_tokens == 68
    assert r.usage["_thinking_billed_separately"] is True
    assert r.usage["token_cost"] == pytest.approx(11 * 1e-6 + (7 + 50) * 2e-6)


def test_gemini_without_thinking_cost_is_unchanged():
    p = bare(GeminiProvider, StubGenai(gemini_response(prompt=11, candidates=7, thoughts=None)))
    r = p.generate("q")
    assert r.thinking_tokens is None
    assert r.usage["token_cost"] == pytest.approx(11 * 1e-6 + 7 * 2e-6)
