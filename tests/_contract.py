"""
The live call-shape contract, written once and run in every access mode.

Each check makes one real call through a provider and asserts what Djinnite
promises: the content, the usage, the cost arithmetic, history, truncation.
They run in direct mode (``tests/test_live_contract.py``, ``--live``) and in
platform mode (``tests/test_e2e_vertexai_*.py``, ``--e2e-platform``), so
platform verification never depends on Vertex alone.

Every check returns the AIResponse (or the partial response of an expected
error) so the caller can record it in a cost ledger.
"""

import json

import pytest

from djinnite.ai_providers.base_provider import AIOutputTruncatedError


SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "country": {"type": "string"},
    },
    "required": ["city", "country"],
}
PROMPT = "Name the capital city of France and its country, as JSON."

HISTORY = [
    {"role": "user", "content": "Remember the number 7. Reply with a short JSON acknowledgement."},
    {"role": "assistant", "content": '{"ack": "noted", "number": 7}'},
    {"role": "user", "content": "Now remember the colour blue as well."},
    {"role": "assistant", "content": '{"ack": "noted", "colour": "blue"}'},
]
HISTORY_SCHEMA = {
    "type": "object",
    "properties": {
        "number": {"type": "integer"},
        "colour": {"type": "string"},
    },
    "required": ["number", "colour"],
}
HISTORY_PROMPT = "Return the number and the colour I asked you to remember, as JSON."

# Room for a thinking model to reason and still answer.
MAX_OUT = 4096


def assert_usage(r, *, multiplier=None):
    """Tokens are reported, and token_cost is exactly what the catalog price says.

    Thinking is billed once: added on top for providers that report it
    separately (Gemini), already inside output_tokens otherwise.
    """
    assert r.input_tokens > 0, r.usage
    assert r.output_tokens > 0 or (r.thinking_tokens or 0) > 0, r.usage
    costing = r_costing(r)
    out = r.output_tokens
    if r.usage.get("_thinking_billed_separately"):
        out += r.thinking_tokens or 0
    expected = (r.input_tokens * costing.input_per_1m + out * costing.output_per_1m) / 1e6
    mult = r.usage.get("price_multiplier", 1.0)
    if multiplier is not None:
        assert mult == pytest.approx(multiplier), r.usage
    assert r.usage["token_cost"] == pytest.approx(expected * mult, rel=1e-6, abs=1e-9), r.usage


def r_costing(r):
    costing = getattr(r, "_costing", None)
    assert costing is not None, "record the provider's costing with attach_costing() first"
    return costing


def attach_costing(r, provider):
    """Remember the catalog price the response was costed against (None: no catalog)."""
    info = provider._model_info
    r._costing = info.costing if info is not None else None
    return r


def check_generate(p):
    r = p.generate("Reply with the single word: ready", max_output_tokens=MAX_OUT)
    assert r.content.strip(), "empty reply"
    return attach_costing(r, p)


def check_generate_json(p, **kwargs):
    r = p.generate_json(PROMPT, SCHEMA, max_output_tokens=MAX_OUT, **kwargs)
    data = json.loads(r.content)
    assert "paris" in data["city"].lower(), data
    assert "france" in data["country"].lower(), data
    return attach_costing(r, p)


def check_history(p):
    """Facts given only in earlier turns come back: the turns reached the model."""
    r = p.generate_json(HISTORY_PROMPT, HISTORY_SCHEMA, max_output_tokens=MAX_OUT,
                        history=HISTORY)
    data = json.loads(r.content)
    assert data["number"] == 7, data
    assert "blue" in data["colour"].lower(), data
    return attach_costing(r, p)


def check_truncation(p, **kwargs):
    """A tiny output cap raises AIOutputTruncatedError carrying billed usage."""
    with pytest.raises(AIOutputTruncatedError) as ei:
        p.generate("Count from 1 to 200, one number per line.", max_output_tokens=16, **kwargs)
    partial = ei.value.partial_response
    assert partial.truncated is True
    assert partial.output_tokens + (partial.thinking_tokens or 0) > 0, partial.usage
    return attach_costing(partial, p)
