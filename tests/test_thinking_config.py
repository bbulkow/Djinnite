"""
Tests for Claude thinking-block construction.

Claude has two mutually exclusive thinking shapes, and mixing them is a
400 from the API:

    {"type": "adaptive"}                    -- model sizes its own reasoning,
                                               accepts NO budget_tokens
    {"type": "enabled", "budget_tokens": N} -- explicit budget,
                                               1024 <= N < max_tokens

Three defects lived here undetected because nothing exercised the builder:

1. ``_get_max_thinking_budget`` returned the catalog's full
   ``max_output_tokens`` while ``_resolve_max_output_tokens`` returned the
   same number, so ``budget >= max_output_tokens`` always tripped and
   ``thinking=True`` raised on every Claude model.
2. The builder attached ``budget_tokens`` to adaptive blocks, which the API
   rejects outright.
3. The int/str rejection messages each recommended the other's shape, so an
   adaptive-only model told callers to pass a string, then to pass an int,
   and accepted neither.

These run offline against the real catalog -- no keys, no network. Building
an Anthropic client does not make a request, so a dummy key is fine.
"""

import sys
from pathlib import Path

import pytest

_project_root = str(Path(__file__).parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from djinnite.config_loader import load_model_catalog
from djinnite.ai_providers import get_provider


CATALOG = load_model_catalog()


def _thinking_models():
    """Every Claude model the catalog says can think."""
    out = []
    for info in CATALOG.list_models("claude"):
        caps = info.capabilities
        if caps and caps.thinking and "on" in caps.thinking:
            out.append(info)
    return out


THINKING_MODELS = _thinking_models()
MODEL_IDS = [m.id for m in THINKING_MODELS]


def _provider(model_id):
    return get_provider("claude", api_key="sk-ant-dummy-not-used", model=model_id)


def test_catalog_has_thinking_models():
    """Guard: the parametrized tests below are meaningless if this is empty."""
    assert THINKING_MODELS, "no thinking-capable Claude models in the catalog"


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_thinking_true_never_raises(model_id):
    """thinking=True must work on every thinking-capable model.

    This is the regression: it used to raise for all of them.
    """
    p = _provider(model_id)
    cap = p._resolve_max_output_tokens(None) or 8192
    block = p._build_claude_thinking(True, cap)
    assert block is not None
    assert block["type"] in ("adaptive", "enabled")


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_block_shape_matches_catalog_style(model_id):
    """Adaptive blocks carry no budget; enabled blocks carry a legal one."""
    p = _provider(model_id)
    cap = p._resolve_max_output_tokens(None) or 8192
    block = p._build_claude_thinking(True, cap)
    styles = CATALOG.get_model("claude", model_id).capabilities.thinking_style

    if block["type"] == "adaptive":
        assert "adaptive" in styles
        # The API rejects budget_tokens on adaptive with
        # "thinking.adaptive.budget_tokens: Extra inputs are not permitted".
        assert "budget_tokens" not in block
    else:
        assert "budget" in styles
        assert block["budget_tokens"] >= p._MIN_THINKING_BUDGET
        assert block["budget_tokens"] < cap


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_explicit_int_selects_budget_shape(model_id):
    """An int budget must produce `enabled`, never `adaptive`.

    Models advertising both styles used to prefer adaptive and then attach
    the caller's budget_tokens to it -- a guaranteed 400.
    """
    p = _provider(model_id)
    styles = CATALOG.get_model("claude", model_id).capabilities.thinking_style
    if "budget" not in styles:
        pytest.skip(f"{model_id} does not accept an int budget")
    block = p._build_claude_thinking(2048, 8192)
    assert block == {"type": "enabled", "budget_tokens": 2048}


def test_budget_reserves_room_for_output():
    """The computed budget must leave headroom, not consume the whole cap."""
    budget_models = [
        m.id for m in THINKING_MODELS
        if "budget" in m.capabilities.thinking_style
        and "adaptive" not in m.capabilities.thinking_style
    ]
    if not budget_models:
        pytest.skip("no budget-only Claude models in the catalog")
    p = _provider(budget_models[0])
    cap = 64000
    budget = p._get_max_thinking_budget(cap)
    assert budget < cap, "budget must leave room for the visible response"
    assert budget == int(cap * p._THINKING_BUDGET_FRACTION)


def test_tiny_cap_raises_actionable_error():
    """A cap too small for a legal budget explains itself."""
    p = _provider(MODEL_IDS[0])
    with pytest.raises(ValueError, match="too small for thinking=True"):
        p._get_max_thinking_budget(100)


def test_rejection_message_only_offers_accepted_forms():
    """Error messages must not recommend a shape the model also rejects."""
    adaptive_only = [
        m.id for m in THINKING_MODELS
        if m.capabilities.thinking_style == ["adaptive"]
    ]
    if not adaptive_only:
        pytest.skip("no adaptive-only Claude models in the catalog")
    p = _provider(adaptive_only[0])

    with pytest.raises(ValueError) as int_err:
        p._resolve_thinking(2048)
    with pytest.raises(ValueError) as str_err:
        p._resolve_thinking("high")

    for err in (str(int_err.value), str(str_err.value)):
        assert "thinking=True" in err
        # The old messages pointed at each other's shape.
        assert "Pass an int token budget" not in err
        assert "Pass a string" not in err


# ------------------------------------------------------------------
# Effort levels (output_config.effort)
#
# Claude carries reasoning effort in `output_config.effort`, NOT in the
# thinking block, and its vocabulary is low/medium/high/xhigh/max -- there
# is no "minimal". Levels also vary per model: Opus 4.5 stops at "high"
# while Opus 5 accepts "max", so the provider-wide set cannot pre-flight a
# request on its own.
# ------------------------------------------------------------------

EFFORT_MODELS = [
    m.id for m in THINKING_MODELS
    if m.capabilities.thinking_style and "effort" in m.capabilities.thinking_style
]


def test_claude_effort_vocabulary_excludes_minimal():
    """Claude rejects 'minimal'; the shared cross-provider set includes it."""
    p = _provider(MODEL_IDS[0])
    assert "minimal" not in p._EFFORT_LEVELS
    assert {"low", "medium", "high", "xhigh", "max"} == set(p._EFFORT_LEVELS)


@pytest.mark.parametrize("model_id", EFFORT_MODELS)
def test_effort_models_declare_their_levels(model_id):
    """A model advertising 'effort' must say which levels it takes."""
    levels = CATALOG.get_model("claude", model_id).capabilities.effort_levels
    assert levels, f"{model_id} advertises effort but declares no effort_levels"
    assert set(levels) <= set(_provider(model_id)._EFFORT_LEVELS)


@pytest.mark.parametrize("model_id", EFFORT_MODELS)
def test_declared_effort_levels_are_accepted(model_id):
    """Every declared level passes pre-flight and lands in output_config."""
    p = _provider(model_id)
    for level in CATALOG.get_model("claude", model_id).capabilities.effort_levels:
        assert p._resolve_thinking(level) == level
        assert p._build_claude_effort(level) == level
        # Effort is not a thinking block -- it must not become one.
        assert p._build_claude_thinking(level, 8192) is None


@pytest.mark.parametrize("model_id", EFFORT_MODELS)
def test_undeclared_effort_levels_rejected_locally(model_id):
    """A level this model lacks must fail locally, not as an API 400."""
    p = _provider(model_id)
    declared = set(CATALOG.get_model("claude", model_id).capabilities.effort_levels)
    missing = set(p._EFFORT_LEVELS) - declared
    if not missing:
        pytest.skip(f"{model_id} accepts every Claude effort level")
    with pytest.raises(ValueError, match="effort level"):
        p._resolve_thinking(sorted(missing)[0])


def test_non_effort_models_reject_effort():
    """Models without effort support reject a level with an accurate message."""
    no_effort = [m.id for m in THINKING_MODELS if m.id not in EFFORT_MODELS]
    if not no_effort:
        pytest.skip("every thinking-capable Claude supports effort")
    p = _provider(no_effort[0])
    with pytest.raises(ValueError, match="not supported by model"):
        p._resolve_thinking("high")


def test_context_window_matches_declared_output_cap():
    """Sanity: context window must exceed the output cap for every model."""
    for info in THINKING_MODELS:
        assert info.context_window > info.max_output_tokens, info.id
