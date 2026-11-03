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
        with pytest.raises(ValueError, match="thinking=int"):
            p._resolve_thinking(2048)
        return
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
    """Error messages must not recommend a shape the model also rejects.

    Uses a synthetic adaptive-only model: the catalog's adaptive-only rows
    were an artifact of missing "effort" styles and disappear on a reprobe,
    and this test must not quietly start skipping when they do.
    """
    from djinnite.ai_providers.claude_provider import ClaudeProvider
    from djinnite.tests._stubs import bare, info
    p = bare(ClaudeProvider, client=None,
             model_info=info(thinking=["on", "off"], thinking_style=["adaptive"]))

    with pytest.raises(ValueError) as int_err:
        p._resolve_thinking(2048)
    err = str(int_err.value)
    assert "thinking=True" in err
    # The old messages pointed at each other's shape.
    assert "Pass an int token budget" not in err
    assert "Pass a string" not in err
    # A level string is no longer a thinking setting at all.
    with pytest.raises(ValueError, match="effort="):
        p._resolve_thinking("high")


# ------------------------------------------------------------------
# Effort levels (output_config.effort)
#
# Claude carries reasoning effort in `output_config.effort`, NOT in the
# thinking block. Its own vocabulary has no "minimal"; request_policy.json
# maps Djinnite's "minimal" to "low". Levels also vary per model: Opus 4.5
# stops at "high" while Opus 5 accepts "max", so a Djinnite level is
# stepped to the nearest level the catalog says the model accepts.
# ------------------------------------------------------------------

from djinnite.config_loader import EFFORT_LEVEL_VALUES, load_request_policy

EFFORT_MODELS = [
    m.id for m in THINKING_MODELS
    if m.capabilities.thinking_style and "effort" in m.capabilities.thinking_style
]


def test_claude_vocabulary_maps_minimal_to_low():
    """Claude rejects 'minimal'; the configured vocabulary steps it down."""
    assert load_request_policy().vocabulary("claude")["minimal"] == "low"
    p = _provider(EFFORT_MODELS[0])
    assert p._map_effort("minimal") == "low"


@pytest.mark.parametrize("model_id", EFFORT_MODELS)
def test_effort_models_declare_their_levels(model_id):
    """A model advertising 'effort' must say which levels it takes."""
    levels = CATALOG.get_model("claude", model_id).capabilities.effort_levels
    assert levels, f"{model_id} advertises effort but declares no effort_levels"
    assert set(levels) <= set(EFFORT_LEVEL_VALUES)


@pytest.mark.parametrize("model_id", EFFORT_MODELS)
def test_declared_effort_levels_are_sent_as_is(model_id):
    """Every declared level maps to itself and never becomes a thinking block."""
    p = _provider(model_id)
    for level in CATALOG.get_model("claude", model_id).capabilities.effort_levels:
        assert p._resolve_effort(None, level) == (None, level)
        assert p._build_claude_thinking(None, 8192) is None


@pytest.mark.parametrize("model_id", EFFORT_MODELS)
def test_undeclared_effort_levels_step_to_the_nearest_declared(model_id):
    """A level this model lacks is mapped locally, never sent as an API 400."""
    p = _provider(model_id)
    declared = CATALOG.get_model("claude", model_id).capabilities.effort_levels
    missing = [lvl for lvl in EFFORT_LEVEL_VALUES if lvl not in declared]
    if not missing:
        pytest.skip(f"{model_id} accepts every effort level")
    ladder = list(EFFORT_LEVEL_VALUES)
    for level in missing:
        sent = p._map_effort(level)
        assert sent in declared
        want = ladder.index("low" if level == "minimal" else level)
        nearest = min(abs(ladder.index(d) - want) for d in declared)
        assert abs(ladder.index(sent) - want) == nearest


def test_non_effort_models_disregard_effort():
    """A budget-only model carries no effort: the level is dropped, not
    converted into a token budget, and no error is raised."""
    no_effort = [m.id for m in THINKING_MODELS if m.id not in EFFORT_MODELS]
    if not no_effort:
        pytest.skip("every thinking-capable Claude supports effort")
    p = _provider(no_effort[0])
    assert p._resolve_effort(None, "high") == (None, None)
    assert p._resolve_effort(True, "high") == (True, None)


def test_context_window_matches_declared_output_cap():
    """Sanity: context window must exceed the output cap for every model."""
    for info in THINKING_MODELS:
        assert info.context_window > info.max_output_tokens, info.id
