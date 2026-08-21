"""
Guards against the catalog schema and its writer drifting apart.

``update_models.py`` REPLACES each model's ``capabilities`` dict on every
refresh. When that dict was built from a hand-maintained key literal, any
capability the literal forgot was silently erased -- which is exactly what
happened to ``effort_levels``: a refresh wiped it from all eight
effort-capable Claude models while leaving ``thinking_style`` claiming the
models still supported effort.

Nothing failed loudly. The catalog just quietly lost a field.

These run offline -- no keys, no network, no catalog rewrite.
"""

import sys
from dataclasses import fields
from pathlib import Path

import pytest

_project_root = str(Path(__file__).parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from djinnite.config_loader import (
    EFFORT_LEVEL_VALUES, ModelCapabilities, load_model_catalog,
)
from djinnite.scripts import update_models


# Text-generation models only. On TTS / image / audio models the
# context window and the output cap count different media, so
# comparing them is meaningless. Mirrors update_models' own list.
_SPECIALIZED = (
    "tts", "embedding", "realtime", "image", "transcribe",
    "audio", "robotics", "computer-use",
)


def _is_specialized(model_id: str) -> bool:
    return any(x in model_id.lower() for x in _SPECIALIZED)


def test_writer_covers_every_capability_field():
    """update_models must round-trip every field the dataclass declares."""
    declared = {f.name for f in fields(ModelCapabilities)}
    written = set(update_models._CAPABILITY_FIELDS)
    missing = declared - written
    assert not missing, (
        f"update_models would silently drop {sorted(missing)} from the "
        f"catalog on the next refresh"
    )


def test_capability_fields_are_derived_not_hardcoded():
    """The key set must come from the dataclass, so it cannot fall behind."""
    assert set(update_models._CAPABILITY_FIELDS) == {
        f.name for f in fields(ModelCapabilities)
    }


@pytest.mark.parametrize("prov", ["gemini", "claude", "chatgpt"])
def test_effort_capability_is_internally_consistent(prov):
    """A model claiming 'effort' must carry the levels it accepts.

    This is the assertion that catches a refresh having wiped
    ``effort_levels`` while leaving ``thinking_style`` advertising effort.
    """
    catalog = load_model_catalog()
    # `None` is a documented state meaning "unknown, skip pre-flight", so a
    # model advertising effort without levels is under-specified, not
    # invalid. What must hold: a populated list is never empty and never
    # contains a token outside the shared vocabulary.
    bad = []
    for info in catalog.list_models(prov):
        levels = info.capabilities.effort_levels
        if levels is None:
            continue
        if not levels or not set(levels) <= set(EFFORT_LEVEL_VALUES):
            bad.append((info.id, levels))
    assert not bad, f"{prov}: malformed effort_levels: {bad}"


def test_claude_effort_levels_are_populated():
    """Anthropic's models endpoint enumerates effort levels, so on Claude a
    missing list means the writer dropped the field again.

    This is the direct guard for the `update_models` regression that wiped
    `effort_levels` from all eight effort-capable Claude models.
    """
    catalog = load_model_catalog()
    missing = [
        info.id for info in catalog.list_models("claude")
        if "effort" in (info.capabilities.thinking_style or [])
        and not info.capabilities.effort_levels
    ]
    assert not missing, (
        f"claude models advertise effort but lost their levels: {missing}"
    )


@pytest.mark.parametrize("prov", ["gemini", "claude", "chatgpt"])
def test_context_window_exceeds_output_cap(prov):
    """A context window smaller than the output cap is stale or wrong.

    A hardcoded 200000 in ClaudeProvider.list_models() wrote the wrong
    context window for every 1M-context model; this is the cheap invariant
    that would have surfaced it.
    """
    catalog = load_model_catalog()
    bad = [
        info.id for info in catalog.list_models(prov)
        if info.max_output_tokens and info.context_window
        and info.context_window <= info.max_output_tokens
        and not _is_specialized(info.id)
    ]
    assert not bad, f"{prov}: context_window <= max_output_tokens for {bad}"


@pytest.mark.parametrize("prov", ["gemini", "claude", "chatgpt"])
def test_text_models_declare_a_context_window(prov):
    """A text model with no context window is a gap, not a pass.

    `test_context_window_exceeds_output_cap` skips falsy values, so a model
    sitting at 0 satisfies it vacuously. That is how a missing number hides,
    the same way a wrong one did: 30 GPT-5-family models lost their context
    window when the invented 128000 was removed and the AI estimator -- fed
    38 models in one request -- silently answered 0 for all of them.
    """
    catalog = load_model_catalog()
    missing = [
        info.id for info in catalog.list_models(prov)
        if not _is_specialized(info.id)
        and not info.disabled
        and info.max_output_tokens          # a model we otherwise know about
        and not info.context_window
    ]
    assert not missing, f"{prov}: no context_window for {missing}"
