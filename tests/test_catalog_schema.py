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

import json
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
        and not info.disabled
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


# ---------------------------------------------------------------------------
# The estimator's output filter must agree with the invariant above.
#
# test_context_window_exceeds_output_cap rejects ctx <= max_out, but the
# writer's filter accepted ctx >= max_out. The two boundaries differed by
# exactly the equal case, so an estimator that answered 128000/128000 for
# gpt-6-astra passed the writer, reached the catalog, and failed the suite.
# The batch degradation that produced those twin numbers is a separate,
# known failure mode; this guards the filter that should have caught it.
# ---------------------------------------------------------------------------

def test_equal_context_and_output_estimate_is_rejected():
    """ctx == max_out leaves zero room for a prompt, so it is not a fact."""
    vals = update_models.sanitize_estimated_limits(
        {"id": "gpt-6-astra", "max_output_tokens": 128000, "context_window": 128000}
    )
    assert "context_window" not in vals, (
        "an estimate whose context window equals its output cap must be "
        "dropped, not written to the catalog"
    )
    assert vals["max_output_tokens"] == 128000, (
        "the output cap is independently usable and should survive"
    )


def test_context_below_output_estimate_is_rejected():
    vals = update_models.sanitize_estimated_limits(
        {"id": "m", "max_output_tokens": 128000, "context_window": 64000}
    )
    assert "context_window" not in vals


def test_plausible_estimate_survives():
    vals = update_models.sanitize_estimated_limits(
        {"id": "m", "max_output_tokens": 128000, "context_window": 1050000}
    )
    assert vals == {"max_output_tokens": 128000, "context_window": 1050000}


def test_context_window_alone_is_kept_without_an_output_cap():
    """With no output cap to contradict, a positive context window stands."""
    vals = update_models.sanitize_estimated_limits(
        {"id": "m", "max_output_tokens": 0, "context_window": 200000}
    )
    assert vals == {"context_window": 200000}


@pytest.mark.parametrize("bad", [0, -1, None, "128000", True])
def test_non_positive_or_non_numeric_limits_are_dropped(bad):
    """Zero is the estimator's own "unsure" answer and must never be stored."""
    vals = update_models.sanitize_estimated_limits(
        {"id": "m", "max_output_tokens": bad, "context_window": bad}
    )
    assert vals == {}


def test_estimator_filter_matches_the_catalog_invariant():
    """Anything the filter accepts must satisfy the schema test above.

    This is the property the two boundaries were supposed to share. If either
    side is edited to disagree again, this fails rather than a live refresh.
    """
    for max_out, ctx in [
        (128000, 128000), (128000, 127999), (128000, 128001),
        (65536, 65536), (65536, 1048576), (32768, 32767),
    ]:
        vals = update_models.sanitize_estimated_limits(
            {"id": "m", "max_output_tokens": max_out, "context_window": ctx}
        )
        if "context_window" in vals:
            assert vals["context_window"] > vals["max_output_tokens"], (
                f"filter accepted ctx={ctx} with max_out={max_out}, which "
                f"test_context_window_exceeds_output_cap would reject"
            )


def test_new_model_does_not_inherit_neighbours_context_window():
    """A new model the API reports no limits for must not take the previous
    model's context_window. gpt-6-sol landed at gpt-4o's 128000 that way."""
    class _Stub:
        PROVIDER_NAME = "chatgpt"

        def discover_modalities(self, model_id):
            return {"input": ["text"], "output": ["text"]}

    class _NoEstimator:
        default_provider = "none"

        def get_provider(self, name):
            return None

    existing = [{"id": "known-model", "context_window": 128000,
                 "max_output_tokens": 16384}]
    listed = [{"id": "known-model"}, {"id": "brand-new-model"}]
    merged = update_models.merge_model_data(
        listed, existing, _Stub(), None, "x", _NoEstimator(),
        reprobe={"no-such-model"},  # restrictive scope: no live probes
    )
    new = next(m for m in merged if m["id"] == "brand-new-model")
    assert not new.get("context_window"), (
        f"new model inherited context_window={new.get('context_window')}"
    )


# ---------------------------------------------------------------------------
# Human decisions: one file, one write path.
#
# model_overrides.json is the only file a human edits for per-model facts;
# model_catalog.json is generated. The invariant that keeps that true is that
# every catalog write goes through model_overrides.save_catalog(), so a refresh
# cannot land with human decisions missing.
#
# The failure this replaces: disable state lived in disabled_models.json AND in
# the catalog, runtime read only the catalog, and the maintenance command
# re-enabled anything absent from the file. Seven models carried reasons
# recorded only in the catalog and were one command away from being silently
# re-enabled.
#
# These run offline against the shipped files.
# ---------------------------------------------------------------------------

from djinnite.scripts import model_overrides as mo


def _overrides():
    return mo.load_overrides()


def test_catalog_matches_the_override_file():
    """Every overridden field in the file must actually be in the catalog."""
    overrides = _overrides()
    catalog = load_model_catalog()

    wrong = []
    for prov in ("gemini", "claude", "chatgpt"):
        for info in catalog.list_models(prov):
            entry = mo.lookup(overrides, prov, info.id)
            if not entry:
                continue
            for path, want in mo.flatten_entry(entry).items():
                if path == "disabled":
                    got = info.disabled
                elif path == "disabled_reason":
                    got = info.disabled_reason
                else:
                    continue
                if got != want:
                    wrong.append(f"{prov}/{info.id}.{path}: catalog={got!r} file={want!r}")
    assert not wrong, (
        "catalog does not match model_overrides.json -- run "
        "`uv run python -m djinnite.scripts.apply_overrides`: " + "; ".join(wrong)
    )


def test_no_disabled_model_lacks_an_override_entry():
    """A disable with no entry in the file is exactly the old drift bug."""
    overrides = _overrides()
    catalog = load_model_catalog()
    orphaned = [
        f"{prov}/{info.id}"
        for prov in ("gemini", "claude", "chatgpt")
        for info in catalog.list_models(prov)
        if info.disabled and not mo.lookup(overrides, prov, info.id).get("disabled")
    ]
    assert not orphaned, (
        "disabled in the catalog with nothing in model_overrides.json backing "
        f"it; the next refresh will re-enable these: {sorted(orphaned)}"
    )


def test_every_write_path_routes_through_save_catalog():
    """Guards the choke point. Bypassing it drops human overrides silently."""
    import inspect
    from djinnite.scripts import update_models, update_model_costs

    offenders = []
    for mod in (update_models, update_model_costs):
        src = inspect.getsource(mod)
        for i, line in enumerate(src.split("\n"), 1):
            if "json.dump(catalog" in line:
                offenders.append(f"{mod.__name__}:{i}: {line.strip()}")
    assert not offenders, (
        "catalog written directly instead of via model_overrides.save_catalog, "
        "so model_overrides.json would not be applied: " + "; ".join(offenders)
    )


# --- unit tests for the override engine ------------------------------------

def test_override_sets_a_field_and_records_what_discovery_said():
    model = {"id": "m", "context_window": 128000}
    mo.apply_to_model(model, {"context_window": 1050000})
    assert model["context_window"] == 1050000
    assert model[mo.PROVENANCE_KEY]["context_window"]["was"] == 128000


def test_override_records_absent_when_discovery_had_no_value():
    model = {"id": "m"}
    mo.apply_to_model(model, {"disabled": True})
    assert model[mo.PROVENANCE_KEY]["disabled"]["was"] == mo._WAS_ABSENT


def test_removing_an_override_restores_the_discovered_value():
    """Deleting a line from the file must take effect without a full refresh."""
    model = {"id": "m", "context_window": 128000}
    mo.apply_to_model(model, {"context_window": 1050000})
    mo.apply_to_model(model, {})
    assert model["context_window"] == 128000, "discovered value was not restored"
    assert mo.PROVENANCE_KEY not in model


def test_removing_an_override_deletes_a_field_discovery_never_set():
    model = {"id": "m"}
    mo.apply_to_model(model, {"disabled": True})
    mo.apply_to_model(model, {})
    assert "disabled" not in model
    assert mo.PROVENANCE_KEY not in model


def test_nested_override_leaves_sibling_fields_alone():
    """costing.input_per_1m must not blow away the discovered source_url."""
    model = {"id": "m", "costing": {"input_per_1m": 1.0, "output_per_1m": 4.0,
                                    "source_url": "https://example/pricing"}}
    mo.apply_to_model(model, {"costing": {"input_per_1m": 2.5}})
    assert model["costing"]["input_per_1m"] == 2.5
    assert model["costing"]["output_per_1m"] == 4.0
    assert model["costing"]["source_url"] == "https://example/pricing"


def test_reapplying_the_same_override_is_idempotent():
    model = {"id": "m", "context_window": 128000}
    mo.apply_to_model(model, {"context_window": 1050000})
    first = json.loads(json.dumps(model))
    mo.apply_to_model(model, {"context_window": 1050000})
    assert model == first


def test_meta_keys_are_notes_not_data():
    model = {"id": "m"}
    mo.apply_to_model(model, {"_note": "why", "disabled": True})
    assert "_note" not in model
    assert model["disabled"] is True


def test_provider_qualified_key_wins_and_bare_key_still_works():
    overrides = {"chatgpt/gpt-x": {"context_window": 1}, "gpt-y": {"context_window": 2}}
    assert mo.lookup(overrides, "chatgpt", "gpt-x") == {"context_window": 1}
    assert mo.lookup(overrides, "chatgpt", "gpt-y") == {"context_window": 2}
    assert mo.lookup(overrides, "chatgpt", "gpt-z") == {}


def test_reapplying_an_unchanged_override_reports_no_change():
    """A no-op run must stay quiet.

    Change detection compares the override against what the CATALOG currently
    holds, not against what discovery originally said. Comparing against
    `discovered` made every single run report all 106 disable fields as
    freshly applied, which buried the one field that actually moved -- the
    opposite of the observability the report exists to provide.
    """
    model = {"id": "m"}
    first = mo.apply_to_model(model, {"disabled": True})
    assert len(first["applied"]) == 1 and not first["noop"]

    second = mo.apply_to_model(model, {"disabled": True})
    assert not second["applied"], (
        "an override already present in the catalog must report as a no-op, "
        f"got {second['applied']}"
    )
    assert len(second["noop"]) == 1
    # Provenance must still say what discovery said, not the human value.
    assert model[mo.PROVENANCE_KEY]["disabled"]["was"] == mo._WAS_ABSENT


def test_noop_run_over_the_real_catalog_is_silent():
    """The shipped catalog and overrides file must already agree."""
    catalog = json.loads(
        (Path(__file__).parent.parent / "config" / "model_catalog.json")
        .read_text(encoding="utf-8")
    )
    stats = mo.apply_overrides(catalog, mo.load_overrides(), verbose=False)
    assert stats["fields"] == 0 and stats["reverted"] == 0, (
        "applying overrides to the shipped catalog changed something, so the "
        f"committed catalog is out of date: {stats}"
    )


def test_disable_gate_guards_every_paid_discovery_path():
    """A disabled model must not reach probing OR the AI estimator.

    These paths cost real money against live APIs. The gate used to be
    computed after the estimation queues were filled, so only capability
    probing respected it -- a refresh still sent disabled models to the
    estimator, and one run spent a web-search call asking for the text output
    limits of ten image and transcription models that have none, getting
    `Got limits for 0/10` back.

    Checked with the AST rather than string matching: an earlier version of
    this test looked for the guard text anywhere in the function and passed
    while both queues were unguarded, because it matched the probe line
    instead. Every append must be lexically INSIDE a conditional that tests
    is_disabled.
    """
    import ast
    import inspect
    import textwrap
    from djinnite.scripts import update_models

    tree = ast.parse(textwrap.dedent(inspect.getsource(update_models.merge_model_data)))

    QUEUES = {"uncertain_models", "unknown_output_limit_models"}

    def mentions_gate(node) -> bool:
        return any(
            isinstance(n, ast.Name) and n.id == "is_disabled"
            for n in ast.walk(node)
        )

    def appends_in(node):
        """Yield queue names appended to anywhere under this node."""
        for n in ast.walk(node):
            if (isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "append"
                    and isinstance(n.func.value, ast.Name)
                    and n.func.value.id in QUEUES):
                yield n.func.value.id

    guarded, seen = set(), set()
    for node in ast.walk(tree):
        for q in appends_in(node):
            seen.add(q)
        if isinstance(node, ast.If) and mentions_gate(node.test):
            for q in appends_in(node):
                guarded.add(q)

    assert seen == QUEUES, f"expected both queues in merge_model_data, found {seen}"
    unguarded = QUEUES - guarded
    assert not unguarded, (
        f"{sorted(unguarded)} appended without an is_disabled guard in the "
        f"enclosing conditional; disabled models would be sent to the paid "
        f"AI estimator"
    )


# ---------------------------------------------------------------------------
# Pricing tiers: the stored number must be the Standard service tier.
#
# A vendor publishes several rates for one model on one page -- Standard, Flex,
# Batch, a long-context tier above N input tokens. ModelCosting holds one pair,
# so the estimator has to pick, and picked differently on consecutive runs:
# gpt-5.4-pro went $30/$180 -> $15/$90 -> $30/$180, reported each time as a
# legitimate price change. $15/$90 is the Flex rate, which Djinnite cannot even
# be billed at -- it never sends service_tier.
#
# See SERVICE_TIER_DESIGN.md for the structural fix that is still open.
# ---------------------------------------------------------------------------

def test_estimator_prompt_pins_the_standard_service_tier():
    """The prompt must name the tiers to reject, not just say 'base tier'.

    The original prompt did say "if a model uses a tiered scheme ... use the
    base tier rate" -- which covers the CONTEXT axis and says nothing about the
    SERVICE axis. That gap is what let Flex through.
    """
    from djinnite.prompts import COST_ANALYST_SYSTEM, COST_ESTIMATION_PROMPT

    prompt = (COST_ESTIMATION_PROMPT + " " + COST_ANALYST_SYSTEM).lower()
    assert "standard" in prompt, "the prompt must name the Standard tier"
    for rejected in ("flex", "batch", "priority"):
        assert rejected in prompt, (
            f"the prompt must explicitly reject the {rejected!r} tier; naming "
            f"only the 'base tier' left the service axis unconstrained"
        )


def test_published_figure_survives_a_catalog_round_trip():
    """The human cross-check field must actually load off disk.

    It was being requested from the estimator and then dropped on the floor --
    which is why a tier swap was invisible.
    """
    from djinnite.config_loader import ModelCosting
    assert "published_figure" in {f.name for f in fields(ModelCosting)}

    catalog = load_model_catalog()
    info = catalog.get_model("chatgpt", "gpt-5.4-pro")
    assert info.costing.published_figure, (
        "gpt-5.4-pro should carry the quoted price text after repricing"
    )
    assert "standard" in info.costing.published_figure.lower(), (
        "the quote must name the tier, or it cannot serve as a cross-check: "
        f"{info.costing.published_figure!r}"
    )


def test_no_priced_model_is_cheaper_than_its_known_standard_rate():
    """Regression pin for the specific model that flip-flopped.

    $15/$90 is gpt-5.4-pro's Flex rate. Djinnite never sends service_tier, so
    being billed at Flex is impossible and storing it guarantees under-reported
    cost.
    """
    info = load_model_catalog().get_model("chatgpt", "gpt-5.4-pro")
    assert info.costing.input_per_1m == 30.0, (
        f"expected the Standard input rate of $30/1M, got "
        f"${info.costing.input_per_1m} -- $15 is the Flex rate"
    )
    assert info.costing.output_per_1m == 180.0
