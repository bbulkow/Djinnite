"""
Named access paths and ``deny`` through a real Vertex AI build (opt-in: ``--e2e-platform``).

Row A1 of PLATFORM_E2E_TEST_DESIGN.md ("Access paths"), default tier,
**unbilled**: availability is a token count, a denied request never leaves
the process, and the canary's 429 is a rejected request.

A temporary ai_config.json on the e2e project declares two platform entries
with user-chosen names (ACCESS_PATHS_DESIGN.md):

* ``claude-vertex`` -- type ``claude``; denies ``web_search`` for every model
  and ``structured_json`` for the e2e Claude model (Haiku);
* ``gemini-vertex`` -- type ``gemini``; no restrictions.

    uv run pytest tests/test_e2e_access_paths.py --e2e-platform -rA -s
"""

import json

import pytest

from djinnite.ai_providers.base_provider import (
    AIModelNotFoundError, AIRateLimitError, DjinniteCapabilityDeniedError,
)
from djinnite.config_loader import LOCATION_STATUS_VALUES, load_ai_config
from djinnite.tests import _contract as contract
from djinnite.tests._e2e import say

pytestmark = pytest.mark.e2e_platform

CLAUDE_ENTRY = "claude-vertex"
GEMINI_ENTRY = "gemini-vertex"
DENY_REASON = "e2e: declared restriction"


@pytest.fixture(scope="module")
def access_cfg(e2e_session, tmp_path_factory):
    """An AIConfig loaded from a temporary ai_config.json with two named platform entries."""
    cfg = e2e_session
    data = {
        "platforms": {"vertexai": {"project_id": cfg.project, "locations": ["global"]}},
        "providers": {
            CLAUDE_ENTRY: {
                "provider": "claude",
                "mode": "platform", "platform": "vertexai", "location": "global",
                "default_model": cfg.claude_model,
                "deny": {"*": ["web_search"], cfg.claude_model: ["structured_json"]},
                "deny_reason": DENY_REASON,
            },
            GEMINI_ENTRY: {
                "provider": "gemini",
                "mode": "platform", "platform": "vertexai", "location": "global",
                "default_model": cfg.gemini_model,
            },
        },
        "default_provider": CLAUDE_ENTRY,
    }
    path = tmp_path_factory.mktemp("access_paths") / "ai_config.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    ai = load_ai_config(path)
    assert ai.provider_type(CLAUDE_ENTRY) == "claude"
    assert ai.provider_type(GEMINI_ENTRY) == "gemini"
    return ai


class _NoNetwork:
    """Stands in for the SDK client: any use of it is a request that should not happen."""

    def __getattr__(self, name):
        raise AssertionError(f"a denied request reached the SDK client (.{name})")


def _cut_off(p):
    """Replace the provider's client so a request that gets past pre-flight fails loudly."""
    p._client = _NoNetwork()
    return p


def _assert_denied(ei, model, capability):
    e = ei.value
    assert e.entry == CLAUDE_ENTRY, e.entry
    assert e.model == model, e.model
    assert capability in e.capabilities, e.capabilities
    assert e.reason == DENY_REASON, e.reason
    assert CLAUDE_ENTRY in str(e) and DENY_REASON in str(e), str(e)


def _is_quota_429(err) -> bool:
    return (isinstance(err, AIRateLimitError)
            and getattr(err.original_error, "status_code", None) == 429)


# ---------------------------------------------------------------- A1

@pytest.mark.parametrize("entry", [CLAUDE_ENTRY, GEMINI_ENTRY])
def test_a1_build_provider_and_availability(access_cfg, entry):
    """build_provider builds each named platform entry; a token count says where it stands."""
    p = access_cfg.build_provider(entry)
    assert p.entry == entry
    assert p.mode == "platform" and p.platform == "vertexai" and p.location == "global"
    assert p.api_key is None
    status, detail = p.probe_availability()
    say(f"[INFO] A1 {entry:<14} {p.model:<28} @global {status}"
        f"{f' ({detail[:120]})' if detail else ''}")
    assert status in LOCATION_STATUS_VALUES
    assert status != "unknown", f"{entry} @global: {detail}"


def test_a1_web_search_denied_for_every_model(access_cfg, e2e_session):
    """``"*": ["web_search"]`` applies to the entry's default model and to the canary."""
    cfg = e2e_session
    for model in (cfg.claude_model, cfg.claude_canary):
        p = _cut_off(access_cfg.build_provider(CLAUDE_ENTRY, model=model))
        with pytest.raises(DjinniteCapabilityDeniedError) as ei:
            p.generate("What is today's top headline? One sentence.",
                       web_search=True, max_output_tokens=256)
        _assert_denied(ei, model, "web_search")


def test_a1_structured_json_denied_for_haiku(access_cfg, e2e_session):
    """The per-model list: generate_json on Haiku is refused before any request."""
    cfg = e2e_session
    p = _cut_off(access_cfg.build_provider(CLAUDE_ENTRY, model=cfg.claude_model))
    with pytest.raises(DjinniteCapabilityDeniedError) as ei:
        p.generate_json(contract.PROMPT, contract.SCHEMA, max_output_tokens=contract.MAX_OUT)
    _assert_denied(ei, cfg.claude_model, "structured_json")


def test_a1_structured_json_not_denied_for_canary(access_cfg, e2e_session, e2e_ledger):
    """A model the map does not name gets only ``"*"``: the request reaches Vertex.

    The canary has no quota, so Vertex answers 429 (not billed).
    """
    cfg = e2e_session
    p = access_cfg.build_provider(CLAUDE_ENTRY, model=cfg.claude_canary)
    try:
        r = p.generate_json(contract.PROMPT, contract.SCHEMA, max_output_tokens=contract.MAX_OUT)
    except DjinniteCapabilityDeniedError as e:
        pytest.fail(f"{cfg.claude_canary} was denied by '{CLAUDE_ENTRY}', but only "
                    f"{cfg.claude_model} is denied structured_json: {e}")
    except AIRateLimitError as e:
        assert _is_quota_429(e), repr(e)
        say(f"[INFO] A1 {cfg.claude_canary} generate_json reached Vertex: 429 (not denied)")
        return
    except AIModelNotFoundError as e:
        pytest.fail(f"canary {cfg.claude_canary} is not enabled in Model Garden: {e}")
    e2e_ledger.record(f"claude A1 canary generate_json {cfg.claude_canary}", r)
    pytest.fail(f"canary {cfg.claude_canary} has quota; set DJINNITE_E2E_CLAUDE_CANARY "
                f"to an enabled Claude model with no quota")
