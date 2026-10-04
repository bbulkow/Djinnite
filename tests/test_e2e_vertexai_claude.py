"""
Claude on Google Vertex AI, end to end (opt-in: ``--e2e-platform``).

Two layers (PLATFORM_E2E_TEST_DESIGN.md, C1-C12):

* **Plumbing (C1-C6) needs no Claude quota.** A 429 RESOURCE_EXHAUSTED naming
  the base model proves ADC, the quota project, the location's endpoint, the
  ``@`` model ID and Model Garden enablement all worked -- and rejected
  requests are not billed.
* **Functional (C7-C12)** runs where the project has Claude quota; otherwise
  each test skips and says so by name.

    uv run pytest tests/test_e2e_vertexai_claude.py --e2e-platform -rA -s
"""

import pytest

from djinnite.ai_providers import get_provider
from djinnite.ai_providers.base_provider import (
    AIAuthenticationError, AIModelNotFoundError, AIProviderError, AIRateLimitError,
)
from djinnite.ai_providers.claude_provider import ClaudeProvider
from djinnite.config_loader import LOCATION_STATUS_VALUES
from djinnite.tests import _contract as contract
from djinnite.tests._e2e import run_without_adc, say

pytestmark = pytest.mark.e2e_platform


def _claude(cfg, location="global", model=None, **kw):
    return get_provider("claude", model=model or cfg.claude_model, platform="vertexai",
                        project_id=cfg.project, location=location, **kw)


def _is_quota_429(err) -> bool:
    return (isinstance(err, AIRateLimitError)
            and getattr(err.original_error, "status_code", None) == 429)


@pytest.fixture(scope="module")
def claude_quota(e2e_session, e2e_ledger):
    """One generate_json @global: ("ok", response) or ("no_quota", error).

    Any other failure propagates -- auth, routing and not-found are defects.
    """
    cfg = e2e_session
    p = _claude(cfg)
    try:
        r = contract.check_generate_json(p)
    except AIRateLimitError as e:
        say(f"[INFO] {cfg.claude_model} @global: no quota ({e})")
        return "no_quota", e
    e2e_ledger.record("claude C1 generate_json @global", r)
    return "ok", r


def _needs_quota(claude_quota, cfg):
    status, _ = claude_quota
    if status != "ok":
        pytest.skip(f"no Claude quota in project {cfg.project} for {cfg.claude_model} @global "
                    f"(request global_online_prediction_requests_per_base_model)")


# ------------------------------------------------------ plumbing (no quota)

def test_c1_keyless_call_reaches_the_model(e2e_session, claude_quota):
    """Success, or a quota 429 -- never auth or not-found."""
    status, result = claude_quota
    if status == "ok":
        assert result.content
        contract.assert_usage(result, multiplier=1.0)
    else:
        assert _is_quota_429(result), repr(result)


def test_c2_zero_quota_canary(e2e_session):
    cfg = e2e_session
    p = _claude(cfg, model=cfg.claude_canary)
    try:
        p.generate("hi", max_output_tokens=16)
    except AIRateLimitError as e:
        assert _is_quota_429(e)
        assert "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower(), str(e)
        return
    except AIModelNotFoundError as e:
        pytest.fail(f"canary {cfg.claude_canary} is not enabled in Model Garden: {e}")
    pytest.fail(f"canary {cfg.claude_canary} has quota; set DJINNITE_E2E_CLAUDE_CANARY "
                f"to an enabled Claude model with no quota")


def test_c3_decoy_quota_project_is_refused(e2e_session):
    cfg = e2e_session
    assert cfg.decoy_project, "set DJINNITE_E2E_DECOY_PROJECT (a project the runner has no rights on)"
    p = _claude(cfg, quota_project=cfg.decoy_project)
    with pytest.raises(AIAuthenticationError) as ei:
        p.generate("hi", max_output_tokens=16)
    assert cfg.decoy_project in str(ei.value), str(ei.value)


def test_c4_unknown_model(e2e_session):
    cfg = e2e_session
    p = ClaudeProvider(model="claude-djinnite-nonexistent-9", platform="vertexai",
                       project_id=cfg.project, location="global", require_pricing=False)
    with pytest.raises(AIModelNotFoundError) as ei:
        p.generate("hi", max_output_tokens=16)
    assert "Model Garden" in str(ei.value)


def test_c5_missing_adc(e2e_session):
    cfg = e2e_session
    assert run_without_adc("claude", cfg.claude_model, cfg.project, "global") == "AIAuthenticationError"


def test_c6_availability_probe(e2e_session, e2e_availability, claude_quota):
    cfg = e2e_session
    for loc in cfg.locations:
        status, detail = e2e_availability[("claude", cfg.claude_model, loc)]
        assert status in LOCATION_STATUS_VALUES
        assert status != "unknown", f"@{loc}: {detail}"
    count_status = e2e_availability[("claude", cfg.claude_model, "global")][0]
    gen_status = claude_quota[0]
    say(f"[INFO] observation: count_tokens @global={count_status}, generate @global={gen_status} "
        f"-> count_tokens {'IS' if count_status == 'no_quota' else 'is NOT'} quota-gated"
        if gen_status == "no_quota" else
        f"[INFO] observation: count_tokens @global={count_status}, generate @global=ok")
    p = _claude(cfg)
    assert p.is_available() is (count_status == "available")


# ------------------------------------------------- functional (needs quota)

def test_c7_generate(e2e_session, claude_quota, e2e_ledger):
    _needs_quota(claude_quota, e2e_session)
    p = _claude(e2e_session)
    r = contract.check_generate(p)
    e2e_ledger.record("claude C7 generate @global", r)
    contract.assert_usage(r, multiplier=1.0)
    assert "price_multiplier" not in r.usage


def test_c8_regional_price_premium(e2e_session, claude_quota, e2e_ledger):
    cfg = e2e_session
    _needs_quota(claude_quota, cfg)
    tried = []
    for loc in [l for l in cfg.locations if l != "global"]:
        p = _claude(cfg, location=loc)
        try:
            r = contract.check_generate_json(p)
        except AIRateLimitError:
            tried.append(f"{loc}=no_quota")
            continue
        e2e_ledger.record(f"claude C8 generate_json @{loc}", r)
        contract.assert_usage(r, multiplier=1.10)
        return
    pytest.skip(f"no Claude quota at any non-global location ({', '.join(tried) or 'none configured'})")


def test_c9_history(e2e_session, claude_quota, e2e_ledger):
    _needs_quota(claude_quota, e2e_session)
    r = contract.check_history(_claude(e2e_session))
    e2e_ledger.record("claude C9 history", r)
    contract.assert_usage(r)


def test_c10_thinking_false_is_accepted(e2e_session, claude_quota, e2e_ledger):
    _needs_quota(claude_quota, e2e_session)
    r = contract.check_generate_json(_claude(e2e_session), thinking=False)
    e2e_ledger.record("claude C10 thinking=False", r)
    assert (r.thinking_tokens or 0) == 0, r.usage


def test_c11_thinking_tokens_on_vertex(e2e_session, claude_quota, e2e_ledger):
    """Records whether Vertex returns usage.output_tokens_details."""
    _needs_quota(claude_quota, e2e_session)
    r = contract.check_generate_json(_claude(e2e_session), thinking=2048)
    e2e_ledger.record("claude C11 thinking=2048", r)
    say(f"[INFO] observation: Vertex thinking_tokens={r.thinking_tokens} "
        f"({'output_tokens_details reported' if r.thinking_tokens is not None else 'NOT reported'})")
    assert r.thinking_tokens is None or r.thinking_tokens > 0
    assert r.usage["_thinking_billed_separately"] is False
    contract.assert_usage(r)


def test_c12_truncation(e2e_session, claude_quota, e2e_ledger):
    _needs_quota(claude_quota, e2e_session)
    partial = contract.check_truncation(_claude(e2e_session))
    e2e_ledger.record("claude C12 truncation (partial)", partial)
    contract.assert_usage(partial)


# ---------------------------------------------------------- extended tier

@pytest.mark.e2e_extended
def test_x3_web_search_on_vertex(e2e_session, claude_quota, e2e_ledger):
    _needs_quota(claude_quota, e2e_session)
    p = _claude(e2e_session)
    r = p.generate("What is today's top headline on a major news site? One sentence.",
                   web_search=True, max_output_tokens=2048)
    e2e_ledger.record("claude X3 web_search", r)
    assert r.content.strip()
    say(f"[INFO] claude web search units={r.search_units} search_cost={r.search_cost}")


@pytest.mark.e2e_extended
def test_x4_claude_5x_thinking_on_vertex(e2e_session, e2e_ledger):
    """between_tools and always-on thinking, delivered through the platform."""
    cfg = e2e_session
    if not cfg.claude_5x_model:
        pytest.skip("DJINNITE_E2E_CLAUDE_5X_MODEL not set (a 5.x Claude with quota)")
    # Built without a catalog entry so the request reaches the API whatever
    # the catalog says today; the API is what is under test here.
    p = ClaudeProvider(model=cfg.claude_5x_model, platform="vertexai", project_id=cfg.project,
                       location="global", require_pricing=False)
    r = p.generate_json(contract.PROMPT, contract.SCHEMA, max_output_tokens=contract.MAX_OUT,
                        thinking="between_tools")
    e2e_ledger.record(f"claude X4 between_tools {cfg.claude_5x_model}", r)
    with pytest.raises(AIProviderError) as ei:
        p.generate("hi", max_output_tokens=64, thinking=False)
    assert "disabled" in str(ei.value).lower(), str(ei.value)
