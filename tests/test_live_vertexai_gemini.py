"""
Gemini on Google Vertex AI, live (opt-in: ``--live-platform``).

Real calls in the Vertex project named in ai_config.json. Billed tests cost fractions of a cent;
rejected requests (403/404) are not billed. Matrix: PLATFORM_LIVE_TEST_DESIGN.md
("Gemini on Vertex", G1-G11; G6 was removed).

    uv run pytest tests/test_live_vertexai_gemini.py --live-platform -rA -s
"""

import pytest

from djinnite.ai_providers import get_provider
from djinnite.ai_providers.base_provider import (
    AIAuthenticationError, AIModelNotFoundError,
)
from djinnite.ai_providers.gemini_provider import GeminiProvider
from djinnite.tests import _contract as contract
from djinnite.tests._platform_live import gemini_call, run_without_adc, say

pytestmark = pytest.mark.live_platform


def _gemini(cfg, location="global", model=None, **kw):
    return get_provider("gemini", model=model or cfg.gemini_model, platform="vertexai",
                        project_id=cfg.project, location=location, **kw)


@pytest.fixture(scope="module")
def gemini(platform_session, platform_availability):
    cfg = platform_session
    status, detail = platform_availability[("gemini", cfg.gemini_model, "global")]
    assert status == "available", (
        f"{cfg.gemini_model} is not available @global in {cfg.project}: {status} {detail}. "
        f"Enable the Vertex AI API and grant roles/aiplatform.user to the runner.")
    p = _gemini(cfg)
    assert p.api_key is None and p.mode == "platform" and p.location == "global"
    return p


def test_g1_keyless_generate(gemini, platform_ledger):
    r = gemini_call(lambda: contract.check_generate(gemini))
    platform_ledger.record("gemini G1 generate @global", r)
    contract.assert_usage(r)
    assert "price_multiplier" not in r.usage  # no Gemini location premium


def test_g2_generate_json(gemini, platform_ledger):
    r = gemini_call(lambda: contract.check_generate_json(gemini))
    platform_ledger.record("gemini G2 generate_json", r)
    contract.assert_usage(r)


def test_g3_history(gemini, platform_ledger):
    r = gemini_call(lambda: contract.check_history(gemini))
    platform_ledger.record("gemini G3 history", r)
    contract.assert_usage(r)


def test_g4_thinking_accounting(gemini, platform_ledger):
    """Off reports no thinking; on is billed on top of output, exactly once."""
    off = gemini_call(lambda: contract.check_generate_json(gemini, thinking=False))
    platform_ledger.record("gemini G4 thinking=False", off)
    assert (off.thinking_tokens or 0) == 0, off.usage
    contract.assert_usage(off)

    on = gemini_call(lambda: contract.check_generate_json(gemini, thinking=True))
    platform_ledger.record("gemini G4 thinking=True", on)
    say(f"[INFO] gemini thinking=True: thinking_tokens={on.thinking_tokens}")
    assert on.usage["_thinking_billed_separately"] is True
    contract.assert_usage(on)  # includes thinking at the output rate


def test_g5_truncation(gemini, platform_ledger):
    partial = gemini_call(lambda: contract.check_truncation(gemini, thinking=False))
    platform_ledger.record("gemini G5 truncation (partial)", partial)
    contract.assert_usage(partial)


def test_g7_quota_project_succeeds(platform_session, platform_ledger):
    cfg = platform_session
    p = _gemini(cfg, quota_project=cfg.quota_project or cfg.project)
    r = gemini_call(lambda: contract.check_generate(p))
    platform_ledger.record("gemini G7 quota_project=configured", r)


def test_g8_decoy_quota_project_is_refused(platform_session):
    """The quota project must travel: naming a project we have no rights on fails."""
    cfg = platform_session
    if not cfg.decoy_project:
        pytest.skip("platforms.vertexai.testing.decoy_project not set in ai_config.json "
                    "(a project the runner has no rights on)")
    p = _gemini(cfg, quota_project=cfg.decoy_project)
    with pytest.raises(AIAuthenticationError) as ei:
        p.generate("hi", max_output_tokens=16)
    assert cfg.decoy_project in str(ei.value), str(ei.value)


def test_g9_unknown_model(platform_session):
    cfg = platform_session
    p = GeminiProvider(api_key=None, model="gemini-0-djinnite-nonexistent", platform="vertexai",
                       project_id=cfg.project, location="global", require_pricing=False)
    with pytest.raises(AIModelNotFoundError):
        p.generate("hi", max_output_tokens=16)


def test_g10_missing_adc(platform_session):
    cfg = platform_session
    assert run_without_adc("gemini", cfg.gemini_model, cfg.project, "global") == "AIAuthenticationError"


@pytest.mark.live_platform_extended
def test_x2_web_search_grounding(gemini, platform_ledger):
    """Google Search grounding through Vertex (billed per grounded prompt)."""
    r = gemini_call(lambda: gemini.generate(
        "What is today's top headline on a major news site? One sentence.",
        web_search=True, max_output_tokens=contract.MAX_OUT))
    platform_ledger.record("gemini X2 web_search", r)
    assert r.content.strip()
    say(f"[INFO] gemini grounding search_units={r.search_units} search_cost={r.search_cost}")


def test_g11_is_available_and_list_models(gemini):
    assert gemini.is_available() is True
    models = gemini.list_models()
    assert models, "list_models() returned nothing on Vertex"
    ids = {m["id"] for m in models}
    say(f"[INFO] Vertex lists {len(ids)} Gemini models; primary listed: "
        f"{gemini.model in ids}")
