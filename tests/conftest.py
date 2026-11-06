"""
Shared pytest fixtures for the Djinnite test suite.

Three kinds of tests live side by side in this directory:

* **Offline tests** -- pure logic, no keys, no network. These run on a plain
  ``uv run pytest tests/``.
* **Live tests (direct mode)** -- they call the providers' own APIs with the
  keys in ai_config.json, which costs money. Any test that requests the
  ``provider`` fixture, or is marked ``live``, is skipped unless you pass
  ``--live``. Each provider type runs through its direct-mode entry
  (``AIConfig.direct_entry``); platform entries are the live platform tier's job.
* **Platform end-to-end tests** -- they call Google Vertex AI in a dedicated
  project named in ai_config.json's ``platforms`` block, with Application
  Default Credentials (see PLATFORM_LIVE_TEST_DESIGN.md). Marked
  ``live_platform``; skipped unless you pass ``--live-platform``. Tests marked
  ``live_platform_extended`` (costlier) also need ``--live-platform-extended``.

Nothing is silently dropped: skipped tests are reported with their reason.

    uv run pytest tests/                        # offline only (free, fast)
    uv run pytest tests/ --live                 # + direct-mode provider APIs
    uv run pytest tests/ --live-platform -rA -s  # + Vertex AI, live
    uv run pytest tests/ --live -k claude
"""

import os
import sys
from pathlib import Path

import pytest

# Support running pytest from anywhere (adds the project's parent to path so
# the `djinnite` package resolves).
_project_root = str(Path(__file__).parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from djinnite.config_loader import load_ai_config


PROVIDER_NAMES = ["gemini", "claude", "chatgpt"]


def pytest_addoption(parser):
    parser.addoption(
        "--live", action="store_true", default=False,
        help="Run tests that call real provider APIs (needs keys, costs money).",
    )
    parser.addoption(
        "--live-platform", action="store_true", default=False,
        help="Run the live platform tier: Djinnite through Google Vertex AI, in the "
             "project ai_config.json's platforms block names, with Application "
             "Default Credentials (costs cents).",
    )
    parser.addoption(
        "--live-platform-extended", action="store_true", default=False,
        help="With --live-platform: also run the costlier extended tier "
             "(capability probes, web search).",
    )


def pytest_configure(config):
    """Opting in to the platform tier without its configuration is an error, not a skip."""
    if config.getoption("--live-platform"):
        from djinnite.tests._platform_live import missing_config
        problem = missing_config(load_ai_config())
        if problem:
            raise pytest.UsageError(f"--live-platform: {problem}")


def pytest_collection_modifyitems(config, items):
    """Gate the live tiers on their opt-in flags."""
    live = config.getoption("--live")
    platform = config.getoption("--live-platform")
    extended = config.getoption("--live-platform-extended")
    skip_live = pytest.mark.skip(reason="live API test -- rerun with --live to include")
    skip_platform = pytest.mark.skip(
        reason="live platform (Vertex AI) test -- rerun with --live-platform to include")
    skip_ext = pytest.mark.skip(
        reason="extended live platform test -- rerun with --live-platform --live-platform-extended")
    for item in items:
        if "live_platform" in item.keywords:
            if not platform:
                item.add_marker(skip_platform)
            elif "live_platform_extended" in item.keywords and not extended:
                item.add_marker(skip_ext)
            continue
        if not live and ("provider" in getattr(item, "fixturenames", ())
                         or "live" in item.keywords):
            item.add_marker(skip_live)


@pytest.fixture(scope="session")
def ai_config():
    return load_ai_config()


# ------------------------------------------------ live platform fixtures
# Only instantiated by tests marked live_platform, which are skipped unless
# --live-platform is given (and then pytest_configure has already checked
# ai_config.json).

@pytest.fixture(scope="session")
def platform_cfg(ai_config):
    from djinnite.tests._platform_live import PlatformLiveConfig
    return PlatformLiveConfig.from_ai_config(ai_config)


@pytest.fixture(scope="session")
def platform_session(platform_cfg):
    """The session's configuration, after the preflight report.

    Credentials are Application Default Credentials, resolved exactly as a
    consumer's would be; the preflight prints which principal and project
    they resolve to so a mismatch with ai_config.json is visible at once.
    """
    from djinnite.tests._platform_live import describe_credentials, say
    try:
        who = describe_credentials()
    except Exception as e:
        pytest.fail(f"--live-platform: no usable Google credentials "
                    f"({type(e).__name__}: {e}). Run 'gcloud auth application-default "
                    f"login' (see PLATFORM_LIVE_TEST_DESIGN.md, 'Credentials').")
    say("")
    say(f"[INFO] platform project={platform_cfg.project} "
        f"quota_project={platform_cfg.quota_project or '-'} "
        f"decoy={platform_cfg.decoy_project or '-'} "
        f"locations={','.join(platform_cfg.locations)}")
    say(f"[INFO] platform credentials: {who}")
    return platform_cfg


@pytest.fixture(scope="session")
def platform_ledger(platform_session):
    """Records every billed platform call; fails the session over the cost cap."""
    from djinnite.tests._platform_live import CostLedger, say
    ledger = CostLedger(platform_session.max_cost)
    yield ledger
    say(ledger.report())
    assert ledger.total <= ledger.cap, (
        f"platform session cost ${ledger.total:.4f} exceeded the cap ${ledger.cap:.2f} "
        f"(platforms.vertexai.testing.max_cost in ai_config.json)")


@pytest.fixture(scope="session")
def platform_availability(platform_session):
    """Token-count availability per (provider, model, location). Unbilled."""
    from djinnite.ai_providers import get_provider
    from djinnite.tests._platform_live import say
    cfg = platform_session
    targets = [("gemini", cfg.gemini_model, loc) for loc in cfg.locations]
    targets += [("claude", cfg.claude_model, loc) for loc in cfg.locations]
    targets += [("claude", cfg.claude_canary, "global")]
    table = {}
    say("[INFO] platform availability (token counts, unbilled)")
    for provider_name, model, loc in targets:
        try:
            p = get_provider(provider_name, model=model, platform="vertexai",
                             project_id=cfg.project, location=loc, require_pricing=False)
            status, detail = p.probe_availability()
        except Exception as e:
            status, detail = "unknown", f"{type(e).__name__}: {e}"
        table[(provider_name, model, loc)] = (status, detail)
        tail = f" ({' '.join(detail.split())[:120]})" if detail else ""
        say(f"  {provider_name:<7} {model:<28} @{loc:<12} {status}{tail}")
    return table


@pytest.fixture(params=PROVIDER_NAMES)
def provider_name(request):
    """Each live test runs once per provider."""
    return request.param


@pytest.fixture
def provider(provider_name, ai_config):
    """A live direct-mode provider built from ai_config.json, or a skip if unconfigured.

    ``provider_name`` is a provider *type*. The entry used is that type's
    direct-mode entry (``AIConfig.direct_entry``): the one named after the
    type, else the only one. Platform entries are the live platform tier's
    job (``--live-platform``). Several direct entries of a type with none named
    after it is a misconfiguration, so it fails rather than skips.
    """
    try:
        entry = ai_config.direct_entry(provider_name)
    except ValueError as e:
        pytest.fail(f"--live: {e}")
    if entry is None:
        pytest.skip(f"no direct-mode entry for {provider_name} in ai_config.json")
    return ai_config.build_provider(entry)
