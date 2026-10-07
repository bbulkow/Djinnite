"""
Shared pytest fixtures for the Djinnite test suite.

Three kinds of tests live side by side in this directory:

* **Offline tests** -- pure logic, no keys, no network. These run on a plain
  ``uv run pytest tests/``.
* **Live tests (direct mode)** -- they call the providers' own APIs with the
  keys in ai_config.json, which costs money. Any test that requests the
  ``provider`` fixture, or is marked ``live``, is skipped unless you pass
  ``--live``. Each provider type runs through its direct-mode entry
  (``AIConfig.direct_entry``); platform entries are the e2e tier's job.
* **Platform end-to-end tests** -- they call Google Vertex AI in a dedicated
  test project with platform credentials (see PLATFORM_E2E_TEST_DESIGN.md).
  Marked ``e2e_platform``; skipped unless you pass ``--e2e-platform``. Tests
  marked ``e2e_extended`` (costlier) also need ``--e2e-extended``.

Nothing is silently dropped: skipped tests are reported with their reason.

    uv run pytest tests/                        # offline only (free, fast)
    uv run pytest tests/ --live                 # + direct-mode provider APIs
    uv run pytest tests/ --e2e-platform -rA -s  # + Vertex AI end-to-end
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
        "--e2e-platform", action="store_true", default=False,
        help="Run platform-mode end-to-end tests against Google Vertex AI "
             "(needs DJINNITE_E2E_PROJECT and platform credentials; costs cents).",
    )
    parser.addoption(
        "--e2e-extended", action="store_true", default=False,
        help="With --e2e-platform: also run the costlier extended tier "
             "(capability probes, web search).",
    )


def pytest_configure(config):
    """Opting in to the e2e tier without its configuration is an error, not a skip."""
    if config.getoption("--e2e-platform"):
        from djinnite.tests._e2e import missing_config
        problem = missing_config()
        if problem:
            raise pytest.UsageError(f"--e2e-platform: {problem}")


def pytest_collection_modifyitems(config, items):
    """Gate live and e2e tests on their opt-in flags."""
    live = config.getoption("--live")
    e2e = config.getoption("--e2e-platform")
    extended = config.getoption("--e2e-extended")
    skip_live = pytest.mark.skip(reason="live API test -- rerun with --live to include")
    skip_e2e = pytest.mark.skip(reason="platform e2e test -- rerun with --e2e-platform to include")
    skip_ext = pytest.mark.skip(
        reason="extended platform e2e test -- rerun with --e2e-platform --e2e-extended")
    for item in items:
        if "e2e_platform" in item.keywords:
            if not e2e:
                item.add_marker(skip_e2e)
            elif "e2e_extended" in item.keywords and not extended:
                item.add_marker(skip_ext)
            continue
        if not live and ("provider" in getattr(item, "fixturenames", ())
                         or "live" in item.keywords):
            item.add_marker(skip_live)


@pytest.fixture(scope="session")
def ai_config():
    return load_ai_config()


# ------------------------------------------------- platform e2e fixtures
# Only instantiated by tests marked e2e_platform, which are skipped unless
# --e2e-platform is given (and then pytest_configure has already checked
# the configuration).

@pytest.fixture(scope="session")
def e2e_cfg():
    from djinnite.tests._e2e import E2EConfig
    return E2EConfig.from_env()


@pytest.fixture(scope="session")
def e2e_session(e2e_cfg):
    """Credentials for the session, plus the preflight report.

    DJINNITE_E2E_CREDENTIALS (an impersonated-service-account ADC file) is
    applied only for this session, so the default ADC of other projects is
    never touched.
    """
    from djinnite.tests._e2e import describe_credentials, say
    saved = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if e2e_cfg.credentials:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = e2e_cfg.credentials
    try:
        try:
            who = describe_credentials()
        except Exception as e:
            pytest.fail(f"--e2e-platform: no usable Google credentials "
                        f"({type(e).__name__}: {e}). See PLATFORM_E2E_TEST_DESIGN.md, "
                        f"'Credentials on a developer machine'.")
        say("")
        say(f"[INFO] e2e project={e2e_cfg.project} decoy={e2e_cfg.decoy_project or '-'} "
            f"locations={','.join(e2e_cfg.locations)}")
        say(f"[INFO] e2e credentials: {who}")
        yield e2e_cfg
    finally:
        if saved is None:
            os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
        else:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = saved


@pytest.fixture(scope="session")
def e2e_ledger(e2e_session):
    """Records every billed e2e call; fails the session over the cost cap."""
    from djinnite.tests._e2e import CostLedger, say
    ledger = CostLedger(e2e_session.max_cost)
    yield ledger
    say(ledger.report())
    assert ledger.total <= ledger.cap, (
        f"e2e session cost ${ledger.total:.4f} exceeded the cap ${ledger.cap:.2f} "
        f"(DJINNITE_E2E_MAX_COST)")


@pytest.fixture(scope="session")
def e2e_availability(e2e_session):
    """Token-count availability per (provider, model, location). Unbilled."""
    from djinnite.ai_providers import get_provider
    from djinnite.tests._e2e import say
    cfg = e2e_session
    targets = [("gemini", cfg.gemini_model, loc) for loc in cfg.locations]
    targets += [("claude", cfg.claude_model, loc) for loc in cfg.locations]
    targets += [("claude", cfg.claude_canary, "global")]
    table = {}
    say("[INFO] e2e availability (token counts, unbilled)")
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
    type, else the only one. Platform entries are the e2e tier's job
    (``--e2e-platform``). Several direct entries of a type with none named
    after it is a misconfiguration, so it fails rather than skips.
    """
    try:
        entry = ai_config.direct_entry(provider_name)
    except ValueError as e:
        pytest.fail(f"--live: {e}")
    if entry is None:
        pytest.skip(f"no direct-mode entry for {provider_name} in ai_config.json")
    return ai_config.build_provider(entry)
