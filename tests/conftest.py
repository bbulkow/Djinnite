"""
Shared pytest fixtures for the Djinnite test suite.

Two kinds of tests live side by side in this directory:

* **Offline tests** -- pure logic, no keys, no network. These run on a plain
  ``uv run pytest tests/``.
* **Live tests** -- they call real provider APIs, which costs money and needs
  network plus configured keys. Any test that requests the ``provider``
  fixture is automatically treated as live and is skipped unless you pass
  ``--live``. Nothing is silently dropped: skipped live tests are reported by
  pytest with their reason.

    uv run pytest tests/                # offline only (free, fast)
    uv run pytest tests/ --live         # everything, hits paid APIs
    uv run pytest tests/ --live -k claude
"""

import sys
from pathlib import Path

import pytest

# Support running pytest from anywhere (adds the project's parent to path so
# the `djinnite` package resolves).
_project_root = str(Path(__file__).parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from djinnite.config_loader import load_ai_config
from djinnite.ai_providers import get_provider


PROVIDER_NAMES = ["gemini", "claude", "chatgpt"]


def pytest_addoption(parser):
    parser.addoption(
        "--live", action="store_true", default=False,
        help="Run tests that call real provider APIs (needs keys, costs money).",
    )


def pytest_collection_modifyitems(config, items):
    """Any test using the `provider` fixture is a live test; gate it on --live."""
    if config.getoption("--live"):
        return
    skip_live = pytest.mark.skip(reason="live API test -- rerun with --live to include")
    for item in items:
        if "provider" in getattr(item, "fixturenames", ()):
            item.add_marker(skip_live)


@pytest.fixture(scope="session")
def ai_config():
    return load_ai_config()


@pytest.fixture(params=PROVIDER_NAMES)
def provider_name(request):
    """Each live test runs once per provider."""
    return request.param


@pytest.fixture
def provider(provider_name, ai_config):
    """A live provider built from ai_config.json, or a skip if unconfigured."""
    p_config = ai_config.get_provider(provider_name)
    if not p_config or not p_config.api_key:
        pytest.skip(f"{provider_name} not configured in ai_config.json")

    kwargs = {}
    if provider_name == "gemini":
        kwargs["backend"] = p_config.backend
        kwargs["project_id"] = p_config.project_id

    return get_provider(
        provider_name, api_key=p_config.api_key,
        model=p_config.default_model, **kwargs,
    )
