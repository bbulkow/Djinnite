"""
Access modes: ``direct`` vs ``platform`` (offline).

A platform (``vertexai`` today) hosts some providers' models. The registry
holds per-platform rules; ai_config.json's ``platforms`` block holds the
settings; the catalog's per-model ``platforms`` block holds what was probed.

    uv run pytest tests/test_platform_mode.py -v
"""

import json
from unittest import mock

import anthropic
import pytest

import djinnite.ai_providers as ap
from djinnite.ai_providers import get_provider, PLATFORMS, resolve_platform
from djinnite.ai_providers.base_provider import AIProviderError
from djinnite.ai_providers.claude_provider import ClaudeProvider
from djinnite.ai_providers.gemini_provider import GeminiProvider
from djinnite.config_loader import (
    load_ai_config, load_model_catalog, ModelCapabilities, ModelCatalog, ModelInfo,
    ModelCosting, PlatformModelInfo, LOCATION_STATUS_VALUES, _parse_platforms,
)
from djinnite.tests._stubs import bare


# --------------------------------------------------------------- registry

def test_vertexai_rules():
    spec = PLATFORMS["vertexai"]
    assert spec.providers == {"gemini", "claude"}
    assert spec.default_location == {"gemini": "us-central1", "claude": "global"}
    assert spec.price_multiplier("claude", "global") == 1.0
    assert spec.price_multiplier("claude", None) == 1.0  # default location is global
    for loc in ("us", "eu", "us-east5", "europe-west1"):
        assert spec.price_multiplier("claude", loc) == pytest.approx(1.10)
    assert spec.price_multiplier("gemini", "us") == 1.0  # no documented premium


@pytest.mark.parametrize("platform, backend, prov, expected", [
    (None, None, "claude", None),
    (None, "gemini", "gemini", None),        # ProviderConfig's default backend
    (None, "anthropic", "claude", None),
    (None, "vertexai", "gemini", "vertexai"),  # legacy alias
    ("vertexai", None, "claude", "vertexai"),
    ("vertexai", "vertexai", "claude", "vertexai"),
])
def test_resolve_platform(platform, backend, prov, expected):
    assert resolve_platform(platform, backend, provider=prov) == expected


@pytest.mark.parametrize("platform, prov, match", [
    ("bedrock", "claude", "Unknown platform"),
    ("vertexai", "chatgpt", "does not host provider 'chatgpt'"),
    ("vertexai", "grok", "does not host provider 'grok'"),
])
def test_resolve_platform_rejects(platform, prov, match):
    # (named 'prov': a test argument called 'provider' is gated as a live test)
    with pytest.raises(AIProviderError, match=match):
        resolve_platform(platform, None, provider=prov)


def test_mode_attribute_defaults_to_direct():
    for cls in (ClaudeProvider, GeminiProvider):
        p = bare(cls, client=None)
        assert p.mode == "direct" and p.platform is None and p._price_multiplier == 1.0


def test_get_provider_rejects_unhosted_provider_cleanly():
    with pytest.raises(AIProviderError, match="does not host provider 'chatgpt'"):
        get_provider("chatgpt", model="gpt-5", platform="vertexai", project_id="p")


# --------------------------------------------------------------- ai_config

def _write_config(tmp_path, data):
    path = tmp_path / "ai_config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_ai_config_platform_mode(tmp_path):
    path = _write_config(tmp_path, {
        "platforms": {
            "_note": "notes are ignored",
            "vertexai": {"project_id": "munin-bbulkow", "quota_project": "munin-bbulkow",
                         "locations": ["global", "us"]},
        },
        "providers": {
            "claude": {"mode": "platform", "platform": "vertexai", "location": "us",
                       "default_model": "claude-sonnet-5-5"},
            "gemini": {"backend": "vertexai", "project_id": "legacy-proj"},
            "chatgpt": {"api_key": "sk-real-looking-key", "default_model": "gpt-5"},
            "grok": {"api_key": "your-xai-key-here"},
        },
    })
    cfg = load_ai_config(path)
    assert cfg.platforms["vertexai"].locations == ["global", "us"]
    assert "_note" not in cfg.platforms

    claude = cfg.providers["claude"]
    assert (claude.mode, claude.platform, claude.location) == ("platform", "vertexai", "us")
    assert claude.project_id == "munin-bbulkow"       # from the platform block
    assert claude.quota_project == "munin-bbulkow"
    assert cfg.provider_kwargs("claude") == {
        "api_key": None, "platform": "vertexai", "project_id": "munin-bbulkow",
        "location": "us", "quota_project": "munin-bbulkow",
    }

    gemini = cfg.providers["gemini"]  # legacy backend=vertexai entry
    assert (gemini.mode, gemini.platform, gemini.project_id) == ("platform", "vertexai", "legacy-proj")
    assert gemini.quota_project == "munin-bbulkow"  # entry lacks it; platform block supplies it

    assert cfg.provider_kwargs("chatgpt") == {"api_key": "sk-real-looking-key"}
    assert cfg.providers["chatgpt"].mode == "direct"

    assert cfg.is_usable("claude") and cfg.is_usable("gemini")  # no key needed
    assert cfg.is_usable("chatgpt")
    assert not cfg.is_usable("grok")  # placeholder key
    assert not cfg.is_usable("missing")


def test_ai_config_without_platforms_loads_as_before(tmp_path):
    path = _write_config(tmp_path, {"providers": {
        "gemini": {"api_key": "k", "backend": "gemini", "project_id": None}}})
    cfg = load_ai_config(path)
    g = cfg.providers["gemini"]
    assert (g.mode, g.platform, g.location, g.quota_project) == ("direct", None, None, None)
    assert cfg.platforms == {}


@pytest.mark.parametrize("entry, match", [
    ({"mode": "platform"}, "requires a 'platform' name"),
    ({"mode": "cloud"}, "must be one of"),
    ({"mode": "direct", "platform": "vertexai"}, "mode is 'direct'"),
])
def test_ai_config_rejects_inconsistent_modes(tmp_path, entry, match):
    path = _write_config(tmp_path, {"providers": {"claude": entry}})
    with pytest.raises(ValueError, match=match):
        load_ai_config(path)


# ------------------------------------------------------------------ catalog

def test_parse_platforms_drops_unknown_statuses():
    parsed = _parse_platforms({"vertexai": {
        "locations": {"global": "available", "us": "maybe"},
        "capabilities": {"thinking": ["on"]},
        "probed": "2026-10-04",
    }})
    pm = parsed["vertexai"]
    assert pm.locations == {"global": "available"}
    assert pm.capabilities.thinking == ["on"]
    assert pm.probed == "2026-10-04"
    assert _parse_platforms(None) == {}


def test_real_catalog_platform_blocks_use_the_vocabulary():
    for models in load_model_catalog().providers.values():
        for m in models:
            for pm in m.platforms.values():
                assert set(pm.locations.values()) <= set(LOCATION_STATUS_VALUES), m.id


def _model(platform_caps=None, locations=None):
    platforms = {}
    if platform_caps is not None or locations is not None:
        platforms["vertexai"] = PlatformModelInfo(
            locations=locations or {}, capabilities=platform_caps)
    return ModelInfo(
        id="claude-sonnet-5-5", name="Sonnet", context_window=1_000_000, max_output_tokens=128_000,
        costing=ModelCosting(input_per_1m=2.0, output_per_1m=10.0, source="published"),
        capabilities=ModelCapabilities(thinking=["on", "off"], thinking_style=["adaptive"],
                                       structured_json=["on", "off"]),
        platforms=platforms,
    )


def test_for_platform_overlays_probed_fields_only():
    m = _model(ModelCapabilities(thinking=["on"], thinking_style=["adaptive", "between_tools"]))
    view = m.for_platform("vertexai")
    assert view.capabilities.thinking == ["on"]
    assert view.capabilities.thinking_style == ["adaptive", "between_tools"]
    assert view.capabilities.structured_json == ["on", "off"]  # not probed: direct value
    assert m.capabilities.thinking == ["on", "off"]             # original untouched
    assert view.costing is m.costing
    assert m.for_platform(None) is m
    no_caps = _model(locations={"global": "available"})
    assert no_caps.for_platform("vertexai") is no_caps  # availability only: direct caps


def _patch_catalog(monkeypatch, info):
    catalog = ModelCatalog(providers={"claude": [info]})
    monkeypatch.setattr(ap, "load_model_catalog", lambda *a, **k: catalog)


def test_get_provider_uses_platform_capabilities(monkeypatch):
    _patch_catalog(monkeypatch, _model(ModelCapabilities(thinking=["on"])))
    with mock.patch.object(anthropic, "AnthropicVertex"):
        p = get_provider("claude", model="claude-sonnet-5-5", platform="vertexai", project_id="p")
    assert p._model_info.capabilities.thinking == ["on"]
    with mock.patch.object(anthropic, "Anthropic"):
        d = get_provider("claude", api_key="k", model="claude-sonnet-5-5")
    assert d._model_info.capabilities.thinking == ["on", "off"]


def test_recorded_availability_never_blocks(monkeypatch):
    _patch_catalog(monkeypatch, _model(locations={"us": "not_found"}))
    with mock.patch.object(anthropic, "AnthropicVertex"):
        p = get_provider("claude", model="claude-sonnet-5-5", platform="vertexai",
                         project_id="p", location="us")
    assert p.location == "us"
