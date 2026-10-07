"""
Access paths: named ai_config entries, several per provider type (offline).

ACCESS_PATHS_DESIGN.md, "Verification -> Offline". No network, no keys:
configs are written to ``tmp_path``, SDK clients are stubbed, and the model
catalog is replaced in memory.

    uv run pytest tests/test_access_paths.py -q -rs

Test arguments are never named ``provider``: conftest treats any test using
that name as a live test and skips it without ``--live``.
"""

import json
from unittest import mock

import anthropic
import pytest

import djinnite
import djinnite.ai_providers as ap
from djinnite.ai_providers import PROVIDERS, get_provider
from djinnite.ai_providers.base_provider import AIProviderError, DjinniteCapabilityDeniedError
from djinnite.ai_providers.claude_provider import ClaudeProvider
from djinnite.ai_providers.gemini_provider import GeminiProvider
from djinnite.ai_providers.grok_provider import GrokProvider
from djinnite.ai_providers.openai_provider import OpenAIProvider
from djinnite.config_loader import (
    DENIABLE_CAPABILITIES, DENY_ALL_MODELS, PROVIDER_TYPES,
    ModelCapabilities, ModelCatalog, ModelChoice, ModelCosting, ModelInfo,
    PlatformModelInfo, ProviderConfig, load_ai_config,
)
from djinnite.tests._stubs import (
    bare, info, claude_response, StubAnthropic, StubGenai, StubResponses,
)


REASON = "org policy vertexai.allowedPartnerModelFeatures in my-project"
SCHEMA = {"type": "object", "properties": {"v": {"type": "integer"}}, "required": ["v"]}


# ------------------------------------------------------------------ helpers

def _write(tmp_path, data):
    path = tmp_path / "ai_config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _load(tmp_path, data):
    return load_ai_config(_write(tmp_path, data))


def _load_text(tmp_path, text):
    """Raw JSON text: json.dumps cannot produce a duplicate key."""
    path = tmp_path / "ai_config.json"
    path.write_text(text, encoding="utf-8")
    return load_ai_config(path)


def _providers(entries, **top):
    return {"providers": entries, **top}


PLATFORMS = {"vertexai": {"project_id": "my-project", "quota_project": "my-quota-project",
                          "locations": ["global", "us"]}}

MIXED = {
    "platforms": PLATFORMS,
    "providers": {
        "_note": "notes are skipped",
        "claude": {"api_key": "sk-ant-test-key", "default_model": "claude-a",
                   "use_cases": {"coding": "claude-b"}},
        "claude-vertex": {
            "provider": "claude", "mode": "platform", "platform": "vertexai",
            "location": "us", "default_model": "claude-a",
            "use_cases": {"coding": "claude-b"},
            "deny": {"*": ["web_search"], "claude-b": ["structured_json"]},
            "deny_reason": REASON,
        },
        "claude-vertex-global": {"provider": "claude", "platform": "vertexai",
                                 "default_model": "claude-a"},
        "claude-old": {"provider": "claude", "api_key": "sk-ant-other", "enabled": False},
        "chatgpt": {"api_key": "sk-test-key", "default_model": "gpt-a"},
        "grok": {"api_key": "your-key-here"},
        "openai": {"api_key": "sk-legacy-entry"},  # implicit unknown type
    },
    "default_provider": "claude-vertex",
}


def _minfo(model_id, caps=None, platform_caps=None):
    platforms = {}
    if platform_caps is not None:
        platforms["vertexai"] = PlatformModelInfo(locations={"global": "available"},
                                                  capabilities=platform_caps)
    return ModelInfo(
        id=model_id, name=model_id, context_window=100_000, max_output_tokens=1000,
        costing=ModelCosting(input_per_1m=1.0, output_per_1m=2.0, source="published"),
        capabilities=caps or ModelCapabilities(),
        platforms=platforms,
    )


def _catalog():
    caps = ModelCapabilities(structured_json=["on", "off"], web_search=["on", "off"],
                             json_with_search=["on", "off"])
    return ModelCatalog(providers={
        "claude": [_minfo(m, caps) for m in ("claude-a", "claude-b", "claude-c")],
        "chatgpt": [_minfo("gpt-a", caps)],
        "grok": [_minfo("grok-a", caps)],
    })


@pytest.fixture
def catalog(monkeypatch):
    """Replace the catalog get_provider reads with an in-memory one."""
    cat = _catalog()
    monkeypatch.setattr(ap, "load_model_catalog", lambda *a, **k: cat)
    return cat


@pytest.fixture
def sdk():
    """Stub every SDK client constructor build_provider could reach."""
    with mock.patch.object(anthropic, "Anthropic") as direct, \
            mock.patch.object(anthropic, "AnthropicVertex") as vertex, \
            mock.patch("openai.OpenAI") as openai_client:
        yield {"Anthropic": direct, "AnthropicVertex": vertex, "OpenAI": openai_client}


# ================================================================== loading

def test_provider_types_match_the_registry():
    assert set(PROVIDER_TYPES) == set(PROVIDERS)


def test_legacy_config_is_unchanged(tmp_path):
    cfg = _load(tmp_path, {
        "platforms": PLATFORMS,
        "providers": {
            "gemini": {"backend": "vertexai", "project_id": "legacy-proj"},
            "claude": {"api_key": "sk-ant-test-key", "default_model": "claude-a"},
            "chatgpt": {"api_key": "sk-test-key"},
            "grok": {"api_key": "xai-test-key"},
        },
        "default_provider": "claude",
    })
    for name in ("gemini", "claude", "chatgpt", "grok"):
        assert cfg.provider_type(name) == name
        pc = cfg.providers[name]
        assert pc.name == name and pc.provider == name
        assert pc.deny == {} and pc.deny_reason == ""
        assert pc.denied_for("anything") == ()
        assert cfg.is_usable(name)
    assert cfg.provider_kwargs("claude") == {"api_key": "sk-ant-test-key"}
    assert cfg.provider_kwargs("chatgpt") == {"api_key": "sk-test-key"}
    assert cfg.provider_kwargs("grok") == {"api_key": "xai-test-key"}
    assert cfg.provider_kwargs("gemini") == {
        "api_key": None, "platform": "vertexai", "project_id": "legacy-proj",
        "quota_project": "my-quota-project",
    }
    assert cfg.get_model_for_use_case("anything") == ("claude", "claude-a")


def test_direct_and_platform_entries_of_one_type_coexist(tmp_path):
    cfg = _load(tmp_path, MIXED)
    direct, vertex = cfg.providers["claude"], cfg.providers["claude-vertex"]
    assert (direct.mode, direct.platform) == ("direct", None)
    assert (vertex.mode, vertex.platform, vertex.location) == ("platform", "vertexai", "us")
    assert vertex.project_id == "my-project"  # from the platform block
    assert cfg.provider_type("claude") == cfg.provider_type("claude-vertex") == "claude"
    assert cfg.provider_kwargs("claude") == {"api_key": "sk-ant-test-key"}
    assert cfg.provider_kwargs("claude-vertex") == {
        "api_key": None, "platform": "vertexai", "project_id": "my-project",
        "location": "us", "quota_project": "my-quota-project",
    }
    assert cfg.is_usable("claude") and cfg.is_usable("claude-vertex")


def test_entries_of_type(tmp_path):
    cfg = _load(tmp_path, MIXED)
    # claude-old is disabled: never listed.
    assert cfg.entries_of_type("claude") == ["claude", "claude-vertex", "claude-vertex-global"]
    assert cfg.entries_of_type("claude", mode="direct") == ["claude"]
    assert cfg.entries_of_type("claude", mode="platform") == ["claude-vertex", "claude-vertex-global"]
    assert cfg.entries_of_type("grok") == ["grok"]
    assert cfg.entries_of_type("grok", usable_only=True) == []  # placeholder key
    assert cfg.entries_of_type("gemini") == []
    assert cfg.entries_of_type("openai") == ["openai"]  # by type string, though unusable
    assert cfg.entries_of_type("openai", usable_only=True) == []


def test_duplicate_provider_key_raises(tmp_path):
    text = """{
      "providers": {
        "claude": {"api_key": "sk-ant-one"},
        "claude": {"mode": "platform", "platform": "vertexai", "project_id": "my-project"}
      }
    }"""
    with pytest.raises(ValueError) as exc:
        _load_text(tmp_path, text)
    msg = str(exc.value)
    assert "'claude'" in msg and "duplicate" in msg.lower()
    assert '"provider"' in msg  # says how to configure one type twice


def test_duplicate_platform_key_raises(tmp_path):
    text = """{
      "platforms": {
        "vertexai": {"project_id": "my-project"},
        "vertexai": {"project_id": "my-other-project"}
      },
      "providers": {"chatgpt": {"api_key": "sk-test-key"}}
    }"""
    with pytest.raises(ValueError, match="duplicate key 'vertexai'"):
        _load_text(tmp_path, text)


def test_explicit_unknown_type_raises(tmp_path):
    with pytest.raises(ValueError) as exc:
        _load(tmp_path, _providers({"my-llm": {"provider": "openai", "api_key": "sk-x"}}))
    msg = str(exc.value)
    assert "my-llm" in msg and "provider" in msg
    for ptype in PROVIDER_TYPES:
        assert ptype in msg


def test_implicit_unknown_type_loads_but_is_unusable(tmp_path, sdk):
    cfg = _load(tmp_path, _providers({"openai": {"api_key": "sk-legacy-entry"}}))
    assert "openai" in cfg.providers
    assert cfg.provider_type("openai") == "openai"
    assert not cfg.is_usable("openai")
    assert cfg.provider_kwargs("openai") == {"api_key": "sk-legacy-entry"}  # unchanged
    with pytest.raises(ValueError) as exc:
        cfg.build_provider("openai")
    assert "'openai'" in str(exc.value) and '"provider"' in str(exc.value)
    assert not sdk["OpenAI"].called


@pytest.mark.parametrize("note", ["a string note", {"text": "a dict note"}, ["a", "list"]])
def test_underscore_provider_keys_are_notes(tmp_path, note):
    cfg = _load(tmp_path, _providers({"_note": note, "chatgpt": {"api_key": "sk-test-key"}}))
    assert list(cfg.providers) == ["chatgpt"]


@pytest.mark.parametrize("value", ["sk-test-key", 5, None, ["a"]])
def test_non_dict_entry_raises(tmp_path, value):
    with pytest.raises(ValueError, match=r"providers\.chatgpt must be an object"):
        _load(tmp_path, _providers({"chatgpt": value}))


def test_deny_list_applies_to_every_model(tmp_path):
    cfg = _load(tmp_path, _providers({"claude": {
        "api_key": "sk-ant-test-key", "deny": ["web_search", "structured_json"]}}))
    pc = cfg.providers["claude"]
    assert pc.deny == {DENY_ALL_MODELS: ("web_search", "structured_json")}
    assert pc.denied_for("claude-a") == ("web_search", "structured_json")
    assert pc.denied_for(None) == ("web_search", "structured_json")


def test_deny_map_with_star_and_model_key(tmp_path):
    cfg = _load(tmp_path, MIXED)
    pc = cfg.providers["claude-vertex"]
    assert pc.deny == {"*": ("web_search",), "claude-b": ("structured_json",)}
    assert pc.deny_reason == REASON


def test_denied_for_is_star_plus_the_models_own_list():
    pc = ProviderConfig(deny={"*": ("web_search",),
                              "claude-b": ("structured_json", "web_search")})
    assert pc.denied_for("claude-a") == ("web_search",)
    assert pc.denied_for("claude-b") == ("web_search", "structured_json")  # union, no repeat
    assert pc.denied_for(None) == ("web_search",)
    only_model = ProviderConfig(deny={"claude-b": ("json_with_search",)})
    assert only_model.denied_for("claude-a") == ()
    assert only_model.denied_for("claude-b") == ("json_with_search",)


def test_deny_duplicates_are_removed(tmp_path):
    cfg = _load(tmp_path, _providers({"claude": {"api_key": "sk-ant-test-key", "deny": {
        "*": ["web_search", "web_search"],
        "claude-b": ["structured_json", "json_with_search", "structured_json"]}}}))
    pc = cfg.providers["claude"]
    assert pc.deny == {"*": ("web_search",), "claude-b": ("structured_json", "json_with_search")}


def test_empty_deny_is_no_restriction(tmp_path):
    cfg = _load(tmp_path, _providers({
        "a": {"provider": "claude", "api_key": "sk-ant-x", "deny": []},
        "b": {"provider": "claude", "api_key": "sk-ant-y", "deny": {"*": []}},
        "c": {"provider": "claude", "api_key": "sk-ant-z", "deny": None},
    }))
    for name in ("a", "b", "c"):
        assert cfg.providers[name].deny == {}


@pytest.mark.parametrize("deny, match", [
    (["bogus"], "not a deniable capability"),
    (["temperature"], "not a deniable capability"),
    (["thinking"], "not a deniable capability"),
    ({"*": ["thinking"]}, "not a deniable capability"),
    ({"claude-b": ["web_search", "temperature"]}, "not a deniable capability"),
    ("web_search", "must be a list"),
    ({"*": "web_search"}, "must be a list"),
    ({"claude-b": {"web_search": True}}, "must be a list"),
    (5, "must be a list"),
    ({"": ["web_search"]}, "non-empty model IDs"),
    ({"   ": ["web_search"]}, "non-empty model IDs"),
])
def test_invalid_deny_raises_naming_the_entry(tmp_path, deny, match):
    with pytest.raises(ValueError, match=match) as exc:
        _load(tmp_path, _providers({"claude-vertex": {
            "provider": "claude", "platform": "vertexai", "project_id": "my-project",
            "deny": deny}}))
    assert "claude-vertex" in str(exc.value)


def test_invalid_capability_error_lists_the_vocabulary(tmp_path):
    with pytest.raises(ValueError) as exc:
        _load(tmp_path, _providers({"claude": {"api_key": "sk-ant-x", "deny": ["thinking"]}}))
    for cap in DENIABLE_CAPABILITIES:
        assert cap in str(exc.value)


@pytest.mark.parametrize("reason", [5, ["a"], {"why": "x"}])
def test_non_string_deny_reason_raises(tmp_path, reason):
    with pytest.raises(ValueError, match=r"claude\.deny_reason must be a string"):
        _load(tmp_path, _providers({"claude": {"api_key": "sk-ant-x", "deny_reason": reason}}))


def test_deny_names_a_model_the_catalog_lacks_still_loads(tmp_path):
    cfg = _load(tmp_path, _providers({"claude": {
        "api_key": "sk-ant-x", "deny": {"claude-retired-1999": ["web_search"]}}}))
    assert cfg.providers["claude"].denied_for("claude-retired-1999") == ("web_search",)


# ============================================================ build_provider

def test_build_direct_entry(tmp_path, catalog, sdk):
    cfg = _load(tmp_path, MIXED)
    p = cfg.build_provider("claude")
    assert isinstance(p, ClaudeProvider)
    assert p.mode == "direct" and p.platform is None
    assert p.entry == "claude"
    assert p.model == "claude-a"  # default_model
    assert p._denied == frozenset() and p._deny_reason is None
    sdk["Anthropic"].assert_called_once_with(api_key="sk-ant-test-key")
    assert not sdk["AnthropicVertex"].called


def test_build_direct_entry_of_another_type(tmp_path, catalog, sdk):
    cfg = _load(tmp_path, MIXED)
    p = cfg.build_provider("chatgpt")
    assert isinstance(p, OpenAIProvider)
    assert (p.entry, p.model, p.mode) == ("chatgpt", "gpt-a", "direct")


def test_build_platform_entry(tmp_path, catalog, sdk):
    cfg = _load(tmp_path, MIXED)
    p = cfg.build_provider("claude-vertex")
    assert isinstance(p, ClaudeProvider)
    assert (p.mode, p.platform, p.location) == ("platform", "vertexai", "us")
    assert p.project_id == "my-project" and p.quota_project == "my-quota-project"
    assert p.entry == "claude-vertex" and p.model == "claude-a"
    assert p._denied == frozenset({"web_search"})
    assert p._deny_reason == REASON
    sdk["AnthropicVertex"].assert_called_once_with(
        project_id="my-project", region="us",
        default_headers={"x-goog-user-project": "my-quota-project"})
    assert not sdk["Anthropic"].called


def test_build_platform_entry_without_location_uses_the_platform_default(tmp_path, catalog, sdk):
    cfg = _load(tmp_path, MIXED)
    p = cfg.build_provider("claude-vertex-global")
    assert (p.platform, p.location, p.project_id) == ("vertexai", "global", "my-project")
    assert p.entry == "claude-vertex-global" and p._denied == frozenset()


def test_build_platform_entry_with_overrides(tmp_path, catalog, sdk):
    cfg = _load(tmp_path, MIXED)
    p = cfg.build_provider("claude-vertex", model="claude-c", location="global",
                           project_id="my-other-project", quota_project="my-billing-project",
                           require_pricing=False)
    assert (p.model, p.location, p.project_id) == ("claude-c", "global", "my-other-project")
    assert p.quota_project == "my-billing-project"
    assert p._require_pricing is False
    assert p.entry == "claude-vertex" and p.platform == "vertexai"
    sdk["AnthropicVertex"].assert_called_once_with(
        project_id="my-other-project", region="global",
        default_headers={"x-goog-user-project": "my-billing-project"})


def test_build_direct_entry_with_allowed_overrides(tmp_path, catalog, sdk):
    cfg = _load(tmp_path, MIXED)
    p = cfg.build_provider("claude", api_key="sk-ant-override", require_pricing=False)
    assert p._require_pricing is False and p.entry == "claude"
    sdk["Anthropic"].assert_called_once_with(api_key="sk-ant-override")


def test_api_key_override_makes_a_placeholder_entry_buildable(tmp_path, catalog, sdk):
    cfg = _load(tmp_path, MIXED)
    p = cfg.build_provider("grok", model="grok-a", api_key="xai-real-key")
    assert isinstance(p, GrokProvider) and p.entry == "grok"


@pytest.mark.parametrize("override", [
    {"platform": "vertexai"},
    {"backend": "vertexai"},
    {"mode": "direct"},
    {"deny": []},
    {"deny_reason": "none"},
    {"entry": "other"},
    {"gemini_api_key": "x"},
    {"bogus": 1},
])
@pytest.mark.parametrize("name", ["claude", "claude-vertex"])
def test_build_refuses_overrides_that_change_the_entry(tmp_path, catalog, sdk, name, override):
    cfg = _load(tmp_path, MIXED)
    with pytest.raises(ValueError, match="cannot be overridden") as exc:
        cfg.build_provider(name, **override)
    assert next(iter(override)) in str(exc.value)
    assert not sdk["Anthropic"].called and not sdk["AnthropicVertex"].called


@pytest.mark.parametrize("override", [
    {"location": "us"}, {"project_id": "my-project"}, {"quota_project": "my-project"},
])
def test_build_refuses_platform_overrides_on_a_direct_entry(tmp_path, catalog, sdk, override):
    cfg = _load(tmp_path, MIXED)
    with pytest.raises(ValueError, match="platform entries only"):
        cfg.build_provider("claude", **override)
    assert not sdk["Anthropic"].called


@pytest.mark.parametrize("name, match", [
    ("claude-old", "disabled"),
    ("nope", "No ai_config entry 'nope'"),
    ("grok", "no usable api_key"),
    ("openai", "provider"),
])
def test_build_refuses_unusable_entries(tmp_path, catalog, sdk, name, match):
    cfg = _load(tmp_path, MIXED)
    with pytest.raises(ValueError, match=match) as exc:
        cfg.build_provider(name)
    assert f"'{name}'" in str(exc.value)
    assert not any(s.called for s in sdk.values())


def test_build_resolves_deny_per_model(tmp_path, catalog, sdk):
    """One entry built for several models: "*" for all, a model's own list for it only."""
    cfg = _load(tmp_path, MIXED)
    a = cfg.build_provider("claude-vertex", model="claude-a")
    b = cfg.build_provider("claude-vertex", model="claude-b")
    c = cfg.build_provider("claude-vertex", model="claude-c")  # not named in the map
    assert a._denied == frozenset({"web_search"})
    assert b._denied == frozenset({"web_search", "structured_json"})
    assert c._denied == frozenset({"web_search"})

    # And it is what the request path enforces.
    for p in (a, b, c):
        p._client = StubAnthropic([claude_response()])
    with pytest.raises(DjinniteCapabilityDeniedError) as exc:
        b.generate_json("q", schema=SCHEMA)
    assert exc.value.model == "claude-b" and exc.value.capabilities == ["structured_json"]
    assert b._client.calls == []
    for p in (a, c):
        p.generate_json("q", schema=SCHEMA)
        assert len(p._client.calls) == 1


def test_provider_kwargs_carries_no_restriction(tmp_path, catalog, sdk):
    """Building by hand from provider_kwargs is deliberately unrestricted."""
    cfg = _load(tmp_path, MIXED)
    p = get_provider(cfg.provider_type("claude-vertex"), model="claude-a",
                     **cfg.provider_kwargs("claude-vertex"))
    assert p.entry is None and p._denied == frozenset()


def test_get_provider_accepts_deny_without_an_entry(catalog, sdk):
    p = get_provider("claude", api_key="sk-ant-x", model="claude-a",
                     deny=["web_search", "web_search"], deny_reason=REASON)
    assert p.entry is None
    assert p._denied == frozenset({"web_search"})
    p._client = StubAnthropic([claude_response()])
    with pytest.raises(DjinniteCapabilityDeniedError) as exc:
        p.generate("q", web_search=True)
    assert exc.value.entry is None
    assert "built with deny" in str(exc.value) and REASON in str(exc.value)
    assert p._client.calls == []


@pytest.mark.parametrize("deny", [["bogus"], ["temperature"], ["thinking"], "web_search"])
def test_get_provider_rejects_invalid_deny(catalog, sdk, deny):
    with pytest.raises(ValueError, match="get_provider"):
        get_provider("claude", api_key="sk-ant-x", model="claude-a", deny=deny)
    assert not sdk["Anthropic"].called


def test_get_provider_hints_at_build_provider_for_an_entry_name():
    with pytest.raises(ValueError, match=r"build_provider\('claude-vertex'\)"):
        get_provider("claude-vertex")


# ====================================================== deny pre-flight

CLASSES = [ClaudeProvider, GeminiProvider, OpenAIProvider, GrokProvider]

ALL_ON = info(structured_json=["on", "off"], web_search=["on", "off"],
              json_with_search=["on", "off"])


def _restricted(cls, deny, model_info=ALL_ON, entry="my-entry", reason=REASON):
    """A provider of ``cls`` with a stub client and ``deny`` applied."""
    if cls is ClaudeProvider:
        client = StubAnthropic([claude_response(), claude_response()])
    elif cls is GeminiProvider:
        client = StubGenai()
    else:
        client = StubResponses()
    p = bare(cls, client, model_info=model_info)
    p._set_access_entry(entry, deny, reason)
    return p, client


def _ncalls(client):
    if isinstance(client, StubAnthropic):
        return len(client.calls) + len(client.create_calls) + len(client.count_calls)
    return len(client.calls)


# (request, denied capability the request uses)
DENIED_REQUESTS = [
    pytest.param(lambda p: p.generate("q", web_search=True), "web_search",
                 id="generate-web_search"),
    pytest.param(lambda p: p.generate_json("q", schema=SCHEMA), "structured_json",
                 id="generate_json"),
    pytest.param(lambda p: p.generate_json("q", schema=SCHEMA, web_search=True),
                 "json_with_search", id="generate_json-web_search"),
]


@pytest.mark.parametrize("call, cap", DENIED_REQUESTS)
@pytest.mark.parametrize("cls", CLASSES)
def test_denied_request_raises_before_any_call(cls, call, cap):
    p, client = _restricted(cls, [cap])
    with pytest.raises(DjinniteCapabilityDeniedError) as exc:
        call(p)
    e = exc.value
    assert (e.entry, e.model, e.reason) == ("my-entry", "test-model", REASON)
    assert e.capabilities == [cap]
    assert e.provider == cls.PROVIDER_NAME
    msg = str(e)
    assert "'my-entry'" in msg and "'test-model'" in msg and REASON in msg
    assert f"{cap}=on" in msg and "deployment restriction" in msg
    assert _ncalls(client) == 0


@pytest.mark.parametrize("cls", CLASSES)
def test_web_search_deny_also_blocks_json_with_search(cls):
    p, client = _restricted(cls, ["web_search"])
    with pytest.raises(DjinniteCapabilityDeniedError) as exc:
        p.generate_json("q", schema=SCHEMA, web_search=True)
    assert exc.value.capabilities == ["web_search"]
    assert _ncalls(client) == 0


@pytest.mark.parametrize("call, cap", DENIED_REQUESTS)
@pytest.mark.parametrize("cls", CLASSES)
def test_deny_is_enforced_without_a_catalog_entry(cls, call, cap):
    p, client = _restricted(cls, [cap], model_info=None)
    with pytest.raises(DjinniteCapabilityDeniedError):
        call(p)
    assert _ncalls(client) == 0


@pytest.mark.parametrize("web_search, cap", [(False, "structured_json"),
                                             (True, "json_with_search"),
                                             (True, "web_search")])
@pytest.mark.parametrize("model_info", [ALL_ON, None], ids=["catalog", "no-catalog"])
@pytest.mark.parametrize("cls", CLASSES)
def test_force_does_not_bypass_deny(cls, model_info, web_search, cap):
    p, client = _restricted(cls, [cap], model_info=model_info)
    with pytest.raises(DjinniteCapabilityDeniedError):
        p.generate_json("q", schema=SCHEMA, web_search=web_search, force=True)
    assert _ncalls(client) == 0


ALLOWED_REQUESTS = [
    # (deny, request) -- the request uses nothing the entry denies
    pytest.param(["json_with_search"], lambda p: p.generate("q"), id="plain"),
    pytest.param(["json_with_search"], lambda p: p.generate("q", web_search=True),
                 id="search-jws-denied"),
    pytest.param(["structured_json"], lambda p: p.generate("q", web_search=True),
                 id="search-json-denied"),
    pytest.param(["json_with_search"], lambda p: p.generate_json("q", schema=SCHEMA),
                 id="json-jws-denied"),
    pytest.param(["web_search"], lambda p: p.generate_json("q", schema=SCHEMA),
                 id="json-search-denied"),
    pytest.param(["web_search", "json_with_search"], lambda p: p.generate("q"),
                 id="plain-both-denied"),
]


@pytest.mark.parametrize("deny, call", ALLOWED_REQUESTS)
@pytest.mark.parametrize("cls", CLASSES)
def test_requests_using_nothing_denied_go_through(cls, deny, call):
    p, client = _restricted(cls, deny)
    resp = call(p)
    assert _ncalls(client) == 1
    assert resp.content


@pytest.mark.parametrize("cls", CLASSES)
def test_no_deny_is_unrestricted(cls):
    p, client = _restricted(cls, [])
    p.generate_json("q", schema=SCHEMA, web_search=True)
    assert _ncalls(client) == 1


def test_denied_error_type_and_export():
    assert issubclass(DjinniteCapabilityDeniedError, AIProviderError)
    assert djinnite.DjinniteCapabilityDeniedError is DjinniteCapabilityDeniedError
    assert ap.DjinniteCapabilityDeniedError is DjinniteCapabilityDeniedError
    assert "DjinniteCapabilityDeniedError" in djinnite.__all__
    assert "DjinniteCapabilityDeniedError" in ap.__all__


def test_catalog_message_wins_on_generate_json():
    """Catalog and deny both reject: the catalog's message is raised."""
    p, client = _restricted(ClaudeProvider, ["structured_json"],
                            model_info=info(structured_json=["off"]))
    with pytest.raises(AIProviderError) as exc:
        p.generate_json("q", schema=SCHEMA)
    assert not isinstance(exc.value, DjinniteCapabilityDeniedError)
    assert "capabilities.structured_json" in str(exc.value)
    assert _ncalls(client) == 0


def test_catalog_message_wins_on_claude_generate_web_search():
    p, client = _restricted(ClaudeProvider, ["web_search"], model_info=info(web_search=["off"]))
    with pytest.raises(AIProviderError) as exc:
        p.generate("q", web_search=True)
    assert _ncalls(client) == 0
    assert not isinstance(exc.value, DjinniteCapabilityDeniedError)
    assert "capabilities.web_search" in str(exc.value)


# ================================================================ resolution

def test_resolve_use_case_returns_the_type_for_a_renamed_entry(tmp_path):
    cfg = _load(tmp_path, MIXED)
    choice = cfg.resolve_use_case("coding")  # default_provider: claude-vertex
    assert isinstance(choice, ModelChoice)
    assert choice == ModelChoice("claude-vertex", "claude", "claude-b")
    assert (choice.entry, choice.provider_type, choice.model) == ("claude-vertex", "claude", "claude-b")
    assert cfg.resolve_use_case("unlisted") == ModelChoice("claude-vertex", "claude", "claude-a")
    assert cfg.resolve_use_case("coding", entry="chatgpt") == ModelChoice("chatgpt", "chatgpt", "gpt-a")
    assert cfg.resolve_use_case("x", entry="claude-vertex-global").provider_type == "claude"


def test_resolve_use_case_refuses_disabled_and_unknown_entries(tmp_path):
    cfg = _load(tmp_path, MIXED)
    for name in ("claude-old", "nope"):
        with pytest.raises(ValueError, match=name):
            cfg.resolve_use_case("coding", entry=name)


def test_get_model_for_use_case_is_still_a_plain_2_tuple(tmp_path):
    cfg = _load(tmp_path, MIXED)
    result = cfg.get_model_for_use_case("coding")
    assert type(result) is tuple and len(result) == 2
    assert result == ("claude-vertex", "claude-b")
    assert cfg.get_model_for_use_case("coding", "claude") == ("claude", "claude-b")


def _caps_catalog():
    direct = ModelCapabilities(thinking=["on", "off"], structured_json=["on", "off"],
                               web_search=["on", "off"])  # json_with_search unknown (None)
    on_vertex = ModelCapabilities(thinking=["on"])
    return ModelCatalog(providers={"claude": [
        _minfo("claude-a", direct, platform_caps=on_vertex),
        _minfo("claude-b", direct, platform_caps=on_vertex),
    ]})


def test_capabilities_for_direct_entry(tmp_path):
    cfg = _load(tmp_path, MIXED)
    cat = _caps_catalog()
    caps = cfg.capabilities_for("claude", "claude-a", cat)
    assert caps.thinking == ["on", "off"]
    assert caps.web_search == ["on", "off"]
    assert caps.structured_json == ["on", "off"]
    assert caps.json_with_search is None


def test_capabilities_for_platform_entry_overlays_platform_facts(tmp_path):
    cfg = _load(tmp_path, MIXED)
    cat = _caps_catalog()
    caps = cfg.capabilities_for("claude-vertex-global", "claude-a", cat)  # no deny
    assert caps.thinking == ["on"]                 # layer 2: probed on Vertex
    assert caps.web_search == ["on", "off"]        # not probed: direct value


def test_capabilities_for_applies_deny_last(tmp_path):
    cfg = _load(tmp_path, MIXED)
    cat = _caps_catalog()
    a = cfg.capabilities_for("claude-vertex", "claude-a", cat)
    assert a.thinking == ["on"]                    # platform overlay kept
    assert a.web_search == ["off"]                 # "*" deny removes "on"
    assert a.structured_json == ["on", "off"]      # only denied for claude-b
    b = cfg.capabilities_for("claude-vertex", "claude-b", cat)
    assert b.web_search == ["off"] and b.structured_json == ["off"]
    # The catalog itself is untouched.
    original = cat.get_model("claude", "claude-b").capabilities
    assert original.web_search == ["on", "off"] and original.structured_json == ["on", "off"]
    assert original.thinking == ["on", "off"]


def test_capabilities_for_denied_unknown_becomes_off(tmp_path):
    cfg = _load(tmp_path, _providers({"claude": {"api_key": "sk-ant-x",
                                                 "deny": ["json_with_search"]}}))
    caps = cfg.capabilities_for("claude", "claude-a", _caps_catalog())
    assert caps.json_with_search == ["off"]        # was None (unknown)
    assert caps.web_search == ["on", "off"]


def test_capabilities_for_unknown_model_or_entry_raises(tmp_path):
    cfg = _load(tmp_path, MIXED)
    with pytest.raises(KeyError, match="claude-z"):
        cfg.capabilities_for("claude", "claude-z", _caps_catalog())
    with pytest.raises(KeyError):
        cfg.capabilities_for("nope", "claude-a", _caps_catalog())


def test_with_denied_leaves_the_original_alone():
    m = _minfo("claude-a", ModelCapabilities(web_search=["on", "off"], structured_json=["on"]))
    view = m.with_denied(("web_search", "structured_json"))
    assert view.capabilities.web_search == ["off"]
    assert view.capabilities.structured_json == ["off"]
    assert m.capabilities.web_search == ["on", "off"] and m.capabilities.structured_json == ["on"]
    assert m.with_denied(()) is m


# ------------------------------------------------------------- direct_entry

def test_direct_entry_prefers_the_entry_named_after_the_type(tmp_path):
    cfg = _load(tmp_path, _providers({
        "claude-2": {"provider": "claude", "api_key": "sk-ant-two"},
        "claude": {"api_key": "sk-ant-one"},
        "claude-vertex": {"provider": "claude", "platform": "vertexai", "project_id": "my-project"},
    }))
    assert cfg.direct_entry("claude") == "claude"


def test_direct_entry_single_differently_named(tmp_path):
    cfg = _load(tmp_path, _providers({
        "claude-direct": {"provider": "claude", "api_key": "sk-ant-one"},
        "claude-vertex": {"provider": "claude", "platform": "vertexai", "project_id": "my-project"},
    }))
    assert cfg.direct_entry("claude") == "claude-direct"


def test_direct_entry_platform_only_is_none(tmp_path):
    cfg = _load(tmp_path, _providers({
        "claude-vertex": {"provider": "claude", "platform": "vertexai", "project_id": "my-project"},
        "claude": {"mode": "platform", "platform": "vertexai", "project_id": "my-project"},
    }))
    assert cfg.direct_entry("claude") is None
    assert cfg.direct_entry("gemini") is None  # not configured at all


def test_direct_entry_ambiguous_raises(tmp_path):
    cfg = _load(tmp_path, _providers({
        "claude-a": {"provider": "claude", "api_key": "sk-ant-one"},
        "claude-b": {"provider": "claude", "api_key": "sk-ant-two"},
    }))
    with pytest.raises(ValueError) as exc:
        cfg.direct_entry("claude")
    msg = str(exc.value)
    assert "claude-a" in msg and "claude-b" in msg
    assert "name one entry 'claude'" in msg and "disable" in msg


def test_direct_entry_ignores_disabled_and_placeholder_entries(tmp_path):
    cfg = _load(tmp_path, _providers({
        "claude": {"api_key": "sk-ant-one", "enabled": False},
        "claude-placeholder": {"provider": "claude", "api_key": "your-key-here"},
        "claude-nokey": {"provider": "claude"},
        "claude-real": {"provider": "claude", "api_key": "sk-ant-two"},
        "claude-off": {"provider": "claude", "api_key": "sk-ant-three", "enabled": False},
    }))
    assert cfg.direct_entry("claude") == "claude-real"


def test_direct_entry_in_the_mixed_config(tmp_path):
    cfg = _load(tmp_path, MIXED)
    assert cfg.direct_entry("claude") == "claude"
    assert cfg.direct_entry("chatgpt") == "chatgpt"
    assert cfg.direct_entry("grok") is None  # placeholder key


@pytest.mark.parametrize("cls", CLASSES)
def test_catalog_incompatible_combination_wins_over_deny(cls):
    """A catalog `incompatible` combination that a deny also blocks gets the catalog message."""
    caps = info(structured_json=["on", "off"], web_search=["on", "off"],
                json_with_search=["on", "off"],
                incompatible=[{"structured_json": "on", "web_search": "on"}])
    p, client = _restricted(cls, ["json_with_search"], model_info=caps)
    with pytest.raises(Exception) as exc:
        p.generate_json("q", schema=SCHEMA, web_search=True, force=True)
    assert not isinstance(exc.value, DjinniteCapabilityDeniedError)
    assert "incompatible" in str(exc.value)
    assert _ncalls(client) == 0


@pytest.mark.parametrize("value", ["", None])
def test_explicit_empty_or_null_provider_is_rejected(tmp_path, value):
    with pytest.raises(ValueError, match=r"providers\.claude\.provider"):
        _load(tmp_path, _providers({"claude": {"provider": value, "api_key": "sk-ant-x"}}))


@pytest.mark.parametrize("denied", [["web_search"], ["structured_json"]])
def test_with_denied_takes_json_with_search_off_with_either_part(denied):
    m = _minfo("claude-a", ModelCapabilities(structured_json=["on", "off"],
                                             web_search=["on", "off"],
                                             json_with_search=["on", "off"]))
    caps = m.with_denied(denied).capabilities
    assert caps.json_with_search == ["off"]
    other = "structured_json" if denied == ["web_search"] else "web_search"
    assert getattr(caps, other) == ["on", "off"]
    assert m.capabilities.json_with_search == ["on", "off"]  # original untouched


def test_platform_block_location_is_the_default_for_its_entries(tmp_path):
    cfg = _load(tmp_path, {
        "platforms": {"vertexai": {"project_id": "my-project", "location": "us"}},
        "providers": {
            "claude-vertex": {"provider": "claude", "platform": "vertexai"},
            "claude-eu": {"provider": "claude", "platform": "vertexai", "location": "eu"},
        },
    })
    assert cfg.provider_kwargs("claude-vertex")["location"] == "us"
    assert cfg.provider_kwargs("claude-eu")["location"] == "eu"  # the entry wins


@pytest.mark.parametrize("deny", [5, {"": ["web_search"]}])
def test_deny_shape_errors_list_the_vocabulary(tmp_path, deny):
    with pytest.raises(ValueError) as exc:
        _load(tmp_path, _providers({"claude": {"api_key": "sk-ant-x", "deny": deny}}))
    assert "providers.claude.deny" in str(exc.value)
    assert all(cap in str(exc.value) for cap in DENIABLE_CAPABILITIES)
