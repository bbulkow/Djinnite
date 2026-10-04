"""
Claude on the ``vertexai`` platform (offline; the SDK clients are mocked).

Platform mode builds ``anthropic.AnthropicVertex`` with Application Default
Credentials -- never ``anthropic.Anthropic``, never an API key -- and keeps
the request body identical to the direct path.

    uv run pytest tests/test_platform_vertexai_claude.py -v
"""

import json
from unittest import mock

import anthropic
import pytest

from djinnite.ai_providers import get_provider
from djinnite.ai_providers.base_provider import (
    AIProviderError, AIRateLimitError, AIAuthenticationError, AIModelNotFoundError,
)
from djinnite.ai_providers.claude_provider import ClaudeProvider
from djinnite.config_loader import ModelCatalog, ModelInfo, PlatformModelInfo, Modalities
from djinnite.tests._stubs import (
    INFO, StubAnthropic, claude_response, anthropic_error,
)

SCHEMA = {"type": "object", "properties": {"v": {"type": "integer"}}, "required": ["v"]}


@pytest.fixture
def sdk():
    """Patch both Anthropic client classes; yields (Anthropic, AnthropicVertex) mocks."""
    with mock.patch.object(anthropic, "Anthropic") as direct, \
         mock.patch.object(anthropic, "AnthropicVertex") as vertex:
        yield direct, vertex


def _vertex(location="global", responses=(), model="claude-sonnet-5-5", **stub_kw):
    """A Vertex-mode provider wired to a recording stub client."""
    with mock.patch.object(anthropic, "AnthropicVertex"):
        p = ClaudeProvider(model=model, model_info=INFO, platform="vertexai",
                           project_id="proj", location=location)
    stub = StubAnthropic(responses, **stub_kw)
    p._client = stub
    return p, stub


# ---------------------------------------------------------------- client

def test_vertex_client_gets_project_region_and_quota_header_no_key(sdk):
    direct, vertex = sdk
    p = ClaudeProvider(model="claude-sonnet-5-5", platform="vertexai",
                       project_id="my-project", location="us",
                       quota_project="my-quota-project")
    vertex.assert_called_once_with(
        project_id="my-project", region="us",
        default_headers={"x-goog-user-project": "my-quota-project"},
    )
    direct.assert_not_called()
    assert p.mode == "platform" and p.platform == "vertexai"


def test_vertex_without_quota_project_sends_no_header(sdk):
    _, vertex = sdk
    ClaudeProvider(model="claude-sonnet-5-5", platform="vertexai", project_id="p")
    assert vertex.call_args.kwargs == {"project_id": "p", "region": "global"}


def test_backend_alias_and_default_location(sdk):
    _, vertex = sdk
    p = ClaudeProvider(model="claude-opus-5-5", backend="vertexai", project_id="p")
    assert p.platform == "vertexai" and p.location == "global"
    vertex.assert_called_once()


def test_api_key_is_never_sent_on_vertex(sdk):
    direct, vertex = sdk
    ClaudeProvider(api_key="sk-ant-should-not-travel", model="claude-sonnet-5-5",
                   platform="vertexai", project_id="p")
    assert "sk-ant-should-not-travel" not in repr(vertex.call_args)
    direct.assert_not_called()


def test_vertex_requires_project_id(sdk):
    with pytest.raises(AIProviderError, match="project_id is required"):
        ClaudeProvider(model="claude-sonnet-5-5", platform="vertexai")


@pytest.mark.parametrize("backend", [None, "gemini", "anthropic"])
def test_direct_mode_construction_unchanged(sdk, backend):
    """Anything but vertexai is direct: exactly Anthropic(api_key=...)."""
    direct, vertex = sdk
    kw = {} if backend is None else {"backend": backend}
    p = ClaudeProvider(api_key="k", model="claude-sonnet-5-5", **kw)
    direct.assert_called_once_with(api_key="k")
    vertex.assert_not_called()
    assert p.mode == "direct" and p._price_multiplier == 1.0


def test_get_provider_constructs_with_no_key(sdk):
    _, vertex = sdk
    p = get_provider("claude", model="claude-sonnet-5-5", platform="vertexai",
                     project_id="p", location="us")
    assert isinstance(p, ClaudeProvider) and p.api_key is None
    vertex.assert_called_once()


def test_dated_snapshot_ids_use_at_on_vertex():
    p, _ = _vertex(model="claude-haiku-4-5-20251001")
    assert p._wire_model() == "claude-haiku-4-5@20251001"
    p, _ = _vertex(model="claude-sonnet-5-5")
    assert p._wire_model() == "claude-sonnet-5-5"


# ------------------------------------------------------- generate_json trip

def test_generate_json_round_trip():
    p, stub = _vertex(responses=[claude_response(
        text='{"v": 3}', input_tokens=1000, output_tokens=500, thinking_tokens=120)])
    r = p.generate_json("q", schema=SCHEMA, force=True)
    req = stub.calls[0]  # sent through messages.stream (streaming)
    assert req["model"] == "claude-sonnet-5-5"
    assert req["output_config"]["format"] == {
        "type": "json_schema",
        "schema": {"type": "object", "properties": {"v": {"type": "integer"}},
                   "required": ["v"], "additionalProperties": False},
    }
    assert json.loads(r.content) == {"v": 3}
    assert r.finish_reason == "end_turn"
    assert (r.input_tokens, r.output_tokens, r.thinking_tokens) == (1000, 500, 120)
    # $1 / $2 per 1M at global: 1000*1e-6 + 500*2e-6 -- thinking counted once.
    assert r.usage["token_cost"] == pytest.approx(0.002)
    assert "price_multiplier" not in r.usage


@pytest.mark.parametrize("location, multiplier", [
    ("global", 1.0), ("us", 1.10), ("eu", 1.10), ("us-east5", 1.10), ("GLOBAL", 1.0),
])
def test_cost_depends_on_location(location, multiplier):
    p, _ = _vertex(location=location, responses=[claude_response(
        input_tokens=1_000_000, output_tokens=1_000_000, thinking_tokens=0)])
    r = p.generate_json("q", schema=SCHEMA, force=True)
    assert r.usage["token_cost"] == pytest.approx(3.0 * multiplier)
    if multiplier == 1.0:
        assert "price_multiplier" not in r.usage
    else:
        assert r.usage["price_multiplier"] == pytest.approx(multiplier)


def test_sonnet_55_price_at_global_and_us():
    """The catalog price at global; +10% at the us multi-region."""
    for location, expected in (("global", 2.0 + 10.0), ("us", (2.0 + 10.0) * 1.1)):
        with mock.patch.object(anthropic, "AnthropicVertex"):
            p = get_provider("claude", model="claude-sonnet-5-5", platform="vertexai",
                             project_id="p", location=location)
        p._client = StubAnthropic([claude_response(input_tokens=1_000_000,
                                                   output_tokens=1_000_000)])
        r = p.generate_json("q", schema=SCHEMA)
        assert r.usage["token_cost"] == pytest.approx(expected)


def test_web_search_uses_the_platform_tool_version():
    p, stub = _vertex(responses=[claude_response(text="hi")])
    p.generate("q", web_search=True)
    assert stub.calls[0]["tools"] == [{"type": "web_search_20250305", "name": "web_search"}]


# --------------------------------------------------------------- errors

@pytest.mark.parametrize("json_mode", [False, True])
@pytest.mark.parametrize("err, expected, text", [
    (anthropic_error(anthropic.RateLimitError, 429, "Quota exceeded for base_model", "RESOURCE_EXHAUSTED"),
     AIRateLimitError, "Quota exceeded for base_model"),
    (anthropic_error(anthropic.PermissionDeniedError, 403,
                     "Permission 'aiplatform.endpoints.predict' denied", "PERMISSION_DENIED"),
     AIAuthenticationError, "Permission 'aiplatform.endpoints.predict' denied"),
    (anthropic_error(anthropic.NotFoundError, 404, "Publisher model was not found", "NOT_FOUND"),
     AIModelNotFoundError, "Model Garden"),
])
def test_vertex_errors_are_mapped(json_mode, err, expected, text):
    p, _ = _vertex(responses=[err])
    with pytest.raises(expected) as ei:
        if json_mode:
            p.generate_json("q", schema=SCHEMA, force=True)
        else:
            p.generate("q")
    assert text in str(ei.value)
    assert ei.value.original_error is err


def test_missing_adc_maps_to_authentication_error():
    from google.auth.exceptions import DefaultCredentialsError
    p, _ = _vertex(responses=[DefaultCredentialsError("Could not automatically determine credentials")])
    with pytest.raises(AIAuthenticationError, match="Application Default Credentials"):
        p.generate_json("q", schema=SCHEMA, force=True)


def test_direct_mode_error_mapping_unchanged():
    """Direct generate_json still wraps a 429 as a plain AIProviderError."""
    err = anthropic_error(anthropic.RateLimitError, 429, "rate limited")
    p, stub = _vertex(responses=[err])
    p.platform = None  # same instance, direct mode
    with pytest.raises(AIProviderError) as ei:
        p.generate_json("q", schema=SCHEMA, force=True)
    assert type(ei.value) is AIProviderError
    assert str(ei.value).startswith("[claude] JSON generation failed")
    stub._responses = [err]
    with pytest.raises(AIRateLimitError, match="Rate limit exceeded"):
        p.generate("q")


# ------------------------------------------- is_available / list / probe

def test_is_available_on_vertex_is_a_live_call():
    p, stub = _vertex()
    assert p.is_available() is True
    assert stub.count_calls == [{"model": "claude-sonnet-5-5",
                                 "messages": [{"role": "user", "content": "test"}]}]
    p, _ = _vertex(count_error=anthropic_error(anthropic.RateLimitError, 429, "zero quota"))
    assert p.is_available() is False


@pytest.mark.parametrize("err, status", [
    (None, "available"),
    (anthropic_error(anthropic.RateLimitError, 429, "q", "RESOURCE_EXHAUSTED"), "no_quota"),
    (anthropic_error(anthropic.PermissionDeniedError, 403, "denied"), "no_access"),
    (anthropic_error(anthropic.NotFoundError, 404, "not found"), "not_found"),
    (anthropic_error(anthropic.InternalServerError, 500, "oops"), "unknown"),
])
def test_probe_availability_statuses(err, status):
    p, _ = _vertex(count_error=err)
    got, detail = p.probe_availability()
    assert got == status
    assert (detail == "") == (err is None)
    assert detail.isascii()


def test_list_models_on_vertex_reads_the_platform_block(monkeypatch):
    def model(mid, statuses):
        return ModelInfo(id=mid, name=mid, context_window=1_000_000, max_output_tokens=128_000,
                         modalities=Modalities(input=["text", "vision"]),
                         platforms={"vertexai": PlatformModelInfo(locations=statuses)} if statuses else {})
    catalog = ModelCatalog(providers={"claude": [
        model("claude-sonnet-5-5", {"global": "available", "us": "no_quota"}),
        model("claude-opus-5-5", {"global": "not_found"}),
        model("claude-sonnet-5", None),
    ]})
    import djinnite.config_loader as cl
    monkeypatch.setattr(cl, "load_model_catalog", lambda *a, **k: catalog)

    p, _ = _vertex(location="global")
    assert [m["id"] for m in p.list_models()] == ["claude-sonnet-5-5"]
    p, _ = _vertex(location="us")
    assert p.list_models() == []
