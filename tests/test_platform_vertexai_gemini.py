"""
Gemini on the ``vertexai`` platform (offline; ``genai.Client`` is mocked).

Covers Munin's acceptance criteria: ``genai.Client`` receives
``vertexai=True``, the project and ``location="global"`` and no ``api_key``;
the default location is unchanged (``us-central1``); and ``get_provider``
constructs with no key.

    uv run pytest tests/test_platform_vertexai_gemini.py -v
"""

from types import SimpleNamespace as NS
from unittest import mock

import pytest
from google import genai
from google.genai import errors as genai_errors

from djinnite.ai_providers import get_provider
from djinnite.ai_providers.base_provider import (
    AIProviderError, AIRateLimitError, AIAuthenticationError, AIModelNotFoundError,
)
from djinnite.ai_providers.gemini_provider import GeminiProvider
from djinnite.tests._stubs import INFO, StubGenai

SCHEMA = {"type": "object", "properties": {"v": {"type": "integer"}}, "required": ["v"]}


@pytest.fixture
def client_cls():
    with mock.patch.object(genai, "Client") as cls:
        yield cls


def _err(code, status, message="boom"):
    return genai_errors.ClientError(code, {"error": {"code": code, "status": status, "message": message}})


def _vertex(error=None, location="global"):
    with mock.patch.object(genai, "Client"):
        p = GeminiProvider(api_key=None, model="gemini-3.5-flash", model_info=INFO,
                           platform="vertexai", project_id="proj", location=location)
    p._client = StubGenai(error=error)
    return p


# ---------------------------------------------------------------- client

def test_vertex_adc_client_kwargs(client_cls):
    GeminiProvider(api_key=None, model="gemini-3.5-flash", backend="vertexai",
                   project_id="my-project", location="global")
    client_cls.assert_called_once_with(vertexai=True, project="my-project", location="global")
    assert "api_key" not in client_cls.call_args.kwargs


def test_default_location_is_unchanged(client_cls):
    p = GeminiProvider(api_key=None, model="gemini-3.5-flash", backend="vertexai", project_id="p")
    assert client_cls.call_args.kwargs["location"] == "us-central1"
    assert p.location == "us-central1"


def test_keyed_vertex_kwargs_are_todays(client_cls):
    GeminiProvider(api_key="AIza-key", model="gemini-3.5-flash", backend="vertexai", project_id="p")
    assert client_cls.call_args.kwargs == {
        "api_key": "AIza-key", "vertexai": True, "project": "p", "location": "us-central1",
    }


def test_ai_studio_kwargs_are_todays(client_cls):
    p = GeminiProvider(api_key="AIza-key", model="gemini-3.5-flash")
    client_cls.assert_called_once_with(api_key="AIza-key")
    assert p.mode == "direct" and p.backend == "gemini"


def test_platform_kwarg_is_the_same_as_the_backend_alias(client_cls):
    p = GeminiProvider(api_key=None, model="gemini-3.5-flash", platform="vertexai",
                       project_id="p", location="global")
    assert p.backend == "vertexai" and p.platform == "vertexai" and p.mode == "platform"
    assert client_cls.call_args.kwargs["vertexai"] is True


def test_get_provider_constructs_with_no_key(client_cls):
    p = get_provider("gemini", model="gemini-3.5-flash", backend="vertexai",
                     project_id="p", location="global")
    assert isinstance(p, GeminiProvider) and p.api_key is None
    client_cls.assert_called_once_with(vertexai=True, project="p", location="global")


def test_quota_project_travels_on_the_credentials(client_cls):
    creds = object()
    with mock.patch("google.auth.default", return_value=(creds, None)) as adc:
        GeminiProvider(api_key=None, model="gemini-3.5-flash", platform="vertexai",
                       project_id="p", location="global", quota_project="billing-proj")
    adc.assert_called_once_with(
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
        quota_project_id="billing-proj",
    )
    assert client_cls.call_args.kwargs["credentials"] is creds
    assert "api_key" not in client_cls.call_args.kwargs


def test_quota_project_with_api_key_is_rejected(client_cls):
    with pytest.raises(AIProviderError, match="quota_project applies to Application Default"):
        GeminiProvider(api_key="AIza-key", model="gemini-3.5-flash", platform="vertexai",
                       project_id="p", quota_project="q")


def test_vertex_requires_project_id(client_cls):
    with pytest.raises(AIProviderError, match="project_id is required"):
        GeminiProvider(api_key=None, model="gemini-3.5-flash", platform="vertexai")


# ------------------------------------------------------------ no-key ops

def test_is_available_and_list_models_work_without_a_key():
    p = _vertex()
    listed = [NS(name="publishers/google/models/gemini-3.5-flash", display_name="Gemini 3.5 Flash",
                 input_token_limit=1_048_576, output_token_limit=65_536)]
    p._client.models.list = lambda **kw: listed
    assert p.is_available() is True
    models = p.list_models()
    assert [m["id"] for m in models] == ["gemini-3.5-flash"]


def test_direct_mode_still_needs_a_key():
    p = _vertex()
    p.platform = None
    p.api_key = None
    assert p.is_available() is False
    assert p.list_models() == []


# --------------------------------------------------------------- errors

@pytest.mark.parametrize("json_mode", [False, True])
@pytest.mark.parametrize("err, expected", [
    (_err(429, "RESOURCE_EXHAUSTED", "Quota exceeded"), AIRateLimitError),
    # Mentions "quota" but is an IAM refusal: must NOT become a rate limit.
    (_err(403, "PERMISSION_DENIED", "Your application is authenticating by using local "
          "Application Default Credentials. The aiplatform.googleapis.com API requires a quota project"),
     AIAuthenticationError),
    (_err(404, "NOT_FOUND", "Publisher Model `gemini-3.5-flash` was not found"), AIModelNotFoundError),
])
def test_vertex_errors_are_mapped(json_mode, err, expected):
    p = _vertex(error=err)
    with pytest.raises(expected) as ei:
        if json_mode:
            p.generate_json("q", schema=SCHEMA, force=True)
        else:
            p.generate("q")
    assert type(ei.value) is expected
    assert ei.value.original_error is err


def test_missing_adc_maps_to_authentication_error():
    from google.auth.exceptions import DefaultCredentialsError
    p = _vertex(error=DefaultCredentialsError("no ADC"))
    with pytest.raises(AIAuthenticationError, match="Application Default Credentials"):
        p.generate_json("q", schema=SCHEMA, force=True)


def test_ai_studio_error_mapping_unchanged():
    """Direct mode keeps the message-matching map (a 'quota' 403 is a rate limit there)."""
    p = _vertex(error=_err(403, "PERMISSION_DENIED", "quota project missing"))
    p.platform = None
    with pytest.raises(AIRateLimitError, match="Rate limit or quota exceeded"):
        p.generate("q")


@pytest.mark.parametrize("err, status", [
    (None, "available"),
    (_err(429, "RESOURCE_EXHAUSTED"), "no_quota"),
    (_err(403, "PERMISSION_DENIED"), "no_access"),
    (_err(404, "NOT_FOUND"), "not_found"),
])
def test_probe_availability_statuses(err, status):
    p = _vertex(error=err)
    assert p.probe_availability()[0] == status
    assert p._client.calls[-1] == {"model": "gemini-3.5-flash", "contents": "test"}
