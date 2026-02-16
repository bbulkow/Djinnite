"""
Tests for cross-provider JSON schema normalization.

These tests verify the Djinnite caller contract:
  - Callers must NOT include ``additionalProperties`` in schemas.
  - Djinnite adds/removes it per-provider automatically.
  - Pydantic-generated schemas are silently cleaned.
  - OpenAI gets ``additionalProperties: false`` injected + array wrapping.
  - Claude gets ``additionalProperties: false`` injected (no array wrapping).
  - Gemini gets ``additionalProperties`` stripped defensively.

No API keys or network calls required — pure unit tests.
"""

import copy
import json
import pytest

from ai_providers.base_provider import BaseAIProvider
from ai_providers.openai_provider import OpenAIProvider
from ai_providers.gemini_provider import GeminiProvider
from ai_providers.claude_provider import ClaudeProvider


# ---------------------------------------------------------------------------
# Helpers — lightweight concrete subclass so we can instantiate BaseAIProvider
# ---------------------------------------------------------------------------

class _FakeProvider(BaseAIProvider):
    """Minimal concrete provider for testing base class methods."""
    PROVIDER_NAME = "fake"

    def _initialize_client(self):
        self._client = "fake"

    def generate(self, prompt, **kw):
        raise NotImplementedError

    def is_available(self):
        return True

    def list_models(self):
        return []


def _make_fake():
    return _FakeProvider(api_key="fake-key", model="fake-model")


def _make_openai():
    """Create an OpenAI provider without actually connecting."""
    orig_init = OpenAIProvider._initialize_client
    OpenAIProvider._initialize_client = lambda self: setattr(self, '_client', 'fake')
    try:
        prov = OpenAIProvider(api_key="fake", model="gpt-4o")
    finally:
        OpenAIProvider._initialize_client = orig_init
    return prov


def _make_gemini():
    """Create a Gemini provider without actually connecting."""
    orig_init = GeminiProvider._initialize_client
    GeminiProvider._initialize_client = lambda self: setattr(self, '_client', 'fake')
    try:
        prov = GeminiProvider(api_key="fake", model="gemini-2.5-flash")
    finally:
        GeminiProvider._initialize_client = orig_init
    return prov


def _make_claude():
    """Create a Claude provider without actually connecting."""
    orig_init = ClaudeProvider._initialize_client
    ClaudeProvider._initialize_client = lambda self: setattr(self, '_client', 'fake')
    try:
        prov = ClaudeProvider(api_key="fake", model="claude-haiku-4-5-20251001")
    finally:
        ClaudeProvider._initialize_client = orig_init
    return prov


# ---------------------------------------------------------------------------
# _is_pydantic_generated
# ---------------------------------------------------------------------------

class TestIsPydanticGenerated:
    def test_hand_written_schema(self):
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        assert BaseAIProvider._is_pydantic_generated(schema) is False

    def test_pydantic_schema_has_title(self):
        schema = {
            "title": "MyModel",
            "type": "object",
            "properties": {"x": {"type": "integer"}},
        }
        assert BaseAIProvider._is_pydantic_generated(schema) is True


# ---------------------------------------------------------------------------
# _schema_contains_additional_properties
# ---------------------------------------------------------------------------

class TestSchemaContainsAdditionalProperties:
    def test_clean_schema(self):
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        assert BaseAIProvider._schema_contains_additional_properties(schema) is False

    def test_top_level(self):
        schema = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "additionalProperties": False,
        }
        assert BaseAIProvider._schema_contains_additional_properties(schema) is True

    def test_nested_in_property(self):
        schema = {
            "type": "object",
            "properties": {
                "child": {
                    "type": "object",
                    "properties": {"y": {"type": "integer"}},
                    "additionalProperties": False,
                }
            },
        }
        assert BaseAIProvider._schema_contains_additional_properties(schema) is True

    def test_nested_in_array_items(self):
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"z": {"type": "boolean"}},
                "additionalProperties": False,
            },
        }
        assert BaseAIProvider._schema_contains_additional_properties(schema) is True

    def test_nested_in_defs(self):
        schema = {
            "type": "object",
            "properties": {"ref": {"$ref": "#/$defs/Inner"}},
            "$defs": {
                "Inner": {
                    "type": "object",
                    "properties": {"v": {"type": "integer"}},
                    "additionalProperties": False,
                }
            },
        }
        assert BaseAIProvider._schema_contains_additional_properties(schema) is True

    def test_nested_in_anyof(self):
        schema = {
            "anyOf": [
                {"type": "object", "properties": {"a": {"type": "string"}}, "additionalProperties": False},
                {"type": "string"},
            ]
        }
        assert BaseAIProvider._schema_contains_additional_properties(schema) is True


# ---------------------------------------------------------------------------
# _add_additional_properties_false  (now on BaseAIProvider)
# ---------------------------------------------------------------------------

class TestAddAdditionalPropertiesFalse:
    def test_adds_to_simple_object(self):
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        BaseAIProvider._add_additional_properties_false(schema)
        assert schema["additionalProperties"] is False

    def test_adds_recursively(self):
        schema = {
            "type": "object",
            "properties": {
                "child": {
                    "type": "object",
                    "properties": {"y": {"type": "integer"}},
                }
            },
        }
        BaseAIProvider._add_additional_properties_false(schema)
        assert schema["additionalProperties"] is False
        assert schema["properties"]["child"]["additionalProperties"] is False

    def test_adds_to_array_items(self):
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"v": {"type": "integer"}},
            },
        }
        BaseAIProvider._add_additional_properties_false(schema)
        assert schema["items"]["additionalProperties"] is False

    def test_adds_to_defs(self):
        schema = {
            "$defs": {
                "Inner": {
                    "type": "object",
                    "properties": {"v": {"type": "integer"}},
                }
            },
        }
        BaseAIProvider._add_additional_properties_false(schema)
        assert schema["$defs"]["Inner"]["additionalProperties"] is False

    def test_skips_non_objects(self):
        schema = {"type": "string"}
        BaseAIProvider._add_additional_properties_false(schema)
        assert "additionalProperties" not in schema


# ---------------------------------------------------------------------------
# _strip_additional_properties
# ---------------------------------------------------------------------------

class TestStripAdditionalProperties:
    def test_strips_top_level(self):
        schema = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "additionalProperties": False,
        }
        result = BaseAIProvider._strip_additional_properties(schema)
        assert "additionalProperties" not in result
        # Original unchanged
        assert "additionalProperties" in schema

    def test_strips_nested(self):
        schema = {
            "type": "object",
            "properties": {
                "child": {
                    "type": "object",
                    "properties": {"y": {"type": "integer"}},
                    "additionalProperties": False,
                }
            },
            "additionalProperties": False,
        }
        result = BaseAIProvider._strip_additional_properties(schema)
        assert "additionalProperties" not in result
        assert "additionalProperties" not in result["properties"]["child"]

    def test_strips_in_defs(self):
        schema = {
            "$defs": {
                "Inner": {
                    "type": "object",
                    "properties": {"v": {"type": "integer"}},
                    "additionalProperties": False,
                }
            },
            "type": "object",
            "properties": {},
        }
        result = BaseAIProvider._strip_additional_properties(schema)
        assert "additionalProperties" not in result["$defs"]["Inner"]

    def test_strips_in_items(self):
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"z": {"type": "boolean"}},
                "additionalProperties": False,
            },
        }
        result = BaseAIProvider._strip_additional_properties(schema)
        assert "additionalProperties" not in result["items"]


# ---------------------------------------------------------------------------
# _validate_caller_schema
# ---------------------------------------------------------------------------

class TestValidateCallerSchema:
    def test_clean_schema_passes(self):
        prov = _make_fake()
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        result = prov._validate_caller_schema(schema)
        assert result == schema

    def test_hand_written_with_additional_properties_raises(self):
        prov = _make_fake()
        schema = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "additionalProperties": False,
        }
        with pytest.raises(ValueError, match="additionalProperties"):
            prov._validate_caller_schema(schema)

    def test_pydantic_generated_silently_stripped(self):
        prov = _make_fake()
        schema = {
            "title": "MyModel",
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "nested": {
                    "type": "object",
                    "properties": {"v": {"type": "integer"}},
                    "additionalProperties": False,
                },
            },
            "additionalProperties": False,
        }
        result = prov._validate_caller_schema(schema)
        assert "additionalProperties" not in result
        assert "additionalProperties" not in result["properties"]["nested"]
        # title preserved
        assert result["title"] == "MyModel"

    def test_deeply_nested_hand_written_raises(self):
        prov = _make_fake()
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "inner": {
                        "type": "object",
                        "properties": {"v": {"type": "integer"}},
                        "additionalProperties": False,
                    }
                },
            },
        }
        with pytest.raises(ValueError, match="additionalProperties"):
            prov._validate_caller_schema(schema)


# ---------------------------------------------------------------------------
# OpenAI: _prepare_schema_for_provider
# ---------------------------------------------------------------------------

class TestOpenAIPrepareSchema:
    def test_adds_additional_properties_to_simple_object(self):
        prov = _make_openai()
        schema = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        }
        result = prov._prepare_schema_for_provider(schema)
        assert result["additionalProperties"] is False
        assert prov._openai_array_wrapped is False

    def test_adds_additional_properties_recursively(self):
        prov = _make_openai()
        schema = {
            "type": "object",
            "properties": {
                "child": {
                    "type": "object",
                    "properties": {"y": {"type": "integer"}},
                    "required": ["y"],
                }
            },
            "required": ["child"],
        }
        result = prov._prepare_schema_for_provider(schema)
        assert result["additionalProperties"] is False
        assert result["properties"]["child"]["additionalProperties"] is False

    def test_adds_to_array_items(self):
        prov = _make_openai()
        schema = {
            "type": "object",
            "properties": {
                "list": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"v": {"type": "integer"}},
                        "required": ["v"],
                    },
                }
            },
            "required": ["list"],
        }
        result = prov._prepare_schema_for_provider(schema)
        assert result["properties"]["list"]["items"]["additionalProperties"] is False

    def test_wraps_top_level_array(self):
        prov = _make_openai()
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        }
        result = prov._prepare_schema_for_provider(schema)
        # Should be wrapped in an object
        assert result["type"] == "object"
        assert "items" in result["properties"]
        assert result["properties"]["items"]["type"] == "array"
        assert result["additionalProperties"] is False
        # Items' object should also have additionalProperties
        assert result["properties"]["items"]["items"]["additionalProperties"] is False
        assert prov._openai_array_wrapped is True

    def test_does_not_wrap_top_level_object(self):
        prov = _make_openai()
        schema = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }
        result = prov._prepare_schema_for_provider(schema)
        assert result["type"] == "object"
        assert "x" in result["properties"]
        assert prov._openai_array_wrapped is False

    def test_does_not_mutate_original(self):
        prov = _make_openai()
        schema = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        }
        original = copy.deepcopy(schema)
        prov._prepare_schema_for_provider(schema)
        assert schema == original  # No mutation

    def test_adds_to_defs(self):
        prov = _make_openai()
        schema = {
            "type": "object",
            "properties": {"ref": {"$ref": "#/$defs/Inner"}},
            "required": ["ref"],
            "$defs": {
                "Inner": {
                    "type": "object",
                    "properties": {"v": {"type": "integer"}},
                    "required": ["v"],
                }
            },
        }
        result = prov._prepare_schema_for_provider(schema)
        assert result["$defs"]["Inner"]["additionalProperties"] is False

    def test_adds_to_anyof_branches(self):
        prov = _make_openai()
        schema = {
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [
                        {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]},
                        {"type": "string"},
                    ]
                }
            },
            "required": ["value"],
        }
        result = prov._prepare_schema_for_provider(schema)
        obj_branch = result["properties"]["value"]["anyOf"][0]
        assert obj_branch["additionalProperties"] is False


# ---------------------------------------------------------------------------
# Claude: _prepare_schema_for_provider
# ---------------------------------------------------------------------------

class TestClaudePrepareSchema:
    def test_adds_additional_properties_to_simple_object(self):
        prov = _make_claude()
        schema = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        }
        result = prov._prepare_schema_for_provider(schema)
        assert result["additionalProperties"] is False

    def test_adds_additional_properties_recursively(self):
        prov = _make_claude()
        schema = {
            "type": "object",
            "properties": {
                "child": {
                    "type": "object",
                    "properties": {"y": {"type": "integer"}},
                    "required": ["y"],
                }
            },
            "required": ["child"],
        }
        result = prov._prepare_schema_for_provider(schema)
        assert result["additionalProperties"] is False
        assert result["properties"]["child"]["additionalProperties"] is False

    def test_does_not_wrap_top_level_array(self):
        """Claude accepts top-level arrays — no wrapping needed."""
        prov = _make_claude()
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        }
        result = prov._prepare_schema_for_provider(schema)
        # Should remain an array (no wrapping)
        assert result["type"] == "array"
        # But items' object should have additionalProperties
        assert result["items"]["additionalProperties"] is False

    def test_does_not_mutate_original(self):
        prov = _make_claude()
        schema = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        }
        original = copy.deepcopy(schema)
        prov._prepare_schema_for_provider(schema)
        assert schema == original  # No mutation

    def test_adds_to_defs(self):
        prov = _make_claude()
        schema = {
            "type": "object",
            "properties": {"ref": {"$ref": "#/$defs/Inner"}},
            "required": ["ref"],
            "$defs": {
                "Inner": {
                    "type": "object",
                    "properties": {"v": {"type": "integer"}},
                    "required": ["v"],
                }
            },
        }
        result = prov._prepare_schema_for_provider(schema)
        assert result["$defs"]["Inner"]["additionalProperties"] is False


# ---------------------------------------------------------------------------
# Gemini: _prepare_schema_for_provider
# ---------------------------------------------------------------------------

class TestGeminiPrepareSchema:
    def test_strips_additional_properties(self):
        prov = _make_gemini()
        schema = {
            "type": "object",
            "properties": {
                "child": {
                    "type": "object",
                    "properties": {"y": {"type": "integer"}},
                    "additionalProperties": False,
                }
            },
            "additionalProperties": False,
        }
        result = prov._prepare_schema_for_provider(schema)
        assert "additionalProperties" not in result
        assert "additionalProperties" not in result["properties"]["child"]

    def test_clean_schema_passes_through(self):
        prov = _make_gemini()
        schema = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        }
        result = prov._prepare_schema_for_provider(schema)
        assert result == schema

    def test_does_not_mutate_original(self):
        prov = _make_gemini()
        schema = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "additionalProperties": False,
        }
        original = copy.deepcopy(schema)
        prov._prepare_schema_for_provider(schema)
        assert schema == original


# ---------------------------------------------------------------------------
# Base: _prepare_schema_for_provider  — pass-through (default)
# ---------------------------------------------------------------------------

class TestBasePrepareSchema:
    def test_passthrough(self):
        prov = _make_fake()
        schema = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        }
        result = prov._prepare_schema_for_provider(schema)
        assert result is schema  # Same object — identity pass-through


# ---------------------------------------------------------------------------
# End-to-end: full pipeline validation → preparation
# ---------------------------------------------------------------------------

class TestEndToEndPipeline:
    """Simulate what generate_json does: normalize → validate → prepare."""

    def test_same_schema_works_for_all_providers(self):
        """A single clean schema should pass validation for all providers."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "label": {"type": "string"},
                        },
                        "required": ["id", "label"],
                    },
                },
            },
            "required": ["name", "items"],
        }

        for make_fn in [_make_openai, _make_gemini, _make_claude, _make_fake]:
            prov = make_fn()
            validated = prov._validate_caller_schema(copy.deepcopy(schema))
            prepared = prov._prepare_schema_for_provider(validated)

            if prov.PROVIDER_NAME == "chatgpt":
                # OpenAI: all objects have additionalProperties: false
                assert prepared["additionalProperties"] is False
                assert prepared["properties"]["items"]["items"]["additionalProperties"] is False
            elif prov.PROVIDER_NAME == "claude":
                # Claude: all objects have additionalProperties: false
                assert prepared["additionalProperties"] is False
                assert prepared["properties"]["items"]["items"]["additionalProperties"] is False
            elif prov.PROVIDER_NAME == "gemini":
                # Gemini: no additionalProperties anywhere
                assert not BaseAIProvider._schema_contains_additional_properties(prepared)
            else:
                # Fake/base: unchanged
                assert not BaseAIProvider._schema_contains_additional_properties(prepared)

    def test_array_schema_works_across_providers(self):
        """A top-level array schema should work for all providers."""
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "score": {"type": "number"},
                },
                "required": ["name", "score"],
            },
        }

        for make_fn in [_make_openai, _make_gemini, _make_claude, _make_fake]:
            prov = make_fn()
            validated = prov._validate_caller_schema(copy.deepcopy(schema))
            prepared = prov._prepare_schema_for_provider(validated)

            if prov.PROVIDER_NAME == "chatgpt":
                # OpenAI: Should be wrapped in an object
                assert prepared["type"] == "object"
                assert "items" in prepared["properties"]
                inner_items = prepared["properties"]["items"]["items"]
                assert inner_items["additionalProperties"] is False
            elif prov.PROVIDER_NAME == "claude":
                # Claude: array stays, but items get additionalProperties
                assert prepared["type"] == "array"
                assert prepared["items"]["additionalProperties"] is False
            elif prov.PROVIDER_NAME == "gemini":
                # Gemini: Should remain an array, no additionalProperties
                assert prepared["type"] == "array"
                assert not BaseAIProvider._schema_contains_additional_properties(prepared)
            else:
                # Fake/base: Pass-through
                assert prepared["type"] == "array"

    def test_schema_with_additional_properties_rejected_for_all(self):
        """All providers reject caller schemas with additionalProperties."""
        bad_schema = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "additionalProperties": False,
        }

        for make_fn in [_make_openai, _make_gemini, _make_claude, _make_fake]:
            prov = make_fn()
            with pytest.raises(ValueError, match="additionalProperties"):
                prov._validate_caller_schema(bad_schema)
