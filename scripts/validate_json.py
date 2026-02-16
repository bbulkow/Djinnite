"""
Validate Structured JSON (generate_json) Across All Providers

Runs a battery of short generate_json() calls against every enabled
provider to verify that Djinnite's schema normalization pipeline
produces valid, schema-conforming JSON for each.

Test cases use portable schemas (no additionalProperties) and exercise:
  1. Simple object schema
  2. Nested object schema
  3. Top-level array schema

For each test, the script validates:
  - The response contains valid JSON (json.loads succeeds)
  - The JSON structurally conforms to the schema (keys present, types correct)

Requires real API keys in config/ai_config.json.

Usage:
    python -m djinnite.scripts.validate_json
    python -m djinnite.scripts.validate_json --provider gemini
    python -m djinnite.scripts.validate_json --config path/to/ai_config.json
"""

import sys
import json
import argparse
from pathlib import Path

# Support direct execution (adds project root to path)
_project_root = str(Path(__file__).parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from djinnite.config_loader import load_ai_config
from djinnite.ai_providers import get_provider


# ======================================================================
# Test schemas — portable (no additionalProperties)
# ======================================================================

SIMPLE_OBJECT_SCHEMA = {
    "type": "object",
    "properties": {
        "value": {"type": "integer"},
    },
    "required": ["value"],
}

NESTED_OBJECT_SCHEMA = {
    "type": "object",
    "properties": {
        "person": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "city": {"type": "string"},
            },
            "required": ["name", "city"],
        },
    },
    "required": ["person"],
}

ARRAY_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "hex": {"type": "string"},
        },
        "required": ["name", "hex"],
    },
}

# ======================================================================
# Test cases: (name, schema, prompt, validator_fn)
# ======================================================================

def _validate_simple(data):
    """Validate simple object response."""
    assert isinstance(data, dict), f"Expected dict, got {type(data).__name__}"
    assert "value" in data, f"Missing 'value' key. Keys: {list(data.keys())}"
    assert isinstance(data["value"], int), f"'value' should be int, got {type(data['value']).__name__}"
    return True


def _validate_nested(data):
    """Validate nested object response."""
    assert isinstance(data, dict), f"Expected dict, got {type(data).__name__}"
    assert "person" in data, f"Missing 'person' key. Keys: {list(data.keys())}"
    person = data["person"]
    assert isinstance(person, dict), f"'person' should be dict, got {type(person).__name__}"
    assert "name" in person, f"Missing 'person.name'. Keys: {list(person.keys())}"
    assert "city" in person, f"Missing 'person.city'. Keys: {list(person.keys())}"
    assert isinstance(person["name"], str), f"'person.name' should be str"
    assert isinstance(person["city"], str), f"'person.city' should be str"
    return True


def _validate_array(data):
    """Validate array response."""
    assert isinstance(data, list), f"Expected list, got {type(data).__name__}"
    assert len(data) >= 1, f"Expected at least 1 item, got {len(data)}"
    for i, item in enumerate(data):
        assert isinstance(item, dict), f"Item {i} should be dict, got {type(item).__name__}"
        assert "name" in item, f"Item {i} missing 'name'. Keys: {list(item.keys())}"
        assert "hex" in item, f"Item {i} missing 'hex'. Keys: {list(item.keys())}"
    return True


TEST_CASES = [
    (
        "simple_object",
        SIMPLE_OBJECT_SCHEMA,
        "Return the number 42.",
        _validate_simple,
    ),
    (
        "nested_object",
        NESTED_OBJECT_SCHEMA,
        "Return a person named Alice who lives in Seattle.",
        _validate_nested,
    ),
    (
        "top_level_array",
        ARRAY_SCHEMA,
        "Return a list of exactly 2 colors with their name and hex code.",
        _validate_array,
    ),
]


# ======================================================================
# Main
# ======================================================================

def validate_json():
    parser = argparse.ArgumentParser(
        description="Validate generate_json() across all enabled AI providers"
    )
    parser.add_argument("--config", type=str, help="Path to ai_config.json")
    parser.add_argument(
        "--provider", type=str, default=None,
        help="Test only this provider (gemini, claude, chatgpt)"
    )
    args = parser.parse_args()

    print("Loading configuration...")
    config_path = Path(args.config) if args.config else None
    config = load_ai_config(config_path)

    # Determine which providers to test
    provider_names = ["gemini", "claude", "chatgpt"]
    if args.provider:
        if args.provider not in provider_names:
            print(f"Unknown provider '{args.provider}'. Available: {provider_names}")
            sys.exit(1)
        provider_names = [args.provider]

    print(f"\nValidating generate_json() — Structured JSON Mode")
    print("=" * 70)

    total_pass = 0
    total_fail = 0
    total_skip = 0

    for provider_name in provider_names:
        # Check provider config
        if provider_name not in config.providers:
            print(f"\n⚪ {provider_name}: Not configured (skipping)")
            total_skip += 1
            continue

        provider_config = config.providers[provider_name]

        if not provider_config.enabled:
            print(f"\n⚪ {provider_name}: Disabled (skipping)")
            total_skip += 1
            continue

        if not provider_config.api_key or "your" in provider_config.api_key.lower():
            print(f"\n⚪ {provider_name}: No API key (skipping)")
            total_skip += 1
            continue

        model = provider_config.default_model
        print(f"\n▶ {provider_name} ({model})")
        print("-" * 50)

        # Initialize provider
        try:
            provider_kwargs = {}
            if provider_name == "gemini":
                provider_kwargs["backend"] = provider_config.backend
                provider_kwargs["project_id"] = provider_config.project_id

            # Get gemini key for OpenAI web search (not needed here but get_provider expects it)
            gemini_key = None
            if provider_name == "chatgpt":
                gemini_cfg = config.get_provider("gemini")
                if gemini_cfg:
                    gemini_key = gemini_cfg.api_key

            provider = get_provider(
                provider_name=provider_name,
                api_key=provider_config.api_key,
                model=model,
                gemini_api_key=gemini_key,
                **provider_kwargs,
            )
        except Exception as e:
            print(f"  ❌ Init failed: {e}")
            total_fail += len(TEST_CASES)
            continue

        # Run test cases
        for test_name, schema, prompt, validator in TEST_CASES:
            label = f"  {test_name}:"
            print(f"{label:<30}", end="", flush=True)

            try:
                response = provider.generate_json(
                    prompt=prompt,
                    schema=schema,
                    temperature=0.0,
                    max_tokens=256,
                    force=True,  # Skip catalog checks — we're testing the pipeline
                )

                # Parse JSON
                content = response.content.strip()
                data = json.loads(content)

                # Structural validation
                validator(data)

                print(f"✅  ({response.output_tokens} tokens)")
                total_pass += 1

            except json.JSONDecodeError as e:
                print(f"❌  Invalid JSON: {e}")
                print(f"    Raw content: {response.content[:200]!r}")
                total_fail += 1

            except AssertionError as e:
                print(f"❌  Schema mismatch: {e}")
                try:
                    print(f"    Parsed: {json.loads(response.content)}")
                except Exception:
                    print(f"    Raw: {response.content[:200]!r}")
                total_fail += 1

            except Exception as e:
                err_type = type(e).__name__
                err_msg = str(e).replace("\n", " ")[:150]
                print(f"❌  {err_type}: {err_msg}")
                total_fail += 1

    # Summary
    print("\n" + "=" * 70)
    total = total_pass + total_fail
    print(f"Results: {total_pass}/{total} passed, {total_fail} failed, {total_skip} providers skipped")

    if total_fail > 0:
        print("\n💡 If a provider failed, check:")
        print("   - API key is valid in config/ai_config.json")
        print("   - Model supports structured JSON (check model_catalog.json)")
        print("   - Network connectivity to the provider API")
        sys.exit(1)
    elif total_pass == 0:
        print("\n⚠️  No providers were tested. Configure at least one provider in config/ai_config.json")
        sys.exit(1)
    else:
        print("\n✅ All structured JSON tests passed!")


if __name__ == "__main__":
    validate_json()
