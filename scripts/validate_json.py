"""
Validate Structured JSON (generate_json) Across All Providers

Runs a battery of short generate_json() calls against every enabled
ai_config.json entry (an access path, ACCESS_PATHS_DESIGN.md: several may
share a provider type) to verify that Djinnite's schema normalization
pipeline produces valid, schema-conforming JSON for each. An entry whose
``deny`` covers structured_json for its default model is skipped.

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
    python -m djinnite.scripts.validate_json --provider gemini        # every gemini entry
    python -m djinnite.scripts.validate_json --provider claude-vertex # one entry
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

try:
    from djinnite.config_loader import load_ai_config
    from djinnite.scripts.validate_ai import entry_label, entry_skip_reason, select_entries
except ImportError:
    _pkg_root = str(Path(__file__).resolve().parent.parent)
    if _pkg_root not in sys.path:
        sys.path.insert(0, _pkg_root)
    from config_loader import load_ai_config  # type: ignore
    from scripts.validate_ai import entry_label, entry_skip_reason, select_entries  # type: ignore


# ======================================================================
# Test schemas -- portable (no additionalProperties)
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
        description="Validate generate_json() across all enabled ai_config entries"
    )
    parser.add_argument("--config", type=str, help="Path to ai_config.json")
    parser.add_argument(
        "--provider", type=str, default=None,
        help="Test only this ai_config entry, or every entry of this provider "
             "type (gemini, claude, chatgpt, grok)"
    )
    args = parser.parse_args()

    print("Loading configuration...")
    config_path = Path(args.config) if args.config else None
    config = load_ai_config(config_path)

    # Determine which entries to test
    try:
        entry_names = select_entries(config, args.provider)
    except ValueError as e:
        print(f"[FAIL] {e}")
        sys.exit(1)

    print("\nValidating generate_json() -- Structured JSON Mode")
    print("=" * 70)

    total_pass = 0
    total_fail = 0
    total_skip = 0

    for name in entry_names:
        label = entry_label(config, name)
        reason = entry_skip_reason(config, name)
        if reason:
            print(f"\n[SKIP] {label}: {reason}")
            total_skip += 1
            continue

        provider_config = config.providers[name]
        model = provider_config.default_model
        # A deployment restriction is not a model failure: the provider would
        # raise DjinniteCapabilityDeniedError before any request.
        if "structured_json" in provider_config.denied_for(model or None):
            why = f" ({provider_config.deny_reason})" if provider_config.deny_reason else ""
            print(f"\n[SKIP] {label}: entry denies structured_json for model {model}{why}")
            total_skip += 1
            continue

        print(f"\n> {label} -- model {model}")
        print("-" * 50)

        # Initialize provider
        try:
            provider = config.build_provider(name)
        except Exception as e:
            print(f"  [FAIL] Init failed: {e}")
            total_fail += len(TEST_CASES)
            continue

        # Run test cases
        for test_name, schema, prompt, validator in TEST_CASES:
            test_label = f"  {test_name}:"
            print(f"{test_label:<30}", end="", flush=True)

            try:
                response = provider.generate_json(
                    prompt=prompt,
                    schema=schema,
                    # Temperature default (0.3) -- the provider will omit it
                    # automatically if the catalog says the model doesn't
                    # support temperature.
                    # Use 4096 to give reasoning models (GPT-5, o-series)
                    # enough room for thinking tokens + output.
                    max_output_tokens=4096,
                )

                # Parse JSON
                content = response.content.strip()
                data = json.loads(content)

                # Structural validation
                validator(data)

                print(f"[OK]  ({response.output_tokens} tokens)")
                total_pass += 1

            except json.JSONDecodeError as e:
                print(f"[FAIL]  Invalid JSON: {e}")
                print(f"    Raw content: {response.content[:200]!r}")
                total_fail += 1

            except AssertionError as e:
                print(f"[FAIL]  Schema mismatch: {e}")
                try:
                    print(f"    Parsed: {json.loads(response.content)}")
                except Exception:
                    print(f"    Raw: {response.content[:200]!r}")
                total_fail += 1

            except Exception as e:
                err_type = type(e).__name__
                err_msg = str(e).replace("\n", " ")[:150]
                print(f"[FAIL]  {err_type}: {err_msg}")
                total_fail += 1

    # Summary
    print("\n" + "=" * 70)
    total = total_pass + total_fail
    print(f"Results: {total_pass}/{total} passed, {total_fail} failed, {total_skip} entries skipped")

    if total_fail > 0:
        print("\n[TIP] If an entry failed, check:")
        print("   - API key is valid in config/ai_config.json")
        print("   - Model supports structured JSON (check model_catalog.json)")
        print("   - Network connectivity to the provider API")
        sys.exit(1)
    elif total_pass == 0:
        print("\n[WARN]  No entries were tested. Configure at least one provider entry in config/ai_config.json")
        sys.exit(1)
    else:
        print("\n[OK] All structured JSON tests passed!")


if __name__ == "__main__":
    validate_json()
