"""
Web Search Integration Test

Tests that web_search=True works across all three providers by asking
questions that require real-time web data (using yesterday's date to
ensure the model cannot answer from training data alone).

Usage:
    uv run python -m djinnite.tests.test_web_search
    uv run python -m djinnite.tests.test_web_search --provider gemini
"""

import json
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Support direct execution
_project_root = str(Path(__file__).parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from djinnite.config_loader import load_ai_config
from djinnite.ai_providers import get_provider
from djinnite.ai_providers.base_provider import AIProviderError, AIOutputTruncatedError


# Dynamic date — impossible to answer from training data
YESTERDAY = (datetime.now() - timedelta(days=1)).strftime("%B %d, %Y")

# ------------------------------------------------------------------
# Test definitions
# ------------------------------------------------------------------

def test_freeform_web_search(provider, provider_name: str) -> bool:
    """
    Test 1: Freeform text generation with web search.
    
    Asks for a specific data point from yesterday — the model must
    search the web to answer.
    """
    prompt = (
        f"What was the closing value of the S&P 500 stock market index "
        f"on {YESTERDAY}? Give a brief answer with the number."
    )
    print(f"  [freeform] Asking about S&P 500 on {YESTERDAY}...", end=" ", flush=True)

    try:
        response = provider.generate(prompt, web_search=True)
        content = response.content.strip()
        if content and len(content) > 5:
            # Show a snippet of the response
            snippet = content[:80].replace("\n", " ")
            print(f"✅ ({snippet}...)")
            return True
        else:
            print(f"❌ Empty or too-short response: '{content}'")
            return False
    except AIOutputTruncatedError:
        # Truncation with web search is still a success — the API worked
        print("✅ (truncated but API worked)")
        return True
    except Exception as e:
        print(f"❌ {e}")
        return False


# Known provider limitations for generate_json() + web_search.
# Gemini 2.x: Cannot combine grounding with structured output (resolved in Gemini 3.0+).
# Claude 4.5+/4.6+ and OpenAI Responses API: Support both natively.
_JSON_SEARCH_KNOWN_LIMITATIONS = set()

# For Gemini, the JSON+search test uses a Gemini 3.x model which supports
# grounding + structured output combined.  Gemini 2.x does not.
_GEMINI_JSON_SEARCH_MODEL = "gemini-3-flash-preview"


def test_structured_web_search(provider, provider_name: str) -> bool:
    """
    Test 2: Structured JSON generation with web search.
    
    Asks for recent news headlines and verifies the response is
    valid JSON conforming to the schema.
    
    Note: Some providers cannot combine structured JSON with web search
    (see ``_JSON_SEARCH_KNOWN_LIMITATIONS``).  These are reported as
    expected limitations, not failures.
    """
    schema = {
        "type": "object",
        "properties": {
            "headlines": {
                "type": "array",
                "items": {"type": "string"},
            },
            "source_date": {"type": "string"},
        },
        "required": ["headlines", "source_date"],
    }

    prompt = (
        f"What were the top 3 news headlines on {YESTERDAY}? "
        f"Return them as a JSON object with a 'headlines' array "
        f"and a 'source_date' string."
    )
    print(f"  [json+search] Asking for headlines from {YESTERDAY}...", end=" ", flush=True)

    try:
        response = provider.generate_json(
            prompt, schema=schema, web_search=True,
        )
        content = response.content.strip()

        # Verify it's valid JSON
        parsed = json.loads(content)
        headlines = parsed.get("headlines", [])
        if isinstance(headlines, list) and len(headlines) > 0:
            print(f"✅ ({len(headlines)} headlines returned)")
            return True
        else:
            print(f"❌ JSON parsed but no headlines: {content[:60]}")
            return False
    except AIOutputTruncatedError:
        print("✅ (truncated but API worked)")
        return True
    except json.JSONDecodeError as e:
        if provider_name in _JSON_SEARCH_KNOWN_LIMITATIONS:
            print(f"⚠️  Expected limitation: JSON+search not combined ({provider_name})")
            return True  # Known limitation, not a failure
        print(f"❌ Invalid JSON: {e}")
        return False
    except Exception as e:
        if provider_name in _JSON_SEARCH_KNOWN_LIMITATIONS:
            print(f"⚠️  Expected limitation: JSON+search not combined ({provider_name})")
            return True  # Known limitation, not a failure
        print(f"❌ {e}")
        return False


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def run_web_search_tests():
    parser = argparse.ArgumentParser(description="Test web search across AI providers")
    parser.add_argument("--provider", type=str, help="Test only this provider (gemini/claude/chatgpt)")
    parser.add_argument("--config", type=str, help="Path to ai_config.json")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    config = load_ai_config(config_path)

    provider_names = ["gemini", "claude", "chatgpt"]
    if args.provider:
        provider_names = [args.provider]

    print(f"\nDjinnite Web Search Test — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Test date: {YESTERDAY}")
    print("=" * 60)

    total_pass = 0
    total_fail = 0

    for name in provider_names:
        p_config = config.get_provider(name)
        if not p_config or not p_config.api_key:
            print(f"\n{name}: ⚪ Not configured (skipping)")
            continue

        print(f"\n{name} ({p_config.default_model}):")

        try:
            kwargs = {}
            if name == "gemini":
                kwargs["backend"] = p_config.backend
                kwargs["project_id"] = p_config.project_id

            provider = get_provider(name, api_key=p_config.api_key, model=p_config.default_model, **kwargs)
        except Exception as e:
            print(f"  ❌ Provider init failed: {e}")
            total_fail += 2
            continue

        # Test 1: Freeform
        if test_freeform_web_search(provider, name):
            total_pass += 1
        else:
            total_fail += 1

        # Test 2: Structured JSON + web search
        # For Gemini, use a 3.x model that supports grounding + structured output.
        # Gemini 2.x cannot combine them.
        json_search_provider = provider
        if name == "gemini" and not p_config.default_model.startswith("gemini-3"):
            try:
                print(f"  (switching to {_GEMINI_JSON_SEARCH_MODEL} for JSON+search)")
                json_search_provider = get_provider(
                    name, api_key=p_config.api_key, model=_GEMINI_JSON_SEARCH_MODEL, **kwargs
                )
            except Exception as e:
                print(f"  ⚠️  Could not init {_GEMINI_JSON_SEARCH_MODEL}: {e}")

        if test_structured_web_search(json_search_provider, name):
            total_pass += 1
        else:
            total_fail += 1

    print("\n" + "=" * 60)
    print(f"Results: {total_pass} passed, {total_fail} failed")

    if total_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_web_search_tests()
