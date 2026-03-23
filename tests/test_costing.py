"""
Cost Tracking Validation Test

Validates that every configured provider returns real dollar costs from
simple prompts.  Tests token cost, search cost, total cost, and thinking
token billing.

Usage:
    uv run python -m djinnite.tests.test_costing
    uv run python -m djinnite.tests.test_costing --provider claude
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Support direct execution
_project_root = str(Path(__file__).parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from djinnite.config_loader import load_ai_config, load_model_catalog
from djinnite.ai_providers import get_provider
from djinnite.ai_providers.base_provider import AIProviderError


# ------------------------------------------------------------------
# Test definitions
# ------------------------------------------------------------------

def test_token_cost(provider, provider_name: str) -> bool:
    """
    Test 1: Basic generate() returns token_cost and total_cost.
    """
    print(f"  [token_cost] Sending 'Say hi.'...", end=" ", flush=True)

    try:
        response = provider.generate("Say hi.", max_tokens=256)
        tc = response.token_cost
        total = response.total_cost

        if tc is None:
            print("FAIL (token_cost is None -- missing catalog pricing?)")
            return False
        if tc <= 0:
            print(f"FAIL (token_cost={tc}, expected > 0)")
            return False
        if tc > 1.0:
            print(f"FAIL (token_cost=${tc:.6f} -- sanity check: > $1 for 'Say hi')")
            return False
        if total is None or total <= 0:
            print(f"FAIL (total_cost={total})")
            return False

        print(f"OK (in={response.input_tokens}, out={response.output_tokens}, "
              f"token_cost=${tc:.6f}, total=${total:.6f})")
        return True
    except Exception as e:
        print(f"FAIL ({e})")
        return False


def test_search_cost(provider, provider_name: str) -> bool:
    """
    Test 2: generate(web_search=True) returns search_cost > 0.
    """
    print(f"  [search_cost] Sending web search query...", end=" ", flush=True)

    try:
        response = provider.generate(
            "What is the current price of Bitcoin?",
            web_search=True,
            max_tokens=4096,
        )

        su = response.search_units
        sc = response.search_cost
        tc = response.token_cost
        total = response.total_cost

        if su <= 0:
            print(f"FAIL (search_units={su}, expected > 0)")
            return False
        if sc is None or sc <= 0:
            print(f"FAIL (search_cost={sc})")
            return False
        if tc is None:
            print(f"FAIL (token_cost is None)")
            return False
        if total is None or total <= tc:
            print(f"FAIL (total_cost={total} should be > token_cost={tc})")
            return False

        print(f"OK (search_units={su}, search_cost=${sc:.6f}, "
              f"token_cost=${tc:.6f}, total=${total:.6f})")
        return True
    except AIProviderError as e:
        if "not supported" in str(e).lower():
            print(f"SKIP (web search not supported for this model)")
            return True
        print(f"FAIL ({e})")
        return False
    except Exception as e:
        print(f"FAIL ({e})")
        return False


def test_thinking_cost(provider, provider_name: str) -> bool:
    """
    Test 3 (Claude only): generate(thinking=True) bills thinking tokens.
    """
    if provider_name != "claude":
        return True  # Skip for non-Claude

    print(f"  [thinking_cost] Sending thinking request...", end=" ", flush=True)

    try:
        response = provider.generate(
            "What is 2+2? Think step by step.",
            thinking=1024,
            max_tokens=4096,
        )

        tt = response.thinking_tokens
        tc = response.token_cost
        total = response.total_cost

        if tt is None:
            print(f"SKIP (thinking_tokens not reported by this model)")
            return True
        if tt <= 0:
            print(f"FAIL (thinking_tokens={tt}, expected > 0)")
            return False
        if tc is None or tc <= 0:
            print(f"FAIL (token_cost={tc})")
            return False
        if total is None:
            print(f"FAIL (total_cost is None)")
            return False

        print(f"OK (thinking_tokens={tt}, token_cost=${tc:.6f}, total=${total:.6f})")
        return True
    except AIProviderError as e:
        if "not supported" in str(e).lower() or "thinking" in str(e).lower():
            print(f"SKIP (thinking not supported for this model)")
            return True
        print(f"FAIL ({e})")
        return False
    except Exception as e:
        print(f"FAIL ({e})")
        return False


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def run_costing_tests():
    parser = argparse.ArgumentParser(description="Test cost tracking across AI providers")
    parser.add_argument("--provider", type=str, help="Test only this provider (gemini/claude/chatgpt)")
    parser.add_argument("--config", type=str, help="Path to ai_config.json")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    config = load_ai_config(config_path)
    catalog = load_model_catalog()

    provider_names = ["gemini", "claude", "chatgpt"]
    if args.provider:
        provider_names = [args.provider]

    print(f"\nDjinnite Cost Tracking Test -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    total_pass = 0
    total_fail = 0

    for name in provider_names:
        p_config = config.get_provider(name)
        if not p_config or not p_config.api_key:
            print(f"\n{name}: Not configured (skipping)")
            continue

        model_id = p_config.default_model
        model_info = catalog.get_model(name, model_id)

        # Check catalog has pricing
        if model_info and model_info.costing and model_info.costing.input_per_1m is not None:
            inp = model_info.costing.input_per_1m
            out = model_info.costing.output_per_1m
            print(f"\n{name} ({model_id}) -- ${inp}/1M in, ${out}/1M out:")
        else:
            print(f"\n{name} ({model_id}) -- WARNING: no pricing in catalog")

        try:
            kwargs = {}
            if name == "gemini":
                kwargs["backend"] = p_config.backend
                kwargs["project_id"] = p_config.project_id

            provider = get_provider(
                name, api_key=p_config.api_key, model=model_id, **kwargs,
            )
        except Exception as e:
            print(f"  FAIL Provider init failed: {e}")
            total_fail += 3
            continue

        # Test 1: Token cost
        if test_token_cost(provider, name):
            total_pass += 1
        else:
            total_fail += 1

        # Test 2: Search cost
        if test_search_cost(provider, name):
            total_pass += 1
        else:
            total_fail += 1

        # Test 3: Thinking cost (Claude only)
        if test_thinking_cost(provider, name):
            total_pass += 1
        else:
            total_fail += 1

    print("\n" + "=" * 60)
    print(f"Results: {total_pass} passed, {total_fail} failed")

    if total_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_costing_tests()
