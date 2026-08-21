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
from datetime import datetime, date, timedelta
from pathlib import Path

# Support direct execution
_project_root = str(Path(__file__).parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from djinnite.config_loader import load_ai_config, load_model_catalog, ModelInfo, ModelCosting
from djinnite.ai_providers import get_provider
from djinnite.ai_providers.base_provider import AIProviderError, AIPricingError, BaseAIProvider


# ------------------------------------------------------------------
# Offline fast-fail tests (no API keys required)
# ------------------------------------------------------------------

class _StubProvider(BaseAIProvider):
    """Minimal concrete provider for exercising cost computation offline."""
    PROVIDER_NAME = "stub"

    def _initialize_client(self) -> None:
        self._client = None

    def generate(self, *args, **kwargs):  # pragma: no cover - not used
        raise NotImplementedError

    def is_available(self) -> bool:  # pragma: no cover - not used
        return True

    def list_models(self) -> list:  # pragma: no cover - not used
        return []


def _make_provider(costing: ModelCosting, require_pricing: bool = True) -> _StubProvider:
    info = ModelInfo(id="stub-model", name="Stub", context_window=1000, costing=costing)
    return _StubProvider(api_key="x", model="stub-model", model_info=info,
                         require_pricing=require_pricing)


def _fresh_today() -> str:
    return date.today().isoformat()


def _stale_date() -> str:
    return (date.today() - timedelta(days=200)).isoformat()


def _check_fast_fail_offline() -> list:
    """Cost computation fast-fails on missing/unknown/stale price; computes when fresh."""
    print("  [fast_fail] missing/unknown/stale/fresh/lenient...", end=" ", flush=True)
    fails = []
    usage = {"input_tokens": 1000, "output_tokens": 1000}

    # 1. No price -> raise.
    p = _make_provider(ModelCosting(source="failed"))
    try:
        p._compute_token_cost(dict(usage)); fails.append("no raise on missing price")
    except AIPricingError:
        pass

    # 2. source=unknown with prices present -> raise.
    p = _make_provider(ModelCosting(input_per_1m=1.0, output_per_1m=2.0,
                                    source="unknown", updated=_fresh_today()))
    try:
        p._compute_token_cost(dict(usage)); fails.append("no raise on source=unknown")
    except AIPricingError:
        pass

    # 3. Stale (>180d) -> raise.
    p = _make_provider(ModelCosting(input_per_1m=1.0, output_per_1m=2.0,
                                    source="estimated", updated=_stale_date()))
    try:
        p._compute_token_cost(dict(usage)); fails.append("no raise on stale price")
    except AIPricingError:
        pass

    # 4. Fresh + priced -> computes a positive cost.
    p = _make_provider(ModelCosting(input_per_1m=1.0, output_per_1m=2.0,
                                    source="estimated", updated=_fresh_today()))
    u = dict(usage)
    p._compute_token_cost(u)
    if u.get("token_cost") is None or u["token_cost"] <= 0:
        fails.append(f"fresh price did not compute: {u.get('token_cost')}")

    # 5. require_pricing=False + unknown -> no raise, no token_cost.
    p = _make_provider(ModelCosting(source="unknown"), require_pricing=False)
    u = dict(usage)
    p._compute_token_cost(u)
    if u.get("token_cost") is not None:
        fails.append("lenient mode produced a cost")

    print("OK" if not fails else "FAIL")
    return fails


# ------------------------------------------------------------------
# Test definitions
# ------------------------------------------------------------------

def _check_token_cost(provider, provider_name: str) -> bool:
    """
    Test 1: Basic generate() returns token_cost and total_cost.
    """
    print(f"  [token_cost] Sending 'Say hi.'...", end=" ", flush=True)

    try:
        response = provider.generate("Say hi.", max_output_tokens=256)
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


def _check_search_cost(provider, provider_name: str) -> bool:
    """
    Test 2: generate(web_search=True) returns search_cost > 0.
    """
    print(f"  [search_cost] Sending web search query...", end=" ", flush=True)

    try:
        response = provider.generate(
            "What is the current price of Bitcoin?",
            web_search=True,
            max_output_tokens=4096,
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


def _check_thinking_cost(provider, provider_name: str) -> bool:
    """
    Test 3 (Claude only): generate(thinking=True) bills thinking tokens.
    """
    if provider_name != "claude":
        return True  # Skip for non-Claude

    # Claude models take different thinking shapes and reject the others
    # outright: an int budget only works where thinking_style includes
    # "budget". thinking=True is the one form every thinking-capable model
    # accepts, so ask for it that way rather than hardcoding a budget.
    thinking_arg = True
    info = getattr(provider, "_model_info", None)
    styles = info.capabilities.thinking_style if info and info.capabilities else None
    if styles and "budget" in styles:
        thinking_arg = 1024

    print(f"  [thinking_cost] Sending thinking request (thinking={thinking_arg})...",
          end=" ", flush=True)

    try:
        response = provider.generate(
            "What is 2+2? Think step by step.",
            thinking=thinking_arg,
            max_output_tokens=4096,
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

# ------------------------------------------------------------------
# pytest entry points
#   * offline test runs always
#   * live tests use the `provider` fixture (tests/conftest.py) and are
#     skipped unless you pass --live
# ------------------------------------------------------------------

def test_fast_fail_offline():
    """Cost computation fast-fails on missing/unknown/stale price."""
    fails = _check_fast_fail_offline()
    assert not fails, "pricing fast-fail regressions:\n  " + "\n  ".join(fails)


def test_token_cost(provider, provider_name):
    assert _check_token_cost(provider, provider_name)


def test_search_cost(provider, provider_name):
    assert _check_search_cost(provider, provider_name)


def test_thinking_cost(provider, provider_name):
    assert _check_thinking_cost(provider, provider_name)


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

    # Offline fast-fail behavior (no API keys needed).
    print("\noffline fast-fail:")
    if not _check_fast_fail_offline():
        total_pass += 1
    else:
        total_fail += 1

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
        if _check_token_cost(provider, name):
            total_pass += 1
        else:
            total_fail += 1

        # Test 2: Search cost
        if _check_search_cost(provider, name):
            total_pass += 1
        else:
            total_fail += 1

        # Test 3: Thinking cost (Claude only)
        if _check_thinking_cost(provider, name):
            total_pass += 1
        else:
            total_fail += 1

    print("\n" + "=" * 60)
    print(f"Results: {total_pass} passed, {total_fail} failed")

    if total_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_costing_tests()
