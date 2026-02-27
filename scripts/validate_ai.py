"""
Validate AI Configuration

Tests connectivity and authentication for all configured AI providers
by attempting a minimal generation request.

Usage:
    python -m djinnite.scripts.validate_ai
"""

import sys
from pathlib import Path

# Support direct execution (adds project root to path)
_project_root = str(Path(__file__).parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from djinnite.config_loader import load_ai_config
from djinnite.ai_providers import get_provider
from djinnite.ai_providers.base_provider import AIOutputTruncatedError
from djinnite.ai_providers.gemini_provider import GeminiProvider
from djinnite.ai_providers.claude_provider import ClaudeProvider
from djinnite.ai_providers.openai_provider import OpenAIProvider

import argparse

def validate_ai():
    parser = argparse.ArgumentParser(description="Validate AI provider configuration")
    parser.add_argument("--config", type=str, help="Path to ai_config.json")
    args = parser.parse_args()

    print("Loading configuration...")
    config_path = Path(args.config) if args.config else None
    config = load_ai_config(config_path)
    
    provider_names = ["gemini", "claude", "chatgpt"]
    
    print("\nValidating AI Providers...")
    print("=" * 60)
    
    success_count = 0
    fail_count = 0
    
    for name in provider_names:
        # Check raw config directly to distinguish between missing and disabled
        # (config.get_provider returns None for disabled providers)
        if name not in config.providers:
            print(f"⚪ {name}: Not found in ai_config.json (skipping)")
            continue
            
        provider_config = config.providers[name]
        
        if not provider_config.enabled:
            print(f"⚪ {name}: Disabled in config (skipping)")
            continue
            
        if not provider_config.api_key or "your" in provider_config.api_key.lower():
            print(f"⚠️  {name}: API key is missing or default")
            fail_count += 1
            continue
            
        # Use configured model if available
        model = provider_config.default_model
        model_info = f" ({model})" if model else ""
        print(f"Testing {name}{model_info}...", end=" ", flush=True)
        
        try:
            # Use get_provider() to auto-load model_info from catalog.
            # This enables catalog-aware features like temperature stripping
            # for reasoning models that reject non-default temperature.
            provider_kwargs = {}
            if name == "gemini":
                provider_kwargs["backend"] = provider_config.backend
                provider_kwargs["project_id"] = provider_config.project_id

            provider = get_provider(name, api_key=provider_config.api_key, model=model, **provider_kwargs)
            
            # Connectivity check (is_available usually does a lightweight check)
            if not provider.is_available():
                print(f"\r❌ {name}: Connection failed (API unreachable or key invalid)")
                fail_count += 1
                continue
                
            # Functional Generation Check (The "Unit Test" part)
            try:
                # Try a small generation to prove auth works.
                # Use max_tokens=50 instead of 1 — reasoning models (e.g.
                # GPT-5) reject very low values.  Truncation is still OK,
                # it proves the API accepted the request (HTTP 200).
                response = provider.generate("Test", max_tokens=50)
                print(f"\r✅ {name}: Success! (Authenticated & Generating)")
                success_count += 1
            except AIOutputTruncatedError:
                # Truncation at max_tokens=1 is expected — the API worked!
                print(f"\r✅ {name}: Success! (Authenticated & Generating)")
                success_count += 1
            except Exception as e:
                # Clean up error message
                error_details = str(e).replace("\n", " ").strip()
                if hasattr(e, 'message'):
                    error_details += f" | Details: {e.message}"
                
                print(f"\r❌ {name}: Generation failed: {error_details}")
                
                # DEBUG: Try to list available models to help user find a valid one
                try:
                    if hasattr(provider, 'list_models'):
                        print(f"   ℹ️  Debugging: Checking available models for {name}...")
                        models = provider.list_models()
                        if models:
                            ids = [m['id'] for m in models]
                            print(f"   ✅ Available models: {', '.join(ids[:5])}...")
                            if model and model not in ids:
                                print(f"   ⚠️  Configured model '{model}' is NOT in this list.")
                        else:
                            print("   ⚠️  No models returned by API.")
                except Exception as list_err:
                    print(f"   ⚠️  Could not list models: {list_err}")
                
                fail_count += 1
                
        except Exception as e:
            error_details = str(e).replace("\n", " ").strip()
            print(f"\r❌ {name}: Initialization failed: {error_details}")
            fail_count += 1

    print("=" * 60)
    print(f"Summary: {success_count} passed, {fail_count} failed.")
    
    # If no success at all, or if there were failures, provide help
    if fail_count > 0 or success_count == 0:
        print("\n💡 TROUBLESHOOTING TIP:")
        print("   1. Ensure you have copied the config template:")
        print("      (Windows) copy config\\ai_config.example.json config\\ai_config.json")
        print("      (Mac/Linux) cp config/ai_config.example.json config/ai_config.json")
        print("   2. Edit config/ai_config.json and add your real API keys")
        print("   3. Ensure 'enabled': true is set for your provider")
    
    if fail_count > 0:
        sys.exit(1)

if __name__ == "__main__":
    validate_ai()
