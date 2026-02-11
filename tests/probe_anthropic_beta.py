"""
Anthropic Web Search Beta Probe

Tests whether the configured Anthropic account has access to the 
'web-search-20250305' beta feature.

Run: python -m djinnite.tests.probe_anthropic_beta
"""

import sys
from pathlib import Path

# Support direct execution (adds project root to path)
_project_root = str(Path(__file__).parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from djinnite.config_loader import load_ai_config


def probe_beta_access():
    """Probe for Anthropic web search beta access."""
    import anthropic
    
    # Load API key from config
    ai_config = load_ai_config()
    claude_config = ai_config.get_provider("claude")
    
    if not claude_config or not claude_config.api_key:
        print("❌ ERROR: No Claude API key configured in ai_config.json")
        return False
    
    client = anthropic.Anthropic(api_key=claude_config.api_key)
    
    print("🔍 Probing for 'web-search-20250305' capability...")
    print(f"   Using model: claude-3-5-sonnet-20241022")
    
    try:
        response = client.beta.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            messages=[{"role": "user", "content": "What is the price of GPT-5 today?"}],
            # THE MAGIC KEYS
            betas=["web-search-20250305"], 
            tools=[{
                "type": "web_search_20250305",
                "name": "web_search"
            }]
        )
        print("✅ SUCCESS: Your account has access to Native Web Search.")
        print(f"   Response stop_reason: {response.stop_reason}")
        
        # Check if search was used
        for block in response.content:
            if hasattr(block, 'type'):
                print(f"   Content block type: {block.type}")
        
        return True
        
    except anthropic.BadRequestError as e:
        error_str = str(e).lower()
        if "invalid header" in error_str or "beta" in error_str or "betas" in error_str:
            print(f"❌ FAILED: Feature flag rejected.")
            print(f"   Reason: {e}")
            print("👉 ACTION: Fallback to 'Manual Tool Injection' (Standard RAG).")
            return False
        else:
            print(f"⚠️ ERROR: Unrelated error occurred: {e}")
            return False
    except anthropic.APIError as e:
        print(f"⚠️ API ERROR: {e}")
        return False
    except Exception as e:
        print(f"⚠️ UNEXPECTED ERROR: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    result = probe_beta_access()
    sys.exit(0 if result else 1)
