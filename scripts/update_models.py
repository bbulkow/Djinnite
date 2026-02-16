"""
Update Models Script

Fetches the latest model lists from AI providers and updates
config/model_catalog.json.

This script preserves existing cost_score and modality values 
when updating models. To update cost scores, use update_model_costs.py.

Usage:
    python -m djinnite.scripts.update_models
"""

import json
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List

# Support direct execution
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from djinnite.config_loader import load_ai_config, CONFIG_DIR, Modalities
from djinnite.ai_providers import get_provider, BaseAIProvider
from djinnite.ai_providers.gemini_provider import GeminiProvider
from djinnite.ai_providers.claude_provider import ClaudeProvider
from djinnite.ai_providers.openai_provider import OpenAIProvider

# ============================================================================
# KNOWN OUTPUT TOKEN LIMITS
# ============================================================================
# These are authoritative max output token values for well-known models.
# Used when the provider API doesn't expose the limit directly.
# Source: Official provider documentation.

KNOWN_OUTPUT_LIMITS = {
    # --- Anthropic Claude ---
    # Claude 3 family
    "claude-3-haiku-20240307": 4096,
    "claude-3-5-haiku-20241022": 8192,
    "claude-3-5-sonnet-20241022": 8192,
    "claude-3-7-sonnet-20250219": 8192,  # 64K with extended thinking, but default is 8192
    # Claude 4 family
    "claude-sonnet-4-20250514": 16384,
    "claude-opus-4-20250514": 16384,
    "claude-opus-4-1-20250805": 16384,
    "claude-sonnet-4-5-20250929": 16384,
    "claude-haiku-4-5-20251001": 8192,
    "claude-opus-4-5-20251101": 16384,
    "claude-opus-4-6": 16384,
    
    # --- OpenAI ---
    # GPT-3.5 family
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-0125": 4096,
    "gpt-3.5-turbo-1106": 4096,
    "gpt-3.5-turbo-16k": 4096,
    "gpt-3.5-turbo-instruct": 4096,
    "gpt-3.5-turbo-instruct-0914": 4096,
    # GPT-4 family
    "gpt-4": 8192,
    "gpt-4-0613": 8192,
    "gpt-4-turbo": 4096,
    "gpt-4-turbo-preview": 4096,
    "gpt-4-turbo-2024-04-09": 4096,
    "gpt-4-0125-preview": 4096,
    "gpt-4-1106-preview": 4096,
    # GPT-4o family
    "gpt-4o": 16384,
    "gpt-4o-2024-05-13": 4096,
    "gpt-4o-2024-08-06": 16384,
    "gpt-4o-2024-11-20": 16384,
    "gpt-4o-mini": 16384,
    "gpt-4o-mini-2024-07-18": 16384,
    "gpt-4o-audio-preview": 16384,
    "gpt-4o-search-preview": 16384,
    "gpt-4o-mini-search-preview": 16384,
    # GPT-4.1 family
    "gpt-4.1": 32768,
    "gpt-4.1-2025-04-14": 32768,
    "gpt-4.1-mini": 32768,
    "gpt-4.1-mini-2025-04-14": 32768,
    "gpt-4.1-nano": 32768,
    "gpt-4.1-nano-2025-04-14": 32768,
    # GPT-5 family (estimates based on publicly available info)
    "gpt-5": 32768,
    "gpt-5-mini": 32768,
    "gpt-5-nano": 16384,
}


# Template for AI-based output limit estimation
OUTPUT_LIMIT_ESTIMATION_PROMPT = """You are an AI model specification expert.
I need the maximum output token limits for these {provider_company} models.

IMPORTANT: Return the MAXIMUM output token count each model can generate in a single response.
This is NOT the context window - it's the max_output_tokens / max_completion_tokens parameter limit.

Return a JSON object where keys are model IDs and values are integers (the max output token count).
If you are unsure about a model, use 0.

MODELS TO ANALYZE:
{model_list}
"""


# Template for AI-based modality estimation
MODALITY_ESTIMATION_PROMPT = """You are an AI model capability expert. 
I have a list of AI model IDs from {provider_company}. 
For each model, determine its primary input and output modalities.

MODALITIES: "text", "vision", "audio", "video", "embedding"

Return a JSON object where keys are model IDs and values are objects with "input" and "output" lists.

Example output:
{{
  "gpt-4o": {{ "input": ["text", "vision"], "output": ["text"] }},
  "tts-1": {{ "input": ["text"], "output": ["audio"] }}
}}

MODELS TO ANALYZE:
{model_list}
"""

def estimate_modalities_with_ai(
    models: list[dict],
    provider_name: str,
    ai_config
) -> dict:
    """Use AI to estimate modalities for models that heuristics missed."""
    default_provider = ai_config.default_provider
    p_config = ai_config.get_provider(default_provider)
    if not p_config or not p_config.api_key:
        return {}

    provider_company = {"gemini": "Google", "claude": "Anthropic", "chatgpt": "OpenAI"}.get(provider_name, provider_name)
    model_list = "\n".join(m["id"] for m in models)
    prompt = MODALITY_ESTIMATION_PROMPT.format(provider_company=provider_company, model_list=model_list)
    
    try:
        # Create a temporary provider for estimation
        instance = get_provider(default_provider, p_config.api_key, p_config.default_model)
        resp = instance.generate_json(prompt)
        content = resp.content.strip()
        # Handle markdown blocks if present
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```json"):
                content = "\n".join(lines[1:-1])
            else:
                content = "\n".join(lines[1:-1])
        
        return json.loads(content)
    except Exception as e:
        print(f"  ⚠️ Modality estimation failed: {e}")
        return {}

def estimate_output_limits_with_ai(
    models: list[dict],
    provider_name: str,
    ai_config
) -> dict[str, int]:
    """Use AI with web search to estimate max output token limits for unknown models."""
    default_provider = ai_config.default_provider
    p_config = ai_config.get_provider(default_provider)
    if not p_config or not p_config.api_key:
        return {}

    provider_company = {"gemini": "Google", "claude": "Anthropic", "chatgpt": "OpenAI"}.get(provider_name, provider_name)
    model_list = "\n".join(m["id"] for m in models)
    prompt = OUTPUT_LIMIT_ESTIMATION_PROMPT.format(provider_company=provider_company, model_list=model_list)
    
    try:
        gemini_config = ai_config.get_provider("gemini")
        gemini_api_key = gemini_config.api_key if gemini_config else None
        instance = get_provider(default_provider, p_config.api_key, p_config.default_model, gemini_api_key=gemini_api_key)
        resp = instance.generate_json(prompt, web_search=True)
        content = resp.content.strip()
        # Handle markdown blocks if present
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        
        estimates = json.loads(content)
        # Validate: only keep positive integers
        cleaned = {}
        for model_id, limit in estimates.items():
            if isinstance(limit, (int, float)) and limit > 0:
                cleaned[model_id] = int(limit)
        return cleaned
    except Exception as e:
        print(f"  ⚠️ Output limit estimation failed: {e}")
        return {}


def _resolve_max_output_tokens(model_id: str, api_value: int, existing_value: int) -> int:
    """
    Resolve the max_output_tokens for a model using the priority:
    1. API value (if the provider returned it, e.g. Gemini)
    2. Known output limits table
    3. Existing catalog value (preserve what we had)
    4. 0 (unknown)
    """
    # 1. API gave us a value
    if api_value and api_value > 0:
        return api_value
    
    # 2. Known table
    if model_id in KNOWN_OUTPUT_LIMITS:
        return KNOWN_OUTPUT_LIMITS[model_id]
    
    # 3. Existing catalog value
    if existing_value and existing_value > 0:
        return existing_value
    
    # 4. Unknown
    return 0


def merge_model_data(
    new_models: list[dict], 
    existing_models: list[dict], 
    provider_instance: BaseAIProvider,
    ai_config
) -> list[dict]:
    """Merge new model data with existing, preserving costing, modality, and output limit data."""
    existing_by_id = {m["id"]: m for m in existing_models}
    
    # Track models where heuristics only found "text" - these might need AI check
    uncertain_models = []
    # Track models with unknown output limits for AI estimation
    unknown_output_limit_models = []
    
    merged = []
    for model in new_models:
        model_id = model["id"]
        
        # 1. Start with Modalities from Provider Heuristics
        discovered = provider_instance.discover_modalities(model_id)
        
        # Resolve max_output_tokens
        api_output_limit = model.get("max_output_tokens", 0) or 0
        existing_output_limit = 0
        
        if model_id in existing_by_id:
            existing = existing_by_id[model_id]
            existing_output_limit = existing.get("max_output_tokens", 0) or 0
            
            # Preserve existing structure if it's already structured
            if "modalities" in existing and isinstance(existing["modalities"], dict):
                model["modalities"] = existing["modalities"]
            else:
                # Migrate or overwrite
                model["modalities"] = discovered
                
            # Preserve costing
            if "costing" in existing:
                model["costing"] = existing["costing"]
            else:
                model["costing"] = {
                    "score": existing.get("cost_score", 1.0),
                    "source": existing.get("cost_source", "default"),
                    "updated": existing.get("cost_updated", ""),
                    "tier": existing.get("cost_tier", "standard")
                }
        else:
            # New model
            model["modalities"] = discovered
            model["costing"] = {
                "score": 1.0,
                "source": "default",
                "updated": "",
                "tier": model.get("cost_tier", "standard")
            }
            # Queue for AI check if it seems too generic
            if discovered["input"] == ["text"] and discovered["output"] == ["text"]:
                 uncertain_models.append(model)
        
        # Resolve max_output_tokens from API, known table, or existing
        resolved_output = _resolve_max_output_tokens(model_id, api_output_limit, existing_output_limit)
        model["max_output_tokens"] = resolved_output
        
        # Queue for AI estimation if still unknown
        if resolved_output == 0:
            unknown_output_limit_models.append(model)
            
        merged.append(model)
        if "cost_tier" in model: del model["cost_tier"]
        if "capabilities" in model: del model["capabilities"]
    
    # 2. AI Estimation Fallback for uncertain new models (modalities)
    if uncertain_models and ai_config.get_provider(ai_config.default_provider):
        print(f"  🤖 Requesting AI estimation for {len(uncertain_models)} uncertain models...")
        estimates = estimate_modalities_with_ai(uncertain_models, provider_instance.PROVIDER_NAME, ai_config)
        for model in uncertain_models:
            if model["id"] in estimates:
                model["modalities"] = estimates[model["id"]]
    
    # 3. AI Estimation Fallback for unknown output limits
    if unknown_output_limit_models and ai_config.get_provider(ai_config.default_provider):
        print(f"  🤖 Estimating output limits for {len(unknown_output_limit_models)} models with AI...")
        limit_estimates = estimate_output_limits_with_ai(
            unknown_output_limit_models, provider_instance.PROVIDER_NAME, ai_config
        )
        for model in unknown_output_limit_models:
            if model["id"] in limit_estimates:
                model["max_output_tokens"] = limit_estimates[model["id"]]
    
    return merged

def update_models():
    parser = argparse.ArgumentParser(description="Update AI model catalog")
    parser.add_argument("--config", type=str, help="Path to ai_config.json")
    parser.add_argument("--catalog", type=str, help="Path to model_catalog.json")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    ai_config = load_ai_config(config_path)
    
    catalog_path = Path(args.catalog) if args.catalog else CONFIG_DIR / "model_catalog.json"
    catalog = {}
    
    if catalog_path.exists():
        try:
            with open(catalog_path, 'r', encoding='utf-8') as f:
                catalog = json.load(f)
        except Exception:
            pass

    providers = {
        "gemini": GeminiProvider,
        "claude": ClaudeProvider,
        "chatgpt": OpenAIProvider
    }

    for name, provider_cls in providers.items():
        print(f"\nUpdating {name} models...")
        p_config = ai_config.get_provider(name)
        if not p_config or not p_config.api_key:
            print(f"  ⚠️ Provider {name} not configured, skipping.")
            continue
            
        try:
            instance = provider_cls(api_key=p_config.api_key, model=p_config.default_model)
            new_list = instance.list_models()
            if new_list:
                existing_list = catalog.get(name, {}).get("models", [])
                merged = merge_model_data(new_list, existing_list, instance, ai_config)
                
                catalog[name] = {
                    "models": merged,
                    "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d")
                }
                print(f"✅ Processed {len(merged)} models")
        except Exception as e:
            print(f"❌ Failed: {e}")

    with open(catalog_path, 'w', encoding='utf-8') as f:
        json.dump(catalog, f, indent=2)
    print(f"\n💾 Saved to {catalog_path}")

if __name__ == "__main__":
    update_models()
