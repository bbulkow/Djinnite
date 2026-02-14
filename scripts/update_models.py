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

def merge_model_data(
    new_models: list[dict], 
    existing_models: list[dict], 
    provider_instance: BaseAIProvider,
    ai_config
) -> list[dict]:
    """Merge new model data with existing, preserving costing and modality data."""
    existing_by_id = {m["id"]: m for m in existing_models}
    
    # Track models where heuristics only found "text" - these might need AI check
    uncertain_models = []
    
    merged = []
    for model in new_models:
        model_id = model["id"]
        
        # 1. Start with Modalities from Provider Heuristics
        discovered = provider_instance.discover_modalities(model_id)
        
        if model_id in existing_by_id:
            existing = existing_by_id[model_id]
            
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
            
        merged.append(model)
        if "cost_tier" in model: del model["cost_tier"]
        if "capabilities" in model: del model["capabilities"]
    
    # 2. AI Estimation Fallback for uncertain new models
    if uncertain_models and ai_config.get_provider(ai_config.default_provider):
        print(f"  🤖 Requesting AI estimation for {len(uncertain_models)} uncertain models...")
        estimates = estimate_modalities_with_ai(uncertain_models, provider_instance.PROVIDER_NAME, ai_config)
        for model in uncertain_models:
            if model["id"] in estimates:
                model["modalities"] = estimates[model["id"]]
    
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
