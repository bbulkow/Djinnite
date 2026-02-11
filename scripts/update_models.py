"""
Update Models Script

Fetches the latest model lists from AI providers and updates
config/model_catalog.json.

This script preserves existing cost_score values when updating models.
To update cost scores, use update_model_costs.py instead.

Usage:
    python -m djinnite.scripts.update_models
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Support direct execution (adds project root to path)
_project_root = str(Path(__file__).parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from djinnite.config_loader import load_ai_config, CONFIG_DIR
from djinnite.ai_providers.gemini_provider import GeminiProvider
from djinnite.ai_providers.claude_provider import ClaudeProvider
from djinnite.ai_providers.openai_provider import OpenAIProvider


# Cost-related fields that should be preserved during model updates
COST_FIELDS = ["cost_score", "cost_source", "cost_updated"]


def merge_model_data(new_models: list[dict], existing_models: list[dict]) -> list[dict]:
    """
    Merge new model data with existing, preserving cost fields.
    
    Args:
        new_models: Fresh model list from API
        existing_models: Existing models from catalog (may have cost data)
        
    Returns:
        Merged model list with cost data preserved
    """
    # Build lookup of existing models by ID
    existing_by_id = {m["id"]: m for m in existing_models}
    
    merged = []
    for model in new_models:
        model_id = model["id"]
        
        # If we have existing data, preserve cost fields
        if model_id in existing_by_id:
            existing = existing_by_id[model_id]
            for field in COST_FIELDS:
                if field in existing:
                    model[field] = existing[field]
        
        merged.append(model)
    
    return merged


def update_models():
    print("Loading configuration...")
    ai_config = load_ai_config()
    
    catalog_path = CONFIG_DIR / "model_catalog.json"
    catalog = {}
    
    # Load existing catalog to preserve data for providers we can't update
    if catalog_path.exists():
        try:
            with open(catalog_path, 'r', encoding='utf-8') as f:
                catalog = json.load(f)
        except json.JSONDecodeError:
            print("⚠️ Existing catalog invalid, starting fresh.")

    providers = {
        "gemini": GeminiProvider,
        "claude": ClaudeProvider,
        "chatgpt": OpenAIProvider
    }

    updated_count = 0

    for name, provider_cls in providers.items():
        print(f"\nUpdating {name} models...")
        
        provider_config = ai_config.get_provider(name)
        
        # Check if provider is configured
        if not provider_config:
            print(f"⚠️ {name} provider not found in ai_config.json")
            continue
            
        # Check if API key is present
        if not provider_config.api_key or "YOUR_" in provider_config.api_key:
            print(f"⚠️ {name} API key not configured. Skipping.")
            continue
            
        try:
            # Initialize provider. We pass the default model, although it's not
            # strictly necessary for list_models(), it is required by the constructor.
            provider = provider_cls(
                api_key=provider_config.api_key,
                model=provider_config.default_model
            )
            
            # List models
            print(f"  Fetching model list...")
            models = provider.list_models()
            
            if models:
                print(f"✅ Found {len(models)} models for {name}")
                
                # Preserve existing cost data during merge
                existing_models = catalog.get(name, {}).get("models", [])
                merged_models = merge_model_data(models, existing_models)
                
                catalog[name] = {
                    "models": merged_models,
                    "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d")
                }
                updated_count += 1
            else:
                print(f"⚠️ No models returned for {name} (check permissions/API key)")
                
        except Exception as e:
            print(f"❌ Failed to update {name}: {e}")

    if updated_count == 0:
        print("\n❌ No providers were successfully updated.")
        print("💡 PREREQUISITE: This command requires valid API keys to fetch live data.")
        print("   Please configure config/ai_config.json first.")

    # Save catalog
    print("\nSaving model catalog...")
    try:
        with open(catalog_path, 'w', encoding='utf-8') as f:
            json.dump(catalog, f, indent=2)
        print(f"✅ Saved to {catalog_path}")
    except Exception as e:
        print(f"❌ Failed to save catalog: {e}")

    # Validate defaults
    print("\nValidating default models...")
    for name, provider_data in catalog.items():
        provider_config = ai_config.get_provider(name)
        if not provider_config:
            continue
            
        default_model = provider_config.default_model
        if not default_model:
            print(f"ℹ️  {name}: No default model configured.")
            continue
            
        available_ids = [m["id"] for m in provider_data.get("models", [])]
        
        if default_model in available_ids:
            print(f"✅ {name}: Default model '{default_model}' is valid.")
        else:
            print(f"⚠️  {name}: Default model '{default_model}' NOT found in current catalog!")
            if available_ids:
                print(f"    Available: {', '.join(available_ids[:5])}...")
            print(f"    Action: Update 'default_model' in config/ai_config.json")

if __name__ == "__main__":
    update_models()
