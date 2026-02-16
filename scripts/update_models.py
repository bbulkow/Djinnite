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
# POLICY: NO STATIC MODEL DATA IN PYTHON
# ============================================================================
# Model capabilities (output limits, structured JSON support, pricing) MUST
# be discovered dynamically via:
#   1. Provider API responses (e.g. Gemini exposes output_token_limit)
#   2. Live probes (e.g. structured JSON support testing)
#   3. AI estimation with web search (for values APIs don't expose)
#   4. Existing model_catalog.json values (persisted between runs)
#
# Do NOT add per-model data tables to Python code.  If an un-discoverable
# override is truly necessary, add it to config/known_model_defaults.json
# with a comment explaining why dynamic discovery is impossible.
# ============================================================================


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
        # Create a temporary provider for estimation.
        # Use generate() with JSON-requesting system prompt because the
        # response schema is dynamic (model IDs as keys).
        instance = get_provider(default_provider, p_config.api_key, p_config.default_model)
        resp = instance.generate(
            prompt=prompt,
            system_prompt="You must respond with valid JSON only. No additional text or explanation.",
            temperature=0.3,
        )
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
        # Use generate() with JSON-requesting system prompt because the
        # response schema is dynamic (model IDs as keys).
        instance = get_provider(default_provider, p_config.api_key, p_config.default_model, gemini_api_key=gemini_api_key)
        resp = instance.generate(
            prompt=prompt,
            system_prompt="You must respond with valid JSON only. No additional text or explanation.",
            temperature=0.3,
            web_search=True,  # Critical: ground with current docs
        )
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
    Resolve the max_output_tokens for a model using dynamic sources only:
    1. Provider API value (e.g. Gemini exposes output_token_limit)
    2. Existing catalog value (persisted from prior AI estimation runs)
    3. 0 (unknown — will trigger AI estimation)
    """
    if api_value and api_value > 0:
        return api_value
    if existing_value and existing_value > 0:
        return existing_value
    return 0


def _resolve_structured_json_support(
    model_id: str,
    existing_value: Optional[bool],
) -> Optional[bool]:
    """
    Resolve supports_structured_json using dynamic sources only:
    1. Existing catalog value (persisted from prior probe runs)
    2. None (unknown — will trigger live probe)
    """
    if existing_value is not None:
        return existing_value
    return None


# Provider-specific thinking styles
THINKING_STYLES = {
    "gemini": "mode",
    "claude": "budget",
    "chatgpt": "effort",
}


def _probe_all_capabilities_for_models(
    models_to_probe: list[dict],
    provider_cls,
    provider_name: str,
    api_key: str,
) -> dict[str, dict]:
    """
    Probe a list of models to discover ALL capabilities at once.
    
    Returns a dict of model_id → {structured_json, temperature, thinking, web_search, thinking_style}.
    """
    results: dict[str, dict] = {}
    for m in models_to_probe:
        model_id = m["id"]
        try:
            instance = provider_cls(api_key=api_key, model=model_id)
            
            ssj = instance.probe_structured_json()
            temp = instance.probe_temperature()
            think = instance.probe_thinking()
            # web_search is a Djinnite-level capability — True for all text models
            ws = True
            ts = THINKING_STYLES.get(provider_name) if think else None
            
            results[model_id] = {
                "structured_json": ssj,
                "temperature": temp,
                "thinking": think,
                "web_search": ws,
                "thinking_style": ts,
            }
            
            parts = []
            parts.append(f"json={'✅' if ssj else '❌' if ssj is False else '❓'}")
            parts.append(f"temp={'✅' if temp else '❌' if temp is False else '❓'}")
            parts.append(f"think={'✅' if think else '❌' if think is False else '❓'}")
            print(f"    {model_id}: {' '.join(parts)}")
            
        except Exception as e:
            results[model_id] = {
                "structured_json": None, "temperature": None,
                "thinking": None, "web_search": True, "thinking_style": None,
            }
            print(f"    ⚠️ {model_id}: probe skipped ({e})")
    return results


def merge_model_data(
    new_models: list[dict], 
    existing_models: list[dict], 
    provider_instance: BaseAIProvider,
    provider_cls,
    api_key: str,
    ai_config
) -> list[dict]:
    """Merge new model data with existing, preserving costing, modality, and output limit data."""
    existing_by_id = {m["id"]: m for m in existing_models}
    
    # Track models where heuristics only found "text" - these might need AI check
    uncertain_models = []
    # Track models with unknown output limits for AI estimation
    unknown_output_limit_models = []
    # Track models needing structured JSON probing
    models_needing_ssj_probe = []
    
    merged = []
    for model in new_models:
        model_id = model["id"]
        
        # 1. Start with Modalities from Provider Heuristics
        discovered = provider_instance.discover_modalities(model_id)
        
        # Resolve max_output_tokens
        api_output_limit = model.get("max_output_tokens", 0) or 0
        existing_output_limit = 0
        existing_ssj = None
        
        if model_id in existing_by_id:
            existing = existing_by_id[model_id]
            existing_output_limit = existing.get("max_output_tokens", 0) or 0
            # Preserve existing capabilities.structured_json value
            # Support both new dict format and old flat format
            raw_caps = existing.get("capabilities")
            if isinstance(raw_caps, dict):
                raw_ssj = raw_caps.get("structured_json")
            else:
                raw_ssj = existing.get("supports_structured_json")
            if raw_ssj is True:
                existing_ssj = True
            elif raw_ssj is False:
                existing_ssj = False
            
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
        
        # Resolve capabilities from existing catalog
        ssj = _resolve_structured_json_support(model_id, existing_ssj)
        # Preserve existing capabilities or initialize fresh
        existing_caps = {}
        if model_id in existing_by_id:
            raw_caps = existing_by_id[model_id].get("capabilities")
            if isinstance(raw_caps, dict):
                existing_caps = raw_caps
        model["capabilities"] = {
            "structured_json": ssj,
            "temperature": existing_caps.get("temperature"),
            "thinking": existing_caps.get("thinking"),
            "web_search": existing_caps.get("web_search"),
            "thinking_style": existing_caps.get("thinking_style"),
        }
        
        # Queue for probing if any capability is still unknown
        needs_probe = (ssj is None or model["capabilities"]["temperature"] is None
                       or model["capabilities"]["thinking"] is None)
        if needs_probe:
            input_mods = model.get("modalities", {})
            if isinstance(input_mods, dict):
                has_text = "text" in input_mods.get("input", [])
            else:
                has_text = True
            # Only probe text-capable, non-specialized models
            is_specialized = any(x in model_id.lower() for x in [
                "tts", "embedding", "realtime", "image", "transcribe",
                "audio", "robotics", "computer-use"
            ])
            if has_text and not is_specialized:
                models_needing_ssj_probe.append(model)
            
        merged.append(model)
        if "cost_tier" in model: del model["cost_tier"]
        # Remove old flat supports_structured_json if present (migrated to capabilities dict)
        if "supports_structured_json" in model: del model["supports_structured_json"]
    
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
    
    # 4. Live probe ALL capabilities on models that need it
    if models_needing_ssj_probe:
        print(f"  🔍 Probing {len(models_needing_ssj_probe)} models for all capabilities...")
        probe_results = _probe_all_capabilities_for_models(
            models_needing_ssj_probe, provider_cls,
            provider_instance.PROVIDER_NAME, api_key,
        )
        for model in models_needing_ssj_probe:
            if model["id"] in probe_results:
                probed = probe_results[model["id"]]
                # Merge probe results into capabilities (probe wins over None)
                caps = model["capabilities"]
                for key in ["structured_json", "temperature", "thinking", "web_search", "thinking_style"]:
                    if probed.get(key) is not None:
                        caps[key] = probed[key]
    
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
                merged = merge_model_data(
                    new_list, existing_list, instance,
                    provider_cls, p_config.api_key, ai_config
                )
                
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
