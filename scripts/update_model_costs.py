"""
Update Model Costs Script

Uses the AI abstraction layer to estimate and update cost_score values
for models in config/model_catalog.json.

The anchor model is defined in config/known_model_defaults.json (default: Gemini 2.5 Flash, cost_score = 1.0).
All non-anchor models are estimated dynamically via AI with web search.

By default, this script updates all models in the catalog that are not
manually overridden.

Usage:
    python -m djinnite.scripts.update_model_costs           # Update all possible costs
    python -m djinnite.scripts.update_model_costs --dry-run # Show changes without saving
    python -m djinnite.scripts.update_model_costs --estimator gemini-2.5-pro  # Use specific model
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    from djinnite.config_loader import load_ai_config, CONFIG_DIR
    from djinnite.ai_providers import get_provider
    from djinnite.llm_logger import LLMLogger
    from djinnite.prompts import COST_ESTIMATION_CONFIG
except ImportError:
    # Fallback for direct execution when package is not installed
    # Adds the project root (one level up from scripts/) to sys.path
    # to allow importing modules as if they were local
    import sys
    _project_root = str(Path(__file__).resolve().parent.parent)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    
    from config_loader import load_ai_config, CONFIG_DIR
    from ai_providers import get_provider
    from llm_logger import LLMLogger
    from prompts import COST_ESTIMATION_CONFIG


# ============================================================================
# ANCHOR CONFIGURATION (loaded from config/known_model_defaults.json)
# ============================================================================
# The cost anchor is the ONLY static model data we maintain.  It defines
# the reference point for all relative cost scores.  Everything else is
# discovered dynamically via AI estimation.
#
# POLICY: Do NOT add per-model pricing tables to Python code.
# See DEVELOPMENT.md for the full policy on static model data.
# ============================================================================

def _load_anchor_config() -> dict:
    """Load cost anchor from config/known_model_defaults.json."""
    defaults_path = CONFIG_DIR / "known_model_defaults.json"
    if defaults_path.exists():
        with open(defaults_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("cost_anchor", {})
    return {}

_anchor = _load_anchor_config()
ANCHOR_MODEL_ID = _anchor.get("model_id", "gemini-2.5-flash")
ANCHOR_PROVIDER = _anchor.get("provider", "gemini")
ANCHOR_COST_SCORE = _anchor.get("cost_score", 1.0)
_anchor_pricing = _anchor.get("pricing", {})
ANCHOR_INPUT_PRICE = _anchor_pricing.get("input_per_1m_tokens", 0.075)
ANCHOR_OUTPUT_PRICE = _anchor_pricing.get("output_per_1m_tokens", 0.30)

# Default cost scores by tier (used as TEMPORARY fallback only when estimation fails)
# These are NOT used as success - they indicate the model needs manual review
TIER_DEFAULTS = {
    "economical": 0.5,
    "standard": 1.0,
    "premium": 10.0,
}

# NOTE: Prompts are externalized in djinnite/prompts/__init__.py
# COST_ESTIMATION_PROMPT and COST_ANALYST_SYSTEM are imported at top


# ============================================================================
# COST CALCULATION
# ============================================================================

def calculate_cost_score(input_price: float, output_price: float) -> float:
    """
    Calculate cost_score from input/output prices ($ per 1M tokens).
    
    Uses weighted average: 25% input, 75% output (typical extraction pattern).
    Anchor pricing is loaded from config/known_model_defaults.json.
    """
    # Anchor effective price ($ per 1M tokens, weighted)
    anchor_effective = (ANCHOR_INPUT_PRICE * 0.25) + (ANCHOR_OUTPUT_PRICE * 0.75)
    
    # Model effective price
    model_effective = (input_price * 0.25) + (output_price * 0.75)
    
    return model_effective / anchor_effective


# Provider company names for clearer prompts
PROVIDER_COMPANIES = {
    "gemini": "Google Gemini",
    "claude": "Anthropic Claude", 
    "chatgpt": "OpenAI ChatGPT",
}


def calculate_gemini_cost_heuristic(model_id: str) -> float:
    """
    Algorithmic cost estimation for Gemini models relative to gemini-2.5-flash (score=1.0).
    Based on standard pricing tiers.
    """
    model_id = model_id.lower()
    
    # Base multipliers (relative to Flash)
    if "pro" in model_id:
        multiplier = 4.0  # Pro is typically ~4x Flash pricing
    elif "flash-lite" in model_id or "flash-8b" in model_id:
        multiplier = 0.5  # Lite versions are typically half price
    elif "flash" in model_id:
        multiplier = 1.0  # Other Flash models assumed similar
    elif "nano" in model_id:
        multiplier = 0.1  # Nano is very cheap/on-device (often free, but keep non-zero)
    elif "embedding" in model_id:
        multiplier = 0.1  # Embeddings are cheap
    else:
        multiplier = 1.0  # Default fallback
        
    return multiplier


def estimate_costs_with_ai(
    models_to_estimate: list[dict],
    provider_name: str,
    estimator_provider: str,
    estimator_model: str,
    api_key: str,
    logger: Optional[LLMLogger] = None,
    gemini_api_key: Optional[str] = None
) -> dict[str, float]:
    """
    Use AI to estimate cost_scores for models, with batching support.
    """
    if not models_to_estimate:
        return {}
    
    # Initialize logger if not provided
    if logger is None:
        logger = LLMLogger("cost_estimation")
    
    # BATCHING: Process in chunks of 5 to avoid overloading context/search
    BATCH_SIZE = 5
    all_estimates = {}
    
    # Break list into batches
    batches = [models_to_estimate[i:i + BATCH_SIZE] for i in range(0, len(models_to_estimate), BATCH_SIZE)]
    
    print(f"  ... Processing {len(models_to_estimate)} models in {len(batches)} batches ...")

    for i, batch in enumerate(batches):
        batch_ids = [m["id"] for m in batch]
        # Simple list of model IDs for the prompt
        model_list = "\n".join(batch_ids)
        provider_company = PROVIDER_COMPANIES.get(provider_name, provider_name.title())
        
        # Get prompt config values
        prompt_template = COST_ESTIMATION_CONFIG["prompt"]
        system_prompt = COST_ESTIMATION_CONFIG["system_prompt"]
        temperature = COST_ESTIMATION_CONFIG["temperature"]
        max_tokens = COST_ESTIMATION_CONFIG["max_tokens"]
        web_search = COST_ESTIMATION_CONFIG.get("web_search", False)
        
        # Render prompt from externalized template
        prompt = prompt_template.format(
            anchor_model=ANCHOR_MODEL_ID,
            anchor_score=ANCHOR_COST_SCORE,
            provider_name=provider_name,
            provider_company=provider_company,
            model_list=model_list
        )
        
        # Log the request BEFORE sending
        request_id = logger.log_request(
            prompt=prompt,
            system_prompt=system_prompt,
            model=estimator_model,
            provider=estimator_provider,
            metadata={"batch_index": i, "batch_size": len(batch), "max_tokens": max_tokens}
        )
        
        raw_response = ""
        
        try:
            provider = get_provider(estimator_provider, api_key, estimator_model, gemini_api_key=gemini_api_key)
            # Use generate() with JSON-requesting system prompt because the
            # response schema is dynamic (model IDs as keys).  Strict schema
            # enforcement via generate_json() requires a fixed schema.
            json_system = system_prompt or ""
            json_system += "\n\nYou must respond with valid JSON only. No additional text or explanation."
            response = provider.generate(
                prompt=prompt,
                system_prompt=json_system.strip(),
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            raw_response = response.content
            content = raw_response.strip()
            
            # Clean up common LLM output issues
            if content.startswith("```"):
                lines = content.split("\n")
                lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines)
            
            start_idx = content.find("{")
            end_idx = content.rfind("}")
            if start_idx != -1 and end_idx != -1:
                content = content[start_idx:end_idx + 1]
            
            batch_estimates = json.loads(content)
            
            # Merge into main results
            for model_id, score in batch_estimates.items():
                if isinstance(score, (int, float)) and score > 0:
                    all_estimates[model_id] = float(score)
            
            logger.log_response(
                request_id=request_id,
                response_content=raw_response,
                success=True,
                usage=response.usage if hasattr(response, 'usage') else None,
                parsed_result=batch_estimates
            )
            
        except Exception as e:
            logger.log_response(
                request_id=request_id,
                response_content=raw_response,
                success=False,
                error=str(e)
            )
            print(f"    ⚠️ Batch {i+1} failed: {e}")
            # Continue to next batch
    
    return all_estimates


def get_tier_default(model: dict) -> float:
    """Get default cost_score based on model's cost_tier."""
    costing = model.get("costing", {})
    tier = costing.get("tier", model.get("cost_tier", "standard"))
    return TIER_DEFAULTS.get(tier, 1.0)


# ============================================================================
# MAIN UPDATE LOGIC
# ============================================================================

def load_catalog(catalog_path: Optional[Path] = None) -> dict:
    """Load the model catalog from disk."""
    path = catalog_path or CONFIG_DIR / "model_catalog.json"
    if not path.exists():
        print(f"❌ Model catalog not found at {path}. Run 'python -m djinnite.scripts.update_models' first.")
        sys.exit(1)
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_catalog(catalog: dict, catalog_path: Optional[Path] = None) -> None:
    """Save the model catalog to disk."""
    path = catalog_path or CONFIG_DIR / "model_catalog.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(catalog, f, indent=2)


def update_model_costs(
    force: bool = True,
    dry_run: bool = False,
    estimator_model: Optional[str] = None,
    provider_filter: Optional[str] = None,
    config_path: Optional[Path] = None,
    catalog_path: Optional[Path] = None
) -> None:
    """
    Main function to update cost_score values in the model catalog.
    
    By default, force=True means it will update all existing models (except manual ones).
    """
    print("🔧 Model Cost Updater")
    print("━" * 40)
    print(f"Anchor: {ANCHOR_MODEL_ID} = {ANCHOR_COST_SCORE}")
    if provider_filter:
        print(f"Provider filter: {provider_filter}")
    print()
    
    ai_config = load_ai_config(config_path)
    catalog = load_catalog(catalog_path)
    
    # Determine estimator model
    if estimator_model:
        est_provider, est_model = None, estimator_model
        for prov_name, prov_data in catalog.items():
            model_ids = [m["id"] for m in prov_data.get("models", [])]
            if estimator_model in model_ids:
                est_provider = prov_name
                break
        if not est_provider:
            est_provider = ai_config.default_provider
            prov_config = ai_config.get_provider(est_provider)
            est_model = prov_config.default_model if prov_config else None
    else:
        est_provider = ai_config.default_provider
        prov_config = ai_config.get_provider(est_provider)
        est_model = prov_config.default_model if prov_config else None
    
    est_api_key = None
    if est_provider:
        prov_config = ai_config.get_provider(est_provider)
        if prov_config:
            est_api_key = prov_config.api_key
    
    print(f"Estimator: {est_provider}/{est_model}")
    print()
    
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    stats = {"updated": 0, "new": 0, "unchanged": 0, "estimated": 0, "failed": 0, "disabled": 0}
    
    for provider_name, provider_data in catalog.items():
        if provider_filter and provider_name != provider_filter:
            continue
            
        models = provider_data.get("models", [])
        if not models:
            continue
        
        print(f"📊 {provider_name.upper()} ({len(models)} models)")
        
        provider_config = ai_config.get_provider(provider_name)
        provider_api_key = provider_config.api_key if provider_config else None
        
        models_needing_estimation = []
        
        for model in models:
            model_id = model["id"]
            
            # Migrate old schema if needed
            if "costing" not in model:
                model["costing"] = {
                    "score": model.get("cost_score", 1.0),
                    "source": model.get("cost_source", "default"),
                    "updated": model.get("cost_updated", ""),
                    "tier": model.get("cost_tier", "standard")
                }
                # Remove old fields
                for f in ["cost_score", "cost_source", "cost_updated", "cost_tier"]:
                    if f in model: del model[f]

            costing = model["costing"]
            score = costing.get("score")
            has_cost = costing.get("updated") != "" and score is not None
            is_manual = costing.get("source") == "manual"
            is_disabled = model.get("disabled", False)
            
            # Skip disabled models
            if is_disabled:
                reason = model.get("disabled_reason", "unknown")
                print(f"  🚫 {model_id}: DISABLED ({reason})")
                stats["unchanged"] += 1
                continue
            
            # Skip manual overrides
            if is_manual:
                val = f"{score:.2f}" if score is not None else "None"
                print(f"  ⏭️ {model_id}: {val} (manual, preserved)")
                stats["unchanged"] += 1
                continue
            
            # Check if this is the anchor
            if model_id == ANCHOR_MODEL_ID and provider_name == ANCHOR_PROVIDER:
                costing["score"] = ANCHOR_COST_SCORE
                costing["source"] = "anchor"
                costing["updated"] = today
                print(f"  ⚓ {model_id}: {ANCHOR_COST_SCORE} (anchor)")
                stats["updated" if has_cost else "new"] += 1
                continue
            
            # Logic: If force=True, we update everything (except manual).
            # If force=False, we only update if score is None or missing.
            should_estimate = force or (score is None) or (not has_cost)
            
            if should_estimate:
                # GEMINI ALGORITHMIC OVERRIDE
                if provider_name == "gemini":
                    heuristic_score = calculate_gemini_cost_heuristic(model_id)
                    costing["score"] = heuristic_score
                    costing["source"] = "algorithmic"
                    costing["updated"] = today
                    print(f"  🧮 {model_id}: {heuristic_score:.2f} (algorithmic)")
                    stats["updated" if has_cost else "new"] += 1
                    continue

                models_needing_estimation.append({
                    "id": model_id,
                    "provider": provider_name,
                    "name": model.get("name", model_id),
                    "cost_tier": costing.get("tier", "standard"),
                    "_model_ref": model
                })
            else:
                val = f"{score:.2f}" if score is not None else "None"
                print(f"  ⏭️ {model_id}: {val} (skipped)")
                stats["unchanged"] += 1
        
        if models_needing_estimation and est_api_key:
            print(f"  🤖 Estimating {len(models_needing_estimation)} models with AI...")
            gemini_config = ai_config.get_provider("gemini")
            gemini_api_key = gemini_config.api_key if gemini_config else None
            
            estimates = estimate_costs_with_ai(
                models_needing_estimation,
                provider_name,
                est_provider,
                est_model,
                est_api_key,
                gemini_api_key=gemini_api_key
            )
            
            for m in models_needing_estimation:
                model_id = m["id"]
                model_ref = m["_model_ref"]
                costing = model_ref["costing"]
                has_prev_cost = costing.get("updated") != ""
                
                if model_id in estimates:
                    costing["score"] = round(estimates[model_id], 2)
                    costing["source"] = "estimated"
                    costing["updated"] = today
                    print(f"  ✓ {model_id}: {estimates[model_id]:.2f} (estimated)")
                    stats["estimated"] += 1
                    stats["updated" if has_prev_cost else "new"] += 1
                else:
                    default_score = get_tier_default(model_ref)
                    costing["score"] = default_score
                    costing["source"] = "failed"
                    costing["updated"] = today
                    print(f"  ❌ {model_id}: {default_score:.2f} (FAILED - needs manual review)")
                    stats["failed"] += 1
        
        elif models_needing_estimation:
            print(f"  ❌ No AI estimator available - marking as FAILED...")
            for m in models_needing_estimation:
                model_ref = m["_model_ref"]
                costing = model_ref["costing"]
                default_score = get_tier_default(model_ref)
                costing["score"] = default_score
                costing["source"] = "failed"
                costing["updated"] = today
                print(f"  ❌ {m['id']}: {default_score:.2f} (FAILED - needs manual review)")
                stats["failed"] += 1
        
        print()
    
    print("━" * 40)
    if dry_run:
        print("🔍 DRY RUN - No changes saved")
    else:
        save_catalog(catalog, catalog_path)
        print(f"💾 Saved to {catalog_path or CONFIG_DIR / 'model_catalog.json'}")
    
    print(f"   Updated: {stats['updated']} models")
    print(f"   New: {stats['new']} models")
    print(f"   Estimated by AI: {stats['estimated']} models")
    print(f"   Unchanged: {stats['unchanged']} models")
    if stats["failed"] > 0:
        print(f"   ❌ FAILED: {stats['failed']} models (need manual review)")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Update cost_score values for AI models in the catalog."
    )
    parser.add_argument(
        "--no-force",
        dest="force",
        action="store_false",
        help="Only update models without existing cost data"
    )
    parser.set_defaults(force=True)
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would change without saving"
    )
    parser.add_argument(
        "--estimator",
        type=str,
        default=None,
        help="Specific model to use for AI estimation (e.g., gemini-2.5-pro)"
    )
    parser.add_argument(
        "--provider", "-p",
        type=str,
        default=None,
        help="Only process this provider (e.g., chatgpt, claude, gemini)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to ai_config.json"
    )
    parser.add_argument(
        "--catalog",
        type=str,
        help="Path to model_catalog.json"
    )
    
    args = parser.parse_args()
    
    update_model_costs(
        force=args.force,
        dry_run=args.dry_run,
        estimator_model=args.estimator,
        provider_filter=args.provider,
        config_path=Path(args.config) if args.config else None,
        catalog_path=Path(args.catalog) if args.catalog else None
    )


if __name__ == "__main__":
    main()
