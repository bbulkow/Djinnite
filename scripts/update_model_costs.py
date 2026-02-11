"""
Update Model Costs Script

Uses the AI abstraction layer to estimate and update cost_score values
for models in config/model_catalog.json.

The anchor model is Gemini 2.5 Flash (cost_score = 1.0).
- Gemini models: Calculate from API pricing data where available
- Other providers: Use AI to estimate based on published pricing

Usage:
    python -m djinnite.scripts.update_model_costs           # Update models without cost_score
    python -m djinnite.scripts.update_model_costs --force   # Recalculate all costs
    python -m djinnite.scripts.update_model_costs --dry-run # Show changes without saving
    python -m djinnite.scripts.update_model_costs --estimator gemini-2.5-pro  # Use specific model
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Support direct execution (adds project root to path)
_project_root = str(Path(__file__).parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from djinnite.config_loader import load_ai_config, CONFIG_DIR
from djinnite.ai_providers import get_provider
from djinnite.llm_logger import LLMLogger
from djinnite.prompts import COST_ESTIMATION_CONFIG


# ============================================================================
# ANCHOR CONFIGURATION
# ============================================================================

# The anchor model - all costs are relative to this
ANCHOR_MODEL_ID = "gemini-2.5-flash"
ANCHOR_PROVIDER = "gemini"
ANCHOR_COST_SCORE = 1.0

# Default cost scores by tier (used as TEMPORARY fallback only when estimation fails)
# These are NOT used as success - they indicate the model needs manual review
TIER_DEFAULTS = {
    "economical": 0.5,
    "standard": 1.0,
    "premium": 10.0,
}

# Known Gemini pricing ($ per 1M tokens) - fallback when API doesn't provide pricing
# Source: https://ai.google.dev/pricing
GEMINI_KNOWN_PRICES = {
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    # Lite variants typically same as flash
    "gemini-2.5-flash-lite": {"input": 0.075, "output": 0.30},
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
    Anchor: gemini-2.5-flash ($0.075 input, $0.30 output) = 1.0
    """
    # Anchor effective price ($ per 1M tokens, weighted)
    anchor_effective = (0.075 * 0.25) + (0.30 * 0.75)  # = 0.24375
    
    # Model effective price
    model_effective = (input_price * 0.25) + (output_price * 0.75)
    
    return model_effective / anchor_effective


def calculate_gemini_cost(model_id: str, api_key: str = None) -> Optional[tuple[float, str]]:
    """
    Calculate cost_score for a Gemini model.
    
    Uses known prices table FIRST (fast), only falls back to API if needed.
    
    Returns:
        Tuple of (cost_score, source) or None if pricing unavailable.
        source is "api" or "known_prices"
    """
    input_price = None
    output_price = None
    source = None
    
    # Check known prices FIRST (no API call needed!)
    if input_price is None or output_price is None:
        # Try exact match first
        if model_id in GEMINI_KNOWN_PRICES:
            prices = GEMINI_KNOWN_PRICES[model_id]
            input_price = prices["input"]
            output_price = prices["output"]
            source = "known_prices"
        else:
            # Try to match base model (e.g., gemini-2.5-flash-001 -> gemini-2.5-flash)
            for known_id, prices in GEMINI_KNOWN_PRICES.items():
                if model_id.startswith(known_id):
                    input_price = prices["input"]
                    output_price = prices["output"]
                    source = "known_prices"
                    break
    
    if input_price is not None and output_price is not None:
        cost_score = calculate_cost_score(input_price, output_price)
        return (cost_score, source)
    
    return None


# Provider company names for clearer prompts
PROVIDER_COMPANIES = {
    "gemini": "Google Gemini",
    "claude": "Anthropic Claude", 
    "chatgpt": "OpenAI ChatGPT",
}


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
    Use AI to estimate cost_scores for models.
    
    Uses externalized prompts from djinnite/prompts/__init__.py and logs
    all requests/responses for debugging.
    
    Args:
        models_to_estimate: List of model dicts with id, provider info
        provider_name: Name of the provider being estimated (e.g., "chatgpt")
        estimator_provider: Provider to use for estimation (e.g., "gemini")
        estimator_model: Model to use for estimation
        api_key: API key for the estimator provider
        logger: Optional LLMLogger for observability
        gemini_api_key: Optional Gemini API key for web search (used by OpenAI)
        
    Returns:
        Dict mapping model_id to estimated cost_score
    """
    if not models_to_estimate:
        return {}
    
    # Initialize logger if not provided
    if logger is None:
        logger = LLMLogger("cost_estimation")
    
    # Simple list of model IDs (no redundant JSON structure)
    model_list = "\n".join(m["id"] for m in models_to_estimate)
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
    
    # Log the request BEFORE sending (observability)
    request_id = logger.log_request(
        prompt=prompt,
        system_prompt=system_prompt,
        model=estimator_model,
        provider=estimator_provider,
        metadata={"model_count": len(models_to_estimate), "max_tokens": max_tokens}
    )
    
    raw_response = ""
    
    try:
        provider = get_provider(estimator_provider, api_key, estimator_model, gemini_api_key=gemini_api_key)
        response = provider.generate_json(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            web_search=web_search  # Enable web search for current pricing
        )
        
        raw_response = response.content
        content = raw_response.strip()
        
        # Clean up common LLM output issues
        # Remove markdown code fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            lines = lines[1:]  # Remove first line (```json or ```)
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]  # Remove last line
            content = "\n".join(lines)
        
        # Try to find JSON object in the response
        start_idx = content.find("{")
        end_idx = content.rfind("}")
        if start_idx != -1 and end_idx != -1:
            content = content[start_idx:end_idx + 1]
        
        # Parse the JSON response
        estimates = json.loads(content)
        
        # Validate and clean up
        cleaned = {}
        for model_id, score in estimates.items():
            if isinstance(score, (int, float)) and score > 0:
                cleaned[model_id] = float(score)
        
        # Log successful response
        logger.log_response(
            request_id=request_id,
            response_content=raw_response,
            success=True,
            usage=response.usage if hasattr(response, 'usage') else None,
            parsed_result=cleaned
        )
        
        return cleaned
        
    except json.JSONDecodeError as e:
        error_msg = f"JSON parse error: {e}"
        print(f"  ⚠️ Failed to parse AI response as JSON: {e}")
        # Log failed response for debugging
        logger.log_response(
            request_id=request_id,
            response_content=raw_response,
            success=False,
            error=error_msg
        )
        return {}
    except Exception as e:
        error_msg = f"AI estimation error: {e}"
        print(f"  ⚠️ AI estimation failed: {e}")
        logger.log_response(
            request_id=request_id,
            response_content=raw_response,
            success=False,
            error=error_msg
        )
        return {}


def get_tier_default(model: dict) -> float:
    """Get default cost_score based on model's cost_tier."""
    tier = model.get("cost_tier", "standard")
    return TIER_DEFAULTS.get(tier, 1.0)


# ============================================================================
# MAIN UPDATE LOGIC
# ============================================================================

def load_catalog() -> dict:
    """Load the model catalog from disk."""
    catalog_path = CONFIG_DIR / "model_catalog.json"
    if not catalog_path.exists():
        print("❌ Model catalog not found. Run 'python -m djinnite.scripts.update_models' first.")
        sys.exit(1)
    
    with open(catalog_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_catalog(catalog: dict) -> None:
    """Save the model catalog to disk."""
    catalog_path = CONFIG_DIR / "model_catalog.json"
    with open(catalog_path, 'w', encoding='utf-8') as f:
        json.dump(catalog, f, indent=2)


def update_model_costs(
    force: bool = False,
    dry_run: bool = False,
    estimator_model: Optional[str] = None,
    provider_filter: Optional[str] = None
) -> None:
    """
    Main function to update cost_score values in the model catalog.
    
    Args:
        force: If True, recalculate all costs (not just missing ones)
        dry_run: If True, show changes without saving
        estimator_model: Specific model to use for AI estimation
        provider_filter: If specified, only process this provider (e.g., "chatgpt")
    """
    print("🔧 Model Cost Updater")
    print("━" * 40)
    print(f"Anchor: {ANCHOR_MODEL_ID} = {ANCHOR_COST_SCORE}")
    if provider_filter:
        print(f"Provider filter: {provider_filter}")
    print()
    
    # Load configuration and catalog
    ai_config = load_ai_config()
    catalog = load_catalog()
    
    # Determine estimator model
    if estimator_model:
        est_provider, est_model = None, estimator_model
        # Figure out which provider has this model
        for prov_name, prov_data in catalog.items():
            model_ids = [m["id"] for m in prov_data.get("models", [])]
            if estimator_model in model_ids:
                est_provider = prov_name
                break
        if not est_provider:
            print(f"⚠️ Estimator model '{estimator_model}' not found in catalog.")
            print("   Using default provider's default model.")
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
    
    # Process each provider
    for provider_name, provider_data in catalog.items():
        # Skip if provider filter is set and doesn't match
        if provider_filter and provider_name != provider_filter:
            continue
            
        models = provider_data.get("models", [])
        if not models:
            continue
        
        print(f"📊 {provider_name.upper()} ({len(models)} models)")
        
        # Get API key for this provider (for Gemini pricing lookup)
        provider_config = ai_config.get_provider(provider_name)
        provider_api_key = provider_config.api_key if provider_config else None
        
        # Separate models that need estimation
        models_needing_estimation = []
        
        for model in models:
            model_id = model["id"]
            has_cost = "cost_score" in model
            is_manual = model.get("cost_source") == "manual"
            is_disabled = model.get("disabled", False)
            
            # Skip already-disabled models (unless forcing)
            if is_disabled and not force:
                reason = model.get("disabled_reason", "unknown")
                print(f"  🚫 {model_id}: DISABLED ({reason})")
                stats["unchanged"] += 1
                continue
            
            # Skip if already has cost and not forcing (unless manual)
            if has_cost and not force:
                if is_manual:
                    print(f"  ⏭️ {model_id}: {model['cost_score']:.2f} (manual, preserved)")
                else:
                    print(f"  ✓ {model_id}: {model['cost_score']:.2f} (existing)")
                stats["unchanged"] += 1
                continue
            
            # Skip manual overrides even with --force
            if is_manual:
                print(f"  ⏭️ {model_id}: {model['cost_score']:.2f} (manual, preserved)")
                stats["unchanged"] += 1
                continue
            
            # Check if this is the anchor
            if model_id == ANCHOR_MODEL_ID and provider_name == ANCHOR_PROVIDER:
                model["cost_score"] = ANCHOR_COST_SCORE
                model["cost_source"] = "anchor"
                model["cost_updated"] = today
                print(f"  ⚓ {model_id}: {ANCHOR_COST_SCORE} (anchor)")
                stats["updated" if has_cost else "new"] += 1
                continue
            
            # For Gemini, try to calculate from known prices
            if provider_name == "gemini" and provider_api_key:
                result = calculate_gemini_cost(model_id, provider_api_key)
                if result is not None:
                    cost_score, source = result
                    model["cost_score"] = round(cost_score, 2)
                    model["cost_source"] = source
                    model["cost_updated"] = today
                    print(f"  ✓ {model_id}: {cost_score:.2f} ({source})")
                    stats["updated" if has_cost else "new"] += 1
                    continue
            
            # Queue for AI estimation (non-Gemini models, or Gemini without known price)
            models_needing_estimation.append({
                "id": model_id,
                "provider": provider_name,
                "name": model.get("name", model_id),
                "cost_tier": model.get("cost_tier", "standard"),
                "_model_ref": model  # Reference to update later
            })
        
        # Batch AI estimation for this provider
        if models_needing_estimation and est_api_key:
            print(f"  🤖 Estimating {len(models_needing_estimation)} models with AI...")
            
            # Get Gemini API key for OpenAI web search fallback
            gemini_config = ai_config.get_provider("gemini")
            gemini_api_key = gemini_config.api_key if gemini_config else None
            
            estimates = estimate_costs_with_ai(
                models_needing_estimation,
                provider_name,  # The provider being estimated
                est_provider,   # The estimator provider
                est_model,
                est_api_key,
                gemini_api_key=gemini_api_key
            )
            
            for m in models_needing_estimation:
                model_id = m["id"]
                model_ref = m["_model_ref"]
                has_cost = "cost_score" in model_ref
                
                if model_id in estimates:
                    model_ref["cost_score"] = round(estimates[model_id], 2)
                    model_ref["cost_source"] = "estimated"
                    model_ref["cost_updated"] = today
                    print(f"  ✓ {model_id}: {estimates[model_id]:.2f} (estimated)")
                    stats["estimated"] += 1
                    stats["updated" if has_cost else "new"] += 1
                else:
                    # FAILED - AI estimation didn't return this model
                    # Still set a tier default as placeholder, but mark as FAILED
                    default_score = get_tier_default(model_ref)
                    model_ref["cost_score"] = default_score
                    model_ref["cost_source"] = "failed"  # Mark as failed, not tier_default
                    model_ref["cost_updated"] = today
                    print(f"  ❌ {model_id}: {default_score:.2f} (FAILED - needs manual review)")
                    stats["failed"] += 1
        
        elif models_needing_estimation:
            # No AI available - this is a failure
            print(f"  ❌ No AI estimator available - marking as FAILED...")
            for m in models_needing_estimation:
                model_ref = m["_model_ref"]
                default_score = get_tier_default(model_ref)
                model_ref["cost_score"] = default_score
                model_ref["cost_source"] = "failed"  # Mark as failed
                model_ref["cost_updated"] = today
                print(f"  ❌ {m['id']}: {default_score:.2f} (FAILED - needs manual review)")
                stats["failed"] += 1
        
        print()
    
    # Save or report
    print("━" * 40)
    if dry_run:
        print("🔍 DRY RUN - No changes saved")
    else:
        save_catalog(catalog)
        print(f"💾 Saved to {CONFIG_DIR / 'model_catalog.json'}")
    
    print(f"   Updated: {stats['updated']} models")
    print(f"   New: {stats['new']} models")
    print(f"   Estimated by AI: {stats['estimated']} models")
    print(f"   Unchanged: {stats['unchanged']} models")
    if stats["disabled"] > 0:
        print(f"   🚫 Disabled: {stats['disabled']} models (specialty/legacy)")
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
        "--force", "-f",
        action="store_true",
        help="Recalculate all costs (not just missing ones)"
    )
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
    
    args = parser.parse_args()
    
    update_model_costs(
        force=args.force,
        dry_run=args.dry_run,
        estimator_model=args.estimator,
        provider_filter=args.provider
    )


if __name__ == "__main__":
    main()
