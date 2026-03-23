"""
Update Model Costs Script

Uses AI with web search to discover current per-token pricing ($/1M tokens)
for all models in config/model_catalog.json.  Every provider is estimated
the same way -- no hardcoded pricing or heuristics.

Usage:
    python -m djinnite.scripts.update_model_costs           # Estimate only new/unknown models
    python -m djinnite.scripts.update_model_costs --all     # Re-estimate ALL models
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
    from djinnite.config_loader import load_ai_config, CONFIG_DIR, _resolve_config_file
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
    
    from config_loader import load_ai_config, CONFIG_DIR, _resolve_config_file
    from ai_providers import get_provider
    from llm_logger import LLMLogger
    from prompts import COST_ESTIMATION_CONFIG


def _load_estimator_config() -> dict:
    """Load Djinnite-internal estimator config from known_model_defaults.json."""
    defaults_path = _resolve_config_file("known_model_defaults.json")
    if defaults_path.exists():
        with open(defaults_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("estimator", {})
    return {}

_estimator_config = _load_estimator_config()


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
) -> dict[str, dict]:
    """
    Use AI with web search to discover per-model pricing.

    Returns:
        dict mapping model_id to {"input_per_1m": float, "output_per_1m": float,
        "search_cost_per_unit": float | None}.
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
            provider_name=provider_name,
            provider_company=provider_company,
            model_list=model_list,
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
                web_search=web_search,
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
            for model_id, pricing in batch_estimates.items():
                if isinstance(pricing, dict):
                    inp = pricing.get("input_per_1m")
                    out = pricing.get("output_per_1m")
                    if isinstance(inp, (int, float)) and isinstance(out, (int, float)) and inp > 0 and out > 0:
                        raw_search = pricing.get("search_cost_per_unit")
                        search = float(raw_search) if isinstance(raw_search, (int, float)) and raw_search > 0 else None
                        all_estimates[model_id] = {
                            "input_per_1m": float(inp),
                            "output_per_1m": float(out),
                            "search_cost_per_unit": search,
                        }
            
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
            print(f"    [WARN] Batch {i+1} failed: {e}")
            # Continue to next batch
    
    return all_estimates


# ============================================================================
# MAIN UPDATE LOGIC
# ============================================================================

def load_catalog(catalog_path: Optional[Path] = None) -> dict:
    """Load the model catalog from disk."""
    path = catalog_path or CONFIG_DIR / "model_catalog.json"
    if not path.exists():
        print(f"[FAIL] Model catalog not found at {path}. Run 'python -m djinnite.scripts.update_models' first.")
        sys.exit(1)
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_catalog(catalog: dict, catalog_path: Optional[Path] = None) -> None:
    """Save the model catalog to disk."""
    path = catalog_path or CONFIG_DIR / "model_catalog.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(catalog, f, indent=2)


def update_model_costs(
    force: bool = False,
    dry_run: bool = False,
    estimator_model: Optional[str] = None,
    provider_filter: Optional[str] = None,
    config_path: Optional[Path] = None,
    catalog_path: Optional[Path] = None
) -> None:
    """
    Main function to update per-token pricing in the model catalog.

    Args:
        force: If False (default), only estimate models that have no cost
               data yet (new/unknown models).  If True (``--all``), re-estimate
               all models except manual overrides.
    """
    print("[TOOL] Model Cost Updater")
    print("-" * 40)
    mode = "ALL models (--all)" if force else "NEW/unknown models only"
    print(f"Mode: {mode}")
    if provider_filter:
        print(f"Provider filter: {provider_filter}")
    print()
    
    ai_config = load_ai_config(config_path)
    catalog = load_catalog(catalog_path)
    
    # Determine estimator model.
    # Priority: 1) --estimator CLI flag  2) known_model_defaults.json  3) ai_config default
    if estimator_model:
        # CLI override
        est_provider, est_model = None, estimator_model
        for prov_name, prov_data in catalog.items():
            model_ids = [m["id"] for m in prov_data.get("models", [])]
            if estimator_model in model_ids:
                est_provider = prov_name
                break
        if not est_provider:
            est_provider = ai_config.default_provider
    elif _estimator_config.get("model"):
        # Djinnite-internal estimator from known_model_defaults.json
        est_provider = _estimator_config.get("provider", "gemini")
        est_model = _estimator_config["model"]
    else:
        # Fallback to user's default provider/model
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
        
        print(f"{provider_name.upper()} ({len(models)} models)")
        
        provider_config = ai_config.get_provider(provider_name)
        provider_api_key = provider_config.api_key if provider_config else None
        
        models_needing_estimation = []
        
        for model in models:
            model_id = model["id"]
            
            # Ensure costing block exists
            if "costing" not in model:
                model["costing"] = {
                    "input_per_1m": None,
                    "output_per_1m": None,
                    "source": "",
                    "updated": "",
                }

            costing = model["costing"]
            has_pricing = costing.get("input_per_1m") is not None and costing.get("updated") != ""
            is_manual = costing.get("source") == "manual"
            is_disabled = model.get("disabled", False)

            # Skip disabled models
            if is_disabled:
                reason = model.get("disabled_reason", "unknown")
                print(f"  [DISABLED] {model_id}: DISABLED ({reason})")
                stats["unchanged"] += 1
                continue

            # Skip manual overrides
            if is_manual:
                inp = costing.get("input_per_1m")
                out = costing.get("output_per_1m")
                print(f"  [SKIP] {model_id}: ${inp}/{out} per 1M (manual, preserved)")
                stats["unchanged"] += 1
                continue

            should_estimate = force or not has_pricing

            if should_estimate:
                models_needing_estimation.append({
                    "id": model_id,
                    "provider": provider_name,
                    "name": model.get("name", model_id),
                    "_model_ref": model,
                })
            else:
                inp = costing.get("input_per_1m")
                out = costing.get("output_per_1m")
                print(f"  [SKIP] {model_id}: ${inp}/{out} per 1M (skipped)")
                stats["unchanged"] += 1

        if models_needing_estimation and est_api_key:
            print(f"  [AI] Estimating {len(models_needing_estimation)} models with AI...")
            gemini_config = ai_config.get_provider("gemini")
            gemini_api_key = gemini_config.api_key if gemini_config else None

            estimates = estimate_costs_with_ai(
                models_needing_estimation,
                provider_name,
                est_provider,
                est_model,
                est_api_key,
                gemini_api_key=gemini_api_key,
            )

            for m in models_needing_estimation:
                model_id = m["id"]
                model_ref = m["_model_ref"]
                costing = model_ref["costing"]
                has_prev = costing.get("updated") != ""

                if model_id in estimates:
                    est = estimates[model_id]
                    costing["input_per_1m"] = round(est["input_per_1m"], 4)
                    costing["output_per_1m"] = round(est["output_per_1m"], 4)
                    costing["search_cost_per_unit"] = est["search_cost_per_unit"]
                    costing["source"] = "estimated"
                    costing["updated"] = today
                    search_str = f", search=${est['search_cost_per_unit']}" if est["search_cost_per_unit"] else ""
                    print(f"  [OK] {model_id}: ${est['input_per_1m']}/{est['output_per_1m']} per 1M{search_str} (estimated)")
                    stats["estimated"] += 1
                    stats["updated" if has_prev else "new"] += 1
                else:
                    costing["input_per_1m"] = None
                    costing["output_per_1m"] = None
                    costing["search_cost_per_unit"] = None
                    costing["source"] = "failed"
                    costing["updated"] = today
                    print(f"  [FAIL] {model_id}: None (FAILED - needs re-estimation or manual review)")
                    stats["failed"] += 1

        elif models_needing_estimation:
            print(f"  [FAIL] No AI estimator available - marking as FAILED...")
            for m in models_needing_estimation:
                model_ref = m["_model_ref"]
                costing = model_ref["costing"]
                costing["input_per_1m"] = None
                costing["output_per_1m"] = None
                costing["source"] = "failed"
                costing["updated"] = today
                print(f"  [FAIL] {m['id']}: None (FAILED - no estimator available)")
                stats["failed"] += 1
        
        print()
    
    print("-" * 40)
    if dry_run:
        print("[CHECK] DRY RUN - No changes saved")
    else:
        save_catalog(catalog, catalog_path)
        print(f"[SAVE] Saved to {catalog_path or CONFIG_DIR / 'model_catalog.json'}")
    
    print(f"   Updated: {stats['updated']} models")
    print(f"   New: {stats['new']} models")
    print(f"   Estimated by AI: {stats['estimated']} models")
    print(f"   Unchanged: {stats['unchanged']} models")
    if stats["failed"] > 0:
        print(f"   [FAIL] FAILED: {stats['failed']} models (need manual review)")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Update per-token dollar pricing for AI models in the catalog. "
            "By default, only estimates pricing for NEW models (those without "
            "existing cost data). Use --all to re-estimate everything."
        )
    )
    parser.add_argument(
        "--all",
        dest="force",
        action="store_true",
        default=False,
        help="Re-estimate ALL models (default: only new/unknown models)"
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
