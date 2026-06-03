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
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Optional

try:
    from djinnite.config_loader import load_ai_config, CONFIG_DIR, _resolve_config_file
    from djinnite.ai_providers import get_provider
    from djinnite.llm_logger import LLMLogger
    from djinnite.prompts import COST_ESTIMATION_CONFIG
    from djinnite.pricing_class import classify_model
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
    from pricing_class import classify_model


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


# --------------------------------------------------------------------------
# Tunable thresholds (overridable via known_model_defaults.json -> estimator)
# --------------------------------------------------------------------------
_bounds = _estimator_config.get("bounds", {})
PRICE_MAX = _bounds.get("price_max", 1000.0)                  # $/1M sanity ceiling
OUTPUT_INPUT_RATIO_MIN = _bounds.get("output_input_ratio_min", 0.5)  # output >= input*this
SEARCH_MAX = _bounds.get("search_max", 1.0)                   # $/search ceiling
DIVERGENCE_THRESHOLD = _estimator_config.get("divergence_threshold", 0.40)  # hold for review
DEFAULT_STALENESS_DAYS = _estimator_config.get("staleness_days", 180)
VERIFY_TOLERANCE = _estimator_config.get("verify_tolerance", 0.10)  # 2-pass agreement


def _check_bounds(inp: float, out: float, search: Optional[float]) -> Optional[str]:
    """Return a violation message if the estimate is implausible, else None."""
    if not (0 < inp <= PRICE_MAX):
        return f"input {inp} outside (0, {PRICE_MAX}]"
    if not (0 < out <= PRICE_MAX):
        return f"output {out} outside (0, {PRICE_MAX}]"
    if out < inp * OUTPUT_INPUT_RATIO_MIN:
        return f"output {out} < input {inp} * {OUTPUT_INPUT_RATIO_MIN} (likely transposed/garbage)"
    if search is not None and not (0 < search <= SEARCH_MAX):
        return f"search_cost {search} outside (0, {SEARCH_MAX}]"
    return None


def _is_stale(updated: str, today: date, max_age_days: int) -> bool:
    """True if an ISO ``updated`` date is missing, unparseable, or too old."""
    if not updated:
        return True
    try:
        return (today - date.fromisoformat(updated)).days > max_age_days
    except ValueError:
        return True


def _pct_change(old: Optional[float], new: float) -> Optional[float]:
    """Fractional change of ``new`` vs ``old``; None if no usable prior."""
    if old is None or old <= 0:
        return None
    return abs(new - old) / old


def _within(a: float, b: float, tol: float) -> bool:
    """True if a and b agree within fractional tolerance ``tol``."""
    base = max(abs(a), abs(b))
    if base == 0:
        return True
    return abs(a - b) / base <= tol



def _run_estimation_pass(
    models_to_estimate: list[dict],
    provider_name: str,
    estimator_provider: str,
    estimator_model: str,
    api_key: str,
    logger: LLMLogger,
    gemini_api_key: Optional[str] = None,
) -> dict[str, dict]:
    """
    One web-search estimation pass over ``models_to_estimate`` (batched).

    Returns dict mapping model_id to one of:
        {"no_public_price": True}                       -- specialty model, no price
        {"input_per_1m", "output_per_1m",
         "search_cost_per_unit", "source_url",
         "no_public_price": False}                      -- a usable estimate
    Models the estimator omits or returns malformed are absent from the result.
    """
    BATCH_SIZE = 5
    all_estimates: dict[str, dict] = {}

    batches = [models_to_estimate[i:i + BATCH_SIZE] for i in range(0, len(models_to_estimate), BATCH_SIZE)]
    print(f"  ... Processing {len(models_to_estimate)} models in {len(batches)} batches ...")

    for i, batch in enumerate(batches):
        batch_ids = [m["id"] for m in batch]
        model_list = "\n".join(batch_ids)
        provider_company = PROVIDER_COMPANIES.get(provider_name, provider_name.title())

        prompt_template = COST_ESTIMATION_CONFIG["prompt"]
        system_prompt = COST_ESTIMATION_CONFIG["system_prompt"]
        temperature = COST_ESTIMATION_CONFIG["temperature"]
        max_output_tokens = COST_ESTIMATION_CONFIG["max_output_tokens"]
        web_search = COST_ESTIMATION_CONFIG.get("web_search", False)

        prompt = prompt_template.format(
            provider_name=provider_name,
            provider_company=provider_company,
            model_list=model_list,
        )

        request_id = logger.log_request(
            prompt=prompt,
            system_prompt=system_prompt,
            model=estimator_model,
            provider=estimator_provider,
            metadata={"batch_index": i, "batch_size": len(batch), "max_output_tokens": max_output_tokens}
        )

        raw_response = ""

        try:
            # require_pricing=False: the estimator must run even if its own model
            # has no/stale price, otherwise cost updates could deadlock.
            provider = get_provider(
                estimator_provider, api_key, estimator_model,
                gemini_api_key=gemini_api_key, require_pricing=False,
            )
            # Use generate() with JSON-requesting system prompt because the
            # response schema is dynamic (model IDs as keys).  Strict schema
            # enforcement via generate_json() requires a fixed schema.
            json_system = system_prompt or ""
            json_system += "\n\nYou must respond with valid JSON only. No additional text or explanation."
            response = provider.generate(
                prompt=prompt,
                system_prompt=json_system.strip(),
                temperature=temperature,
                max_output_tokens=max_output_tokens,
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

            for model_id, pricing in batch_estimates.items():
                if not isinstance(pricing, dict):
                    continue
                # Specialty model with no public per-token price -> stable unknown.
                if pricing.get("no_public_price"):
                    all_estimates[model_id] = {"no_public_price": True}
                    continue
                inp = pricing.get("input_per_1m")
                out = pricing.get("output_per_1m")
                if isinstance(inp, (int, float)) and isinstance(out, (int, float)):
                    raw_search = pricing.get("search_cost_per_unit")
                    search = float(raw_search) if isinstance(raw_search, (int, float)) and raw_search > 0 else None
                    url = pricing.get("source_url")
                    all_estimates[model_id] = {
                        "input_per_1m": float(inp),
                        "output_per_1m": float(out),
                        "search_cost_per_unit": search,
                        "source_url": url if isinstance(url, str) and url.strip() else None,
                        "no_public_price": False,
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


def estimate_costs_with_ai(
    models_to_estimate: list[dict],
    provider_name: str,
    estimator_provider: str,
    estimator_model: str,
    api_key: str,
    logger: Optional[LLMLogger] = None,
    gemini_api_key: Optional[str] = None,
    verify: bool = False,
) -> dict[str, dict]:
    """
    Use AI with web search to discover per-model pricing.

    When ``verify`` is True a second independent pass is run and each priced
    estimate is accepted only if both passes agree within ``VERIFY_TOLERANCE``;
    otherwise the entry is flagged ``verification_mismatch`` (the caller keeps
    the prior price and surfaces it for human review).

    Returns dict mapping model_id to a per-model result (see
    ``_run_estimation_pass``), possibly carrying ``verification_mismatch``.
    """
    if not models_to_estimate:
        return {}

    if logger is None:
        logger = LLMLogger("cost_estimation")

    first = _run_estimation_pass(
        models_to_estimate, provider_name, estimator_provider,
        estimator_model, api_key, logger, gemini_api_key,
    )
    if not verify:
        return first

    print("  ... Verification pass (second independent estimate) ...")
    second = _run_estimation_pass(
        models_to_estimate, provider_name, estimator_provider,
        estimator_model, api_key, logger, gemini_api_key,
    )

    for mid, e in first.items():
        if e.get("no_public_price"):
            continue
        s = second.get(mid)
        if not s or s.get("no_public_price"):
            e["verification_mismatch"] = True
            continue
        if not _within(e["input_per_1m"], s["input_per_1m"], VERIFY_TOLERANCE) or \
                not _within(e["output_per_1m"], s["output_per_1m"], VERIFY_TOLERANCE):
            e["verification_mismatch"] = True

    return first


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
    catalog_path: Optional[Path] = None,
    staleness_days: int = DEFAULT_STALENESS_DAYS,
    verify: bool = False,
    hold_divergent: bool = False,
    refresh_unknown: bool = False,
) -> None:
    """
    Main function to update per-token pricing in the model catalog.

    Re-pricing policy (per model, after classifying fixed vs floating):
        - ``manual`` / ``disabled``           -> never touched.
        - ``source == "unknown"``             -> skipped (stable terminal state)
          unless ``force`` or ``refresh_unknown``.
        - ``floating``                        -> always re-priced.
        - ``fixed``                           -> re-priced if missing or older
          than ``staleness_days``, or when ``force``.

    A new estimate that diverges from the stored price by more than
    ``DIVERGENCE_THRESHOLD`` is applied by default but always reported in the
    audit.  Pass ``hold_divergent`` to quarantine large changes (keep the prior
    price, list them for human sign-off) instead of applying them.  Estimates
    that fail sanity bounds or a verification second pass are always held.

    Args:
        force: ``--all`` -- re-estimate every non-manual model.
        verify: Run a second independent estimation pass and require agreement.
        hold_divergent: Hold large-divergence changes for review instead of
            applying them (default applies and reports them).
        refresh_unknown: Re-attempt models stuck in the ``unknown`` state.
    """
    print("[TOOL] Model Cost Updater")
    print("-" * 40)
    mode = "ALL models (--all)" if force else "floating + new + stale-fixed"
    print(f"Mode: {mode}")
    print(f"Staleness: {staleness_days} days | Verify: {verify} | Hold-divergent: {hold_divergent}")
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
    today_date = datetime.now(timezone.utc).date()
    stats = {"updated": 0, "new": 0, "unchanged": 0, "estimated": 0, "failed": 0}
    # Human-review buckets surfaced in the audit report at the end.
    report = {
        "re_evaluated": [], "divergent_applied": [], "divergent_held": [],
        "sanity_violations": [], "unknown": [], "verification_mismatch": [],
        "class_drift": [],
    }

    for provider_name, provider_data in catalog.items():
        if provider_filter and provider_name != provider_filter:
            continue

        models = provider_data.get("models", [])
        if not models:
            continue

        print(f"{provider_name.upper()} ({len(models)} models)")

        sibling_ids = [m["id"] for m in models]
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

            # Classify fixed vs floating and persist (respect a manual pin).
            auto_class = classify_model(provider_name, model_id, sibling_ids)
            if costing.get("pricing_class_source") == "manual":
                stored = costing.get("pricing_class")
                pricing_class = stored or auto_class
                if stored and stored != auto_class:
                    report["class_drift"].append(
                        f"{model_id}: manual={stored} but classifier={auto_class}")
            else:
                pricing_class = auto_class
                costing["pricing_class"] = auto_class
                costing["pricing_class_source"] = "auto"

            has_pricing = costing.get("input_per_1m") is not None and costing.get("updated") != ""
            source = costing.get("source")
            is_manual = source == "manual"
            is_disabled = model.get("disabled", False)

            if is_disabled:
                reason = model.get("disabled_reason", "unknown")
                print(f"  [DISABLED] {model_id}: DISABLED ({reason})")
                stats["unchanged"] += 1
                continue

            if is_manual:
                inp = costing.get("input_per_1m")
                out = costing.get("output_per_1m")
                print(f"  [SKIP] {model_id}: ${inp}/{out} per 1M (manual, preserved)")
                stats["unchanged"] += 1
                continue

            # Unknown is a STABLE terminal state -- don't retry like 'failed'.
            if source == "unknown" and not (force or refresh_unknown):
                print(f"  [SKIP] {model_id}: unknown (no public price; --refresh-unknown to retry)")
                stats["unchanged"] += 1
                continue

            if pricing_class == "floating":
                should_estimate = True
                reason = "floating"
            else:
                stale = _is_stale(costing.get("updated", ""), today_date, staleness_days)
                should_estimate = force or not has_pricing or stale
                reason = "forced" if force else ("missing" if not has_pricing else ("stale" if stale else ""))

            if should_estimate:
                models_needing_estimation.append({
                    "id": model_id,
                    "provider": provider_name,
                    "name": model.get("name", model_id),
                    "_model_ref": model,
                    "_prior_in": costing.get("input_per_1m"),
                    "_prior_out": costing.get("output_per_1m"),
                    "_prior_search": costing.get("search_cost_per_unit"),
                    "_reason": reason,
                })
            else:
                inp = costing.get("input_per_1m")
                out = costing.get("output_per_1m")
                print(f"  [SKIP] {model_id}: ${inp}/{out} per 1M (fixed, fresh)")
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
                verify=verify,
            )

            for m in models_needing_estimation:
                model_id = m["id"]
                costing = m["_model_ref"]["costing"]
                prior_in, prior_out = m["_prior_in"], m["_prior_out"]
                has_prev = costing.get("updated") != ""

                est = estimates.get(model_id)

                # 1. Estimator omitted / errored -> transient failure.
                if est is None:
                    costing["input_per_1m"] = None
                    costing["output_per_1m"] = None
                    costing["search_cost_per_unit"] = None
                    costing["source_url"] = None
                    costing["source"] = "failed"
                    costing["updated"] = today
                    print(f"  [FAIL] {model_id}: None (FAILED - needs re-estimation)")
                    stats["failed"] += 1
                    continue

                # 2. No public per-token price -> stable unknown (human review).
                if est.get("no_public_price"):
                    costing["input_per_1m"] = None
                    costing["output_per_1m"] = None
                    costing["search_cost_per_unit"] = None
                    costing["source_url"] = None
                    costing["source"] = "unknown"
                    costing["updated"] = today
                    print(f"  [UNKNOWN] {model_id}: no public per-1M price (needs manual search)")
                    report["unknown"].append(model_id)
                    continue

                inp, out = est["input_per_1m"], est["output_per_1m"]
                search = est["search_cost_per_unit"]
                prior_search = m["_prior_search"]
                # Don't wipe a known search price if the estimator didn't return
                # one this pass -- re-evaluation must not silently drop it (that
                # would become silent under-billing of web search).
                if search is None and prior_search is not None:
                    search = prior_search

                # A human-readable "old -> new" string covering all three prices.
                price_str = f"${prior_in}/{prior_out} -> ${inp}/{out}"
                if prior_search != search:
                    price_str += f", search ${prior_search} -> ${search}"

                # 3. Verification second pass disagreed -> hold prior.
                if est.get("verification_mismatch"):
                    print(f"  [HOLD] {model_id}: ${inp}/{out} held (verification mismatch)")
                    report["verification_mismatch"].append(
                        f"{model_id}: ${inp}/{out} (prior ${prior_in}/{prior_out})")
                    stats["unchanged"] += 1
                    continue

                # 4. Sanity bounds -> hold prior.
                violation = _check_bounds(inp, out, search)
                if violation:
                    print(f"  [HOLD] {model_id}: ${inp}/{out} held ({violation})")
                    report["sanity_violations"].append(f"{model_id}: {violation}")
                    stats["unchanged"] += 1
                    continue

                # 5. Large divergence in input, output, OR search -> applied by
                #    default, held only with --hold-divergent.
                pc_in = _pct_change(prior_in, inp)
                pc_out = _pct_change(prior_out, out)
                pc_search = _pct_change(prior_search, search)
                worst = max(pc_in or 0, pc_out or 0, pc_search or 0)
                divergent = worst > DIVERGENCE_THRESHOLD
                if divergent and hold_divergent:
                    print(f"  [HOLD] {model_id}: {price_str} "
                          f"DIVERGENT {worst:.0%} (held; remove --hold-divergent to apply)")
                    report["divergent_held"].append(
                        f"{model_id}: {price_str} ({worst:.0%})"
                        + (f"  {est.get('source_url')}" if est.get("source_url") else ""))
                    stats["unchanged"] += 1
                    continue

                # 6. Accept.
                costing["input_per_1m"] = round(inp, 4)
                costing["output_per_1m"] = round(out, 4)
                costing["search_cost_per_unit"] = search
                costing["source_url"] = est.get("source_url")
                costing["source"] = "published" if est.get("source_url") else "estimated"
                costing["updated"] = today
                search_str = f", search=${search}" if search else ""
                div_str = f" [DIVERGENT {worst:.0%} applied]" if divergent else ""
                print(f"  [OK] {model_id}: ${inp}/{out} per 1M{search_str} ({costing['source']}){div_str}")
                if divergent:
                    report["divergent_applied"].append(
                        f"{model_id}: {price_str} ({worst:.0%}, {costing['source']})"
                        + (f"  {est.get('source_url')}" if est.get("source_url") else ""))
                else:
                    changed = not (prior_in == inp and prior_out == out and prior_search == search)
                    note = "changed" if changed else "unchanged"
                    report["re_evaluated"].append(
                        f"{model_id}: {price_str} ({note}, {costing['source']})")
                stats["estimated"] += 1
                stats["updated" if has_prev else "new"] += 1

        elif models_needing_estimation:
            print(f"  [FAIL] No AI estimator available - marking as FAILED...")
            for m in models_needing_estimation:
                costing = m["_model_ref"]["costing"]
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

    _print_audit_report(report)


def _print_audit_report(report: dict) -> None:
    """Print the human-review section: what changed and what needs attention."""
    sections = [
        ("RE-EVALUATED (re-checked against source; see changed/unchanged)", "re_evaluated"),
        ("DIVERGENT (APPLIED -- large change, please verify)", "divergent_applied"),
        ("DIVERGENT (HELD for review -- remove --hold-divergent to apply)", "divergent_held"),
        ("SANITY VIOLATIONS (held)", "sanity_violations"),
        ("VERIFICATION MISMATCH (held)", "verification_mismatch"),
        ("UNKNOWN (no public price -- needs manual search)", "unknown"),
        ("CLASS DRIFT (manual pricing_class disagrees with classifier)", "class_drift"),
    ]
    if not any(report.get(key) for _, key in sections):
        return
    print()
    print("=" * 40)
    print("PRICING AUDIT -- HUMAN REVIEW")
    print("=" * 40)
    for title, key in sections:
        items = report.get(key, [])
        if not items:
            continue
        print(f"\n{title} ({len(items)}):")
        for item in items:
            print(f"  - {item}")


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
    parser.add_argument(
        "--staleness-days",
        type=int,
        default=DEFAULT_STALENESS_DAYS,
        help=f"Re-price fixed models older than this many days (default: {DEFAULT_STALENESS_DAYS})"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=False,
        help="Run a second independent estimation pass and require agreement"
    )
    parser.add_argument(
        "--hold-divergent",
        action="store_true",
        default=False,
        help="Hold large-divergence price changes for review instead of applying them "
             "(default: apply and report them)"
    )
    parser.add_argument(
        "--refresh-unknown",
        action="store_true",
        default=False,
        help="Re-attempt models stuck in the 'unknown' (no public price) state"
    )

    args = parser.parse_args()

    update_model_costs(
        force=args.force,
        dry_run=args.dry_run,
        estimator_model=args.estimator,
        provider_filter=args.provider,
        config_path=Path(args.config) if args.config else None,
        catalog_path=Path(args.catalog) if args.catalog else None,
        staleness_days=args.staleness_days,
        verify=args.verify,
        hold_divergent=args.hold_divergent,
        refresh_unknown=args.refresh_unknown,
    )


if __name__ == "__main__":
    main()
