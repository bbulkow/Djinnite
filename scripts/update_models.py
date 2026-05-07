"""
Update Models Script

Fetches the latest model lists from AI providers and updates
config/model_catalog.json.

This script preserves existing pricing and modality values
when updating models. To update pricing, use update_model_costs.py.

Usage:
    python -m djinnite.scripts.update_models
"""

import json
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List

try:
    from djinnite.config_loader import load_ai_config, CONFIG_DIR, Modalities, _serialize_vision_limit, _resolve_config_file
    from djinnite.ai_providers import get_provider, BaseAIProvider
    from djinnite.ai_providers.gemini_provider import GeminiProvider
    from djinnite.ai_providers.claude_provider import ClaudeProvider
    from djinnite.ai_providers.openai_provider import OpenAIProvider
except ImportError:
    # Fallback for direct execution
    import sys
    _project_root = str(Path(__file__).resolve().parent.parent)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    
    from config_loader import load_ai_config, CONFIG_DIR, Modalities, _serialize_vision_limit, _resolve_config_file
    from ai_providers import get_provider, BaseAIProvider
    from ai_providers.gemini_provider import GeminiProvider
    from ai_providers.claude_provider import ClaudeProvider
    from ai_providers.openai_provider import OpenAIProvider

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


def _load_known_defaults() -> dict:
    """Load known_model_defaults.json (estimator config, vision defaults, etc.)."""
    defaults_path = _resolve_config_file("known_model_defaults.json")
    if defaults_path.exists():
        with open(defaults_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

_known_defaults = _load_known_defaults()
_estimator_config = _known_defaults.get("estimator", {})
_vision_defaults = _known_defaults.get("vision_defaults", {})


def _resolve_estimator(ai_config) -> tuple:
    """
    Resolve the estimator provider/model for AI estimation tasks.
    
    Priority: 1) known_model_defaults.json  2) ai_config default
    
    Returns:
        (provider_name, model_id, api_key) or (None, None, None)
    """
    if _estimator_config.get("model"):
        est_provider = _estimator_config.get("provider", "gemini")
        est_model = _estimator_config["model"]
    else:
        est_provider = ai_config.default_provider
        p_config = ai_config.get_provider(est_provider)
        est_model = p_config.default_model if p_config else None

    p_config = ai_config.get_provider(est_provider)
    est_api_key = p_config.api_key if p_config else None
    return est_provider, est_model, est_api_key


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
    est_provider, est_model, est_api_key = _resolve_estimator(ai_config)
    if not est_api_key:
        return {}

    provider_company = {"gemini": "Google", "claude": "Anthropic", "chatgpt": "OpenAI"}.get(provider_name, provider_name)
    model_list = "\n".join(m["id"] for m in models)
    prompt = MODALITY_ESTIMATION_PROMPT.format(provider_company=provider_company, model_list=model_list)
    
    try:
        # Use the Djinnite-internal estimator model.
        print(f"    Querying {est_provider}/{est_model} for {len(models)} models...")
        instance = get_provider(est_provider, est_api_key, est_model)
        resp = instance.generate(
            prompt=prompt,
            system_prompt="You must respond with valid JSON only. No additional text or explanation.",
            temperature=0.3,
        )
        print(f"    Response received, parsing...")
        content = resp.content.strip()
        # Handle markdown blocks if present
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```json"):
                content = "\n".join(lines[1:-1])
            else:
                content = "\n".join(lines[1:-1])

        result = json.loads(content)
        print(f"    Done. Got modalities for {len(result)} models")
        return result
    except Exception as e:
        print(f"  [WARN] Modality estimation failed: {e}")
        return {}

def estimate_output_limits_with_ai(
    models: list[dict],
    provider_name: str,
    ai_config
) -> dict[str, int]:
    """Use AI with web search to estimate max output token limits for unknown models."""
    est_provider, est_model, est_api_key = _resolve_estimator(ai_config)
    if not est_api_key:
        return {}

    provider_company = {"gemini": "Google", "claude": "Anthropic", "chatgpt": "OpenAI"}.get(provider_name, provider_name)
    model_list = "\n".join(m["id"] for m in models)
    prompt = OUTPUT_LIMIT_ESTIMATION_PROMPT.format(provider_company=provider_company, model_list=model_list)
    
    try:
        # Use the Djinnite-internal estimator model with web search.
        print(f"    Querying {est_provider}/{est_model} (web search) for {len(models)} models...")
        instance = get_provider(est_provider, est_api_key, est_model)
        resp = instance.generate(
            prompt=prompt,
            system_prompt="You must respond with valid JSON only. No additional text or explanation.",
            temperature=0.3,
            web_search=True,  # Critical: ground with current docs
        )
        print(f"    Response received, parsing...")
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
        print(f"    Done. Got output limits for {len(cleaned)} models")
        return cleaned
    except Exception as e:
        print(f"  [WARN] Output limit estimation failed: {e}")
        return {}



def _resolve_max_output_tokens(model_id: str, api_value: int, existing_value: int) -> int:
    """
    Resolve the max_output_tokens for a model using dynamic sources only:
    1. Provider API value (e.g. Gemini exposes output_token_limit)
    2. Existing catalog value (persisted from prior AI estimation runs)
    3. 0 (unknown -- will trigger AI estimation)
    """
    if api_value and api_value > 0:
        return api_value
    if existing_value and existing_value > 0:
        return existing_value
    return 0


def _resolve_structured_json_support(
    model_id: str,
    existing_value,
):
    """
    Resolve structured_json using dynamic sources only:
    1. Existing catalog value (persisted from prior probe runs) — already a list.
    2. None (unknown — will trigger live probe).
    """
    if existing_value is not None:
        return existing_value
    return None


def _onoff_states(probe_result: Optional[bool]) -> Optional[list[str]]:
    """Translate a True/False/None probe result into the on/off list shape.

    * ``True``  → ``["on", "off"]`` — capability supported, off-state always
                  works (omit-the-param semantics).
    * ``False`` → ``["off"]``       — capability cleanly unsupported.
    * ``None``  → ``None``          — inconclusive / unknown.
    """
    if probe_result is True:
        return ["on", "off"]
    if probe_result is False:
        return ["off"]
    return None


def _temperature_states(probe_result: Optional[bool]) -> Optional[list[str]]:
    """Translate a temperature probe result into the temperature list shape."""
    if probe_result is True:
        return ["any", "default"]
    if probe_result is False:
        return ["default"]
    return None


def _probe_all_capabilities_for_models(
    models_to_probe: list[dict],
    provider_cls,
    provider_name: str,
    api_key: str,
) -> dict[str, dict]:
    """
    Probe a list of models to discover ALL capabilities at once.

    Each capability is emitted as ``list[str] | None`` matching the
    ``ModelCapabilities`` schema:
    * ``thinking`` / ``structured_json`` / ``web_search`` / ``json_with_search``
      → subset of ``("on", "off")`` (or None for unknown).
    * ``temperature`` → subset of ``("any", "default")``.
    * ``thinking_style`` → subset of ``("adaptive", "budget", "effort")``.

    Returns a dict of model_id -> capability dict.
    """
    results: dict[str, dict] = {}
    for m in models_to_probe:
        model_id = m["id"]
        try:
            instance = provider_cls(api_key=api_key, model=model_id)

            # ---- Per-capability raw probes -----------------------------
            ssj_raw = instance.probe_structured_json()
            temp_raw = instance.probe_temperature()
            jws_raw = instance.probe_json_with_search()

            if hasattr(instance, "probe_web_search"):
                ws_raw = instance.probe_web_search()
            else:
                # Conservative default: assume supported. Unknown providers
                # without a probe shouldn't block callers from web_search.
                ws_raw = True

            # Thinking is two-dimensional: which styles work + can disable?
            styles_raw = instance.probe_thinking_style()    # list[str] | None
            disable_raw = instance.probe_thinking_disable()  # bool | None

            # ---- Assemble list-shaped fields ---------------------------
            ssj_states = _onoff_states(ssj_raw)
            temp_states = _temperature_states(temp_raw)
            ws_states = _onoff_states(ws_raw)
            jws_states = _onoff_states(jws_raw)

            # thinking field: combine on-state (any style succeeded) and
            # off-state (disable accepted). Conservative default for a
            # missing/unknown disable probe is to include "off" — matches
            # current de-facto behavior of every model in the catalog.
            if styles_raw is None:
                # Inconclusive on the on-state — we cannot tell if thinking
                # works. Mark as unknown to leave runtime pre-flight off.
                thinking_states: Optional[list[str]] = None
                thinking_style_states: Optional[list[str]] = None
            else:
                on_supported = bool(styles_raw)
                states: list[str] = []
                if on_supported:
                    states.append("on")
                # If the model has no thinking, "off" is the only state.
                # If it has thinking, "off" depends on the disable probe.
                if not on_supported:
                    states.append("off")
                else:
                    if disable_raw is True or disable_raw is None:
                        states.append("off")
                    # disable_raw is False → always-on; no "off" token.
                thinking_states = states
                thinking_style_states = list(styles_raw) if styles_raw else None

            results[model_id] = {
                "structured_json": ssj_states,
                "temperature": temp_states,
                "thinking": thinking_states,
                "web_search": ws_states,
                "json_with_search": jws_states,
                "thinking_style": thinking_style_states,
            }

            def _mark(v):
                if v is None:
                    return "[?]"
                return "[OK]" if "on" in v else "[FAIL]"

            parts = [
                f"json={_mark(ssj_states)}",
                f"temp={'[OK]' if temp_states and 'any' in temp_states else '[FAIL]' if temp_states else '[?]'}",
            ]
            think_label = _mark(thinking_states)
            if thinking_states and "off" not in thinking_states:
                think_label += "(always-on)"
            if thinking_style_states:
                think_label += f"({'|'.join(thinking_style_states)})"
            parts.append(f"think={think_label}")
            parts.append(f"web={_mark(ws_states)}")
            parts.append(f"json+search={_mark(jws_states)}")
            print(f"    {model_id}: {' '.join(parts)}")

        except Exception as e:
            results[model_id] = {
                "structured_json": None, "temperature": None,
                "thinking": None, "web_search": ["on", "off"],
                "json_with_search": None, "thinking_style": None,
            }
            print(f"    [WARN] {model_id}: probe skipped ({e})")
    return results


def merge_model_data(
    new_models: list[dict],
    existing_models: list[dict],
    provider_instance: BaseAIProvider,
    provider_cls,
    api_key: str,
    ai_config,
    reprobe: Optional[set] = None,
) -> list[dict]:
    """Merge new model data with existing, preserving costing, modality, and output limit data.

    ``reprobe``: if set, any model whose ID is in this set has its cached
    capabilities wiped to ``None`` before the probe gate runs, forcing
    rediscovery. Accepted forms:
        * ``"all"``                  – reprobe every model across every provider
        * ``"<provider>:all"``       – reprobe every model under one provider
                                       (e.g. ``"claude:all"``, ``"gemini:all"``)
        * ``"<model_id>"``           – reprobe a single model

    Use this to recover from poisoned capability values (e.g. Opus 4.7 marked
    ``thinking=false`` because older probes sent ``temperature``).
    """
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
            # Preserve existing capabilities.structured_json — pass the raw
            # value (list / bool / None) through; the loader's coerce shim
            # accepts all three, and the resolver below stores it as-is.
            raw_caps = existing.get("capabilities")
            if isinstance(raw_caps, dict):
                existing_ssj = raw_caps.get("structured_json")
            else:
                existing_ssj = existing.get("supports_structured_json")
            
            # Preserve existing structure if it's already structured
            if "modalities" in existing and isinstance(existing["modalities"], dict):
                model["modalities"] = existing["modalities"]
            else:
                # Migrate or overwrite
                model["modalities"] = discovered
                
            # Preserve disabled status
            if existing.get("disabled"):
                model["disabled"] = True
                model["disabled_reason"] = existing.get("disabled_reason", "")

            # Preserve costing
            if "costing" in existing:
                model["costing"] = existing["costing"]
            else:
                model["costing"] = {
                    "input_per_1m": None,
                    "output_per_1m": None,
                    "source": "",
                    "updated": "",
                }
        else:
            # New model
            model["modalities"] = discovered
            model["costing"] = {
                "input_per_1m": None,
                "output_per_1m": None,
                "source": "",
                "updated": "",
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

        # Force-reprobe: wipe cached capabilities so the needs_probe gate
        # below re-queues this model and the fresh probe results win.
        provider_scoped = f"{provider_instance.PROVIDER_NAME}:all"
        force_reprobe = reprobe and (
            "all" in reprobe
            or provider_scoped in reprobe
            or model_id in reprobe
        )
        if force_reprobe:
            existing_caps = {}
            ssj = None
            print(f"  [REPROBE] Resetting capabilities for {model_id}")

        model["capabilities"] = {
            "structured_json": ssj,
            "temperature": existing_caps.get("temperature"),
            "thinking": existing_caps.get("thinking"),
            "web_search": existing_caps.get("web_search"),
            "json_with_search": existing_caps.get("json_with_search"),
            "thinking_style": existing_caps.get("thinking_style"),
        }
        
        # Resolve vision_limits for vision-capable models
        input_modalities = model.get("modalities", {})
        if isinstance(input_modalities, dict):
            is_vision = "vision" in input_modalities.get("input", [])
        else:
            is_vision = False

        if is_vision:
            # Preserve existing vision_limits from catalog
            existing_vl = None
            if model_id in existing_by_id:
                existing_vl = existing_by_id[model_id].get("vision_limits")

            if existing_vl and isinstance(existing_vl, dict):
                model["vision_limits"] = existing_vl
            else:
                # Apply provider defaults
                provider_vl = _vision_defaults.get(provider_instance.PROVIDER_NAME, {})
                if provider_vl:
                    model["vision_limits"] = dict(provider_vl)  # Copy, not reference
                else:
                    model["vision_limits"] = None

        else:
            model["vision_limits"] = None

        # Queue for probing if any capability is still unknown
        needs_probe = (ssj is None or model["capabilities"]["temperature"] is None
                       or model["capabilities"]["thinking"] is None
                       or model["capabilities"]["json_with_search"] is None)
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
        print(f"  [AI] Requesting AI estimation for {len(uncertain_models)} uncertain models...")
        estimates = estimate_modalities_with_ai(uncertain_models, provider_instance.PROVIDER_NAME, ai_config)
        for model in uncertain_models:
            if model["id"] in estimates:
                model["modalities"] = estimates[model["id"]]
    
    # 3. AI Estimation Fallback for unknown output limits
    if unknown_output_limit_models and ai_config.get_provider(ai_config.default_provider):
        print(f"  [AI] Estimating output limits for {len(unknown_output_limit_models)} models with AI...")
        limit_estimates = estimate_output_limits_with_ai(
            unknown_output_limit_models, provider_instance.PROVIDER_NAME, ai_config
        )
        for model in unknown_output_limit_models:
            if model["id"] in limit_estimates:
                model["max_output_tokens"] = limit_estimates[model["id"]]
    
    # 4. Live probe ALL capabilities on models that need it
    if models_needing_ssj_probe:
        print(f"  [CHECK] Probing {len(models_needing_ssj_probe)} models for all capabilities...")
        probe_results = _probe_all_capabilities_for_models(
            models_needing_ssj_probe, provider_cls,
            provider_instance.PROVIDER_NAME, api_key,
        )
        for model in models_needing_ssj_probe:
            if model["id"] in probe_results:
                probed = probe_results[model["id"]]
                # Merge probe results into capabilities (probe wins over None)
                caps = model["capabilities"]
                for key in ["structured_json", "temperature", "thinking", "web_search", "json_with_search", "thinking_style"]:
                    if probed.get(key) is not None:
                        caps[key] = probed[key]
    
    return merged

def update_models():
    parser = argparse.ArgumentParser(description="Update AI model catalog")
    parser.add_argument("--config", type=str, help="Path to ai_config.json")
    parser.add_argument("--catalog", type=str, help="Path to model_catalog.json")
    parser.add_argument(
        "--reprobe",
        action="append",
        default=None,
        metavar="TARGET",
        help="Force re-probing of model capabilities (clears cached values). "
             "Accepts: a model ID (e.g. 'claude-opus-4-7'), a provider-scoped "
             "wildcard (e.g. 'claude:all', 'gemini:all', 'chatgpt:all'), or "
             "'all' for every model across every provider. Repeatable: "
             "--reprobe claude:all --reprobe gemini-2.5-pro.",
    )
    args = parser.parse_args()

    reprobe_set = set(args.reprobe) if args.reprobe else None

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
            print(f"  [WARN] Provider {name} not configured, skipping.")
            continue

        try:
            instance = provider_cls(api_key=p_config.api_key, model=p_config.default_model)
            new_list = instance.list_models()
            if new_list:
                existing_list = catalog.get(name, {}).get("models", [])
                merged = merge_model_data(
                    new_list, existing_list, instance,
                    provider_cls, p_config.api_key, ai_config,
                    reprobe=reprobe_set,
                )
                
                catalog[name] = {
                    "models": merged,
                    "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d")
                }
                print(f"[OK] Processed {len(merged)} models")
        except Exception as e:
            print(f"[FAIL] Failed: {e}")

    with open(catalog_path, 'w', encoding='utf-8') as f:
        json.dump(catalog, f, indent=2)
    print(f"\n[SAVE] Saved to {catalog_path}")

    # Auto-estimate costs for new/unknown models
    try:
        try:
            from djinnite.scripts.update_model_costs import update_model_costs
        except ImportError:
            from scripts.update_model_costs import update_model_costs

        print("\n[TOOL] Estimating costs for new models...")
        update_model_costs(
            force=False,
            dry_run=False,
            catalog_path=catalog_path,
            config_path=config_path,
        )
    except Exception as e:
        print(f"[WARN] Cost estimation failed: {e}")
        print("  Run 'python -m djinnite.scripts.update_model_costs' manually to retry.")

if __name__ == "__main__":
    update_models()
