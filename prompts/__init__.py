"""
Prompts Module

Centralized storage for all LLM prompts used by Djinnite.
Prompts are treated as configuration, not code.

This includes not just the prompt text, but also the parameters needed
to execute the prompt (temperature, max_output_tokens, etc).

Usage:
    from djinnite.prompts import COST_ESTIMATION_CONFIG
    prompt = COST_ESTIMATION_CONFIG["prompt"].format(...)
    max_output_tokens = COST_ESTIMATION_CONFIG["max_output_tokens"]
"""

# ============================================================================
# COST ESTIMATION PROMPT CONFIG
# ============================================================================

# Full configuration for cost estimation task
COST_ESTIMATION_CONFIG = {
    # The prompt template — asks for actual dollar prices per 1M tokens
    "prompt": """Look up CURRENT pricing for {provider_company} models from their official pricing page.

IMPORTANT: Use your search capability to find the LATEST published pricing. Do NOT rely on training data.

{provider_name} models to find pricing for:
{model_list}

For each model, find its current pricing in USD and report:
- "input_per_1m": dollars per 1 million input tokens
- "output_per_1m": dollars per 1 million output tokens
- "search_cost_per_unit": dollars per web search invocation (null if the model does not support web search)
- "source_url": the official pricing page URL you took the figures from
- "published_figure": the exact price text as published (e.g. "$1.50 / $9.00 per 1M tokens"), for human cross-check
- "no_public_price": true if NO public per-1M-token price exists for this model, otherwise false

CRITICAL RULES:
- If the model is billed per-image, per-minute of audio, per-character, or by any unit OTHER than per-token,
  OR you cannot find an official published per-1M-token price, set "no_public_price": true and leave
  "input_per_1m"/"output_per_1m" as null. Do NOT guess or fabricate a per-token equivalent.
- Always include "source_url" pointing to the official page. If you cannot find an official page with a
  per-token price, set "no_public_price": true.
- If a model has separate "cached input" pricing, use the standard (non-cached) rate.
- If a model uses a tiered scheme (different rate after N tokens), use the base tier rate.

Return JSON mapping model_id to an object with these fields.
Example: {{"model-a": {{"input_per_1m": 3.00, "output_per_1m": 15.00, "search_cost_per_unit": 0.01, "source_url": "https://example.com/pricing", "published_figure": "$3 / $15 per 1M", "no_public_price": false}}, "model-b": {{"input_per_1m": null, "output_per_1m": null, "search_cost_per_unit": null, "source_url": "https://example.com/pricing", "published_figure": "priced per image", "no_public_price": true}}}}""",

    # System prompt for JSON-only responses
    "system_prompt": """You are a precise AI model pricing analyst with access to current information.
ALWAYS search for the LATEST pricing data - never rely on training knowledge for pricing.
Use your search tools to look up current prices from official sources like openai.com/pricing, anthropic.com/pricing, ai.google.dev/pricing.
NEVER fabricate a per-token price for a model that is not billed per token (image, audio-minute, character) or that you cannot find an official published price for -- report "no_public_price": true instead.
Always respond with valid JSON only, no additional text or explanation.""",

    # Generation parameters
    "temperature": 0.3,  # Low for consistency

    # Cap on output tokens — generous to avoid truncation
    "max_output_tokens": 65536,

    # Enable web search for current pricing info (avoids knowledge cutoff issues)
    "web_search": True,
}

# Convenience aliases
COST_ESTIMATION_PROMPT = COST_ESTIMATION_CONFIG["prompt"]
COST_ANALYST_SYSTEM = COST_ESTIMATION_CONFIG["system_prompt"]
