"""
Prompts Module

Centralized storage for all LLM prompts used by Djinnite.
Prompts are treated as configuration, not code.

This includes not just the prompt text, but also the parameters needed
to execute the prompt (temperature, max_tokens, etc).

Usage:
    from djinnite.prompts import COST_ESTIMATION_CONFIG
    prompt = COST_ESTIMATION_CONFIG["prompt"].format(...)
    max_tokens = COST_ESTIMATION_CONFIG["max_tokens"]
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

For each model, find its current pricing in USD:
- "input_per_1m": dollars per 1 million input tokens
- "output_per_1m": dollars per 1 million output tokens
- "search_cost_per_unit": dollars per web search invocation (null if model does not support web search)

Return JSON mapping model_id to an object with these three fields.
Example: {{"model-a": {{"input_per_1m": 3.00, "output_per_1m": 15.00, "search_cost_per_unit": 0.01}}, "model-b": {{"input_per_1m": 0.15, "output_per_1m": 0.60, "search_cost_per_unit": null}}}}

If a model has separate "cached input" pricing, use the standard (non-cached) rate.
If a model uses a tiered pricing scheme (e.g. different rate after N tokens), use the base tier rate.""",

    # System prompt for JSON-only responses
    "system_prompt": """You are a precise AI model pricing analyst with access to current information.
ALWAYS search for the LATEST pricing data - never rely on training knowledge for pricing.
Use your search tools to look up current prices from official sources like openai.com/pricing, anthropic.com/pricing, ai.google.dev/pricing.
Always respond with valid JSON only, no additional text or explanation.""",

    # Generation parameters
    "temperature": 0.3,  # Low for consistency

    # Max tokens for output — generous to avoid truncation
    "max_tokens": 65536,

    # Enable web search for current pricing info (avoids knowledge cutoff issues)
    "web_search": True,
}

# Convenience aliases
COST_ESTIMATION_PROMPT = COST_ESTIMATION_CONFIG["prompt"]
COST_ANALYST_SYSTEM = COST_ESTIMATION_CONFIG["system_prompt"]
