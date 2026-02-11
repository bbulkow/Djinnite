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
    # The prompt template
    "prompt": """Look up CURRENT pricing for {provider_company} models from their official pricing page.

IMPORTANT: Use your search capability to find the LATEST published pricing. Do NOT rely on training data.

ANCHOR: {anchor_model} = {anchor_score} (≈$0.075/1M input, $0.30/1M output)

SCALE: Same cost as anchor = 1.0, 10x more expensive = 10.0, half = 0.5

{provider_name} models to estimate:
{model_list}

For each model, search for its current pricing and calculate the cost_score relative to the anchor.
Return JSON mapping model_id to cost_score. Example: {{"model-a": 1.5, "model-b": 8.0}}""",
    
    # System prompt for JSON-only responses
    "system_prompt": """You are a precise AI model pricing analyst with access to current information.
ALWAYS search for the LATEST pricing data - never rely on training knowledge for pricing.
Use your search tools to look up current prices from official sources like openai.com/pricing.
Always respond with valid JSON only, no additional text or explanation.""",
    
    # Generation parameters
    "temperature": 0.3,  # Low for consistency
    
    # Max tokens for output - set very high to avoid truncation
    # 86 models * ~35 chars each = ~3000 chars = ~1000 tokens
    # Setting to 65536 to ensure no artificial limit
    "max_tokens": 65536,
    
    # Enable web search for current pricing info (avoids knowledge cutoff issues)
    "web_search": True,
}

# Convenience aliases for backward compatibility
COST_ESTIMATION_PROMPT = COST_ESTIMATION_CONFIG["prompt"]
COST_ANALYST_SYSTEM = COST_ESTIMATION_CONFIG["system_prompt"]
