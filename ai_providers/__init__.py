"""
AI Providers Module

Lightweight abstraction layer for multiple AI providers.
Each provider wraps the native SDK directly without heavy frameworks.
"""

import json
from pathlib import Path
from typing import Optional

from .base_provider import BaseAIProvider, AIResponse, AIProviderError, DjinniteModalityError
from .gemini_provider import GeminiProvider
from .claude_provider import ClaudeProvider
from .openai_provider import OpenAIProvider

# Try absolute or relative import for CONFIG_DIR
try:
    from djinnite.config_loader import CONFIG_DIR
except ImportError:
    try:
        from ..config_loader import CONFIG_DIR
    except ImportError:
        CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


# Registry of available providers
PROVIDERS = {
    "gemini": GeminiProvider,
    "claude": ClaudeProvider,
    "chatgpt": OpenAIProvider,
}

# Path to model catalog (in host project's config directory)
MODEL_CATALOG_PATH = CONFIG_DIR / "model_catalog.json"


def _is_model_disabled(provider_name: str, model_id: str) -> tuple[bool, str]:
    """
    Check if a model is disabled in the catalog.
    
    Returns:
        Tuple of (is_disabled, reason)
    """
    if not model_id or not MODEL_CATALOG_PATH.exists():
        return (False, "")
    
    try:
        with open(MODEL_CATALOG_PATH, 'r', encoding='utf-8') as f:
            catalog = json.load(f)
        
        provider_data = catalog.get(provider_name, {})
        models = provider_data.get("models", [])
        
        for model in models:
            if model.get("id") == model_id:
                if model.get("disabled", False):
                    reason = model.get("disabled_reason", "model is disabled")
                    return (True, reason)
                return (False, "")
        
        # Model not found in catalog - allow it (might be new)
        return (False, "")
    except Exception:
        # If catalog can't be read, allow the request
        return (False, "")


def get_provider(
    provider_name: str,
    api_key: str,
    model: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    **kwargs
) -> BaseAIProvider:
    """
    Factory function to create an AI provider instance.
    
    Args:
        provider_name: Name of the provider (gemini, claude, chatgpt)
        api_key: API key for the provider
        model: Optional model ID to use
        gemini_api_key: Optional Gemini API key for web search (used by OpenAI)
        **kwargs: Additional provider-specific arguments (e.g., backend, project_id for Gemini)
        
    Returns:
        Configured provider instance
        
    Raises:
        ValueError: If provider name is not recognized
        AIProviderError: If the requested model is disabled
    """
    if provider_name not in PROVIDERS:
        available = ", ".join(PROVIDERS.keys())
        raise ValueError(
            f"Unknown provider: {provider_name}. Available: {available}"
        )
    
    # Check if model is disabled
    if model:
        is_disabled, reason = _is_model_disabled(provider_name, model)
        if is_disabled:
            raise AIProviderError(
                f"Model '{model}' is disabled: {reason}",
                provider=provider_name
            )
    
    provider_class = PROVIDERS[provider_name]
    
    # OpenAI needs Gemini API key for web search capability
    if provider_name == "chatgpt" and gemini_api_key:
        return provider_class(api_key=api_key, model=model, gemini_api_key=gemini_api_key, **kwargs)
    
    return provider_class(api_key=api_key, model=model, **kwargs)


def list_available_providers() -> list[str]:
    """Return list of available provider names."""
    return list(PROVIDERS.keys())


__all__ = [
    "BaseAIProvider",
    "AIResponse",
    "AIProviderError",
    "DjinniteModalityError",
    "GeminiProvider",
    "ClaudeProvider",
    "OpenAIProvider",
    "get_provider",
    "list_available_providers",
]
