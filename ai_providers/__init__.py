"""
AI Providers Module

Lightweight abstraction layer for multiple AI providers.
Each provider wraps the native SDK directly without heavy frameworks.
"""

import json
from pathlib import Path
from typing import Optional

from .base_provider import (
    BaseAIProvider,
    AIResponse,
    AIProviderError,
    AIOutputTruncatedError,
    AIEmptyResponseError,
    AIContextLengthError,
    AIRateLimitError,
    AIAuthenticationError,
    AIModelNotFoundError,
    AIPricingError,
    DjinniteModalityError,
)
from .gemini_provider import GeminiProvider
from .claude_provider import ClaudeProvider
from .openai_provider import OpenAIProvider
from .grok_provider import GrokProvider
from .platforms import ACCESS_MODES, PLATFORMS, resolve_platform

# Import config resolution and catalog loader.
# _resolve_config_file checks the host project's config/ first, then falls
# back to the package's own config/ -- so host projects only need ai_config.json
# while model_catalog.json is inherited from the distribution.
try:
    from djinnite.config_loader import _resolve_config_file, load_model_catalog
except ImportError:
    try:
        from ..config_loader import _resolve_config_file, load_model_catalog
    except (ImportError, ValueError):
        # Direct execution: ai_providers was imported as a top-level package
        # (e.g. via update_models.py adding the project root to sys.path).
        # Fall back to a sibling import.
        from config_loader import _resolve_config_file, load_model_catalog


# Registry of available providers
PROVIDERS = {
    "gemini": GeminiProvider,
    "claude": ClaudeProvider,
    "chatgpt": OpenAIProvider,
    "grok": GrokProvider,
}


def _get_model_catalog_path() -> Path:
    """Resolve model_catalog.json with project-local → package fallback."""
    return _resolve_config_file("model_catalog.json")


def _is_model_disabled(provider_name: str, model_id: str) -> tuple[bool, str]:
    """
    Check if a model is disabled in the catalog.

    Returns:
        Tuple of (is_disabled, reason)
    """
    catalog_path = _get_model_catalog_path()
    if not model_id or not catalog_path.exists():
        return (False, "")

    try:
        with open(catalog_path, 'r', encoding='utf-8') as f:
            catalog = json.load(f)

        provider_data = catalog.get(provider_name, {})
        models = provider_data.get("models", [])

        for model in models:
            if model.get("id") == model_id:
                if model.get("disabled", False):
                    reason = model.get("disabled_reason", "model is disabled")
                    return (True, reason)
                return (False, "")

        return (False, "")
    except Exception:
        return (False, "")


def get_provider(
    provider_name: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    **kwargs
) -> BaseAIProvider:
    """
    Factory function to create an AI provider instance.

    Requires the model catalog to exist and the requested model to be
    present in it.  This ensures pre-flight capability checks (vision
    limits, structured JSON support, etc.) are always active.

    Access modes:

    * **direct** (default): the provider's own API; pass ``api_key``.
    * **platform**: a cloud platform serving the provider's models; pass
      ``platform="vertexai"`` (or the legacy ``backend="vertexai"``) with
      ``project_id``, and optionally ``location`` / ``quota_project``. No
      ``api_key`` is needed -- the platform's own credentials (Google
      Application Default Credentials) are used. If the catalog records
      capabilities probed on that platform, they are used for pre-flight.

    Args:
        provider_name: Name of the provider (gemini, claude, chatgpt, grok)
        api_key: API key for the provider. Required in direct mode (a
            missing key fails as before, at the provider); not needed in
            platform mode.
        model: Optional model ID to use
        gemini_api_key: Optional Gemini API key for web search (used by OpenAI)
        **kwargs: Additional provider-specific arguments: ``platform``,
            ``backend``, ``project_id``, ``location``, ``quota_project``,
            ``require_pricing``.

    Returns:
        Configured provider instance

    Raises:
        ValueError: If provider name is not recognized
        AIProviderError: If the catalog is missing/unreadable, model is
            not found, the model is disabled, or the platform is unknown
            or does not host this provider
    """
    if provider_name not in PROVIDERS:
        available = ", ".join(PROVIDERS.keys())
        raise ValueError(
            f"Unknown provider: {provider_name}. Available: {available}"
        )

    # Platform mode: validate before building anything, so an unhosted
    # provider fails with a clear message instead of a constructor TypeError.
    platform = resolve_platform(
        kwargs.get("platform"), kwargs.get("backend"), provider=provider_name
    )

    # Check if model is disabled
    if model:
        is_disabled, reason = _is_model_disabled(provider_name, model)
        if is_disabled:
            raise AIProviderError(
                f"Model '{model}' is disabled: {reason}",
                provider=provider_name
            )

    # Load model_info from catalog -- hard failure if catalog is missing
    # or unreadable, or if the model is not found.
    model_info = None
    catalog_path = _get_model_catalog_path()

    if model:
        if not catalog_path.exists():
            raise AIProviderError(
                f"Model catalog not found at {catalog_path}. "
                f"Run the model update script to generate it.",
                provider=provider_name
            )

        catalog = load_model_catalog(catalog_path)
        model_info = catalog.get_model(provider_name, model)
        if model_info is None:
            raise AIModelNotFoundError(
                f"Model '{model}' not found in catalog for provider '{provider_name}'. "
                f"Run the model update script or check your ai_config.json.",
                provider=provider_name
            )
        # Capabilities probed on the platform (scripts/probe_platform.py)
        # override the direct-mode ones field by field. Recorded location
        # availability is informational and never blocks the call.
        model_info = model_info.for_platform(platform)

    provider_class = PROVIDERS[provider_name]
    
    # OpenAI needs Gemini API key for web search capability
    if provider_name == "chatgpt" and gemini_api_key:
        return provider_class(api_key=api_key, model=model, gemini_api_key=gemini_api_key, model_info=model_info, **kwargs)
    
    return provider_class(api_key=api_key, model=model, model_info=model_info, **kwargs)


def list_available_providers() -> list[str]:
    """Return list of available provider names."""
    return list(PROVIDERS.keys())


__all__ = [
    "BaseAIProvider",
    "AIResponse",
    "AIProviderError",
    "AIOutputTruncatedError",
    "AIEmptyResponseError",
    "AIContextLengthError",
    "AIRateLimitError",
    "AIAuthenticationError",
    "AIModelNotFoundError",
    "AIPricingError",
    "DjinniteModalityError",
    "GeminiProvider",
    "ClaudeProvider",
    "OpenAIProvider",
    "GrokProvider",
    "get_provider",
    "list_available_providers",
    "ACCESS_MODES",
    "PLATFORMS",
    "resolve_platform",
]
