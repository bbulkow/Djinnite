"""
Djinnite AI Abstraction Layer

Standardized, multimodal interface for Gemini, Claude, and OpenAI.
"""

__version__ = "0.5.0"

from .ai_providers import (
    get_provider,
    BaseAIProvider,
    AIResponse,
    AIProviderError,
    AIOutputTruncatedError,
    AIEmptyResponseError,
    AIContextLengthError,
    AIRateLimitError,
    AIAuthenticationError,
    AIModelNotFoundError,
    DjinniteModalityError,
    DjinniteCapabilityDeniedError,
)
from .config_loader import load_ai_config, load_model_catalog

__all__ = [
    "get_provider",
    "BaseAIProvider",
    "AIResponse",
    "AIProviderError",
    "AIOutputTruncatedError",
    "AIEmptyResponseError",
    "AIContextLengthError",
    "AIRateLimitError",
    "AIAuthenticationError",
    "AIModelNotFoundError",
    "DjinniteModalityError",
    "DjinniteCapabilityDeniedError",
    "load_ai_config",
    "load_model_catalog",
]
