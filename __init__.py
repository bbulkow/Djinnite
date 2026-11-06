"""
Djinnite AI Abstraction Layer

Standardized, multimodal interface for Gemini, Claude, and OpenAI.
"""

__version__ = "0.7.0"

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
    DjinniteEffortMappingError,
)
from .config_loader import load_ai_config, load_model_catalog, load_request_policy, RequestPolicy

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
    "DjinniteEffortMappingError",
    "load_ai_config",
    "load_model_catalog",
    "load_request_policy",
    "RequestPolicy",
]
