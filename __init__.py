"""
Djinnite AI Abstraction Layer

Standardized, multimodal interface for Gemini, Claude, and OpenAI.
"""

__version__ = "0.3.0"

from .ai_providers import (
    get_provider,
    BaseAIProvider,
    AIResponse,
    AIProviderError,
)
from .ai_providers.base_provider import DjinniteModalityError
from .config_loader import load_ai_config, load_model_catalog

__all__ = [
    "get_provider",
    "BaseAIProvider",
    "AIResponse",
    "AIProviderError",
    "DjinniteModalityError",
    "load_ai_config",
    "load_model_catalog",
]
