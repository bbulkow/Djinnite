"""
Djinnite - AI Abstraction Layer

Lightweight abstraction layer for multiple AI providers (Gemini, Claude, OpenAI).
Each provider wraps the native SDK directly without heavy frameworks.

Designed to be used as a git submodule across multiple projects.

Usage:
    from djinnite.ai_providers import get_provider, AIResponse
    from djinnite.config_loader import load_ai_config
    from djinnite.llm_logger import LLMLogger
"""

__version__ = "0.1.0"
