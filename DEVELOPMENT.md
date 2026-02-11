# Djinnite Development Guide

## ⚠️ This is a Shared Package — Breaking Changes Affect Multiple Projects

Djinnite is used as a git submodule by multiple projects. Any change to its public API will impact **all** consuming projects when they update the submodule.

**Think twice before changing anything. Think three times before deleting anything.**

---

## Public API (Do Not Break)

These are the stable import paths and signatures that consuming projects depend on:

### Core Imports

```python
# Provider factory and types
from djinnite.ai_providers import get_provider, list_available_providers
from djinnite.ai_providers import BaseAIProvider, AIResponse, AIProviderError
from djinnite.ai_providers.base_provider import AIRateLimitError, AIAuthenticationError, AIModelNotFoundError

# Individual providers
from djinnite.ai_providers.gemini_provider import GeminiProvider
from djinnite.ai_providers.claude_provider import ClaudeProvider
from djinnite.ai_providers.openai_provider import OpenAIProvider

# Configuration
from djinnite.config_loader import load_ai_config, load_model_catalog
from djinnite.config_loader import AIConfig, ProviderConfig, ModelInfo, ModelCatalog
from djinnite.config_loader import CONFIG_DIR, PROJECT_ROOT

# Logging
from djinnite.llm_logger import LLMLogger

# Prompts
from djinnite.prompts import COST_ESTIMATION_CONFIG
```

### Key Function Signatures

```python
# These signatures are contracts — do not change without coordinating across projects

get_provider(provider_name: str, api_key: str, model: Optional[str], gemini_api_key: Optional[str]) -> BaseAIProvider

BaseAIProvider.generate(prompt: str, system_prompt: Optional[str], temperature: float, max_tokens: Optional[int]) -> AIResponse

BaseAIProvider.generate_json(prompt: str, system_prompt: Optional[str], temperature: float, max_tokens: Optional[int], web_search: bool) -> AIResponse

load_ai_config(config_path: Optional[Path]) -> AIConfig
load_model_catalog(catalog_path: Optional[Path]) -> ModelCatalog

LLMLogger.log_request(prompt, system_prompt, model, provider, metadata) -> str
LLMLogger.log_response(request_id, response_content, success, error, usage, parsed_result) -> None
```

---

## Rules for Changes

### ✅ Safe Changes (Go Ahead)

- **Adding** new functions, methods, or classes
- **Adding** new optional parameters with defaults to existing functions
- **Adding** new modules (new `.py` files)
- **Adding** new provider implementations
- Bug fixes that don't change behavior
- Internal refactoring that doesn't change public interfaces
- Updating docstrings and comments
- Adding new prompt configs to `prompts/__init__.py`

### ⚠️ Requires Coordination (Ask First)

- Changing the **return type** of any public function
- Adding **required** parameters to existing functions
- Changing the **behavior** of existing functions (even if signature is same)
- Changing how `CONFIG_DIR` or `PROJECT_ROOT` are computed
- Modifying `AIResponse` fields
- Changing error class hierarchies

### 🚫 Breaking Changes (Do Not Do Without Explicit Approval)

- **Renaming** any module, class, or function in the public API
- **Removing** any module, class, or function
- **Moving** code between modules (changes import paths)
- Changing **required** parameter names
- Changing the `pyproject.toml` package name or structure
- Removing or renaming entries from `__all__`

---

## Project Structure

```
djinnite/
├── __init__.py              # Package version and docstring
├── pyproject.toml           # Build/install configuration
├── config_loader.py         # AI config loading (ProviderConfig, AIConfig, etc.)
├── llm_logger.py            # LLM request/response observability
├── ai_providers/
│   ├── __init__.py          # Provider factory (get_provider) + registry
│   ├── base_provider.py     # Abstract base + AIResponse + error classes
│   ├── gemini_provider.py   # Google Gemini implementation
│   ├── claude_provider.py   # Anthropic Claude implementation
│   └── openai_provider.py   # OpenAI ChatGPT implementation
├── prompts/
│   └── __init__.py          # Externalized prompt templates
├── scripts/
│   ├── validate_ai.py       # Test provider connectivity
│   ├── update_models.py     # Refresh model catalog from APIs
│   ├── update_model_costs.py # AI-estimated cost scoring
│   └── clean_disabled_reasons.py  # Catalog maintenance
├── tests/
│   └── probe_anthropic_beta.py    # Anthropic beta feature probe
├── config/
│   └── ai_config.example.json     # Example configuration template
├── requirements.txt         # Direct dependencies
├── DEVELOPMENT.md           # This file
└── README.md                # Package overview (TODO)
```

## Config Path Convention

Djinnite expects config files in the **host project's** `config/` directory:

```
<project_root>/config/ai_config.json
<project_root>/config/model_catalog.json
```

Where `<project_root>` = `Path(__file__).parent.parent` (parent of the `djinnite/` directory).

This works both when djinnite is a subdirectory and when it's a git submodule.

## Adding a New Provider

1. Create `djinnite/ai_providers/new_provider.py`
2. Subclass `BaseAIProvider` and implement all abstract methods
3. Register in `djinnite/ai_providers/__init__.py` → `PROVIDERS` dict
4. Add SDK dependency to `pyproject.toml` and `requirements.txt`
5. Test with `python -m djinnite.scripts.validate_ai`

## Running Scripts

```bash
# From the host project root:
python -m djinnite.scripts.validate_ai
python -m djinnite.scripts.update_models
python -m djinnite.scripts.update_model_costs --dry-run
```

## Version Policy

- Version is in `djinnite/__init__.py` (`__version__`)
- Bump version when making notable changes
- Use semantic versioning: `MAJOR.MINOR.PATCH`
  - PATCH: bug fixes, safe additions
  - MINOR: new features, new optional parameters
  - MAJOR: breaking changes (should be rare and coordinated)
