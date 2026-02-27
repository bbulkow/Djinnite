# Djinnite Development Guide

## ⚠️ CRITICAL: UV REQUIRED

**This project uses `uv` for dependency management.**
ALL Python commands must be executed via `uv run` to ensure the correct environment and dependencies (including `google-genai`, `anthropic`, `openai`) are loaded.

```bash
✅ CORRECT:   uv run python -m djinnite.scripts.update_models
❌ WRONG:     python -m djinnite.scripts.update_models
```

## 🔧 Developer Setup

After cloning, install **all** dependencies including dev tools (pytest, pytest-cov):

```bash
uv sync --extra dev
```

Without `--extra dev`, only the runtime dependencies are installed. Running
tests (`uv run python -m pytest`) will fail with "No module named pytest".

The `--extra dev` flag installs everything declared in `[project.optional-dependencies] dev`
in `pyproject.toml`. You only need to run this once (or after `uv lock --upgrade`).

```bash
# Quick reference — full setup from scratch:
uv sync --extra dev                              # Install all deps + dev tools
uv run python -m djinnite.scripts.validate_ai    # Verify API keys & connectivity
uv run python -m pytest tests/ -v                # Run unit tests
```

## ⚠️ This is a Shared Package — Breaking Changes Affect Multiple Projects

Djinnite is used as a git submodule by multiple projects. Any change to its public API will impact **all** consuming projects when they update the submodule.

**Think twice before changing anything. Think three times before deleting anything.**

---

## 📜 THE CONTRACT (Public API)

These are the **only** stable import paths and signatures that consuming projects should depend on. Anything not listed here is an internal implementation detail and may change in a PATCH release.

### Public Imports (Stable)

```python
# The Primary Interface
from djinnite import get_provider, load_ai_config, load_model_catalog
from djinnite import BaseAIProvider, AIResponse, AIProviderError, DjinniteModalityError

# Error Hierarchy (all subclass AIProviderError)
from djinnite import (
    AIOutputTruncatedError,   # Output hit max token limit (HTTP 200, partial content)
    AIContextLengthError,     # Input exceeds context window (HTTP 400)
    AIRateLimitError,         # Rate limit / quota exceeded (HTTP 429)
    AIAuthenticationError,    # Invalid API key (HTTP 401)
    AIModelNotFoundError,     # Model does not exist (HTTP 404)
    DjinniteModalityError,    # Unsupported modality requested (client-side)
)

# Configuration Types
from djinnite.config_loader import AIConfig, ProviderConfig, ModelInfo, ModelCatalog
```

### Public Function Signatures

```python
# Provider factory
get_provider(provider_name, api_key, model, **kwargs) -> BaseAIProvider

# Generation (two distinct methods)
BaseAIProvider.generate(prompt, system_prompt, temperature, max_tokens, web_search, thinking) -> AIResponse
BaseAIProvider.generate_json(prompt, schema, system_prompt, temperature, max_tokens, web_search, force, thinking) -> AIResponse

# Discovery
load_ai_config() -> AIConfig
load_model_catalog() -> ModelCatalog
ModelCatalog.find_models(input_modality, output_modality) -> list[tuple[str, ModelInfo]]
```

### Thinking Parameter

The `thinking` parameter on `generate()` and `generate_json()` provides a unified
interface for controlling model reasoning/thinking across all providers:

```python
thinking: Union[bool, int, str, None] = None
```

| Value | Description |
|---|---|
| `True` (**recommended**) | Enable thinking at maximum budget |
| `False` | Explicitly disable thinking |
| `None` (default) | No thinking requested — standard generation |
| `int` (e.g. `8192`) | Specific token budget for reasoning |
| `str` (`"low"`, `"medium"`, `"high"`) | Effort level hint |

Djinnite translates this into the provider-native format automatically:

| Provider | Native Mechanism | Style Values |
|---|---|---|
| Claude | `thinking={"type": "adaptive"\|"enabled", "budget_tokens": N}` | `"adaptive"`, `"budget"` |
| OpenAI | `reasoning_effort="low"\|"medium"\|"high"` | `"effort"` |
| Gemini | `thinking_config={"thinking_budget": N}` | `"budget"` |

**Temperature conflicts** are handled automatically:
- Claude forces `temperature=1` when thinking is active (SDK requirement).
- Claude enforces `max_tokens > budget_tokens` automatically (adjusts upward if needed).
- OpenAI strips temperature entirely when thinking is active (reasoning models reject it).
- Models with `capabilities.temperature=false` in the catalog always have temperature omitted.

**Token budget guidance:**
Token budgets are highly unpredictable — they depend on prompt complexity, model
version, and task type.  The recommended approach is `thinking=True` (maximum budget
from the model catalog's `max_output_tokens`).  Only use explicit `int` budgets
after profiling specific workloads.  Low budgets cause partial/useless reasoning
that is still charged.

### max_tokens Parameter

The `max_tokens` parameter on `generate()` and `generate_json()` controls the
maximum number of output tokens the model can generate.

**Auto-resolution:** If the caller passes `max_tokens=None` (the default), Djinnite
automatically fills it from the model catalog's `max_output_tokens` value.  This
ensures the model has its full output capacity available and reduces expensive
truncated responses.

Resolution order:
1. Caller's explicit value (if provided)
2. Model catalog `max_output_tokens` (auto-filled from `model_catalog.json`)
3. Provider SDK default (if no catalog is available)

**Recommendation:** For most use cases, omit `max_tokens` entirely and let Djinnite
use the model's maximum from the catalog.  Only pass an explicit value when you need
to constrain output size for cost or latency reasons.  Setting it too low results
in truncated responses — which are useless but still charged.

### Error Contract

Every call to `generate()` or `generate_json()` can raise the following exceptions.
Consumers **must** handle at least `AIOutputTruncatedError` and `AIContextLengthError`
to avoid acting on incomplete data.

| Exception | HTTP Status | When | Data Available |
|---|---|---|---|
| `AIOutputTruncatedError` | 200 OK | Model output was cut short by the max output token limit | `e.partial_response` — the incomplete `AIResponse` with `truncated=True`, usage info, and partial content |
| `AIContextLengthError` | 400 Bad Request | Input prompt exceeds the model's context window | Standard error info |
| `AIRateLimitError` | 429 | Rate limit or quota exceeded | Standard error info |
| `AIAuthenticationError` | 401 | Invalid or missing API key | Standard error info |
| `AIModelNotFoundError` | 404 | Requested model doesn't exist | Standard error info |
| `DjinniteModalityError` | N/A (client) | Prompt contains unsupported modalities | `e.requested_modalities`, `e.supported_modalities` |

All exceptions inherit from `AIProviderError`, which itself inherits from `Exception`.
Every `AIProviderError` carries `e.provider` (str) and `e.original_error` (Optional[Exception]).

### AIResponse Fields

```python
@dataclass
class AIResponse:
    content: str                          # Generated text
    model: str                            # Model ID
    provider: str                         # Provider name
    usage: dict[str, int | None]          # Token usage (see below)
    parts: list[dict]                     # Multimodal output parts
    raw_response: Any                     # Original SDK response
    truncated: bool = False               # True if output was cut short
    finish_reason: Optional[str] = None   # Provider-native stop reason
```

**Token usage** (`response.usage` dict and convenience properties):

| Key / Property | Type | Description |
|---|---|---|
| `input_tokens` | `int` | Input/prompt tokens |
| `output_tokens` | `int` | Output/completion tokens |
| `total_tokens` | `int` | Total tokens (from provider or computed) |
| `thinking_tokens` | `int \| None` | Reasoning/thinking tokens. **`None` = unknown** (distinct from 0 = no thinking) |

Check `response.thinking_tokens is None` to know if `total_tokens` may be
incomplete (i.e., the provider didn't report thinking tokens separately).

The `truncated` and `finish_reason` fields are **always populated** — even when
`AIOutputTruncatedError` is raised, the partial `AIResponse` on the exception
will have `truncated=True` and the provider-native finish reason.

Provider-specific `finish_reason` values:

| Provider | Normal Completion | Truncated |
|---|---|---|
| OpenAI | `"stop"` | `"length"` |
| Anthropic | `"end_turn"` | `"max_tokens"` |
| Gemini | `"STOP"` | `"MAX_TOKENS"` |

### ModelInfo Fields

```python
@dataclass
class ModelInfo:
    id: str                               # Model ID (e.g. "gemini-2.5-flash")
    name: str                             # Human-readable display name
    context_window: int                   # Max input tokens (context window)
    max_output_tokens: int = 0            # Max output tokens (0 = unknown)
    capabilities: list[str]               # Legacy field
    modalities: Modalities                # Input/output modality capabilities
    costing: ModelCosting                 # Cost scoring information
```

**`max_output_tokens`** is the maximum number of tokens a model can generate in a
single response. Callers **should** use this value when setting `max_tokens` on
`generate()` / `generate_json()` to avoid truncation. A value of `0` means the
limit is unknown — callers should use a conservative default.

The field is populated by `update_models.py` dynamically (in priority order):
1. **Provider API** — Gemini exposes `output_token_limit` directly
2. **Existing catalog value** — Persisted from prior estimation runs
3. **AI estimation** — Web search-powered estimation for new/unknown models

---

## 🛠 INTERNAL IMPLEMENTATION (Do Not Depend On)

The following are internal tools used for maintenance scripts. Host projects **must not** latch onto these as they lack stability guarantees.

- `djinnite.llm_logger.LLMLogger`: Internal observability for Djinnite scripts.
- `djinnite.ai_providers.gemini_provider.*`: Use the `get_provider` factory instead.
- `djinnite.prompts.*`: Internal template system for maintenance.
- `djinnite.scripts.*`: CLI utility implementation details.

### Key Function Signatures (Maintenance Only)

```python
# These signatures are contracts — do not change without coordinating across projects

get_provider(provider_name: str, api_key: str, model: Optional[str], gemini_api_key: Optional[str]) -> BaseAIProvider

BaseAIProvider.generate(prompt: str, system_prompt: Optional[str], temperature: float, max_tokens: Optional[int]) -> AIResponse

BaseAIProvider.generate_json(prompt: str, schema: Union[Dict, Type], system_prompt: Optional[str], temperature: float, max_tokens: Optional[int], web_search: bool, force: bool) -> AIResponse

load_ai_config(config_path: Optional[Path]) -> AIConfig
load_model_catalog(catalog_path: Optional[Path]) -> ModelCatalog
ModelCatalog.find_models(input_modality, output_modality, provider) -> list[tuple[str, ModelInfo]]

LLMLogger.log_request(prompt, system_prompt, model, provider, metadata) -> str
LLMLogger.log_response(request_id, response_content, success, error, usage, parsed_result) -> None
```

---

## Rules for Changes

### ⛔ No Static Model Data in Python Code

Model capabilities (output token limits, structured JSON support, pricing, modalities) **must be discovered dynamically** via:
1. **Provider API responses** (e.g. Gemini exposes `output_token_limit`)
2. **Live probes** (e.g. structured JSON support testing)
3. **AI estimation with web search** (for values APIs don't expose)
4. **Existing `model_catalog.json` values** (persisted between runs)

Do **NOT** add per-model data tables (dicts, lists of model IDs with hardcoded values) to Python code. The model catalog is the database — it is populated dynamically by `update_models.py` and persists between runs.

If a truly un-discoverable override is needed (e.g. the cost anchor reference point), place it in `config/known_model_defaults.json` with a comment explaining why dynamic discovery is impossible. This file should remain **minimal**.

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
5. Test with `uv run python -m djinnite.scripts.validate_ai`

## Running Python Commands

⚠️ **CRITICAL FOR AI AGENTS AND DEVELOPERS:** This project uses **`uv`** for dependency
management. **ALL** Python commands — scripts, one-liners, imports, validation — **MUST**
be executed via `uv run`. Never use bare `python` or `python -c` directly.

```
✅ CORRECT:   uv run python -m djinnite.scripts.validate_ai
✅ CORRECT:   uv run python -c "from djinnite.config_loader import ..."
❌ WRONG:     python -m djinnite.scripts.validate_ai
❌ WRONG:     python -c "from config_loader import ..."
```

**Why:** Bare `python` may resolve to a system interpreter that lacks the project's
virtual environment and SDK dependencies (`google-genai`, `anthropic`, `openai`).
The `uv run` prefix ensures the correct venv, Python version, and all dependencies
are loaded — even for quick ad-hoc checks. There is **no exception** to this rule.

```bash
# From the host project root:
uv run python -m djinnite.scripts.validate_ai
uv run python -m djinnite.scripts.update_models
uv run python -m djinnite.scripts.update_model_costs --dry-run
uv run python -m djinnite.scripts.validate_models --multimodal
```

### Validation Notes for AI Agents

When running validation scripts (like `validate_models.py`), it is critical to **look at the actual command output text**, not just the exit code. Provider SDKs may be missing or API keys may be invalid, which are reported as successes in the process but failures in the output logic.

- **Check for ✅**: Indicates a successful end-to-end round trip.
- **Check for ❌**: Indicates a failure in initialization or generation.
- **Dependency Failures**: If a provider SDK (e.g., `google-genai`, `openai`, `anthropic`) is missing, the script will report an initialization failure with the specific package name.

## Version Policy

- Version is in `djinnite/__init__.py` (`__version__`)
- Bump version when making notable changes
- Use semantic versioning: `MAJOR.MINOR.PATCH`
  - PATCH: bug fixes, safe additions
  - MINOR: new features, new optional parameters
  - MAJOR: breaking changes (should be rare and coordinated)
