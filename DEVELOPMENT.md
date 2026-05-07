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
tests (`uv run pytest`) will fail with "No module named pytest".

The `--extra dev` flag installs everything declared in `[project.optional-dependencies] dev`
in `pyproject.toml`. You only need to run this once (or after `uv lock --upgrade`).

```bash
# Quick reference — full setup from scratch:
uv sync --extra dev                              # Install all deps + dev tools
uv run python -m djinnite.scripts.validate_ai    # Verify API keys & connectivity
uv run pytest tests/ -v                           # Run unit tests
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

### AIResponse Properties (Stable)

```python
# Content
response.content         # str — generated text
response.model           # str — model ID
response.provider        # str — provider name
response.parts           # list[dict] — multimodal output parts
response.raw_response    # provider SDK response object
response.truncated       # bool — True if output was cut short
response.finish_reason   # str — provider-native stop reason

# Token counts
response.input_tokens    # int
response.output_tokens   # int
response.thinking_tokens # Optional[int] — None if not reported
response.total_tokens    # int

# Dollar costs (None if model pricing unknown)
response.token_cost      # Optional[float] — input + output + thinking tokens
response.search_cost     # Optional[float] — web search events
response.total_cost      # Optional[float] — token_cost + search_cost
response.search_units    # int — number of billable search events
```

### ModelCosting Fields (Stable)

```python
model_info.costing.input_per_1m        # Optional[float] — $/1M input tokens
model_info.costing.output_per_1m       # Optional[float] — $/1M output tokens
model_info.costing.search_cost_per_unit # Optional[float] — $ per search event
model_info.costing.source              # str — "estimated", "manual", "failed"
model_info.costing.updated             # str — ISO date
```

### Thinking Parameter

The `thinking` parameter on `generate()` and `generate_json()` provides a unified
interface for controlling model reasoning/thinking across all providers:

```python
thinking: Union[bool, int, str, None] = None
```

| Value | Description |
|---|---|
| `None` (default) | **No opinion** — let the model use its default behavior. Some models default to thinking ON (e.g., Gemini 3 Flash), others default to OFF. |
| `False` | **Explicitly disable thinking.** Sends a provider-specific "no thinking" signal. Errors on reasoning-only models that cannot disable thinking. |
| `True` (**recommended**) | **Enable thinking at maximum budget.** |
| `int` (e.g. `8192`) | Specific token budget for reasoning. |
| `str` (`"low"`, `"medium"`, `"high"`) | Effort level hint. |

**`None` vs `False`:** These are semantically different. `None` = "I don't care, do whatever
the model normally does." `False` = "I explicitly do NOT want thinking." If you need
predictable behavior, always pass `True` or `False` — never rely on `None` for production code.

**Error behavior:** Each capability in the catalog is a list of supported
states drawn from a fixed Djinnite vocabulary (see `ModelCapabilities` in
`config_loader.py`). For thinking, the vocabulary is `{"on", "off"}`. The
caller's `thinking` argument maps to a state — `False` → `"off"`, anything
truthy → `"on"` — and Djinnite raises `AIProviderError` if the catalog says
that state is not in the model's list (e.g. `thinking=True` on a model with
`capabilities.thinking=["off"]`, or `thinking=False` on an always-on
reasoning model with `capabilities.thinking=["on"]`).

Djinnite translates this into the provider-native format automatically.
The semantic of `True` is **"enable thinking — let the model decide how
much"**, which Djinnite emits as the most accurate native expression on
each provider:

| Provider | `True` → "let the model decide" | `int` budget | `str` effort | `False` → disable | `None` → |
|---|---|---|---|---|---|
| **Claude** | `thinking={"type": "adaptive"}` *(when `thinking_style` includes adaptive; else `enabled` at max budget)* | `thinking={"type": "enabled", "budget_tokens": N}` | translated to budget via `_effort_to_budget` | omit `thinking` block *(opt-in design: omission = disabled)* | omit (model default = off) |
| **Gemini** | `thinking_config={"thinking_budget": -1}` *(dynamic — model picks budget)* | `thinking_config={"thinking_budget": N}` | translated to budget via `_effort_to_budget` | `thinking_config={"thinking_budget": 0}` | omit (model default) |
| **OpenAI** | `reasoning={"effort": "high"}` *(no true "model decides" mode — high is the closest)* | translated via `_budget_to_effort` to low/medium/high | `reasoning={"effort": "low"\|"medium"\|"high"}` | `reasoning={"effort": "none"}` *(GPT-5.x hybrid; rejected on reasoning-only models like o1/o3 — pre-flight catches this)* | omit (model default) |

#### Design note: why `True` means "adaptive," not "max budget"

Anthropic's `thinking.type=adaptive` looks like a third on-mode but is
semantically the same thing Djinnite already meant by `thinking=True` — *enable
thinking, you decide how much*. The unified API gives callers exactly two
levers:

* `True` — "I want thinking, you handle it." → adaptive / dynamic / high effort, depending on what the provider offers.
* `int` or `str` — "I want a specific depth." → fixed budget or specific effort tier.

Adaptive is therefore **not** exposed as a separate caller-facing value
(`thinking="adaptive"` is intentionally not accepted). Doing so would force
callers to learn a vendor-native concept the abstraction is meant to hide,
and it would be redundant with `True`.

This is also why earlier Gemini behavior — `True` mapping to `thinking_budget=N`
where `N` was the model's `max_output_tokens` — was a bug: it burned the
full budget on every prompt, regardless of complexity. The fix sends
`thinking_budget=-1` (dynamic), which mirrors Claude's adaptive shape and
matches what the caller intended when they wrote `True`.

The catalog field `capabilities.thinking_style` records which native shapes
each model accepts (`{"adaptive","budget","effort"}`). This is provider-internal
information used by the translators above to pick the right shape — it does
not surface in the caller's vocabulary.

**Temperature conflicts** are handled automatically:
- Claude forces `temperature=1` when thinking is active (SDK requirement).
- Claude enforces `max_tokens > budget_tokens` automatically (adjusts upward if needed).
- OpenAI strips temperature entirely when thinking is active (reasoning models reject it).
- Models whose `capabilities.temperature` list does not include `"any"` (e.g. `["default"]`) have the caller's temperature stripped automatically.

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
    capabilities: ModelCapabilities       # Per-model lists of supported Djinnite-API states
    modalities: Modalities                # Input/output modality capabilities
    costing: ModelCosting                 # Dollar-based pricing ($/1M tokens)
    vision_limits: Optional[VisionLimits] = None  # Image input constraints (None for non-vision models)
```

**`vision_limits`** constrains image inputs for vision-capable models. Each field uses a three-value convention:
- `None` = unknown (fail-open: no validation)
- `float('inf')` = confirmed unlimited (stored as `"inf"` in JSON)
- positive number = hard limit (enforced by pre-flight validation)

```python
@dataclass
class VisionLimits:
    max_image_bytes: Optional[float]       # Max bytes per image (e.g. 5242880 for 5 MB)
    max_dimension_px: Optional[float]      # Max width or height in pixels (e.g. 8000)
    max_images_per_request: Optional[float] # Max images in a single request
    supported_formats: list[str]           # e.g. ["jpeg", "png", "gif", "webp"]
```

Images are validated in `_validate_vision_limits()` (called in every provider's `generate()` and `generate_json()`) before any API call is made. Oversized images raise `AIProviderError` immediately.

**`max_output_tokens`** is the maximum number of tokens a model can generate in a
single response. Callers **should** use this value when setting `max_tokens` on
`generate()` / `generate_json()` to avoid truncation. A value of `0` means the
limit is unknown — callers should use a conservative default.

The field is populated by `update_models.py` dynamically (in priority order):
1. **Provider API** — Gemini exposes `output_token_limit` directly
2. **Existing catalog value** — Persisted from prior estimation runs
3. **AI estimation** — Web search-powered estimation for new/unknown models

### ModelCapabilities — list of supported Djinnite-API states

Every field on `ModelCapabilities` is `Optional[list[str]]`. **A non-null list is
the per-model subset of the capability's fixed Djinnite vocabulary that the
model accepts.** `None` means unknown — runtime pre-flight is skipped.

```python
@dataclass
class ModelCapabilities:
    structured_json:  Optional[list[str]] = None   # subset of {"on","off"}
    temperature:      Optional[list[str]] = None   # subset of {"any","default"}
    thinking:         Optional[list[str]] = None   # subset of {"on","off"}
    web_search:       Optional[list[str]] = None   # subset of {"on","off"}
    json_with_search: Optional[list[str]] = None   # subset of {"on","off"}
    thinking_style:   Optional[list[str]] = None   # subset of {"adaptive","budget","effort"}
```

The vocabularies are exported as `Final` constants from `config_loader`
(`THINKING_STATES`, `TEMPERATURE_STATES`, `THINKING_STYLE_VALUES`, …) — models
do not invent new vocabulary; they only declare which subset they support.

**Why a list and not a bool:** A bool can't tell apart toggleable, always-on,
and never-thinks. A list does — and the same shape extends naturally if a
future API gains another mode (e.g., `"auto"`).

#### Caller arg → state mapping

The runtime maps the caller's argument on `generate()` / `generate_json()` to
a vocabulary token and rejects with `AIProviderError` if the token is not in
the catalog list. Mapping is enforced in `_resolve_thinking`,
`_resolve_temperature`, and `_check_capability` in `base_provider.py`.

| Capability | Caller arg → state | Pre-flight rule |
|---|---|---|
| `thinking` | `None` → no check; `False` → `"off"`; `True`/`int`/`str` → `"on"` | Required token must be in `caps.thinking` |
| `temperature` | caller-passed float → `"any"`; caller-omitted → `"default"` | If `"any"` not in `caps.temperature`, the float is silently stripped (no error) |
| `structured_json` | schema present → `"on"` | `"on"` must be in `caps.structured_json` |
| `web_search` | `web_search=True` → `"on"` | `"on"` must be in `caps.web_search` |
| `json_with_search` | schema + `web_search=True` → `"on"` | `"on"` must be in `caps.json_with_search` |

**`thinking_style` is informational, not enforced.** Djinnite cross-translates
between budget integers and effort strings transparently
(`_effort_to_budget` / `_budget_to_effort`), so callers can pass `int` or
`str` regardless of the model's native style. Provider subclasses consult
`caps.thinking_style` only to choose the right native request shape.

#### Catalog examples

```jsonc
// Toggleable thinking model (Claude Sonnet 4, Gemini 2.5 Flash)
"capabilities": {
  "thinking":         ["on", "off"],
  "thinking_style":   ["adaptive", "budget"],
  "temperature":      ["any", "default"],
  "structured_json":  ["on", "off"],
  "web_search":       ["on", "off"],
  "json_with_search": ["on", "off"]
}

// Always-on reasoning model (cannot be disabled)
"capabilities": {
  "thinking":       ["on"],
  "thinking_style": ["effort"],
  "temperature":    ["default"]
}

// Non-thinking model
"capabilities": {
  "thinking":       ["off"],
  "thinking_style": null
}
```

#### Pre-flight error semantics

The runtime raises distinct error messages for each failure mode:

* `thinking=True` on `["off"]` → "does not support thinking/reasoning"
* `thinking=False` on `["on"]` → "does not support disabling thinking … reasoning is always on"
* schema on `structured_json=["off"]` → "does not support structured JSON"
* schema + search on `json_with_search=["off"]` → "does not support combining structured JSON output with web search"

Callers that need to bypass a pre-flight rejection use `force=True` on
`generate_json()` (existing escape hatch — unchanged).

#### Capability discovery

`update_models.py` populates these lists by combining per-provider probes:

* `probe_thinking_style()` returns `list[str]` of styles confirmed to work.
* `probe_thinking_disable()` returns whether explicit-disable is accepted.
* `probe_structured_json()`, `probe_temperature()`, `probe_web_search()`,
  `probe_json_with_search()` each return tri-state `True/False/None`, which
  the orchestrator translates to the on/off list shape.

Run `uv run python -m djinnite.scripts.update_models --reprobe all` to refresh
the catalog with current provider capabilities.

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
- Changing how `CONFIG_DIR`, `PACKAGE_CONFIG_DIR`, `PROJECT_CONFIG_DIR`, or `_resolve_config_file` are computed
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
│   ├── update_model_costs.py # AI-discovered per-token pricing
│   └── clean_disabled_reasons.py  # Catalog maintenance
├── tests/
│   └── probe_anthropic_beta.py    # Anthropic beta feature probe
├── config/
│   ├── ai_config.example.json     # Example configuration template
│   ├── model_catalog.json         # Package default model catalog (fallback for projects)
│   └── known_model_defaults.json  # Package default model defaults (fallback for projects)
├── requirements.txt         # Direct dependencies
├── DEVELOPMENT.md           # This file
└── README.md                # Package overview (TODO)
```

## Config Path Convention

Djinnite uses **local project config with package fallback**. Two config directories are defined:

- **`PACKAGE_CONFIG_DIR`** (`Path(__file__).parent / "config"`) -- Djinnite's own `config/` directory, ships with the distribution. Contains `model_catalog.json`, `known_model_defaults.json`, and `ai_config.example.json`.
- **`PROJECT_CONFIG_DIR`** -- The host project's `config/` directory (discovered from CWD or parent-of-package). Contains `ai_config.json` (secrets) and optional overrides.

**Read resolution** (`_resolve_config_file(filename)`):
1. If `PROJECT_CONFIG_DIR` exists and contains the file, use it.
2. Otherwise, fall back to `PACKAGE_CONFIG_DIR`.

**Write behavior** (`CONFIG_DIR`):
- `CONFIG_DIR = PROJECT_CONFIG_DIR or PACKAGE_CONFIG_DIR`
- Scripts write to the project dir when it exists, otherwise to the package dir.

This means consuming projects only need `ai_config.json` in their `config/` directory. The model catalog and known defaults are inherited from the package unless explicitly overridden.

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

## Breaking Changes Log

### May 2026: `thinking=True` means "let the model decide"

**Changed:** On Gemini, `thinking=True` now translates to `thinking_config={"thinking_budget": -1}` (dynamic budget — model self-regulates) instead of `thinking_budget=N` where `N` was the model's full `max_output_tokens`. Claude (already adaptive) and OpenAI (`effort: "high"`) are unchanged.

**Why:** The unified `True` semantic is "enable thinking, you decide how much." Anthropic's `thinking.type=adaptive` and Gemini's `thinking_budget=-1` are the provider-native expressions of that intent. Sending the full max-output budget on every Gemini call burned tokens regardless of prompt complexity — the model now picks its own budget, matching Claude's adaptive behavior.

**Behavior change for callers:**
* Gemini calls with `thinking=True` will produce *less* reasoning depth on simple prompts and *similar* depth on hard ones. Total cost will drop.
* Callers who specifically want a fixed deep thinking budget should pass `thinking=<int>` or `thinking="high"` — both unchanged.

**`thinking="adaptive"` is intentionally not a valid caller value.** Adaptive is what `True` already means; exposing it as a separate string would force callers to learn vendor concepts the abstraction is meant to hide.

### May 2026: Capability fields are now lists of supported states

**Changed:** Every field on `ModelCapabilities` (`structured_json`, `temperature`, `thinking`, `web_search`, `json_with_search`, `thinking_style`) is now `Optional[list[str]]` instead of `Optional[bool]` (or `Optional[str]` for `thinking_style`). A non-null list is the per-model subset of a fixed Djinnite vocabulary; `None` still means "unknown".

**Why:** The boolean shape couldn't distinguish toggleable, always-on, and never-thinks reasoning models — they all collapsed to `thinking=true`. The list shape expresses *which* states the model accepts (`["on","off"]`, `["on"]`, `["off"]`) and extends to future modes without per-capability flags.

**New behavior:**
* `thinking=False` on an always-on model raises a distinct `AIProviderError` ("does not support disabling thinking") instead of silently passing through to the vendor SDK.
* Models with `temperature=["default"]` (no `"any"`) have caller-passed temperature stripped automatically.
* `thinking_style` may now list multiple styles when a model supports more than one (e.g. Claude 4.7 = `["adaptive","budget"]`).

**Migration:**
* Existing catalogs are read through a back-compat shim in `config_loader._coerce_states` that converts the old bool / single-string shapes to lists on load — no manual migration required to keep loading.
* To rewrite the JSON on disk, run `uv run python scripts/migrate_capabilities_to_lists.py`. After that, run `uv run python -m djinnite.scripts.update_models --reprobe all` to tighten always-on / multi-style cases the conservative migration assumed toggleable.
* Consumer code that read `caps.thinking is True` / `is False` should switch to `"on" in caps.thinking` / `"off" in caps.thinking`. The `ModelInfo.supports_structured_json` convenience property (returns `Optional[bool]`) is unchanged — it now derives the bool from the list.

### March 2026: Dollar-Based Cost Tracking

**Removed:** `ModelCosting.score`, `ModelCosting.tier`, `cost_anchor` config, Gemini algorithmic heuristic, all anchor/relative-scoring infrastructure. Gemini-proxy web search fallback for Claude (Claude now uses native GA web search). Beta header (`anthropic-beta: web-search-...`) for Claude web search.

**Added:** `ModelCosting.input_per_1m`, `ModelCosting.output_per_1m`, `ModelCosting.search_cost_per_unit` (all in dollars). `AIResponse.token_cost`, `AIResponse.search_cost`, `AIResponse.total_cost` properties.

**Behavior change:** `web_search=True` on models that don't support native web search now raises `AIProviderError`. Previously, some providers would silently proxy through Gemini or fall back to system-prompt guidance.

**Why:** The old system stored costs as relative scores (gemini-2.5-flash = 1.0). This made it impossible to compute actual dollar costs or add token costs to web search costs. All providers now publish clear per-token pricing, so Djinnite stores and reports costs in dollars.

**Migration:**
- Run `uv run python -m djinnite.scripts.update_model_costs --all` after updating
- Replace `model_info.costing.score` with `model_info.costing.input_per_1m` / `output_per_1m`
- Use `response.token_cost` / `response.total_cost` for dollar costs
- The `update_model_costs` script now uses AI + web search for ALL providers (including Gemini)
