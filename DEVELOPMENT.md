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
uv run pytest tests/ -v                           # Offline tests (free, no keys)
uv run pytest tests/ -v --live                    # Adds live provider tests (costs tokens)
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
    AIEmptyResponseError,     # No usable content (HTTP 200): blocked, filtered, empty JSON
    AIContextLengthError,     # Input exceeds context window (HTTP 400)
    AIRateLimitError,         # Rate limit / quota exceeded (HTTP 429)
    AIAuthenticationError,    # Invalid API key (HTTP 401)
    AIModelNotFoundError,     # Model does not exist (HTTP 404)
    DjinniteModalityError,    # Unsupported modality requested (client-side)
)

# Configuration Types
from djinnite.config_loader import AIConfig, ProviderConfig, ModelInfo, ModelCatalog
from djinnite.config_loader import PlatformConfig, PlatformModelInfo   # platform mode
```

### Public Function Signatures

```python
# Provider factory. api_key is optional: platform mode needs none.
get_provider(provider_name, api_key=None, model=None, **kwargs) -> BaseAIProvider
#   platform mode kwargs: platform="vertexai" (legacy alias backend="vertexai"),
#                         project_id, location, quota_project

# Generation (two distinct methods). history is keyword-only.
BaseAIProvider.generate(prompt, system_prompt, temperature, max_output_tokens, web_search, thinking, *, history=None) -> AIResponse
BaseAIProvider.generate_json(prompt, schema, system_prompt, temperature, max_output_tokens, web_search, force, thinking, *, history=None) -> AIResponse

# Access mode of a provider instance
provider.mode       # "direct" | "platform"
provider.platform   # None | "vertexai"
provider.location   # platform location, or None in direct mode

# Config -> constructor kwargs (the one place that mapping lives)
AIConfig.provider_kwargs(name) -> dict
AIConfig.is_usable(name) -> bool

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
response.thinking_tokens # Optional[int] — None if not reported (unknown, never 0)
                         #   Claude/OpenAI/Grok: a SUBSET of output_tokens
                         #   Gemini: SEPARATE from output_tokens
response.total_tokens    # int

# Dollar costs (None if model pricing unknown)
response.token_cost      # Optional[float] — input + output + thinking tokens (thinking billed once),
                         #   times usage["price_multiplier"] when present (platform location premium)
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

### Catalog & validation philosophy

Four governing principles for how the catalog, probes, and runtime checks
fit together. New work touching capabilities or validation must respect
these; the rules exist because an unwritten gap let a known Anthropic
constraint (temperature + thinking ≠ 1) escape into production.

1. **The catalog must be expressive enough to encode every constraint a
   caller needs to build a valid request.** If a real API constraint
   cannot be expressed in the schema, the schema is wrong — extend it.
   We do not paper over schema gaps with hardcoded provider logic,
   silent rewrites, or out-of-band rules.
2. **Fail fast in the Djinnite layer.** When a caller's request is
   incoherent with the catalog, raise a clear error before any HTTP
   call. Never silently rewrite caller-supplied values.
3. **Probes discover reality, not assumptions.** Even constraints that
   look like API-class rules get re-validated per-model on every probe
   pass, so provider relaxations are picked up automatically.
4. **Cross-capability constraints are first-class.** They get a generic
   schema slot (`capabilities.incompatible`), a generic combinatorial
   probe orchestrator (`BaseAIProvider.probe_incompatible_combinations`),
   and a generic runtime check
   (`BaseAIProvider._validate_incompatible_combinations`). No
   hand-listing of specific known-bad pairs in provider code.

#### `capabilities.incompatible`

Each entry is a `dict[str, str]` mapping capability name to a state
token from that capability's vocabulary. Semantics: if a request
simultaneously selects every state in the dict, the request is invalid
and Djinnite raises before any HTTP call.

```python
# Example for every current Claude thinking model:
caps.incompatible == [{"temperature": "any", "thinking": "on"}]
```

Allowed capability names are listed in `ACTIVATABLE_CAPABILITIES`
(`config_loader.py`):

| capability        | "active" token | "inactive" token |
|-------------------|----------------|------------------|
| `temperature`     | `"any"`        | `"default"`      |
| `thinking`        | `"on"`         | `"off"`          |
| `structured_json` | `"on"`         | `"off"`          |
| `web_search`      | `"on"`         | `"off"`          |

`incompatible: None` means "not yet probed"; `[]` means "probed, no
incompatibilities found." The combinatorial probe runs pairwise across
the activatable capabilities each model supports, with conservative
classification (any inconclusive failure returns `None` so a partial
result never overwrites a good cached value).

Callers introspecting the catalog can preempt rejections themselves:

```python
caps = info.capabilities
for combo in (caps.incompatible or []):
    # combo is e.g. {"temperature": "any", "thinking": "on"}
    # caller should avoid producing every state in `combo` at once
    ...
```

### Thinking Parameter

The `thinking` parameter on `generate()` and `generate_json()` provides a unified
interface for controlling model reasoning/thinking across all providers:

```python
thinking: Union[bool, int, str, None] = None
```

| Value | Description |
|---|---|
| `None` (default) | **Provider default** — nothing is sent. The default varies: Gemini 3 Flash and every Claude 5.x model think (adaptive) by default; Claude Opus 4.x does not. |
| `False` | **Explicitly disable thinking.** Sends a provider-specific "no thinking" signal (Claude: `{"type": "disabled"}`). Raises locally on models that cannot disable thinking (`"off"` absent from `capabilities.thinking`: Claude Sonnet 5.5, Opus 5.5, Fable). |
| `True` (**recommended**) | **Enable thinking at maximum budget.** |
| `int` (e.g. `8192`) | Specific token budget for reasoning. |
| `str` (`"low"`, `"medium"`, `"high"`) | Effort level hint. |
| `"between_tools"` | Claude only, on models whose `thinking_style` lists it (Sonnet 5.5): the lowest thinking setting, `{"type": "between_tools"}` — Anthropic's replacement for `disabled` on that model. No effort is sent; the model default (`high`) is the highest level it accepts. Counts as thinking `"off"` for `capabilities.incompatible`. |

**`None` vs `False`:** These are semantically different. `None` = "I don't care, do whatever
the model normally does." `False` = "I explicitly do NOT want thinking." If you need
predictable behavior, always pass `True` or `False` — never rely on `None` for production code.
On Claude 5.x, `None` is **not** off: omitting the block runs adaptive thinking.

**Error behavior:** Each capability in the catalog is a list of supported
states drawn from a fixed Djinnite vocabulary (see `ModelCapabilities` in
`config_loader.py`). For thinking, the vocabulary is `{"on", "off"}`. The
caller's `thinking` argument maps to a state — `False` → `"off"`, anything
truthy → `"on"` — and Djinnite raises `AIProviderError` if the catalog says
that state is not in the model's list (e.g. `thinking=True` on a model with
`capabilities.thinking=["off"]`, or `thinking=False` on an always-on
reasoning model with `capabilities.thinking=["on"]`).

Djinnite dispatches the caller's value to the provider's native field —
**without silent translation between shapes**. Each provider only accepts
the shapes its native API exposes; mismatches raise `ValueError` locally
before any API call is made. The caller introspects
`capabilities.thinking_style` to know which shapes a given model accepts.

| Provider | `True` → "let the model decide" | `int` budget | `str` effort | `False` → disable | `None` → |
|---|---|---|---|---|---|
| **Claude** | `thinking={"type": "adaptive"}` *(when `thinking_style` includes adaptive; else `enabled` with a budget sized to leave room for output)* | `thinking={"type": "enabled", "budget_tokens": N}` *(requires `"budget"` in `thinking_style`)* | `output_config={"effort": "low"\|"medium"\|"high"\|"xhigh"\|"max"}` *(requires `"effort"`; **not** a `thinking` block, and no `"minimal"`)*; `"between_tools"` → `thinking={"type": "between_tools"}` *(requires `"between_tools"`)* | `thinking={"type": "disabled"}` *(requires `"off"`; omission is **not** off on 5.x)* | omit (model default: off on Opus 4.x, **adaptive on every 5.x**) |
| **Gemini** | `thinking_config={"thinking_budget": -1}` *(dynamic — model picks budget)* | `thinking_config={"thinking_budget": N}` | `thinking_config={"thinking_level": ThinkingLevel.<UPPER>}` *(native enum)* | `thinking_config={"thinking_budget": 0}` | omit (model default) |
| **OpenAI** | `reasoning={"effort": "high"}` *(no true "model decides" mode — high is the closest)* | **`ValueError`** — OpenAI is effort-only | `reasoning={"effort": "minimal"\|"low"\|"medium"\|"high"}` | `reasoning={"effort": "none"}` *(GPT-5.x hybrid; rejected on reasoning-only models like o1/o3 — pre-flight catches this)* | omit (model default) |

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

#### How callers know which shape a model accepts

The catalog field `capabilities.thinking_style` records the *union* of
native shapes each model accepts, drawn from
`{"adaptive","budget","effort","between_tools"}`:

* `"adaptive"` — caller passes `True`; provider has a "model decides" mode.
* `"budget"` — caller passes `int`; provider has an integer-budget field.
* `"effort"` — caller passes `str` (`"minimal"`/`"low"`/`"medium"`/`"high"`);
  provider has a string-effort field (OpenAI's `reasoning.effort`,
  Gemini's `thinking_level`, Claude's `output_config.effort`).
  `update_models` adds it whenever the provider enumerates `effort_levels`.
* `"between_tools"` — caller passes the string `"between_tools"`; Claude's
  lowest thinking setting (Sonnet 5.5). A *setting*, not evidence that the
  model thinks, so it never establishes the thinking `"on"` state.

`"between_tools"` is accepted because it names a distinct native mode with
its own constraints (no other field, effort high or below), not a synonym for
`True` — so the design note above (no `thinking="adaptive"`) does not
exclude it.

Discovery notes (Claude): the Models API reports
`thinking.types.{adaptive,enabled}` but nothing about `disabled` or
`between_tools`, so those come from probes (`probe_thinking_disable` sends
`{"type": "disabled"}`; `probe_thinking_style` tries `between_tools`). A
rejected probe is a 400, which Anthropic does not bill.

A model can list multiple — Claude 4.7 is `["adaptive","budget"]`, Gemini
2.5+ is `["budget","effort"]` — meaning either shape is accepted on
different calls. The "exactly one per call" constraint is enforced by
the type system (the `thinking` parameter is a single value of type
`bool | int | str | None`).

Calling code introspects via the public catalog API:

```python
from djinnite import load_model_catalog
shapes = load_model_catalog().get_model("gemini", "gemini-2.5-flash").capabilities.thinking_style
# → ["budget", "effort"]
if "effort" in shapes:
    resp = provider.generate(prompt, thinking="low")
elif "budget" in shapes:
    resp = provider.generate(prompt, thinking=2048)
```

Mismatches surface as a local `ValueError` whose message names the
offending shape and lists `caps.thinking_style`, so a caller who skipped
the introspection sees what to do without burning an API call.

**Temperature conflicts** and other cross-capability constraints are
recorded in `capabilities.incompatible` on each model and enforced at
call time — see the [Catalog & validation philosophy](#catalog--validation-philosophy)
section. Notes:
- Caller is responsible for `max_output_tokens > thinking_budget` on Claude — Djinnite raises `ValueError` if not, rather than silently bumping the cap.
- Models whose `capabilities.temperature` list does not include `"any"` (e.g. `["default"]`) have the caller's temperature stripped automatically. (The strip is independent of cross-capability checks.)

**Token budget guidance:**
Token budgets are highly unpredictable — they depend on prompt complexity, model
version, and task type.  The recommended approach is `thinking=True` (maximum budget
from the model catalog's `max_output_tokens`).  Only use explicit `int` budgets
after profiling specific workloads.  Low budgets cause partial/useless reasoning
that is still charged.

### max_output_tokens Parameter

The `max_output_tokens` parameter on `generate()` and `generate_json()` caps
the output tokens the model may emit. **Auto-resolution:** when the caller
passes `None` (the default), Djinnite fills it from the model catalog's
`ModelInfo.max_output_tokens`. Resolution order:

1. Caller's explicit value (if provided and > 0)
2. Model catalog `max_output_tokens` (from `model_catalog.json`)
3. `None` (the provider SDK uses its own default)

**Recommendation:** for most use cases, omit `max_output_tokens` entirely
and let Djinnite use the catalog value. Pass an explicit value only when
you need to constrain output for cost or latency.

Per-provider semantic note (see the [Token Budgets](#token-budgets) section
below for the full mapping): Claude and Gemini cap visible output only;
OpenAI's `max_output_tokens` caps visible output **and reasoning combined**.

### Debug request inspection (`DJINNITE_DEBUG_REQUESTS`)

Set the environment variable `DJINNITE_DEBUG_REQUESTS=1` (also accepts
`true`/`yes`/`on`) to dump the resolved per-request config to stderr
immediately before each provider SDK call. Useful for debugging cost
and budget surprises ("what did Djinnite actually hand to the SDK?")
without modifying code or attaching a debugger.

The dump is a single line per request, written to stderr (so it doesn't
pollute stdout used by piped tools), and contains both the caller's
**original** args (`thinking`, `max_output_tokens`, `temperature`,
`web_search`, `system_prompt`, `force` where applicable) and the
**resolved** provider-native config dict (Gemini's `config`, Claude's
`kwargs`, OpenAI's `kwargs`). Long strings (prompts, system
instructions) and large lists (multimodal parts, message histories) are
elided. Bytes are reported as `<bytes len=N>`.

Example:

```
$ DJINNITE_DEBUG_REQUESTS=1 uv run python my_script.py
[DJINNITE_REQUEST] gemini/gemini-2.5-flash generate caller={"thinking": "low", "max_output_tokens": null, "temperature": 0.7, "web_search": false, "system_prompt": null} native={"temperature": 0.7, "max_output_tokens": 8192, "thinking_config": {"thinking_level": "ThinkingLevel.LOW"}}
```

The check is a single `os.environ.get` per call; zero overhead when the
env var is unset. No logging-library dependency. Exceptions raised
during dump rendering never propagate to the request path — they
print a `<dump failed: ...>` line instead.

### Error Contract

Every call to `generate()` or `generate_json()` can raise the following exceptions.
Consumers **must** handle at least `AIOutputTruncatedError` and `AIContextLengthError`
to avoid acting on incomplete data.

| Exception | HTTP Status | When | Data Available |
|---|---|---|---|
| `AIOutputTruncatedError` | 200 OK | Model output was cut short by the max output token limit | `e.partial_response` — the incomplete `AIResponse` with `truncated=True`, usage info, and partial content |
| `AIEmptyResponseError` | 200 OK | Provider blocked/filtered the output, or `generate_json()` got no usable JSON (empty or refusal) | `e.reason`; `e.partial_response` with billed `usage` and costs, `finish_reason`, `block_reason` |
| `AIContextLengthError` | 400 Bad Request | Input prompt exceeds the model's context window | Standard error info |
| `AIRateLimitError` | 429 | Rate limit or quota exceeded | Standard error info |
| `AIAuthenticationError` | 401 | Invalid or missing API key | Standard error info |
| `AIModelNotFoundError` | 404 | Requested model doesn't exist | Standard error info |
| `DjinniteModalityError` | N/A (client) | Prompt contains unsupported modalities | `e.requested_modalities`, `e.supported_modalities` |

All exceptions inherit from `AIProviderError`, which itself inherits from `Exception`.
Every `AIProviderError` carries `e.provider` (str) and `e.original_error` (Optional[Exception]).

#### Empty, blocked, and refused responses

Rule: **raise when the response cannot be used as the method promises; return
when the model produced its own answer, even an empty one or a refusal.**

| Case | `generate_json()` | `generate()` |
|---|---|---|
| Provider blocked the prompt / no candidates | raise | raise |
| Provider filtered the output (even with partial text) | raise | raise |
| Model refusal | raise | return (`finish_reason="refusal"`, text in `content`) |
| Normal stop, zero text | raise | return (`content=""`) |

The caller-facing version, with retry guidance, is in USE.md
("Empty, Blocked, and Refused Responses").

**When adding or changing a provider:**
- Decide every empty/blocked/refusal case with the table above.
- Compute usage and costs (`_compute_costs()`) *before* raising, and raise via
  `self._raise_empty(ai_response, reason=..., details=..., web_search=..., thinking=...)`
  so the billed usage travels on `e.partial_response` and the message format
  is uniform.
- Check truncation first (`AIOutputTruncatedError` is more specific), then empty/blocked.
- Set `AIResponse.block_reason` when the provider blocked or filtered.
- Never let an SDK convenience accessor (e.g. Gemini `response.text`) put `None`
  into `content`.
- Put `except AIProviderError: raise` ahead of any generic `except Exception`
  mapping, so Djinnite's own errors are never rewrapped or misclassified.

### AIResponse Fields

```python
@dataclass
class AIResponse:
    content: str                          # Generated text (never None; may be "" from generate())
    model: str                            # Model ID
    provider: str                         # Provider name
    usage: dict[str, int | None]          # Token usage (see below)
    parts: list[dict]                     # Multimodal output parts
    raw_response: Any                     # Original SDK response
    truncated: bool = False               # True if output was cut short
    finish_reason: Optional[str] = None   # Provider-native stop reason
    block_reason: Optional[str] = None    # Provider block/filter reason, if any
```

**Token usage** (`response.usage` dict and convenience properties):

| Key / Property | Type | Description |
|---|---|---|
| `input_tokens` | `int` | Input/prompt tokens |
| `output_tokens` | `int` | Output/completion tokens |
| `total_tokens` | `int` | Total tokens (from provider or computed) |
| `thinking_tokens` | `int \| None` | Reasoning/thinking tokens. **`None` = unknown** (distinct from 0 = no thinking). A **subset of `output_tokens`** on Claude, OpenAI and Grok; **separate from** `output_tokens` on Gemini |
| `price_multiplier` | `float` (absent = 1.0) | Present only when the access path prices off the catalog, e.g. Claude on Vertex AI at `us` / `eu` / a region (1.10). Already applied to `token_cost` |

`total_tokens` includes thinking on every provider (the provider's own total,
or input + output where output already includes thinking). `token_cost`
bills thinking exactly once, at the output rate.

The `truncated` and `finish_reason` fields are **always populated** — even when
`AIOutputTruncatedError` is raised, the partial `AIResponse` on the exception
will have `truncated=True` and the provider-native finish reason.

Provider-specific `finish_reason` values:

| Provider | Normal Completion | Truncated | Blocked / Filtered | Refusal |
|---|---|---|---|---|
| OpenAI / Grok | `"completed"` | `"max_output_tokens"` | `"content_filter"` | `"refusal"` |
| Anthropic | `"end_turn"` | `"max_tokens"` | — | `"refusal"` |
| Gemini | `"STOP"` | `"MAX_TOKENS"` | `prompt_feedback.block_reason`, or `SAFETY` / `RECITATION` / `PROHIBITED_CONTENT` / ... | — |

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
single response. Callers **should** use this value when setting `max_output_tokens`
on `generate()` / `generate_json()` to avoid truncation. A value of `0` means the
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
    thinking_style:   Optional[list[str]] = None   # subset of {"adaptive","budget","effort","between_tools"}
    effort_levels:    Optional[list[str]] = None   # subset of {"minimal","low","medium","high","xhigh","max"}
    incompatible:     Optional[list[dict[str, str]]] = None  # forbidden cross-capability combos
```

The `incompatible` field encodes cross-capability constraints — see
[Catalog & validation philosophy](#catalog--validation-philosophy).

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
| `thinking` | `None` → no check (provider default); `False` → `"off"`; `"between_tools"` → requires `"between_tools"` in `caps.thinking_style` (Claude only; counts as `"off"` for `incompatible`); `True`/`int`/`str` → `"on"`; additionally `int` → requires `"budget"` in `caps.thinking_style`, `str` → requires `"effort"` in `caps.thinking_style` | Required token must be in `caps.thinking`; required shape must be in `caps.thinking_style` |
| `temperature` | caller-passed float → `"any"`; caller-omitted → `"default"` | If `"any"` not in `caps.temperature`, the float is silently stripped (no error) |
| `structured_json` | schema present → `"on"` | `"on"` must be in `caps.structured_json` |
| `web_search` | `web_search=True` → `"on"` | `"on"` must be in `caps.web_search` |
| `json_with_search` | schema + `web_search=True` → `"on"` | `"on"` must be in `caps.json_with_search` |

**`thinking_style` is enforced.** The caller must match the shape to the
model's `thinking_style`. Mismatches raise `ValueError` locally before any
API call is made — no silent shape conversion. Each provider's native
field (Claude `budget_tokens`, OpenAI `reasoning.effort`, Gemini
`thinking_budget`/`thinking_level`) accepts the corresponding caller
shape unchanged. To write portable code across providers, branch on
`info.capabilities.thinking_style` to pick a shape the model accepts.

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
* `probe_thinking_disable()` returns whether explicit-disable is accepted
  (Claude sends `{"type": "disabled"}`; it used to return True
  unconditionally, which marked Sonnet/Opus 5.5 and Fable as `"off"`-capable).
* `probe_incompatible_combinations()` receives the probed `thinking_style`
  so the request builder sends a thinking shape the model accepts. Sending a
  budget block to an adaptive-only model recorded the *shape's* rejection as
  `{thinking:on, structured_json:on}` / `{thinking:on, web_search:on}` on every
  Opus 4.7+ / 5.x model; a reprobe clears them.
* `probe_structured_json()`, `probe_temperature()`, `probe_web_search()`,
  `probe_json_with_search()` each return tri-state `True/False/None`, which
  the orchestrator translates to the on/off list shape.

Run `uv run python -m djinnite.scripts.update_models --reprobe all` to refresh
the catalog with current provider capabilities.

### Token Budgets

Djinnite has five distinct conceptual budgets in play at request time. The
table below maps each to the Djinnite-side surface (request param, catalog
field, internal helper, response field) and the native parameter the
underlying SDK uses. Read this when you need to know "which knob controls
X, and what does it become at the wire?"

| Budget | Djinnite surface | Anthropic Messages API | OpenAI Responses API | Gemini GenerationConfig |
|---|---|---|---|---|
| **Output budget** | request param: `max_output_tokens` (`Optional[int]`)<br>catalog: `ModelInfo.max_output_tokens`<br>internal resolver: `_resolve_max_output_tokens`<br>response: `AIResponse.output_tokens` | `max_tokens` *(required int)* — caps **visible output only**; thinking is counted under a separate budget. Anthropic enforces `max_tokens > thinking.budget_tokens`. | `max_output_tokens` *(int)* — caps **visible output + reasoning combined**; OpenAI does not expose them separately. | `GenerationConfig.max_output_tokens` *(int)* — caps **visible output only**; thinking is counted under a separate budget. |
| **Thinking budget** | request param: `thinking: Union[bool, int, str, None]` *(int form = budget tokens; str form = effort tier; bool/None = on/off/no-opinion)*<br>internal helpers: `_resolve_thinking`, `_get_max_thinking_budget`, `_EFFORT_LEVELS`<br>catalog: shape support in `ModelCapabilities.thinking_style`; max for `True` read from `ModelInfo.max_output_tokens`<br>response: `AIResponse.thinking_tokens` | `thinking={"type":"enabled","budget_tokens":N}` — **explicit numeric budget**, separate from `max_tokens` (rejected by Opus 4.7+ and every 5.x). `type:"adaptive"` has no explicit budget (model decides). `str` effort rides in `output_config.effort` (requires `"effort"` in `thinking_style`); `"between_tools"` sends `thinking={"type":"between_tools"}`. Thinking tokens are reported in `usage.output_tokens_details.thinking_tokens`, a subset of `output_tokens`. | `reasoning.effort: "minimal"\|"low"\|"medium"\|"high"\|"none"` — **opaque tier label, no numeric knob**. Reasoning consumption is folded into `max_output_tokens`. Caller must pass `str` or `True`/`False`/`None`; `int` raises `ValueError` (no native budget field). | `thinking_config.thinking_budget` *(int)*: `-1`=dynamic (model decides), `0`=disable, `N>0`=fixed budget. *Or* `thinking_config.thinking_level` *(`ThinkingLevel.MINIMAL\|LOW\|MEDIUM\|HIGH`)*. **Both fields are alternatives — Djinnite sets exactly one per request based on the caller's shape.** Both are separate from `max_output_tokens`. |
| **Total token budget**<br>(input + output + thinking + tool round-trips) | request param: *not exposed — model property, server-enforced*<br>catalog: `ModelInfo.context_window`<br>response: `AIResponse.total_tokens` *(post-hoc usage, not a cap)* | not a request parameter — model property | not a request parameter — model property *(`truncation: "auto"\|"disabled"` chooses overflow handling, not a numeric cap)* | not a request parameter — model property |
| **Input budget**<br>(max input tokens) | request param: *not exposed*<br>catalog: not stored — derivable from `context_window` minus reserved output / thinking<br>response: `AIResponse.input_tokens` *(post-hoc usage)* | not a request parameter | not a request parameter; `truncation: "auto"` lets the server drop earliest turns on overflow but does not set a cap | not a request parameter |
| **Search budget**<br>(billing events — *not tokens*) | request param: `web_search: bool`<br>catalog: `ModelCosting.search_cost_per_unit`<br>response: `AIResponse.search_units`, `AIResponse.search_cost` | `tools=[{...web_search…}]` *(enable/disable; no per-call event cap)* | `tools=[{"type":"web_search_preview"}]` *(enable/disable; no per-call event cap)* | `tools=[Tool(google_search=...)]` *(enable/disable; no per-call event cap)* |

#### Notable semantic asymmetries

1. **Output-budget meaning differs across providers.** A caller passing
   `max_output_tokens=2000` gets:
   * Claude / Gemini → up to 2000 *visible* output tokens; thinking is
     additional and capped under a separate budget.
   * OpenAI → up to 2000 tokens *combined* across reasoning + visible
     output. So a high-reasoning prompt with `max_output_tokens=2000`
     may produce far less visible text than the same number on
     Claude/Gemini.

   This is an SDK-level inconsistency Djinnite is exposing transparently;
   it is **not** something Djinnite normalizes today.

2. **Thinking-budget granularity differs, and Djinnite does NOT
   cross-translate.** Anthropic accepts only a numeric token budget;
   OpenAI accepts only a string effort tier (`"minimal"`/`"low"`/
   `"medium"`/`"high"`/`"none"`); Gemini accepts either (numeric
   `thinking_budget` *or* the `ThinkingLevel` enum). The caller is
   responsible for matching the shape to the model's
   `capabilities.thinking_style` — passing the wrong shape raises
   `ValueError` locally before the API call. To write portable code,
   branch on `info.capabilities.thinking_style`. Djinnite removed the
   silent two-way translation that older versions performed: it tied
   thinking budgets to unrelated caps (`max_output_tokens`) and obscured
   the caller's intent. The new contract is fail-fast and never "do what
   I mean." See the Breaking Changes Log for the migration.

3. **`thinking="adaptive"` is intentionally not a Djinnite caller value.**
   Adaptive is what `True` already means semantically (let the model
   decide). Exposing it would force callers to learn vendor-native
   concepts the abstraction is meant to hide.

#### What Djinnite does *not* expose

* **No request parameter for total or input budget.** Both are model
  properties enforced server-side; exceeding them surfaces as
  `AIContextLengthError` (HTTP 400). Adding a Djinnite-side cap would
  duplicate the server's enforcement.
* **No separate `thinking_budget=N` parameter.** The `thinking`
  parameter's int form covers it, and splitting would force callers to
  coordinate two parameters with rules like "`thinking_budget` is
  honored only when `thinking=True`."

### Multi-turn input (`history`)

`generate()` and `generate_json()` take an optional keyword-only
`history=[{"role": "user" | "assistant", "content": str | parts}, ...]` —
the earlier turns, oldest first. `prompt` is always the final user turn.
Djinnite keeps no session state: the caller replays the transcript on every
call. `history=None` (or `[]`) sends exactly the single-turn request.

Rules (each a local `ValueError`): the first turn is `user`; assistant turns
are text only and non-empty. Vision limits count images across all turns.

| Provider | Native shape |
|---|---|
| Claude | `messages`: the turns, then the prompt; assistant content is a plain string |
| Gemini | `contents`: `types.Content(role="user" \| "model", parts=...)` per turn, then the prompt |
| OpenAI / Grok | `input`: `{"role", "content"}` items; assistant content is a plain string (the Responses API rejects `input_text` there) |

With `generate_json`, the schema constrains only the turn being generated
(`output_config.format` / `response_schema` / `text.format` are
request-level); earlier assistant turns — including JSON the caller is
replaying — are sent as plain text. Only text is replayed, so Claude's
"preserved thinking" history checks have no thinking blocks to bind.

### Access modes: direct and platform

Djinnite reaches a model in one of two **access modes**:

* **direct** — the provider's own API (Anthropic, Google AI Studio, OpenAI,
  xAI), authenticated with that provider's API key. The default.
* **platform** — a cloud platform serving the provider's models,
  authenticated with the platform's own credentials. Google Vertex AI
  (`"vertexai"`) is the first; the design expects more (Bedrock, Foundry),
  so the *mode* is "platform" and `vertexai` is one platform.

A platform serves some subset of each hosted provider's models, and that
subset grows and shrinks. So Djinnite **implements the platform's mechanics**
(per-platform rules in `ai_providers/platforms.py`, client code in each
provider) and **probes** what the platform offers per model
(`scripts/probe_platform.py`), rather than encoding model lists.

#### Selecting platform mode

```python
# Runtime: no api_key.
p = get_provider("claude", model="claude-sonnet-5-5", platform="vertexai",
                 project_id="my-project", location="us",
                 quota_project="my-project")
p.mode, p.platform, p.location   # ("platform", "vertexai", "us")

# Legacy alias, unchanged: backend="vertexai" == platform="vertexai".
g = get_provider("gemini", model="gemini-3.5-flash", backend="vertexai",
                 project_id="my-project", location="global")
```

```jsonc
// ai_config.json
"platforms": {
  "vertexai": {"project_id": "my-project", "quota_project": "my-project",
               "locations": ["global", "us"]}   // what probe_platform checks
},
"providers": {
  "claude": {"mode": "platform", "platform": "vertexai", "location": "us",
             "default_model": "claude-sonnet-5-5"}   // no api_key
}
// get_provider(name, model=..., **cfg.provider_kwargs(name))
```

A provider entry's `project_id` / `location` / `quota_project` override the
platform block's. A legacy entry with `backend: "vertexai"` loads as platform
mode. `update_models` and `update_model_costs` are **direct-mode only**: they
skip a platform-mode entry (`[SKIP] <name>: platform mode -- use
probe_platform`), because the catalog's top-level fields are direct-mode facts.

#### Vertex AI rules

| | Gemini | Claude |
|---|---|---|
| Client | `genai.Client(vertexai=True, project, location)` | `anthropic.AnthropicVertex(project_id, region=location)` |
| Credentials | ADC; `api_key` passed only if given (legacy keyed Vertex) | ADC; `api_key` never sent |
| Default location | `us-central1` (unchanged) | `global` (premium-free; 5.x is not served at single-region endpoints) |
| `quota_project` | applied to the ADC credentials (`google.auth.default(quota_project_id=...)`), because google-genai **overwrites** an `x-goog-user-project` header with the credentials' quota project when they carry one; rejected with `api_key` | sent as `default_headers={"x-goog-user-project": ...}`: AnthropicVertex never derives that header from the credentials, so the explicit header always wins |
| Price | catalog price (no documented location premium) | catalog price at `global`; **×1.10** at `us`, `eu` and regional endpoints (`usage["price_multiplier"]`) |
| Model IDs | as catalog | as catalog; dated snapshots rewritten `-YYYYMMDD` → `@YYYYMMDD` |
| Web search | `google_search` (as direct) | `web_search_20250305` (the only version Vertex serves); search price is the catalog's first-party $10/1k, **unverified for Vertex** |
| `list_models()` | the platform's model list | Vertex has no Models API: catalog models recorded `available` at this location by `probe_platform` (`[]` if never probed) |
| `is_available()` | live call | live call (token count); any failure, including a zero quota, is False |

Errors (platform mode only; the direct mappings are unchanged): 429 /
`RESOURCE_EXHAUSTED` → `AIRateLimitError`; 401 / 403 →
`AIAuthenticationError` with Google's message; 404 → `AIModelNotFoundError`
(not served at that location, or not enabled for the project); missing ADC
(`DefaultCredentialsError` / `RefreshError`) → `AIAuthenticationError`.

#### Catalog: the `platforms` block

Generated per model by `probe_platform --write`, never hand-edited (pin a
value through `model_overrides.json`, e.g.
`platforms.vertexai.capabilities.web_search`). `update_models` carries it
forward untouched.

```json
"platforms": {"vertexai": {
  "locations": {"global": "available", "us": "no_quota", "eu": "not_found"},
  "capabilities": { "thinking": ["on"], "...": "..." },
  "probed": "2026-10-04"
}}
```

Location statuses (`LOCATION_STATUS_VALUES`): `available`, `not_found`
(404), `no_access` (401/403), `no_quota` (429 / RESOURCE_EXHAUSTED),
`unknown`. **Availability is informational** — runtime never blocks on it,
because a stale `not_found` would hide a model the platform has since added.
**Capabilities** probed on the platform replace the direct-mode values field
by field (`ModelInfo.for_platform`); `get_provider` applies that view in
platform mode. Costing is always the direct-mode costing times the
platform's multiplier.

#### `scripts/probe_platform.py`

```
uv run python -u -m djinnite.scripts.probe_platform --platform vertexai \
    [--provider claude] [--model ID] [--location us] [--capabilities] [--write]
```

Default: one token-count call per model × location (unbilled) → statuses.
`--capabilities`: the update_models capability suite through the platform
at the first `available` location, printing `[DIFF] <field>: direct=..
platform=..` — real generation calls, billed to the platform project.
Without `--write` nothing is saved. Run it observably (AGENTS.md).

#### Verifying platform mode

Three layers, so verification never rests on one platform:

| Layer | Command | Calls | Proves |
|---|---|---|---|
| Offline | `uv run pytest tests/` | none (SDK clients stubbed) | what Djinnite hands each SDK |
| Direct live | `uv run pytest tests/ --live` | provider APIs, your keys | the call-shape contract (`tests/_contract.py`) per provider; Claude 5.x thinking semantics |
| Platform e2e | `uv run pytest tests/ --e2e-platform -rA -s` | Vertex AI, dedicated e2e project | the same contract through the platform, plus auth, quota project, routing, error mapping, `probe_platform` |

The e2e tier is manual (no CI) and needs the GCP setup in
[PLATFORM_E2E_TEST_DESIGN.md](PLATFORM_E2E_TEST_DESIGN.md). A change to
platform-mode code is finished only when it passes (AGENTS.md).

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

BaseAIProvider.generate(prompt: str, system_prompt: Optional[str], temperature: float, max_output_tokens: Optional[int]) -> AIResponse

BaseAIProvider.generate_json(prompt: str, schema: Union[Dict, Type], system_prompt: Optional[str], temperature: float, max_output_tokens: Optional[int], web_search: bool, force: bool) -> AIResponse

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

### ⚠️ Known limitation: context-length pricing tiers

**`ModelCosting.input_per_1m` / `output_per_1m` are always the STANDARD service
tier at the BASE context tier.** `AIResponse.token_cost` therefore
under-reports for a request whose input exceeds a model's context-tier
threshold.

Vendors publish several prices for one model on the same page:

| axis | example (`gpt-5.4-pro`) | applies to Djinnite? |
|---|---|---|
| Standard, base context | $30 / $180 per 1M | **yes — this is what we store** |
| Context length above threshold | $60 / $270 above 272k input tokens | **yes, and it is not modelled** |
| Flex (off-peak) | $15 / $90 | no — Djinnite never sends `service_tier` |
| Batch / Priority | varies | no — same reason |
| Regional data residency (direct APIs) | +10% | no |
| Platform location premium (Claude on Vertex AI off `global`) | +10% | **yes** — `usage["price_multiplier"]`, a per-platform rule in `ai_providers/platforms.py` |

**Why this is tolerable for now.** Djinnite keeps no accumulating
conversation: a request is one `prompt` plus whatever `history` the caller
chooses to replay, so input size is bounded by what one caller passes in one
call. Requests above 272k input tokens are rare in practice -- but a caller
replaying a long transcript through `history` gets there sooner.

**It is not impossible, though.** A caller may legitimately pass a single
300k-token prompt — that is well under the 1,050,000 context window, so it
succeeds, and `token_cost` will report roughly half the true amount. Treat
`token_cost` as exact for ordinary requests and as a *lower bound* for very
large ones.

**How a tier mix-up is caught.** `ModelCosting.published_figure` stores the
vendor's price text verbatim, naming the tier it came from. Without it, reading
the Flex row instead of the Standard row looks identical to a price cut:
`gpt-5.4-pro` oscillated $30/$180 -> $15/$90 -> $30/$180 across consecutive
runs, reported each time as a legitimate DIVERGENT change. The estimator prompt
now pins the Standard tier by name and the divergence report prints the quoted
figure.

**Fixing it properly** means adding a `context_tiers` list to `costing` and
selecting the bracket from actual `input_tokens` in
`BaseAIProvider._compute_token_cost`. Note the subtlety: the threshold is on
*input* tokens but changes the *output* rate too. See
[SERVICE_TIER_DESIGN.md](SERVICE_TIER_DESIGN.md).

### ⛔ `model_catalog.json` Is Generated — Do Not Hand-Edit It

The companion to the rule above. Discovery writes the catalog; humans write
`config/model_overrides.json`. Four config files, separated by **role**, which
is what stops this becoming a file per parameter:

| file | role | edited by |
|---|---|---|
| `ai_config.json` | which providers, which keys | human |
| `known_model_defaults.json` | **inputs to** discovery (estimator choice, provider vision defaults) | human |
| `model_overrides.json` | **decisions on top of** discovery — any field, any model | human |
| `model_catalog.json` | generated output of the above plus the provider APIs | **nobody** |

*Defaults feed into discovery; overrides sit on top of it.* Anything a human
pins about a specific model goes in `model_overrides.json` whatever the field —
disabled state, a corrected context window, a hand-verified price. The file is
named for the relationship, not the parameter, so it never needs a sibling.

Entries are keyed `provider/model-id`; nested fields may be nested, so
`{"costing": {"input_per_1m": 2.5}}` overrides that number alone. Keys starting
with `_` are notes.

**One write path.** `scripts/model_overrides.save_catalog()` applies the
overrides and then writes; every script that persists the catalog routes
through it. Calling `json.dump(catalog, ...)` directly bypasses human decisions
and is caught by `test_every_write_path_routes_through_save_catalog`.

**The generated file stays readable.** Each overridden model carries an
`_overridden` block recording which fields a human set and what discovery had
said, so you read the catalog and edit the overrides. Removing an override
restores the discovered value immediately, without waiting for a refresh.

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
│   ├── platforms.py         # Platform registry (access mode "platform")
│   ├── gemini_provider.py   # Google Gemini implementation
│   ├── claude_provider.py   # Anthropic Claude implementation
│   ├── openai_provider.py   # OpenAI ChatGPT implementation
│   └── grok_provider.py     # xAI Grok implementation
├── prompts/
│   └── __init__.py          # Externalized prompt templates
├── scripts/
│   ├── validate_ai.py       # Test provider connectivity
│   ├── update_models.py     # Refresh model catalog from APIs (direct mode)
│   ├── probe_platform.py    # Probe platform availability/capabilities
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

### October 2026: platform access mode, multi-turn `history`, thinking accounting (0.5.0)

**Added (no change for existing callers):** platform access mode
(`platform="vertexai"`, `location`, `quota_project`; `api_key` optional on
`get_provider`), Claude on Vertex AI, the `ai_config.json` `platforms` block,
the catalog's per-model `platforms` block, `scripts/probe_platform.py`,
the `--e2e-platform` test tier, keyword-only `history=` on `generate()` /
`generate_json()`, and the Claude sentinel `thinking="between_tools"`.

**Behavior changes (approved):**

* **Claude `thinking=False` now sends `{"type": "disabled"}`.** It used to
  omit the block — which on Sonnet 5 / Opus 5 (and every 5.x) runs
  *adaptive* thinking, so `False` silently thought. Opus 4.x: same meaning,
  different request bytes. On models that cannot disable thinking (Sonnet
  5.5, Opus 5.5, Fable) `False` raises locally once the catalog is
  reprobed; until then the API returns a 400 rather than silently thinking.
* **Gemini thinking is now costed.** `candidates_token_count` excludes
  `thoughts_token_count`, but Gemini was marked "not billed separately", so
  `token_cost` left thinking out. It is now billed at the output rate:
  reported Gemini cost rises to the correct value for thinking calls.
* **Claude `thinking_tokens` is now reported** from
  `usage.output_tokens_details.thinking_tokens` (it read a nonexistent field
  and was always `None`). It is a subset of `output_tokens`, so cost and
  `total_tokens` do not change; `_thinking_billed_separately` is now False.
* **Discovery fixes** (take effect on the next reprobe): the combination
  probe sends a thinking shape the model accepts (budget blocks on Opus 4.7+
  / 5.x produced bogus `{thinking:on, structured_json:on}` and
  `{thinking:on, web_search:on}` entries); `probe_thinking_disable` actually
  probes (it returned True unconditionally); `between_tools` is detected;
  `effort` joins `thinking_style` whenever `effort_levels` is set.

**Migration:** none required. To correct the current catalog, run
`update_models --reprobe claude:all` (observably, per AGENTS.md); to record
what a platform serves, run `probe_platform`.

**Why:** Munin (Cloud Run, Vertex AI only, no API keys) needed Djinnite to
reach Gemini and Claude through Vertex with ADC, and needed accurate thinking
and cost numbers for its spend cap. Vertex is modelled as the first of
several platforms rather than a backend flag, because the set of models a
platform serves changes over time and has to be probed, not listed.

### September 2026: human overrides consolidated into `model_overrides.json`

**Removed:** `config/disabled_models.json`. `scripts/disable_models.py` is now
a shim that exits non-zero with a pointer to the replacement rather than
silently doing nothing.

**Added:** `config/model_overrides.json` — the single human-editable record of
per-model decisions — plus `scripts/model_overrides.py` (the engine and the
sole catalog write path) and `scripts/apply_overrides.py` (the CLI).

**Migration:** all 69 disable entries were migrated automatically and verified
byte-for-byte against the old file; keys gained a `provider/` prefix
(`gpt-4o` -> `chatgpt/gpt-4o`), and bare model IDs are still accepted. No
model's effective state changed. If you call `disable_models` from a script,
switch to `apply_overrides`.

**Why:** disable state lived in `disabled_models.json` *and* in the catalog.
Runtime read only the catalog, while the maintenance command re-enabled
anything absent from the file — so seven models (`gpt-4o`, `gpt-4o-mini`, four
`*-search-preview` variants, `gemini-3.1-flash-live-preview`) carried disable
reasons recorded only in the catalog and were one command away from being
silently re-enabled. `merge_model_data` copying `disabled` forward on every
refresh is what let that state renew itself indefinitely.

Making the catalog value a projection of the disable file was not a sufficient
fix: the catalog is still a readable, editable JSON file, so that only turned
"your edit drifts" into "your edit vanishes silently." It also left three
override mechanisms coexisting — `disabled_models.json`, the `known_model_defaults.json`
sidecar, and an in-catalog `costing.source: "manual"` sentinel with zero
users. A file per parameter does not scale: "disabled models" has no sensible
sibling for a pinned context window or a verified price. Naming the file for
the relationship rather than the parameter fixes that permanently.


### May 2026: drop silent shape translation for `thinking`

**Removed:** the magic translation tables that converted between effort
strings and integer token budgets — `_EFFORT_FRACTIONS`,
`_effort_to_budget`, `_budget_to_effort`, and `_DEFAULT_THINKING_BUDGET`
in `BaseAIProvider`. The Gemini provider also no longer derives a
`thinking_budget` from a string effort by computing a fraction of
`max_output_tokens`. The Claude provider no longer silently raises
`max_output_tokens` when it's smaller than the requested thinking budget.

**Why:** the previous behavior silently converted shapes the provider's
native API didn't accept, tying budgets to unrelated caps and obscuring
caller intent. The motivating bug: `thinking="low"` against Gemini was
ignoring the SDK's native `thinking_level` enum and computing a budget
from `max_output_tokens × 0.25`, losing the "let Google decide what
'low' means" intent. The new contract is fail-fast and never "do what I
mean": each provider passes the caller's value through to its native
field unchanged, OR raises `ValueError` locally if the shape isn't
expressible. No API call is made on a mismatch.

**New behavior:**
* Gemini `thinking="low"` → `thinking_config={"thinking_level": ThinkingLevel.LOW}` (was: budget computed from `max_output_tokens`).
* Gemini `thinking="minimal"` is now valid (was: `ValueError`).
* Claude `thinking="low"` → `ValueError` (Claude has no native effort field; was: budget computed from `max_output_tokens`).
* OpenAI `thinking=8192` → `ValueError` (OpenAI has no native budget field; was: silently translated to `effort:"high"` via thresholds).
* Claude with `budget >= max_output_tokens` → `ValueError` (was: silently raised `max_output_tokens` to `budget + max(1024, budget//4)`).
* `thinking=True` with neither catalog `max_output_tokens` nor an explicit `max_output_tokens` argument → `ValueError` (was: defaulted to `_DEFAULT_THINKING_BUDGET = 32768`).

**Migration:** introspect the model's `capabilities.thinking_style`
before picking a shape:

```python
from djinnite import load_model_catalog
shapes = load_model_catalog().get_model("gemini", "gemini-2.5-flash").capabilities.thinking_style
if "effort" in shapes:
    resp = provider.generate(prompt, thinking="low")
elif "budget" in shapes:
    resp = provider.generate(prompt, thinking=2048)
elif "adaptive" in shapes:
    resp = provider.generate(prompt, thinking=True)
```

Or just pass `thinking=True` (and nothing else) — that works on every
thinking-capable model. The `ValueError` on a mismatched shape names the
offending shape and lists `caps.thinking_style`, so callers can fix
their code without burning an API call to discover the constraint.

**Catalog change:** Gemini 2.5+ models now advertise
`thinking_style: ["budget", "effort"]` in `model_catalog.json` (was just
`["budget"]`). Re-probe via `update_models --reprobe` to regenerate.

### May 2026: `max_tokens` → `max_output_tokens` on the public API

**Renamed:** the request parameter on `BaseAIProvider.generate()` and `BaseAIProvider.generate_json()` is now `max_output_tokens` instead of `max_tokens`. The internal helper `_resolve_max_tokens` is now `_resolve_max_output_tokens`.

**Why:** `max_tokens` was a leftover from when Djinnite tracked Claude's Messages API one-for-one. The parameter actually caps *output* tokens (and on OpenAI, output + reasoning combined) — never the total token budget. The new name says what it does. OpenAI's Responses API and Gemini's GenerationConfig already use `max_output_tokens` natively; only Anthropic's SDK keyword stays `max_tokens` and that is unchanged at the wire.

**Also corrected (related):** `ModelInfo.context_window` was previously docstring-claimed as "max input tokens." That was wrong — the catalog values are total token budgets (input + output + thinking + tool round-trips combined). Field name unchanged; docstring fixed.

**Migration:** rename every `max_tokens=` keyword in callers of `generate()` / `generate_json()` to `max_output_tokens=`. The Anthropic SDK keyword (the dict key Djinnite passes to `client.messages.create(...)`) stays as `"max_tokens"` — that's the wire-level name and is not Djinnite's concern. Submodule consumers must update call sites in the same release as Djinnite.

See the new [Token Budgets](#token-budgets) section above for the full
cross-provider mapping table.

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
