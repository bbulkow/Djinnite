# Changelog

## Unreleased

### 0.5.0 -- Added: platform access mode (Google Vertex AI)

Djinnite now reaches models in one of two access modes: **direct** (the
provider's own API and key, as before) or **platform** (a cloud platform
serving the provider's models, with the platform's credentials). Google
Vertex AI (`"vertexai"`) is the first platform. See DEVELOPMENT.md "Access
modes: direct and platform".

- **`get_provider(..., platform="vertexai", project_id=..., location=...,
  quota_project=...)`** with no `api_key`; `api_key` is now optional. The
  legacy `backend="vertexai"` still works and means the same thing.
- **Gemini on Vertex:** configurable `location`, **default now `global`**
  (was `us-central1`, where the current Gemini models are not served; pass
  `location="us"` to keep processing in the US), Application Default Credentials when no key is given,
  `quota_project` applied to the credentials, and `is_available()` /
  `list_models()` without a key.
- **Claude on Vertex:** `anthropic.AnthropicVertex` with ADC (never an API
  key), same request body as the direct path, dated-snapshot IDs rewritten to
  `@` form, Vertex's `web_search_20250305` tool, and a **10% price premium**
  off `global` (`usage["price_multiplier"]`).
- **Platform errors are mapped**: 429 / RESOURCE_EXHAUSTED ->
  `AIRateLimitError`, 401 / 403 -> `AIAuthenticationError` (Google's message
  kept), 404 -> `AIModelNotFoundError`, missing ADC ->
  `AIAuthenticationError`. Direct-mode mapping is unchanged.
- **`ai_config.json` `platforms` block** and provider `"mode": "platform"`;
  `AIConfig.provider_kwargs(name)` and `AIConfig.is_usable(name)`.
- **Catalog `platforms.<name>` block** per model (availability per location,
  optional platform capabilities), written by the new
  **`scripts/probe_platform.py`** and preserved by `update_models`.
- **Verification tiers**: `--e2e-platform` runs the platform end to end
  against a dedicated Vertex AI test project (manual; see
  PLATFORM_E2E_TEST_DESIGN.md), and `--live` now also runs the same
  call-shape contract in direct mode plus Claude 5.x thinking checks.
- `anthropic` dependency is now `anthropic[vertex]` (google-auth was already
  present via google-genai).

### 0.5.0 -- Added: several access paths per provider type

One provider can now be configured through more than one path at once, for
example Claude direct and Claude on Vertex AI. See ACCESS_PATHS_DESIGN.md and
DEVELOPMENT.md "Several access paths for one provider type". No existing
config needs editing.

- **Named entries:** `ai_config.json` `providers` keys are entry names. An
  entry's provider type is its new **`"provider"`** field, defaulting to the
  entry name, so existing configs load and behave as before. `default_provider`
  names an entry. Keys starting with `_` are notes.
- **`AIConfig.build_provider(entry, model=None, **overrides)`** builds a ready
  provider from an entry (type, credentials, platform settings, `deny`);
  overrides are limited to `api_key`, `require_pricing`, and for platform
  entries `location`, `project_id`, `quota_project`. New
  `provider.entry`. Djinnite never picks or falls back between entries.
- **New `AIConfig` methods:** `provider_type(name)`, `entries_of_type(type,
  *, mode=None, usable_only=False)`, `direct_entry(type)`,
  `resolve_use_case(use_case, entry=None)` returning
  `ModelChoice(entry, provider_type, model)`, and
  `capabilities_for(entry, model)`. `get_model_for_use_case`,
  `provider_kwargs` and `is_usable` are unchanged (`is_usable` is also False
  for an entry whose type cannot be determined).
- **`deny` / `deny_reason`** on an entry declare capabilities the deployment
  does not allow (`structured_json`, `web_search`, `json_with_search`), as a
  list for every model or a map per model with `"*"`. A request that uses one
  raises the new **`DjinniteCapabilityDeniedError`** (exported from
  `djinnite`; `e.entry`, `e.model`, `e.capabilities`, `e.reason`) before any
  network call, on every provider; `force=True` does not bypass it.
  `get_provider(..., entry=, deny=, deny_reason=)` accepts the same.
- **Duplicate keys in `ai_config.json` are rejected** at load with a
  `ValueError` (they used to drop one entry silently), as is an explicit
  unknown `"provider"` or a malformed `deny`.
- **`update_model_costs` no longer nulls prices without an estimator:** with
  no usable direct-mode estimator entry it stops before any write (exit 1).
  It used to set existing prices to `None` with `source: "failed"`. It
  also stops when the estimator's entry denies web search, which every
  price request uses. When the estimator runs but a model's estimate
  fails, a price the catalog already has is kept (only an unpriced model
  is marked `failed`). `update_models` and the estimator use each type's
  direct entry (`AIConfig.direct_entry`) and report which one they used.
- **`platforms.<name>.location`** is now the default location for that
  platform's entries (an entry's own `location` wins), as the docs said.
- **`validate_ai`** reports every enabled entry it cannot build -- a
  placeholder key or an unknown type -- as `[FAIL]`.

### 0.5.0 -- Added: multi-turn `history`

- **`generate(..., history=[...])` / `generate_json(..., history=[...])`**
  (keyword-only): earlier `{"role": "user" | "assistant", "content": ...}`
  turns, oldest first; `prompt` is the final user turn. Mapped natively on
  Claude, Gemini, OpenAI and Grok. `history=None` sends exactly the
  single-turn request.

### 0.5.0 -- Claude 5.x thinking

- **New `thinking="between_tools"`** (Claude, on models whose catalog
  `thinking_style` lists it -- Sonnet 5.5): the lowest thinking setting.
- **Behavior change: `thinking=False` sends `{"type": "disabled"}` on
  Claude.** It used to omit the block, which on Sonnet 5 / Opus 5 runs
  adaptive thinking. Sonnet 5.5, Opus 5.5 and Fable cannot disable thinking.
- `thinking=None` is documented as the provider default (adaptive on 5.x).
- **Discovery fixes** (apply on the next `update_models --reprobe claude:all`):
  the combination probe no longer sends `budget_tokens` to adaptive-only
  models (that recorded bogus `{thinking:on, structured_json:on}` /
  `{thinking:on, web_search:on}` incompatibilities on Opus 4.7+ and 5.x);
  `probe_thinking_disable` really probes; `between_tools` is detected; and
  `effort` joins `thinking_style` whenever `effort_levels` is set (so
  `thinking="high"` works on Sonnet 5.5, Opus 5.5, Fable 5.1).

### 0.5.0 -- Thinking-token accounting

- **Claude `thinking_tokens` is now reported** from
  `usage.output_tokens_details.thinking_tokens` (it was always `None`). It is
  a subset of `output_tokens`; cost and `total_tokens` are unchanged. A
  response that does not report it gives `None`, never 0.
- **Behavior change: Gemini thinking is now included in `token_cost`.**
  Gemini reports thoughts separately from `output_tokens` and they are billed
  at the output rate; Djinnite left them out, under-reporting cost.

### Behavior change: empty, blocked, and refused responses

Djinnite now follows one rule for responses that carry no usable content:
**raise when the response cannot be used as the method promises; return when
the model produced its own answer, even an empty one or a refusal.** See
"Empty, Blocked, and Refused Responses" in USE.md.

- **New `AIEmptyResponseError(AIProviderError)`.** It carries `e.reason` and
  `e.partial_response`, whose `usage` holds the billed tokens and costs.
  Existing `except AIProviderError` handlers keep working.
- **Blocked or filtered output now raises in both `generate()` and
  `generate_json()`.** Previously, Gemini could return `content=None,
  finish_reason=None` when it sent no candidates (for example, a prompt
  blocked for SAFETY), and the block reason was lost. The error message now
  includes Gemini's `block_reason`, finish reason, and safety ratings, the
  model id, and whether web search and thinking were active.
- **`generate_json()` now raises on an empty reply or a model refusal**, since
  neither can be schema-conforming JSON. A returned `generate_json()` response
  always has non-empty `content`.
- **`generate()` still returns** a model's own empty answer (`content == ""`)
  or refusal. OpenAI and Grok refusal text is now placed in `content` with
  `finish_reason == "refusal"`; previously it was dropped and `content` was
  empty.
- **OpenAI and Grok `content_filter` is no longer reported as
  `AIOutputTruncatedError`**; it raises `AIEmptyResponseError` with
  `reason="content_filter"`.
- **New `AIResponse.block_reason`** field, set when the provider blocked or
  filtered the output.
- **`AIResponse.content` is never `None`.**

Any caller that parsed `content` was already failing on the responses that
now raise, so no working behavior is lost.
