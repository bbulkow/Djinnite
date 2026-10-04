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
- **Gemini on Vertex:** configurable `location` (default unchanged,
  `us-central1`), Application Default Credentials when no key is given,
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
