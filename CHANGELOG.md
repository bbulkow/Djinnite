# Changelog

## Unreleased

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
