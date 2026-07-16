# Grok (xAI) Provider — Design Document

**Status:** Implemented
**Audience:** Code agents and maintainers working on Djinnite provider support
**Source:** Architectural session, July 2026
**Confidence on overall approach:** High

---

## TL;DR

Add Grok (xAI) as a fourth Djinnite provider. Existing public API is unchanged —
`get_provider("grok", model=...)` returns a `BaseAIProvider` with the identical contract.

The enabling fact: **xAI fully implements the OpenAI Responses API (`POST /v1/responses`)** at
`https://api.x.ai/v1` with bearer auth. Djinnite's `OpenAIProvider` already targets that exact wire
API, so Grok speaks a format Djinnite already knows.

**Architecture: parallel stacks.** `GrokProvider` is a standalone `BaseAIProvider` subclass. It
shares **no code** with `OpenAIProvider` — not by inheritance, not by mixin, not by a shared helper
module. The genuinely cross-provider logic already lives in `BaseAIProvider` (schema normalization,
cost computation, modality/vision validation, thinking *resolution*, probe *orchestration*) and is
reused. What each provider implements itself is its own wire-mapping. The ~150 lines of Responses
mapping in `GrokProvider` resemble `OpenAIProvider`'s but are intentionally independent, because the
two dialects already diverge and will keep diverging.

**No new public error classes. No changes to `AIResponse` shape. `OpenAIProvider` is not touched.**

---

## Goals

- Add Grok with the same contract as Gemini/Claude/OpenAI.
- Reuse `BaseAIProvider`'s cross-provider machinery; duplicate only the xAI wire-mapping.
- Expose Grok's native web search (Live Search) through Djinnite's unified `web_search=True`.
- Preserve all existing invariants: catalog-driven (no static model data in Python), submodule API
  stability, config discovery.

## Non-goals

- **No shared "OpenAI-REST" library.** Deliberately rejected to avoid coupling two dialects that
  diverge (see "Why parallel stacks"). If a *third* Responses-API provider ever appears, revisit.
- **No refactor of `OpenAIProvider`.** Zero regression risk to the working ~93-model path.
- **No X/Twitter-specific search surface.** Live Search's X source is reachable but exposed only as
  generic "web search on"; a dedicated social-search API is out of scope.
- **No Anthropic-compatible xAI endpoint.** xAI also offers one; we use the Responses endpoint only.

---

## Architectural decisions

### 1. Parallel stacks, not shared code

`GrokProvider(BaseAIProvider)` reimplements the Responses-API mapping it needs. **Confidence: High.**

**Why not subclass `OpenAIProvider`?** Inheritance would couple two vendors whose dialects already
differ (Live Search vs `web_search_preview`, source-billing vs search-billing) and force
`OpenAIProvider` to grow override hooks for Grok's sake. **Why not a shared helper module?** A
shared "Responses REST" library is the classic premature-DRY that accretes escape hatches as the
dialects drift; the identical surface is only ~150 mechanical lines, and `BaseAIProvider` already
holds everything genuinely common. Parallel stacks match how the existing three providers are
structured (each maps to its own SDK, sharing only `BaseAIProvider`).

### 2. Transport: OpenAI SDK pointed at xAI

`_initialize_client()` constructs `OpenAI(api_key=..., base_url="https://api.x.ai/v1")`. The `openai`
SDK is already a Djinnite dependency (`>=1.6.0`), so no new dependency is added. All calls go through
`self._client.responses.create(...)`. **Confidence: High.**

### 3. Web search = Live Search, billed per source

Grok's Live Search is the analogue of Gemini grounding / Claude web search, enabled on the Responses
API with a server-side `web_search` tool. The one real divergence from OpenAI is the **billable
unit**:

| Provider | Native mechanism | Billable unit | Rough price |
|---|---|---|---|
| Gemini | Google Search grounding | grounded request | ~$35 / 1k |
| Claude | `web_search` server tool | search executed | ~$10 / 1k |
| OpenAI | `web_search_preview` tool | search call | ~$10 / 1k |
| **xAI Grok** | **Live Search (`web_search` tool)** | **source returned** | **~$25 / 1k** |

Djinnite already models this as `search_units * search_cost_per_unit`, and `_count_search_units` is
provider-specific by design. Grok's implementation counts **returned source citations** (distinct
cited URLs), falling back to counting search-tool-call items. **Confidence: High on the abstraction;
Medium on the exact response field** — see Open Items.

### 4. Capabilities are probed, never hardcoded

Per the repo's "NO STATIC MODEL DATA IN PYTHON" rule, Grok's per-model capabilities
(`structured_json`, `temperature`, `thinking`, `thinking_style`, `web_search`, `json_with_search`,
incompatible combinations) are discovered by the live probe methods, which the catalog updater
drives. Grok reasons vary by model (some flagship models reason unconditionally and reject an
explicit effort/`none`); the probes surface this so the runtime pre-flight rejects unsupported
requests with a clear message instead of a vendor 400. **Confidence: High.**

### 5. Pricing: uniform estimation, conservative classification

Djinnite prices every provider the same way (AI + web-search estimation; no per-provider pricing
logic). Grok needs only to appear as a catalog key. Two small touch-ups: a `PROVIDER_COMPANIES`
display name, and a `pricing_class._classify_grok` rule. We have **no confirmed evidence** xAI
silently re-prices a stable id, and xAI's snapshot ids use an `MMDD` stamp the shared date regex
does not match — so the rule conservatively pins every non-`-latest` id as `fixed`, relying on the
universal `-latest` float rule and the 180-day staleness re-check. **Confidence: Medium** (revise
when xAI's re-pricing behaviour is confirmed).

---

## File layout

### New files
```
ai_providers/grok_provider.py     # GrokProvider(BaseAIProvider) — standalone
GROK_PROVIDER_DESIGN.md           # this document
```

### Edited files
```
ai_providers/__init__.py          # PROVIDERS["grok"], import, __all__
scripts/update_models.py          # providers["grok"], imports (both branches)
config/ai_config.example.json     # grok provider block (enabled:false, default grok-4.5)
scripts/update_model_costs.py     # PROVIDER_COMPANIES["grok"]
pricing_class.py                  # _classify_grok + _VENDOR_RULES["grok"]
config/known_model_defaults.json  # vision_defaults["grok"]
```

### Not touched
```
ai_providers/openai_provider.py   # explicitly — parallel-stacks decision
```

### Generated (not hand-edited)
```
config/model_catalog.json         # "grok" top-level key, populated by update_models
```

---

## `GrokProvider` surface

Standalone `BaseAIProvider` subclass with `PROVIDER_NAME = "grok"`,
`BASE_URL = "https://api.x.ai/v1"`. Implements:

- `_initialize_client()` — OpenAI SDK with xAI base_url.
- `generate()` / `generate_json()` — build the Responses request (parts → `input_text`/`input_image`,
  system → `instructions`, temperature/thinking/max-tokens via inherited `BaseAIProvider` resolvers,
  `text.format` json_schema strict for JSON), parse `output` → text/usage, `_compute_costs`, detect
  truncation → `AIOutputTruncatedError`.
- `_web_search_tools()` / `_count_search_units()` — Live Search enablement and per-source counting
  (the isolated divergence points).
- `is_available()` / `list_models()` — `grok-*` ids, specialized ids dropped.
- `probe_*` + `discover_modalities()` + `_build/_run_combination_probe` — live capability discovery.
- `_map_error()` — SDK exception → Djinnite hierarchy.

---

## Implementation phases (shipped)

1. **Provider + registration** — `grok_provider.py`, `PROVIDERS["grok"]`, `update_models` dict.
2. **Config + pricing scaffolding** — example config, `PROVIDER_COMPANIES`, `_classify_grok`,
   vision defaults.
3. **Catalog population** — `update_models --reprobe grok:all` writes the `grok` catalog section
   (requires a live xAI key).
4. **Pricing** — `update_model_costs --provider grok`.
5. **Verification** — text / structured-JSON / Live-Search smoke tests + regression suite.

---

## What this preserves / changes / does NOT do

**Preserves:** public API (`get_provider("grok", ...)`), catalog-driven discovery, capability
pre-flight (`_check_capability`), the error contract, config discovery, submodule API stability.

**Changes:** one new provider class; one new top-level `model_catalog.json` key; small config and
pricing-metadata additions; a new example-config block. No new dependency (`openai` SDK reused).

**Does NOT do:** touch `OpenAIProvider`; add error classes; change `AIResponse`; introduce a shared
Responses-API library; hardcode model or capability data in Python.

---

## Open items (resolve against the live API)

1. **Live Search response shape.** Confirm, on `/v1/responses`, that Live Search is enabled by the
   `web_search` server tool (vs a `search_parameters` field) and which field carries returned
   sources for unit counting. Both are isolated to `_web_search_tools()` and `_count_search_units()`
   — a one-line adjustment each.
2. **Reasoning effort.** Confirm which Grok models accept `reasoning.effort` and which reason
   unconditionally (rejecting `effort`/`none`). The probes reveal this; no code change expected.
3. **Model ids / context windows.** Confirm current flagship/fast ids and real context windows from
   a live `list_models()` + catalog update, rather than the rough defaults in `list_models()`.

---

## References

- `OLLAMA_PROVIDER_DESIGN.md` — sibling design doc; this one follows its shape.
- `ai_providers/base_provider.py` — the reused `BaseAIProvider` machinery.
- `ai_providers/openai_provider.py` — the Responses-API reference (NOT a dependency of this provider).
- `ai_providers/__init__.py`, `scripts/update_models.py`, `scripts/update_model_costs.py`,
  `pricing_class.py`, `config/known_model_defaults.json` — the wiring touch points.
- `DEVELOPMENT.md` — public API contract, capability schema, "no static model data" rule.
- `USE.md` — configuration discovery, maintenance-script usage.
