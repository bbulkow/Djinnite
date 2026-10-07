# Access Paths -- Requirements and Design

> **Status:** Proposed 2026-10-06, awaiting review
> **Scope:** configuring one provider type through more than one access path
> at the same time (for example Claude direct *and* Claude on Vertex AI), and
> declaring what a given path may not do.
> **Builds on:** DEVELOPMENT.md "Access modes: direct and platform" (0.5.0).

## Problem

0.5.0 added platform mode: a provider can be reached through its own API
(**direct**) or through a cloud platform such as Google Vertex AI
(**platform**). But `ai_config.json` keys `providers` by provider *type*, so
each type has one entry and therefore one mode.

That is not enough. The same provider is needed through more than one path at
once, because the paths are not interchangeable:

| | Claude direct (Anthropic) | Claude on Vertex AI |
|---|---|---|
| Credentials | Anthropic API key | Google ADC (service account); no key |
| Billing, data handling | Anthropic account | the Google Cloud project |
| Structured outputs, web search | available | denied by default in an organization (org policy, per project and model) |
| Web search tool | current version | `web_search_20250305` only |
| Model listing | Models API | none; only what `probe_platform` recorded |
| Token counting | everywhere | `global` and `us` only |
| Price | catalog | catalog at `global`; +10% at `us`, `eu`, regions |
| GCP integration | none | native (out of scope here, see Non-goals) |

An application may want Claude on Vertex for work that must stay in its
Google Cloud project, and Claude direct for work that needs a capability the
project's Vertex setup does not allow. Today it cannot say so in config. A
second `"claude"` key is not an error either: `json.load` keeps the last
duplicate, so one entry disappears silently.

## Terms

| Term | Meaning |
|---|---|
| **Provider type** | A key in `ai_providers.PROVIDERS`: `gemini`, `claude`, `chatgpt`, `grok`. It selects the SDK code and the catalog section. |
| **Entry** (access path) | A named object under `providers` in `ai_config.json`. Its name is chosen by the user. It has one type and one mode. |
| **Access mode** | `direct` (the provider's API and key) or `platform` (a cloud platform's credentials). |
| **Platform fact** | What a platform can do for a model, independent of who uses it. Recorded in the catalog by `probe_platform`. |
| **Deployment fact** | What *this* project, account or organization allows. Declared in `ai_config.json` by the operator. |

Before this change, entry name and provider type were the same string. After
it, they are the same string **by default**.

## Requirements

* **R1.** Any number of entries per provider type, in any mix of modes.
* **R2.** An entry's type defaults to its name. Every existing `ai_config.json`
  loads and behaves exactly as before.
* **R3.** The caller names the entry. Djinnite never picks an entry for the
  caller and never falls back from one entry to another.
* **R4.** One call builds a ready provider from an entry
  (`AIConfig.build_provider`), so callers stop assembling
  `get_provider(type, **provider_kwargs(name))` by hand.
* **R5.** An entry can declare capabilities it does not allow (`deny`), for
  every model it serves or per model, matching how platform policy is set
  (Vertex org policy names each model). Pre-flight rejects a request that
  uses one **before any network call**, with an error that names the entry,
  the model and the declared reason.
* **R5a.** `deny` is the only source of deployment restrictions. Djinnite
  does not probe, detect or learn them, at runtime or in maintenance
  scripts. The operator writes them from their policy.
* **R6.** The catalog holds platform facts; `ai_config.json` holds deployment
  facts. Neither is written into the other.
* **R7.** Configuration mistakes fail when the config loads, with a message
  naming the entry: duplicate keys, an explicit unknown type, a `deny` value
  outside the vocabulary.
* **R8.** Maintenance scripts (`update_models`, `update_model_costs`) use the
  direct-mode entry of each type by a stated rule, report which entry they
  used, and never change prices when they cannot estimate them.
* **R9.** Public API changes are additive. Existing signatures and return
  values stay as they are.

## Non-goals

* **GCP-only features**: Vertex AI Search grounding, RAG Engine, BigQuery,
  `gs://` file inputs. They are platform-only capabilities and will need
  their own design. Nothing here should make them harder to add.
* **Automatic selection or fallback** between entries, such as retrying on
  direct when Vertex returns 429. It would silently change billing and data
  handling. Callers who want it write the loop themselves.
* **Detecting restrictions.** No probe, runtime check or catalog field
  discovers what a deployment's policy denies (R5a). A request that `deny`
  does not cover goes out, and if the platform refuses it, the platform's
  error is mapped as today (Vertex: `AIAuthenticationError`, "Vertex AI
  refused the request").
* **Per-entry model allow-lists** (Model Garden `:predict` policy).
* **Use-case routing across entries** (a top-level `use_cases` map naming
  entries). `use_cases` stays inside each entry.

## Configuration

```jsonc
{
  "platforms": {
    "vertexai": {"project_id": "my-project", "locations": ["global", "us"]}
  },
  "providers": {
    "claude": {                                  // type "claude" (implicit)
      "api_key": "sk-ant-...",
      "default_model": "claude-sonnet-5-5"
    },
    "claude-vertex": {
      "provider": "claude",
      "mode": "platform", "platform": "vertexai", "location": "global",
      "default_model": "claude-sonnet-5-5",
      "deny": {
        "*": ["web_search"],
        "claude-haiku-4-5-20251001": ["structured_json"]
      },
      "deny_reason": "org policy vertexai.allowedPartnerModelFeatures in my-project"
    },
    "chatgpt": {"api_key": "sk-...", "default_model": "gpt-5"}
  },
  "default_provider": "claude"
}
```

### Entry fields

| Field | Default | Meaning |
|---|---|---|
| *(the key)* | -- | Entry name, unique in the file. |
| `provider` | the entry name | Provider type. **New.** |
| `mode` | `platform` if `platform` is set, else `direct` | Access mode. |
| `platform` | -- | Platform name (`vertexai`); required in platform mode. |
| `api_key` | -- | Direct mode credential. Ignored by Claude in platform mode. |
| `location`, `project_id`, `quota_project` | from `platforms.<name>` | Platform settings; the entry wins over the platform block. |
| `default_model`, `use_cases`, `enabled`, `modality_policy` | as today | Unchanged. |
| `deny` | none | Capabilities this entry does not allow: a list (every model) or a map of model ID to list, where `"*"` means every model. **New.** See "Restrictions". |
| `deny_reason` | -- | Free text shown in the error, one per entry. **New.** |

`default_provider` names an **entry**. So does the name passed to
`AIConfig.get_provider(name)`, `is_usable(name)` and `provider_kwargs(name)`;
these methods keep their names for compatibility.

### Validation at load

| Problem | Result |
|---|---|
| The same key twice anywhere in the file | `ValueError` naming the key, and saying how to configure a type twice (another entry name plus `"provider"`). |
| `provider` set explicitly to an unknown type | `ValueError` naming the entry and the known types. |
| No `provider`, and the entry name is not a known type (for example an old `"openai"` entry) | Loads, as today. `is_usable` is False and `build_provider` raises, saying to add `"provider"`. Keeps configs that load today loading. |
| `deny` neither a list nor a map of lists, a capability outside the vocabulary, or an empty model key | `ValueError` naming the entry (and model key) and the vocabulary. |
| `deny` names a model the catalog does not have | Loads (the config is read without the catalog, and a model the catalog later drops must not stop the config loading). `validate_ai` reports it as `[WARN]`. |
| A key under `providers` starting with `_` | Treated as a note and skipped, as under `platforms`. (It crashes today.) |
| Mode and platform errors | Unchanged. |

## API (additive)

```python
cfg = load_ai_config()

p = cfg.build_provider("claude-vertex")                 # default_model
p = cfg.build_provider("claude-vertex", model="claude-opus-5-5",
                       location="us")                   # per-call settings
p.entry      # "claude-vertex"   (None when built with get_provider directly)
p.mode       # "platform"

cfg.provider_type("claude-vertex")                      # "claude"
cfg.entries_of_type("claude")                           # ["claude", "claude-vertex"]
cfg.entries_of_type("claude", mode="direct")            # ["claude"]
cfg.direct_entry("claude")                              # "claude"  (maintenance rule)

choice = cfg.resolve_use_case("coding")                 # default entry
choice = cfg.resolve_use_case("coding", entry="claude-vertex")
choice.entry, choice.provider_type, choice.model
catalog.get_model(choice.provider_type, choice.model)

cfg.capabilities_for("claude-vertex", "claude-sonnet-5-5")   # effective view
```

* **`build_provider(entry, model=None, **overrides)`** calls
  `get_provider(type, model=model or default_model, entry=entry, deny=...,
  deny_reason=..., **provider_kwargs(entry), **overrides)`.
  * Overrides may set `location`, `project_id`, `quota_project`, `api_key`,
    `require_pricing`.
  * Overriding `platform`, `backend`, `mode`, `deny`, `deny_reason` or
    `entry` raises: that would make the provider something other than the
    entry. Configure another entry instead.
  * An unknown, disabled or unusable entry (placeholder key, unknown type)
    raises `ValueError` naming it.
  * `build_provider` resolves the entry's `deny` for the model being built
    (`"*"` plus that model's list) and passes the result on.
* **`get_provider(type, ..., entry=None, deny=None, deny_reason=None)`**: new
  optional keywords, so code that does not use `ai_config.json` can declare
  restrictions too. A provider serves one model, so here `deny` is a plain
  list. The unknown-name error adds a hint: if the name is an
  ai_config entry, use `build_provider`.
* **`DjinniteCapabilityDeniedError(AIProviderError)`**, with `e.entry`,
  `e.capabilities`, `e.reason`. Exported from `djinnite`.
* **`ModelChoice(entry, provider_type, model)`**, returned by
  `resolve_use_case`. The existing `get_model_for_use_case` keeps returning
  `(entry_name, model)`.

Unchanged: `get_provider(type, ...)` without the new keywords,
`provider_kwargs` (exact same dict), `is_usable` (plus: False for an unknown
type), `get_provider(name)` / `get_default_provider()` on `AIConfig`,
`get_model_for_use_case`.

## Capability resolution

What a request may use is decided in three layers, each narrower than the
one before:

| Layer | Source | Written by | Example |
|---|---|---|---|
| 1. Model, direct | catalog, top-level `capabilities` | `update_models` | Sonnet 5.5 supports web search |
| 2. Model, on the platform | catalog `platforms.<name>.capabilities`, overlaid field by field (`ModelInfo.for_platform`) | `probe_platform --capabilities` | Vertex serves no newer web-search tool |
| 3. Entry | `ai_config.json` `deny` | the operator | this project's org policy blocks web search |

Layers 1 and 2 are enforced by today's catalog pre-flight. Layer 3 is new and
enforced separately (below), so its error can say *whose* restriction it is.
`capabilities_for(entry, model)` returns the combined view for inspection.

## Restrictions (`deny`)

`deny` is configuration: the operator's statement of what this deployment's
policy does not allow, copied from that policy. It is not discovered
(R5a). Djinnite enforces exactly what it says, nothing more.

**Shape.** A list applies to every model the entry serves. A map applies per
model, with `"*"` for every model. The models' lists are added to `"*"`:

```jsonc
"deny": ["web_search"]                              // every model

"deny": {
  "*":                         ["web_search"],      // every model
  "claude-haiku-4-5-20251001": ["structured_json"]  // and, for Haiku, this too
}
```

Model keys are Djinnite model IDs, the same IDs as `default_model` and
`get_provider(model=...)`. Vertex org policy names models by Google's short
ID, so the policy value `publishers/anthropic/models/claude-haiku-4-5:structured_outputs`
becomes `"claude-haiku-4-5-20251001": ["structured_json"]`. A model the map
does not name gets only `"*"`.

**Vocabulary:** `structured_json`, `web_search`, `json_with_search`. Each
removes the `"on"` state of that capability. These are the request features
a deployment's policy can switch off. Vertex's
`vertexai.allowedPartnerModelFeatures` gates `structured_outputs` and
`web_search`; `json_with_search` is the two together.

Not deniable, by decision:

* **`thinking`.** No platform policy gates it. Where thinking behaves
  differently on a platform, that is a fact about the model on that platform
  (layer 2: `probe_platform --capabilities`, or a `model_overrides.json`
  pin), not about one deployment. It also could not be enforced truthfully:
  always-on models (Sonnet/Opus 5.5, Fable) think when the caller asks for
  nothing, so a `deny` could only block *requests* for thinking, not
  thinking itself.
* **`temperature`.** Not a policy-gated feature, and most calls send one.

The vocabulary grows when a platform policy gates a new feature.

**When it is enforced.** Before any network call, on every `generate()` and
`generate_json()` path of every provider. The hook is the cross-capability
pre-flight every provider already calls (`_validate_incompatible_combinations`),
so `generate(web_search=True)` is covered on Gemini, OpenAI and Grok too.
Those three have no catalog web-search check today.

**`force=True` does not bypass it.** `force` skips the *catalog* checks so
probes can find out what a model really does. A `deny` is the operator's
statement about the deployment, and the platform would refuse the request
anyway. Probes build providers without an entry, so they are unaffected.

**No catalog needed.** A `deny` is enforced even when the model has no
catalog entry.

**Precedence.** When the catalog and a `deny` both reject a request, the
catalog's message is raised. The case `deny` exists for -- the model can, the
project may not -- always gets the entry's message.

**Message:**

```
[claude] Request uses web_search=on, which ai_config entry 'claude-vertex'
denies for model 'claude-sonnet-5-5' (deny_reason: org policy
vertexai.allowedPartnerModelFeatures in my-project). This is a deployment
restriction, not a model limit. Use an entry that allows it, or drop it
from the request.
```

**Building without the entry.** `deny` belongs to the entry. A provider built
with `get_provider("claude", **cfg.provider_kwargs("claude-vertex"))` carries
no restriction. That is deliberate and visible in the code.

## Maintenance scripts

The catalog's top-level fields are direct-mode facts, so maintenance needs a
**direct** entry per type. The rule (`AIConfig.direct_entry(type)`):

1. Consider enabled, usable, direct-mode entries of that type.
2. If one is named after the type, use it.
3. Otherwise, if there is exactly one, use it.
4. If there are none, the type is not refreshed: `[SKIP] claude: platform
   mode only (claude-vertex) -- use probe_platform`, or the existing
   `[WARN] Provider claude not configured, skipping.`
5. If there are several and none is named after the type: `[FAIL]`, listing
   them and how to resolve it (name one after the type, or disable the
   others).

Each run prints the entry it used (`Updating claude models (entry 'claude')`).

* **Estimator** (the model that estimates prices and limits): one resolver
  shared by `update_models` and `update_model_costs`. It takes the CLI
  `--estimator` first, then `known_model_defaults.estimator`, then the
  default entry. `known_model_defaults.estimator.provider` is a **type**. The
  entry it runs through is `direct_entry(type)`.
* **`update_model_costs` without a usable estimator** stops before any
  write: `[FAIL] Estimator: <reason> -- no prices changed`, exit 1. Today a
  platform-mode estimator entry sets existing prices to `None` with
  `source: "failed"`. DEVELOPMENT.md and USE.md say this script skips
  platform entries; it does not, and the docs will be corrected.
* **`probe_platform`** is unchanged. It reads only the `platforms` block.
* **`validate_ai`, `validate_json`, `validate_models`** iterate entries
  rather than a hard-coded list of types, so second entries (and grok) are
  checked too.
  * Each entry is labelled `claude-vertex (claude, platform vertexai@global)`.
  * `validate_models` skips platform entries.

## Compatibility and migration

No existing config needs editing. Behavior changes:

1. A duplicate key in `ai_config.json` raises at load. (It used to drop one
   entry silently.)
2. An explicit unknown `provider` raises at load.
3. `_` keys under `providers` are notes.
4. `is_usable` is False for an entry whose type cannot be determined.
5. `update_model_costs` stops without writing when it has no usable direct
   estimator. (It used to set prices to `None`.)
6. `update_models` and the estimator choose the direct entry by the rule
   above.
7. The `--live` test fixtures use each type's direct entry. Platform entries
   are the e2e tier's job.

Platform mode has not been released (0.5.0 is untagged, under
"Unreleased"), so this lands as part of 0.5.0, with one Breaking Changes Log
entry.

## Verification

**Offline** (`tests/test_access_paths.py`; no parameter may be named
`provider`, which conftest treats as a live test):

* **Loading:**
  * a legacy config is unchanged, and `provider_kwargs` gives the same dict
  * mixed direct and platform entries of one type
  * duplicate keys are rejected (written as raw JSON text)
  * explicit and implicit unknown types
  * `_` notes are skipped
  * `deny` as a list and as a map; the vocabulary, the shapes and empty
    model keys are checked
  * `PROVIDER_TYPES` matches `PROVIDERS`
* **`build_provider`** for direct and platform entries, with stubbed SDK
  clients:
  * the overrides that are allowed, and the ones that raise
  * disabled, unknown and placeholder-key entries raise
* **`deny`** for each provider class, on `generate(web_search)`,
  `generate_json`, and JSON plus search:
  * the error is raised with zero client calls
  * `force=True` does not bypass it
  * it is enforced without a catalog entry
  * requests that use nothing denied go through
  * per model: one entry built for two models applies `"*"` to both and a
    model's own list only to that model; a model the map does not name gets
    only `"*"`
* **Resolution:**
  * `resolve_use_case` and `capabilities_for`
  * the `direct_entry` rule, including the ambiguous case
  * the estimator precedence
* **Costs:** a cost pass with only a platform-mode Claude entry changes
  nothing.
* **Unchanged:** `tests/test_platform_mode.py` must pass without changes; it
  guards the old behavior.

**E2E** (default tier, unbilled; new row A1 in PLATFORM_E2E_TEST_DESIGN.md):

* A temporary config on the e2e project has two entries:
  * `claude-vertex`, denying `web_search` for every model and
    `structured_json` for Haiku;
  * `gemini-vertex`.
* `build_provider` builds both; `probe_availability()` returns a known
  status for each.
* `claude-vertex` `generate(web_search=True)` raises
  `DjinniteCapabilityDeniedError` before any request is sent.
* Per model, through a real build:
  * `claude-vertex` built for Haiku: `generate_json` raises
    `DjinniteCapabilityDeniedError`.
  * Built for the zero-quota canary (Sonnet 5.5): `generate_json` is not
    denied. It reaches Vertex and gets the canary's 429, which is not billed.
