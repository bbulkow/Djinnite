# OLLAMA Provider — Design Document

**Status:** Draft, pending implementation
**Audience:** Code agents (Claude Code, etc.) implementing OLLAMA support in Djinnite
**Source:** Architectural review session, May 2026
**Confidence on overall approach:** High

---

## TL;DR

Add OLLAMA as a fourth Djinnite provider, supporting a small self-hosted fleet (≤10 hosts, ≤10 machine types). Existing public API is unchanged — `get_provider("ollama", model=...)` returns a `BaseAIProvider` with identical contract.

The hard problem OLLAMA solves differently from cloud providers is **per-host effective context**. We address it by:

1. **Banning floating tags** (`:latest`) and **requiring digest-uniform Modelfiles** across hosts of the same machine type, so model identity is stable across the fleet.
2. A new **`(machine_type, model_id) → effective_max_ctx`** binding table, populated by a binary-search probe and shared as a community knowledge base.
3. **Local client-side tokenization** to preflight context-overflow as `AIContextLengthError` *before* sending — converting OLLAMA's silent-input-truncation failure into Djinnite's standard contract.

Capability asymmetry (no native web search, limited thinking) is reflected honestly in the catalog (`web_search: ["off"]` etc.). The existing `_check_capability` machinery handles enforcement with no new code.

**No new public error classes.** **No changes to `AIResponse` shape.** **No breaking changes to existing providers.**

---

## Goals

- Add OLLAMA as a Djinnite provider with the same contract as Gemini/Claude/OpenAI.
- Support a small heterogeneous fleet of self-hosted hosts.
- Detect and reject context-overflow client-side, matching cloud-provider error semantics.
- Preserve all existing Djinnite invariants: catalog-driven, no static model data in Python, submodule API stability, configuration discovery.
- Position Djinnite to grow a shareable community asset (`fleet_bindings.json`) for hardware × model effective-context measurements.

## Non-goals

- **Custom inference layer (vLLM/SGLang/llama.cpp directly).** Reserved for a future provider when throughput justifies it. The Djinnite contract is what holds the line; alternative backends can swap behind it.
- **Local-to-commercial fallback routing.** Out of scope; provider selection stays explicit at the caller level.
- **Universal feature parity with cloud providers.** OLLAMA models will be flagged with reduced capabilities. Callers can already discover this via the catalog.
- **Per-host queueing, load balancing beyond simple selection, streaming throttling.**
- **Multi-deployment fleet config distribution.** Out of scope (see "Distribution problem" below).

---

## Architectural decisions

### 1. Identity discipline (ban the cause, don't track the symptom)

OLLAMA model tags can resolve to different artifacts on different hosts. Rather than tracking resolved digests in the catalog, we enforce tag discipline at registration.

**Validation rules**, enforced by `validate_ollama_fleet.py`:

- Reject any `model_id` ending in `:latest` or matching `:[a-z]+-latest`.
- Reject any `model_id` without an explicit quantization suffix. `mistral:7b-instruct-v0.3-q4_K_M` is valid; `mistral:7b` is not.
- At registration, query `/api/show` on every host serving a given `model_id` and verify the digests agree across all hosts of the same `machine_type`. Disagreement is a hard failure.

Modelfile uniformity itself is an out-of-band operational concern (Ansible/Make/pull script). Djinnite enforces *digest agreement*; humans enforce *deployment uniformity*. The digest check catches drift if humans get sloppy.

**Confidence: High.** Strictly simpler than digest-tracking and matches how container registries solved analogous problems.

### 2. Machine types layer

Effective deployable context is a property of `(machine_type, model_id)`, not of model alone (varies by hardware) and not of host alone (hosts of the same type with uniform Modelfiles share VRAM behavior).

**Three new entities:**

```
MachineType:
  id: str            # e.g. "rtx-4090-24gb", "h100-80gb", "m3-max-128gb"
  vram_gb: int
  notes: str         # free-form, human readable

Host:
  url: str           # e.g. "http://burning1.local:11434"
  machine_type: str  # references MachineType.id
  enabled: bool

Binding:                    # per (machine_type, model_id, ollama_versions)
  machine_type: str
  model_id: str
  effective_max_ctx: int    # discovered via probe OR specified by operator
  ollama_versions: str      # PEP 440 SpecifierSet, e.g. ">=0.5"
  source: "probed" | "specified"
  probe_date: str           # ISO 8601
```

The existing `model_catalog.json` is unchanged in structure — it gains an `ollama` top-level key with standard `ModelInfo` entries. **The `context_window` field there remains the *architectural* max; the effective deployable max lives in the bindings file.**

### 3. Two-file split for fleet data

This is intentional. The two files have fundamentally different sharability profiles:

| File | Sharability | Authority | Resolution |
|---|---|---|---|
| `fleet_catalog.json` | Deployment-local (never in public repo) | Operator | Project-only, no package fallback |
| `fleet_bindings.json` | Shareable (community-contributed) | Probe scripts + community PRs | Package default, project override permitted |

The mental model is symmetric with existing Djinnite files:
- `fleet_catalog.json` is to deployment topology as `ai_config.json` is to secrets — purely local.
- `fleet_bindings.json` is to hardware × model facts as `model_catalog.json` is to model facts — publishable.

This split positions Djinnite to serve a community knowledge base over time (analogous to a Hardware Compatibility List). Anyone running the same hardware/model/OLLAMA combo benefits from prior measurements without re-probing.

### 4. Effective context probing

A binary-search probe per `(machine_type, model_id)` pair:

```python
def probe_effective_ctx(host, model_id, architectural_max):
    lo, hi = 1024, architectural_max
    while hi - lo > 512:
        mid = (lo + hi) // 2
        if try_request(host, model_id, num_ctx=mid):
            lo = mid
        else:
            hi = mid
    return int(lo * 0.9)  # safety margin
```

`try_request` sends a controlled-size dummy prompt with `num_ctx=mid`. Success = response without truncation indicators; failure = OLLAMA OOM-class error or detected silent truncation.

**When probing runs:**

- At host registration (mandatory).
- When `update_ollama_capabilities.py` is invoked manually.
- **Never per-request.** Cached aggressively in `fleet_bindings.json`.

For each `(machine_type, model_id)`, only one host of that machine type needs probing. The result generalizes to all hosts of the same type — this is the value of the machine_type abstraction.

**Manual override:** if probing isn't desired, operators can specify `effective_max_ctx` directly with `source: "specified"`. The provider treats specified and probed values identically.

**Safety margin: 0.9.** May need adjustment to 0.85 in practice once probe data accumulates. **Confidence: Medium** on the exact value; **High** on the approach.

### 5. Version range tracking via PEP 440

OLLAMA major versions can shift effective context meaningfully (KV cache quantization defaults at 0.5, flash-attention upgrades, etc.). GPU drivers and CUDA versions do *not* meaningfully affect effective context — the 0.9 safety margin absorbs that variance.

Bindings carry an `ollama_versions` field using PEP 440 specifier syntax:

```json
"ollama_versions": ">=0.5"          // open-ended, currently valid
"ollama_versions": ">=0.5,<1.0"     // closed range, superseded by a newer binding
```

Implementation uses `packaging.specifiers.SpecifierSet`, already a transitive dep of any modern Python project.

**Workflow when OLLAMA ships a breaking change at, say, 1.0:**

1. Re-probe on 1.0, measure new effective_max_ctx.
2. Close the old binding: update `ollama_versions` from `">=0.5"` to `">=0.5,<1.0"`.
3. Add new binding with same `(machine_type, model_id)`, `ollama_versions: ">=1.0"`, new measurements.

Both bindings persist. Lookup selects based on the running OLLAMA version's fit within the range. This also gives historical visibility for debugging drift.

**Confidence: High** that PEP 440 is the right mechanism. **Confidence: High** that only OLLAMA major versions need tracking — drivers/CUDA do not.

### 6. Local tokenization for preflight

OLLAMA's worst failure mode is silent input truncation: HTTP 200 with a confidently wrong answer because the prompt was clipped to fit `num_ctx`. To convert this into Djinnite's standard `AIContextLengthError` contract:

```python
def preflight_check(prompt, model_id, host, max_output_tokens):
    tokens = local_tokenize(prompt, model_id)
    needed = tokens + max_output_tokens + SYSTEM_OVERHEAD_TOKENS
    eff_max = lookup_effective_max_ctx(
        machine_type=host.machine_type,
        model_id=model_id,
        ollama_version=host.ollama_version,  # from /api/version
    )
    if needed > eff_max:
        raise AIContextLengthError(
            f"Input requires {needed} tokens, host can serve {eff_max}",
            provider="ollama"
        )
    return needed  # use as num_ctx in actual request, locking the contract
```

**Tokenizer source:**

- **Preferred:** extract from the GGUF file via `/api/show` (canonical — matches the actual model in use).
- **Fallback:** fetch `tokenizer.json` from HuggingFace using the model's HF identifier.
- **Cache** per `model_id` for the life of the process. Tokenizers are stable artifacts.

Use the HuggingFace `tokenizers` Rust library (Python bindings) — it executes any BPE/WordPiece/Unigram tokenizer described by a `tokenizer.json`-style spec. Fast, deterministic, well-tested.

**Residual risk:** if a host loads other models between probe and request, effective context can drop below estimate. The 0.9 safety margin contains this. **No new public error class is added** — silent truncation becomes `AIContextLengthError`, which callers already handle.

### 7. Capability asymmetry, honestly reflected

OLLAMA models lack native web search, native graduated thinking, etc. The catalog reflects this:

```json
"capabilities": {
  "structured_json":  ["on", "off"],
  "temperature":      ["any", "default"],
  "thinking":         ["off"],
  "web_search":       ["off"],
  "json_with_search": ["off"],
  "thinking_style":   null
}
```

The existing `_check_capability` machinery in `BaseAIProvider` will reject `web_search=True` for OLLAMA models with no extra code. This matches Djinnite's existing tier-by-capability discipline.

Some local models do support reasoning (DeepSeek-R-distilled, Llama-thinking variants). Probe these case-by-case using the same probe machinery as commercial providers; default `thinking: ["off"]` and flip on when probed positive.

---

## File layout

### New files in the package

```
djinnite/
├── ai_providers/
│   └── ollama_provider.py              # NEW — OllamaProvider class
├── ollama/                             # NEW subpackage
│   ├── __init__.py
│   ├── fleet.py                        # OllamaHostRegistry, MachineType, Host
│   ├── bindings.py                     # Binding lookup with version-range matching
│   ├── tokenizer.py                    # tokenizer loading & caching
│   └── probe.py                        # binary-search context probe
├── scripts/
│   ├── update_ollama_capabilities.py   # NEW — runs probes, writes bindings file
│   └── validate_ollama_fleet.py        # NEW — validates topology, tags, digests, foreign keys
└── config/
    ├── fleet_bindings.json             # NEW — shipped community catalog (initially seeded)
    ├── fleet_bindings.example.json     # NEW
    └── fleet_catalog.example.json      # NEW
```

The `djinnite/ollama/` subpackage exists because fleet/probe/tokenizer logic is too much to fit cleanly inside a single provider file. `ai_providers/ollama_provider.py` imports from this subpackage and stays thin.

### Project-level config additions (host project's `config/` dir)

```
your-project/config/
├── ai_config.json              # MODIFIED — add ollama provider section
├── fleet_catalog.json          # NEW (project-only, never in public repo)
└── fleet_bindings.json         # OPTIONAL — project override of package defaults
```

Both new files follow the existing config-resolution pattern. `fleet_catalog.json` has no package fallback (deployment-specific by definition). `fleet_bindings.json` follows `model_catalog.json`'s "package default + project override" pattern.

---

## Schema specifications

### `fleet_catalog.json`

```json
{
  "machine_types": [
    {
      "id": "rtx-4090-24gb",
      "vram_gb": 24,
      "notes": "Single RTX 4090, dedicated to inference"
    },
    {
      "id": "m3-max-128gb",
      "vram_gb": 96,
      "notes": "M3 Max, unified memory; effective inference budget ~75% of total"
    }
  ],
  "hosts": [
    {
      "url": "http://burning1.local:11434",
      "machine_type": "rtx-4090-24gb",
      "enabled": true
    },
    {
      "url": "http://burning2.local:11434",
      "machine_type": "rtx-4090-24gb",
      "enabled": true
    },
    {
      "url": "http://studio.local:11434",
      "machine_type": "m3-max-128gb",
      "enabled": true
    }
  ]
}
```

### `fleet_bindings.json`

```json
{
  "bindings": [
    {
      "machine_type": "rtx-4090-24gb",
      "model_id": "mistral:7b-instruct-v0.3-q4_K_M",
      "effective_max_ctx": 28672,
      "ollama_versions": ">=0.5",
      "source": "probed",
      "probe_date": "2026-05-09"
    },
    {
      "machine_type": "m3-max-128gb",
      "model_id": "llama3.3:70b-instruct-q4_K_M",
      "effective_max_ctx": 16384,
      "ollama_versions": ">=0.5",
      "source": "specified",
      "probe_date": "2026-05-09"
    }
  ]
}
```

When a binding is superseded by a newer probe (e.g., OLLAMA 1.0 changes behavior), the old binding's `ollama_versions` is updated to a closed range (`">=0.5,<1.0"`) and a new binding is added for `">=1.0"`. Both persist.

### `ai_config.json` — new `ollama` provider section

```json
{
  "providers": {
    "ollama": {
      "enabled": true,
      "default_model": "mistral:7b-instruct-v0.3-q4_K_M",
      "use_cases": {
        "cheap": "mistral:7b-instruct-v0.3-q4_K_M",
        "code":  "deepseek-coder-v2:16b-instruct-q4_K_M"
      }
    }
  }
}
```

**Note:** no `api_key` — OLLAMA does not authenticate. The `ProviderConfig` schema needs a small adjustment to make `api_key` optional for OLLAMA, or the config loader can supply a sentinel value. Implementer's choice — flag in PR for review.

### `model_catalog.json` — new `ollama` top-level key

Standard `ModelInfo` shape, with `context_window` set to the architectural max (32768 for Mistral 7B v0.3, etc.) and capabilities reflecting OLLAMA's reduced feature set as shown in §7 above.

The package version of `model_catalog.json` should ship with sensible defaults for popular OLLAMA models so other adopters get a head start. Initial seed: top 10–20 models by community usage.

---

## Implementation phases

Each phase should ship as a standalone, mergeable PR.

### Phase 1: Provider scaffold + identity discipline (1–2 days)

- Create `ai_providers/ollama_provider.py`.
- Register `"ollama": OllamaProvider` in the `PROVIDERS` factory dict.
- Implement `_initialize_client` (HTTP client; base URL hardcoded for testing).
- Implement `is_available()` against `/api/version`.
- Implement `list_models()` against `/api/tags`.
- Implement tag-discipline validation (reject `:latest`, require quantization suffix).
- **Goal:** prove the contract integrates. No probing, no fleet logic, single-host hardcoded URL.

### Phase 2: Fleet config + host selection (1–2 days)

- Define `fleet_catalog.json` schema and write `djinnite/ollama/fleet.py` loader.
- Implement `OllamaHostRegistry`: lookup by `model_id`, returns a host serving it.
- OllamaProvider gets host from registry per-request.
- Simple host selection: round-robin among hosts that serve the requested `model_id`.

### Phase 3: Bindings + version-range matching (2–3 days)

- Define `fleet_bindings.json` schema and write `djinnite/ollama/bindings.py` loader.
- Implement `Binding.matches(machine_type, model_id, ollama_version)` using `packaging.specifiers.SpecifierSet`.
- Add `lookup_effective_max_ctx(...)` with package-default + project-override resolution.
- Wire the lookup into the provider (without preflight enforcement yet — just available for inspection).

### Phase 4: Probing (2–3 days)

- Implement binary-search probe in `djinnite/ollama/probe.py`.
- Write `scripts/update_ollama_capabilities.py` — invokes probes for declared `(machine_type, model_id)` pairs, writes results to **project-local** `fleet_bindings.json`. Package version is never touched.
- Probe protocol must keep model warm (`keep_alive: -1`) for the duration of the binary search to avoid reload overhead, then release.
- Capture OLLAMA version via `/api/version` to populate `ollama_versions` field as `">=<major>.<minor>"`.

### Phase 5: Local tokenization + preflight (2–3 days)

- Add `tokenizers` (HuggingFace) to `pyproject.toml` and `requirements.txt`.
- Implement tokenizer loader in `djinnite/ollama/tokenizer.py`: prefer GGUF extraction via `/api/show`, fall back to HF download.
- Tokenizer cache (process-lifetime, keyed by `model_id`).
- Wire preflight check into `OllamaProvider.generate()` — raise `AIContextLengthError` when input exceeds effective ceiling.

### Phase 6: generate() and generate_json() (2 days)

- `generate()` against `/api/chat` with `num_ctx` set per-request from preflight.
- `generate_json()` using OLLAMA's `format=` parameter (object or JSON schema; OLLAMA 0.5+ supports schema-constrained generation).
- Error mapping: OLLAMA's various failure modes → Djinnite's exception hierarchy.
- Token usage extraction from response (`prompt_eval_count`, `eval_count`).
- Cost computation: zero for self-hosted (electricity is real but out of scope).

### Phase 7: Validation script + catalog integration (1 day)

- Write `scripts/validate_ollama_fleet.py`:
  - Tag-discipline checks (no `:latest`, quantization required).
  - Digest-agreement check across hosts of the same `machine_type`.
  - Foreign-key checks: every `machine_type` referenced by `hosts[]` exists in the `machine_types[]` section; every `model_id` referenced by bindings exists in `model_catalog.json`.
- Extend `update_models.py` to refresh OLLAMA section of `model_catalog.json`.

---

## What this preserves

- **Existing public API:** `get_provider("ollama", model=...)` returns `BaseAIProvider`. Identical contract.
- **Catalog-driven discovery:** no hardcoded model data in Python.
- **Capability pre-flight:** existing `_check_capability` machinery handles OLLAMA's reduced feature set.
- **Error contract:** callers handle `AIContextLengthError` exactly as for cloud providers.
- **Configuration discovery:** existing `_resolve_config_file` pattern (project-local override, package fallback) covers both new files.
- **Submodule API stability:** no breaking changes to existing function signatures, return types, or modules.

## What this changes

- Two new project-level config files (`fleet_catalog.json`, optionally `fleet_bindings.json`).
- One new package-level config file (`fleet_bindings.json`, ships as community catalog).
- One new provider class (`OllamaProvider`).
- One new subpackage (`djinnite/ollama/`) for fleet/probe/tokenizer machinery.
- Two new maintenance scripts (`update_ollama_capabilities.py`, `validate_ollama_fleet.py`).
- `model_catalog.json` gains an `ollama` top-level key.
- New runtime dependency: `tokenizers` (HuggingFace).
- `ProviderConfig.api_key` becomes effectively optional (or sentinel) for providers that don't authenticate.

## What this explicitly does NOT do

- **No new error classes.** Silent truncation containment uses existing `AIContextLengthError`.
- **No changes to `AIResponse` shape.**
- **No changes to existing provider classes.**
- **No driver/CUDA/GPU-version tracking** in bindings. The 0.9 safety margin absorbs that variance.
- **No automatic fleet config distribution.** Operator's responsibility (see below).

---

## Out-of-band operational requirements

For this design to work in practice, the human operator must:

- Pull the same OLLAMA model with the same Modelfile across all hosts of a given `machine_type` (Ansible/Make/pull-script recommended; not Djinnite's job).
- Re-run `update_ollama_capabilities.py` when adding hosts of a new machine type, adding new models, or upgrading OLLAMA across a major version boundary.
- Avoid loading other models on inference hosts that would change KV-cache pressure between probe and serve time.
- Distribute `fleet_catalog.json` across deployments using whatever mechanism fits (private git, Ansible inventory, scp). See "Distribution problem" below.

These should be documented in `OLLAMA_OPERATIONS.md` (also new).

## Distribution problem (explicitly out of scope)

Multi-deployment scenarios (e.g., multiple installations each with different fleets) need a way to push `fleet_catalog.json` to each deployment. **This is not Djinnite's job.** Common patterns:

- Private git repo per deployment.
- Ansible/Terraform inventory generating `fleet_catalog.json`.
- Object store (S3/GCS) with deployment ID lookup.
- Manual scp.

Djinnite ships validation tools (`validate_ollama_fleet.py`) that work on configs from any source. The distribution mechanism is the operator's choice.

---

## Open questions deferred to implementation

These should be decided during the relevant PR with a brief rationale recorded in commit messages or DEVELOPMENT.md. None are blockers.

1. **Tokenizer source preference in tie-break cases.** GGUF extraction is canonical (matches the actual model file in use), but extraction tooling adds dependencies. HF fallback is easier but risks drift. **Recommendation:** prefer GGUF, document the decision and rationale.

2. **Probe overhead mitigation.** OLLAMA loads a model on first probe request, may take 30–60s. Probe protocol should keep the model warm for the duration of the binary search (`keep_alive: -1`) and release afterward.

3. **Concurrent host capacity.** Not addressed in this design. If two callers hit the same OLLAMA host simultaneously and the second exceeds remaining VRAM, OLLAMA queues. May be acceptable; if not, host selection grows a load-tracking heuristic. **Defer until felt as pain.**

4. **Initial seed for package `fleet_bindings.json`.** What hardware/model combos to ship as defaults. Maintainer's discretion; suggest top 10 popular OLLAMA models on top 3 consumer GPU classes (4090, 3090, M-series 64GB+).

5. **Safety margin tuning.** 0.9 may be conservative; data over time may indicate 0.95 is fine, or that 0.85 is needed for some hardware. Adjust based on probe results vs. real-world failures.

6. **`ProviderConfig.api_key` optionality.** Either make it `Optional[str]` (cleanest, but touches the config schema) or supply a sentinel `"none"` for OLLAMA (smaller change, less clean). Implementer's call.

---

## References (existing Djinnite docs to consult)

- `DEVELOPMENT.md` — public API contract, capability schema, error hierarchy, "no static model data" rule.
- `USE.md` — configuration discovery pattern, project-vs-package layout.
- `AGENTS.md` — risky-action rules; note that running probe scripts touches real hosts and should be scoped narrowly during development.
- `ai_providers/base_provider.py` — `BaseAIProvider` interface, `_check_capability`, `_validate_vision_limits`, error class hierarchy.
- `ai_providers/__init__.py` — `get_provider` factory, `PROVIDERS` registry, catalog requirement.
- `config_loader.py` — `_resolve_config_file`, `PROJECT_CONFIG_DIR`, `PACKAGE_CONFIG_DIR`.

---

## Appendix: design rationale recap

For agents implementing this who didn't see the conversation:

- **Why not just track digests instead of banning floating tags?** Identity discipline is simpler than tracking. Container registries solved this problem; we copy their answer.
- **Why a separate machine_types layer?** Effective context is keyed on `(hardware, model)`. Hosts of the same hardware class share behavior; encoding this lets probe results generalize across hosts. At the operator's scale (≤10 machine types, ≤100 hosts), this is a vast simplification over per-host tracking.
- **Why two files instead of one?** `fleet_catalog.json` is deployment-private (your basement); `fleet_bindings.json` is shareable hardware × model facts. Different sharability profiles → different files. Mirrors `ai_config.json` vs `model_catalog.json` exactly.
- **Why PEP 440 ranges instead of min/max?** Single-field, well-known syntax, free parser already in scope, expressive enough for any future scenario including disjunctions.
- **Why no driver/CUDA tracking?** Driver-level VRAM allocator differences are ≤5%, absorbed by the 0.9 safety margin. OLLAMA major version differences (KV cache quantization defaults, flash-attention) are 10–30% and worth tracking. Different magnitudes → different treatment.
- **Why no new error class for silent truncation?** The preflight check converts the OLLAMA-specific failure mode into the existing contract. Exposing it as a new class would force every Djinnite caller to handle a fourth exception, breaking submodule API discipline.
- **Why local tokenization instead of just trusting `num_ctx`?** OLLAMA's silent truncation happens when the prompt exceeds the model's actual context, regardless of `num_ctx`. Without local tokenization we can't preflight; without preflight we'd ship truncated answers as success. This is the core abstraction-preservation move.
