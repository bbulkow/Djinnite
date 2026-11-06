# Live Platform Tests (Google Vertex AI) -- Design

> **Status:** Accepted 2026-10-04; reworked 2026-11-06 to be driven by
> `ai_config.json` (see "Configuration" and "Decisions"). Until the rework
> it was named the "e2e platform tier", read `DJINNITE_E2E_*` environment
> variables, and required a GCP project provisioned just for it; it never
> ran. It now runs in the Vertex project the operator already configured.
> **Date:** 2026-10-04 (rework 2026-11-06)
> **Scope:** verify platform mode (Google Vertex AI) inside Djinnite, against
> the real service, as part of Djinnite's own test suite.

**What this tier is, in one line:** the `--live` contract, delivered
through Vertex AI instead of the providers' own APIs, plus the plumbing
only a platform has (ADC, quota project, location routing, Model Garden).
It is **not** a test of any other project.

## Problem

Platform mode (0.5.0) is covered by offline tests only. They prove what
Djinnite *hands the SDK* -- client kwargs, request bodies, error mapping on
synthetic error objects -- but not that Vertex accepts it. Every claim that
depends on Google is currently untested:

* ADC authenticates and the quota project is the one billed;
* the endpoint for each location resolves, and the model IDs (including the
  dated `@` rewrite) exist there;
* Vertex's real error bodies map to the right Djinnite errors;
* `count_tokens` works on Vertex (the availability probe depends on it);
* Vertex returns `usage.output_tokens_details` (Claude thinking tokens);
* Gemini accepts role-tagged `Content` history on Vertex;
* `probe_platform` produces a correct catalog block.

A script a person runs and reads (the retired `scripts/smoke_platform.py`)
is not verification. Verification has to live in Djinnite: a pytest tier
that a maintainer **or an agent** runs, that asserts outcomes, and that
fails loudly.

## Three verification layers

Platform verification must not rest on Vertex alone. The functional checks
are written once (`tests/_contract.py`) and run in every access mode:

| Layer | Flag | Calls | Proves |
|---|---|---|---|
| Offline | *(none)* | none -- SDK clients stubbed | what Djinnite hands each SDK; error mapping on synthetic errors |
| Direct live | `--live` | provider APIs with ai_config keys | the contract (JSON, history, truncation, cost arithmetic) per provider; Claude 5.x thinking semantics |
| Platform live | `--live-platform` | Vertex AI, the project `ai_config.json` names | the same contract through the platform, plus auth, quota project, routing, real error bodies, `probe_platform` |

A contract failure in both live layers is a Djinnite bug; in only one, it
is an access-path bug (or a provider/platform difference worth recording).

## Principles

1. **Pytest, not a script.** Live tests are collected, asserted, counted, and
   reported like every other test. Opt-in flag, same pattern as `--live`.
2. **Configuration is `ai_config.json`.** The tier runs in the Vertex
   project the `platforms.vertexai` block names, exactly as a consumer
   would, and takes its test-only knobs from that block's optional
   `testing` sub-block. Nothing is read from the environment, and nothing
   project-specific is committed (`ai_config.json` is git-ignored). A
   dedicated project and a least-privilege runner identity are still the
   recommended setup; they are no longer prerequisites.
3. **No keys anywhere.** Application Default Credentials, whatever they
   resolve to on the machine (a user login, an impersonated service
   account, a workload identity). The preflight prints the principal and
   project so a mismatch with `ai_config.json` is visible before any call.
4. **Error paths are evidence, and they are free.** A request Vertex rejects
   is not billed. A 429 `RESOURCE_EXHAUSTED` naming the right base model
   proves authentication, routing, location, model ID and Model Garden
   enablement all worked. This is how Claude is verified while its quota is 0.
5. **Bounded cost, measured every run.** A session cost ledger, a hard cap,
   and a GCP budget alert.
6. **No silent skips.** Opted in but misconfigured is a failure. The only
   skips are the environmental one Djinnite cannot fix (no Claude quota
   granted), and those are listed by name in the summary.

## Model choice

| Role | Model | Why |
|---|---|---|
| Gemini, primary | `gemini-3.5-flash` @ `global` | Current flash; supports schema, thinking, history. Shared dynamic quota, so no quota request. ~$0.002 / test call. |
| Claude, functional | `claude-haiku-4-5-20251001` | Cheapest Claude ($1 / $5). A dated snapshot, so every call exercises the `-YYYYMMDD` -> `@YYYYMMDD` rewrite. Supports structured outputs and budget thinking. |
| Claude, zero-quota canary | `claude-sonnet-5-5` | Enabled in Model Garden, **no quota requested**. A call must return 429 -> `AIRateLimitError`. Once Munin-style quota is granted for it, it graduates to the 5.x tier (between_tools, always-on thinking). |

Haiku 4.5 cannot test 5.x-only behavior (adaptive default, `between_tools`,
"cannot disable"). That behavior belongs to the *model*, not the platform,
so it is verified in **direct mode** with the Anthropic key instead (see
"Direct-mode companion"), where it is cheap and needs no Vertex quota.

## Verified Google facts (2026-10-04)

Read from Google Cloud's documentation on 2026-10-04 (Vertex AI is now
documented under "Gemini Enterprise Agent Platform"). Re-verify before
relying on them after a Google pricing or quota change.

**Locations**

* **Claude Haiku 4.5** is served at `global`, `us-east5` and `europe-west1` --
  **not** at the `us` multi-region. Its model ID on Google is `claude-haiku-4-5`.
  ([model card](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/partner-models/claude/haiku-4-5))
* **Claude Sonnet 5.5 and Opus 5.5** are served at `global`, `us` and `eu`,
  not at single regions such as `us-east5`. Multi-region endpoints serve
  Claude 4.7 and later.
* **`gemini-3.5-flash`** (GA) is served at `global`, the `us` and `eu`
  multi-regions, and `northamerica-northeast1`, `europe-west2`,
  `europe-west3`, `asia-northeast1`, `asia-south1`, `asia-southeast1` and
  `australia-southeast1` -- **not** at `us-central1`, Gemini's Vertex default
  before 0.5.0. (Checked 2026-10-05.)
  ([model card](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-5-flash))

**Pricing**

* **Claude:** `us` costs 1.1x `global` -- Sonnet 5.5 $2.00 / $10.00 at
  `global`, $2.20 / $11.00 at `us`; Opus 5.5 $4.00 / $20.00 and $4.40 / $22.00.
* **Gemini:** the 10% non-global premium applies to **GA Gemini 3 and later
  models, from 2026-07-01**, and `us-central1` counts as non-global. It does
  not apply to every Gemini family. `gemini-3.5-flash` is $1.50 / $9.00 per 1M
  tokens at `global` and $1.65 / $9.90 elsewhere; AI Studio (direct mode)
  charges $1.50 / $9.00 too, thinking billed at the output rate.
  ([pricing](https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing))

**Permissions and org policy**

* Enabling partner (Anthropic) models in Model Garden needs
  `roles/consumerprocurement.entitlementManager`; calling them needs
  `roles/aiplatform.user` (a separate role).
  ([partner-model permissions](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/partner-models/use-partner-models))
* Org-policy values name models by the **bare** model ID with publisher and
  action: `publishers/anthropic/models/claude-haiku-4-5:predict` (Model
  Garden access).
  ([control model access](https://cloud.google.com/vertex-ai/generative-ai/docs/control-model-access))
* Structured outputs and web search for Claude are **denied by default for
  projects in an organization** by `constraints/vertexai.allowedPartnerModelFeatures`;
  allow `publishers/anthropic/models/MODEL_NAME:structured_outputs` (and
  `:web_search`) per model. Structured outputs are supported on all Claude
  4.5+ models.

**Quotas and token counting**

* Haiku 4.5 (launched before Google's 2026-05-26 cutoff for shared-lineage
  quotas) has per-endpoint QPM metrics: `global_online_prediction_requests_per_base_model`
  (global), `us_multi_region_online_prediction_requests_per_base_model` (`us`),
  `online_prediction_requests_per_base_model` (regional, e.g. `us-east5`); plus
  input and output TPM quotas with the same prefixes and
  `input_tokens_per_minute_per_base_model` / `output_tokens_per_minute_per_base_model`.
  Claude models launched after the cutoff use shared-lineage quotas at global
  and multi-region endpoints.
  ([Claude quotas](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/partner-models/claude/quotas))
* Claude **count-tokens is free and has its own quota** (default 2,000
  requests per minute), separate from generation. It is offered at `global`
  and `us` only -- **not** at `us-east5`.
  ([count tokens](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/partner-models/claude/count-tokens))

**Not verified:** which web-search tool type strings Vertex accepts for
Claude. Google's example sends `"type": "web_search"`; Anthropic's docs say
only `web_search_20250305` works on Google Cloud, which is what Djinnite
sends. Extended test X3 settles it.

### Consequences (open -- not yet acted on)

1. **Gemini platform pricing is wrong for Gemini 3+ off `global`.** The
   registry applies no Gemini premium, so `gemini-3.5-flash` anywhere but
   `global` (for example `us`) is under-reported by 10%. Because the premium depends on the model (GA
   Gemini 3+), it cannot be a provider-wide rule in `platforms.py`; it needs a
   per-model, catalog-driven source.
2. **Count-tokens cannot prove availability everywhere.** It doesn't exist at
   `us-east5`, so `probe_availability()` and Claude's `is_available()` report
   Haiku there wrongly. And because count-tokens has its own quota,
   `available` never implies generation quota -- the harness must measure
   generation separately (as `claude_quota` already does at `global`).
3. **Haiku's regional-premium test (C8) must use `us-east5`**, picked from a
   generation attempt, not from count-tokens discovery; Sonnet 5.5 covers the
   `us` multi-region.
4. **Resolved 2026-10-05: Gemini's default location is now `global`.**
   `gemini-3.5-flash` is not served at `us-central1`, the old default, so
   testing that default live needed a model weeks from retirement. G6, which
   did, was removed; the offline suite pins the new default, and discovery
   still token-counts Gemini at `us`.
5. **Step 8/9 below** use the metric names and policy values above.

## Configuration

Everything comes from `config/ai_config.json`:

```json
{
  "platforms": {
    "vertexai": {
      "project_id": "<the operator's Vertex project>",
      "quota_project": "<the project to bill, or null>",
      "locations": ["global", "us"],
      "testing": {
        "gemini_model": "gemini-3.5-flash",
        "claude_model": "claude-haiku-4-5-20251001",
        "claude_canary": "claude-sonnet-5-5",
        "claude_5x_model": null,
        "decoy_project": null,
        "max_cost": 0.50
      }
    }
  }
}
```

| Key | Default | Meaning |
|---|---|---|
| `project_id` | **required** | the Vertex project; opting in without it is a `UsageError` |
| `quota_project` | unset | project billed for the calls (test G7 uses it, else `project_id`) |
| `locations` | **must include `global`** | locations the availability discovery checks |
| `testing.gemini_model` | `gemini-3.5-flash` | |
| `testing.claude_model` | `claude-haiku-4-5-20251001` | |
| `testing.claude_canary` | `claude-sonnet-5-5` | an enabled Claude model with **no** quota (must answer 429) |
| `testing.claude_5x_model` | unset -> X4 skips | a 5.x Claude with quota (extended tier) |
| `testing.decoy_project` | unset -> G8 and C3 skip | an existing project the credentials have **no** rights on |
| `testing.max_cost` | `0.50` | session hard cap, dollars |

A missing optional key skips the tests that need it, **by name**, in the
summary (`-rA`). A missing required key fails the session at startup. The
`testing` block is read only by the test harness (`tests/_platform_live.py`);
runtime ignores it. `PlatformConfig.testing` carries it through the loader.

## Credentials

Application Default Credentials, resolved by google-auth as for any consumer:

```powershell
gcloud auth application-default login            # a user login, or
gcloud auth application-default login --impersonate-service-account=<runner SA>
```

The preflight prints what they resolve to
(`[INFO] platform credentials: <type> principal=<who> adc_project=<p> adc_quota_project=<q>`).
If `adc_project` is not the project `ai_config.json` names, the calls still
go to the configured project (Djinnite passes `project_id` explicitly), but
the quota project on the credentials is whatever ADC carries unless
`quota_project` is set -- which is why G7 exists. To keep a test identity
separate from a consumer's, point `GOOGLE_APPLICATION_CREDENTIALS` at the
impersonated ADC file for the shell that runs pytest; Djinnite adds no
mechanism of its own.

## GCP setup (recommended, not required)

The base setup is the user guide, **[VERTEX_AI_SETUP.md](VERTEX_AI_SETUP.md)**.
A dedicated project with a least-privilege runner service account is the
recommended way to run this tier; the choices below are what the matrix
assumes. Setting a project up this way also tests the guide: a step the
guide gets wrong shows up here first.

Every account, project and billing value is **configuration you supply** --
nothing below names a real one:

| Variable | What to supply | Where it goes |
|---|---|---|
| `$PROJECT` | the Vertex project ID | `platforms.vertexai.project_id` |
| `$DECOY` | an **existing** project ID the runner has no rights on | `platforms.vertexai.testing.decoy_project` |
| `$BILLING` | the billing account ID to charge | -- |
| `$OPERATOR` | the operator's Google account, as `user:<email>` | -- |
| `$RUNNER` | a name for the runner service account | -- |
| `$GCLOUD_DIR` | a separate gcloud config directory for the runner's credentials | `GOOGLE_APPLICATION_CREDENTIALS` in the shell that runs pytest |

```powershell
$PROJECT    = "<e2e project id>"
$DECOY      = "<decoy project id>"
$BILLING    = "<billing account id>"
$OPERATOR   = "user:<operator email>"
$RUNNER     = "<runner service account name>"
$GCLOUD_DIR = "<directory for the runner's gcloud config>"
$SA         = "$RUNNER@$PROJECT.iam.gserviceaccount.com"
```

**A. Create the project** (the guide assumes one exists):

```powershell
gcloud projects create $PROJECT
gcloud billing projects link $PROJECT --billing-account=$BILLING
```

The decoy is **not** created. Projects count against an account's project
quota, so use any existing project the runner service account has no rights
on. The decoy tests only name it as the quota project and expect a 403;
nothing is created, enabled or billed there.

**B. Follow VERTEX_AI_SETUP.md steps 1-6 for `$PROJECT`**, with these choices
(the models and locations are those in "Model choice" above and in
`testing`):

| Guide step | e2e choice |
|---|---|
| 1. Sign in, check project | `$PROJECT` is the project created in A |
| 2. Vertex AI API | also enable `iamcredentials.googleapis.com` (impersonation) |
| 3. Model Garden | enable Haiku 4.5 and Sonnet 5.5 |
| 4. Org policy | the guide's file, with the `claude-haiku-4-5` and `claude-sonnet-5-5` lines (`:structured_outputs` and `:web_search`) |
| 5. Quota | request Haiku 4.5 quota at `global`, plus `us-east5` (regional `online_prediction_requests_per_base_model`) for the regional-premium test. **Do not** request quota for Sonnet 5.5: it is the 429 canary. New projects are often refused Claude quota; until it is granted the Claude functional tests skip by name and the plumbing tests still run. |
| 6. Identity | 6a, then 6c **"As a service account"**, with `$RUNNER` as the service account (`roles/aiplatform.user`), the operator as its Token Creator, and the credentials kept in `$GCLOUD_DIR` (next section) |

**C. Test-tier additions:**

```powershell
# The runner names the project as its quota project (test G7).
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:$SA" `
    --role="roles/serviceusage.serviceUsageConsumer"

# Budget alert (alerts only -- GCP budgets do not stop spending)
gcloud billing budgets create --billing-account=$BILLING --display-name="$PROJECT" `
    --budget-amount=10USD --filter-projects="projects/$PROJECT" `
    --threshold-rule=percent=0.5 --threshold-rule=percent=1.0
```

Resulting grants:

| Grant | To | Why |
|---|---|---|
| `roles/aiplatform.user` | runner SA, project | generate, count tokens, model listing |
| `roles/serviceusage.serviceUsageConsumer` | runner SA, project | lets the runner name the project as its quota project (`x-goog-user-project`) |
| *(nothing)* | runner SA, decoy project | deliberately absent: naming the decoy as quota project must fail with 403 |
| `roles/iam.serviceAccountTokenCreator` | the operator, on the runner SA | local runs impersonate the runner -- no key files |
| `roles/consumerprocurement.entitlementManager` | the operator, project | enable Anthropic models in Model Garden |
| `roles/orgpolicy.policyAdmin` | the operator (usually at the org) | set the structured-outputs org policy |

**No CI.** The tier is run manually, by a maintainer or an agent (see
"Who runs it"). Nothing runs it on push or on a schedule.

To keep a dedicated test identity separate from any other ADC on the
machine, give it its own gcloud config directory:

```powershell
$env:CLOUDSDK_CONFIG = $GCLOUD_DIR
gcloud auth application-default login --impersonate-service-account=$SA
Remove-Item Env:CLOUDSDK_CONFIG
# -> $GCLOUD_DIR\application_default_credentials.json
#    (type "impersonated_service_account"; google-auth reads it natively)
```

and point `GOOGLE_APPLICATION_CREDENTIALS` at that file in the shell that
runs pytest. The default `%APPDATA%\gcloud` credentials are not touched.

## Harness

### Invocation

```powershell
uv run pytest tests/ --live-platform -rA -s 2>&1 | tee $env:TEMP\platform.log                          # default tier
uv run pytest tests/ --live-platform --live-platform-extended -rA -s 2>&1 | tee $env:TEMP\platform.log  # + extended
```

Without `--live-platform` the platform tests are skipped ("rerun with
--live-platform"), exactly like `--live`, so a plain `uv run pytest tests/`
stays offline and free. Opted in with no `platforms.vertexai` block, no
`project_id`, or no ADC resolvable: the session **fails** at startup with
the exact missing piece named. Never a skip.

### Session fixtures (`tests/_platform_live.py`, `tests/conftest.py`)

1. **Preflight.** Resolve ADC; print credential type, principal (SA email when
   impersonating), the configured project and quota project, locations. ASCII only.
2. **Location discovery.** `probe_availability()` for every (provider, model,
   location) -- token counts, unbilled. Printed as a table; tests take their
   locations from it. Required: Gemini primary `available` at `global`, else
   the session fails (misconfiguration, not environment).
3. **Cost ledger.** Every billed test records its `AIResponse`. At session end
   the ledger prints per-test tokens and `total_cost`, and the session fails
   if the total exceeds `testing.max_cost`. This also cross-checks
   Djinnite's own cost math against the token counts Vertex returned.
4. **One retry on 429 for Gemini only** (shared dynamic quota can 429 under
   global load). Never for Claude: a Claude 429 is a result, not noise.

## Test matrix

Billed = generates tokens. Everything else is a rejected request or a token
count, which Google does not bill.

### Gemini on Vertex (default tier)

| # | Test | Asserts | Billed |
|---|---|---|---|
| G1 | keyless construct + `generate` @ global | non-empty text; `usage` tokens > 0; `token_cost` > 0; no `price_multiplier` | yes |
| G2 | `generate_json` | `json.loads` passes the schema | yes |
| G3 | `generate_json` + `history` (two exchanges) | recalls facts given only in history (e.g. 7 and "blue") | yes |
| G4 | `thinking=False`, then `thinking="low"` | thinking_tokens 0/None, then > 0; `token_cost` == input + (output + thinking) x rate | yes (2) |
| G5 | `max_output_tokens=5` | `AIOutputTruncatedError` with partial usage | yes (tiny) |
| G6 | *(removed 2026-10-05: tested the old region-specific default location)* | | |
| G7 | `quota_project` = the configured quota project (else the project) | succeeds (credentials carry the quota project) | yes (tiny) |
| G8 | `quota_project` = decoy | `AIAuthenticationError` whose message names the decoy -- proves the quota project actually travels. Skips by name without `testing.decoy_project` | no |
| G9 | unknown model ID | `AIModelNotFoundError` | no |
| G10 | missing ADC (`GOOGLE_APPLICATION_CREDENTIALS` -> nonexistent file, in a subprocess) | `AIAuthenticationError` | no (no network) |
| G11 | `is_available()`, `list_models()` with no key | True; list contains the primary model | no |

### Claude on Vertex -- plumbing (default tier, needs no quota)

| # | Test | Asserts | Billed |
|---|---|---|---|
| C1 | keyless construct + `generate_json` with Haiku @ global | **either** success **or** `AIRateLimitError` naming `anthropic-claude-haiku-4-5` -- **never** auth / not-found. Either outcome proves ADC, routing, location, the `@` model ID and Model Garden enablement. | only if quota |
| C2 | canary (Sonnet 5.5, zero quota) | `AIRateLimitError`, `RESOURCE_EXHAUSTED` in message | no |
| C3 | `quota_project` = decoy (`default_headers` path) | `AIAuthenticationError` naming the decoy. Skips by name without `testing.decoy_project` | no |
| C4 | unknown model ID | `AIModelNotFoundError` mentioning Model Garden | no |
| C5 | missing ADC (subprocess) | `AIAuthenticationError` | no |
| C6 | `probe_availability()` / `is_available()` | status in vocabulary; records whether `count_tokens` is quota-gated (an observation the docs need) | no |

### Claude on Vertex -- functional (runs where discovery found Haiku `available`)

| # | Test | Asserts | Billed |
|---|---|---|---|
| C7 | `generate` + `generate_json` @ global | schema-valid; usage; no multiplier | yes |
| C8 | same @ first available non-global location | `price_multiplier == 1.10`; `token_cost` == formula x 1.10 | yes |
| C9 | `history` recall | as G3 | yes |
| C10 | `thinking=False` | accepted (Haiku takes `disabled`) | yes |
| C11 | `thinking=2048` | succeeds; `thinking_tokens` is `None` or > 0; **prints which** -- settles whether Vertex sends `output_tokens_details` | yes |
| C12 | truncation | `AIOutputTruncatedError` | yes (tiny) |

If Haiku is `no_quota` everywhere, C7-C12 skip with
`"no Claude quota in project <project> ..."`, listed in the summary. C1-C6
still run.

### Scripts (default tier)

| # | Test | Asserts | Billed |
|---|---|---|---|
| S1 | `probe_platform --write` against a **temp copy** of the catalog (subprocess, `-u`) | exit 0; ASCII output; written `platforms.vertexai.locations` equals the discovery table; real catalog untouched | no |

### Access paths (default tier)

Named entries and `deny` (ACCESS_PATHS_DESIGN.md) through a real build. A
temporary `ai_config.json` (pytest `tmp_path`) on the configured project
declares two platform entries: `claude-vertex` (`"provider": "claude"`,
default model `testing.claude_model`, `deny` `{"*": ["web_search"], <Haiku>:
["structured_json"]}`, a `deny_reason`) and `gemini-vertex`
(`"provider": "gemini"`, default model `testing.gemini_model`, no `deny`).

| # | Test | Asserts | Billed |
|---|---|---|---|
| A1 | `build_provider` for both entries; `deny` per model | each builds with `entry` set, platform mode, no key; `probe_availability()` (token count) returns a known status. `claude-vertex` `generate(web_search=True)` raises `DjinniteCapabilityDeniedError` for Haiku **and** the canary (`"*"`); `generate_json` on Haiku raises it (`structured_json`). Denied calls run with the SDK client replaced by a tripwire, so they provably never leave the process; the error names the entry, model and `deny_reason`. `generate_json` on the canary is **not** denied: it reaches Vertex and gets the canary's 429 (`AIRateLimitError`). Nothing is recorded in the cost ledger unless the canary unexpectedly answers. | no |

### Extended tier (`--live-platform-extended`, needs a go-ahead per run)

| # | Test | Billed |
|---|---|---|
| X1 | `probe_platform --capabilities` for Gemini primary (and Haiku if available), temp catalog; `[DIFF]` lines printed | ~10 small calls each |
| X2 | Gemini `web_search=True` (grounding) | per grounded prompt |
| X3 | Claude `web_search=True` (`web_search_20250305`) -- only if org policy allows | per search |
| X4 | Sonnet 5.5 on Vertex: `between_tools`, `thinking=False` rejected locally, effort | only once it has quota |

## Direct-mode companion (model behavior, not platform)

5.x thinking semantics are properties of the model, verifiable cheaply with
the Anthropic key under the existing `--live` flag. A new
`tests/test_live_claude_thinking.py`:

* `thinking={"type": "disabled"}` on Sonnet 5.5 -> 400 (unbilled);
* `thinking="between_tools"` on Sonnet 5.5 -> accepted, `generate_json` works;
* `thinking=None` on Sonnet 5.5 -> `thinking_tokens > 0` (adaptive by default);
* Sonnet 5 `thinking=False` -> `thinking_tokens == 0` (disabled really is off);
* `output_tokens_details.thinking_tokens` reported in direct mode.

Together with the Vertex suite, this splits verification cleanly: the model
behaves as documented (direct), and Djinnite delivers it through the platform
(e2e).

## Cost

| Tier | Billed calls | Per run |
|---|---|---|
| Default, Claude without quota | ~8 Gemini | ~$0.02 |
| Default, Claude with quota | + ~7 Haiku | ~$0.03-0.05 |
| Extended | + capability probes, grounding | ~$0.10-0.30 |

The $0.50 session cap is ten times the expected default cost: generous
enough not to flake, small enough to stop a runaway loop.

## Coverage of 0.5.0 platform features

| Feature | Verified by |
|---|---|
| Keyless `get_provider` / ADC | G1, C1 |
| Location routing (global / multi-region) | G1, C1, C8, discovery |
| Gemini default location `global` | offline (`test_default_location_is_global`) |
| Quota project travels (both mechanisms) | G7, G8, C3 |
| `@` dated-snapshot rewrite | C1 (Haiku is dated) |
| Error mapping 429 / 403 / 404 / ADC | C2, G8/C3, G9/C4, G10/C5 |
| `generate_json` on platform | G2, C7 |
| `history` on platform | G3, C9 |
| Thinking accounting on platform | G4, C11 |
| Regional price multiplier | C8 |
| `is_available` / `list_models` keyless | G11, C6 |
| `probe_platform` | S1, X1 |
| Web search on platform | X2, X3 |
| 5.x thinking semantics | direct-mode companion; X4 once quota exists |
| Named entries / deny via `build_provider` | A1 |

**Not covered until Claude quota exists:** C7-C12 and X4. C1-C6 still
prove the Claude platform plumbing.

## Who runs it

AGENTS.md rule (adopted): **a change to platform-mode code is not finished
until `uv run pytest tests/ --live-platform` passes**, run by the agent doing
the work, not handed to the user. The default tier needs no go-ahead: it is
capped at `testing.max_cost` ($0.50 by default) and runs only in the project
`ai_config.json` names. The extended tier, and any run with a different
`ai_config.json`, needs the user's go-ahead.

## Implementation

| File | Contents |
|---|---|
| `tests/conftest.py` | `--live-platform` / `--live-platform-extended`; gating; opt-in without config is a `UsageError`; session fixtures `platform_session` (preflight), `platform_availability`, `platform_ledger` (cap); the `--live` `provider` fixture builds each type's direct-mode entry (`AIConfig.direct_entry`) |
| `tests/_platform_live.py` | `PlatformLiveConfig` (from `ai_config.platforms.vertexai` and its `testing` block), `missing_config`, credential description, `CostLedger`, Gemini 429 retry, missing-ADC subprocess probe |
| `tests/_contract.py` | the call-shape contract, shared by direct live and platform live |
| `tests/test_live_vertexai_gemini.py` | G1-G11, X2 |
| `tests/test_live_vertexai_claude.py` | C1-C12, X3, X4 |
| `tests/test_live_probe_platform.py` | S1, X1 |
| `tests/test_live_platform_access_paths.py` | A1 (temporary ai_config with named platform entries and `deny`) |
| `tests/test_live_contract.py` | the contract in direct mode (`--live`) |
| `tests/test_live_claude_thinking.py` | 5.x thinking semantics in direct mode (`--live`) |
| `scripts/smoke_platform.py` | removed |

`testing.claude_5x_model` (extended X4) should be `claude-sonnet-5-5` once
it has quota -- and then the canary must move to another enabled,
zero-quota model, since one model cannot be both.

## Decisions (2026-10-04, amended 2026-11-06)

1. **Dedicated project** -- adopted 2026-10-04 as a prerequisite, with its
   own `DJINNITE_E2E_*` environment; **amended 2026-11-06 (Brian):** the
   tier is driven by `ai_config.json` like every other part of Djinnite,
   runs in whatever Vertex project the `platforms` block names, and a
   dedicated project is a recommendation. Why: the environment-variable
   harness and its provisioning were a second configuration surface that
   nobody set up, so the tier never ran and the AGENTS.md rule that depends
   on it was unsatisfiable for a month. The name changed from "e2e" to
   "live platform" at the same time, because "e2e" did not say what was
   being tested (Vertex).
2. **Agents run the default tier** as part of finishing platform work, under
   the $0.50 cap -- adopted.
3. **`smoke_platform.py` retired** -- adopted, with the direct-live layer so
   verification does not rest on Vertex alone.
4. **No CI** -- testing is manual.
