# Platform Mode End-to-End Tests -- Design

> **Status:** Accepted 2026-10-04; harness implemented, GCP project pending
> **Date:** 2026-10-04
> **Scope:** verify platform mode (Google Vertex AI) inside Djinnite, against
> the real service, as part of Djinnite's own test suite.

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
| Platform e2e | `--e2e-platform` | Vertex AI, dedicated project | the same contract through the platform, plus auth, quota project, routing, real error bodies, `probe_platform` |

A contract failure in both live layers is a Djinnite bug; in only one, it
is an access-path bug (or a provider/platform difference worth recording).

## Principles

1. **Pytest, not a script.** Live tests are collected, asserted, counted, and
   reported like every other test. Opt-in flag, same pattern as `--live`.
2. **Own project, least privilege.** A dedicated GCP project and service
   account. Nothing runs against a consumer's project (Munin) by default.
3. **No keys anywhere.** Service-account impersonation locally; Workload
   Identity Federation in CI. No downloaded SA key files.
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
| Gemini, legacy default location | `gemini-2.5-flash` @ `us-central1` | Proves the unchanged default location still works. A model served in us-central1 is required; 3.5 Flash is not. |
| Claude, functional | `claude-haiku-4-5-20251001` | Cheapest Claude ($1 / $5). A dated snapshot, so every call exercises the `-YYYYMMDD` -> `@YYYYMMDD` rewrite. Supports structured outputs and budget thinking. |
| Claude, zero-quota canary | `claude-sonnet-5-5` | Enabled in Model Garden, **no quota requested**. A call must return 429 -> `AIRateLimitError`. Once Munin-style quota is granted for it, it graduates to the 5.x tier (between_tools, always-on thinking). |

Haiku 4.5 cannot test 5.x-only behavior (adaptive default, `between_tools`,
"cannot disable"). That behavior belongs to the *model*, not the platform,
so it is verified in **direct mode** with the Anthropic key instead (see
"Direct-mode companion"), where it is cheap and needs no Vertex quota.

## GCP setup (one time)

Run as a user with Owner on the billing account's org/folder (or equivalent).
PowerShell; set the variables first.

```powershell
$PROJECT = "djinnite-e2e"            # dedicated test project
$DECOY   = "djinnite-e2e-decoy"      # empty project the runner has NO rights on
$BILLING = "XXXXXX-XXXXXX-XXXXXX"    # billing account id
$ME      = "user:brian@bulkowski.org"
$SA      = "djinnite-e2e-runner@$PROJECT.iam.gserviceaccount.com"

# 1. Projects
gcloud projects create $PROJECT
gcloud billing projects link $PROJECT --billing-account=$BILLING
gcloud projects create $DECOY          # no billing, no APIs, no grants -- by design

# 2. APIs on the test project
gcloud services enable aiplatform.googleapis.com iamcredentials.googleapis.com --project=$PROJECT

# 3. The runner identity (no keys are ever created for it)
gcloud iam service-accounts create djinnite-e2e-runner --project=$PROJECT `
    --display-name="Djinnite e2e runner"

# 4. Runner grants: call models; use the project for quota/billing
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:$SA" --role="roles/aiplatform.user"
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:$SA" --role="roles/serviceusage.serviceUsageConsumer"

# 5. Human grants: impersonate the runner; enable partner models in Model Garden
gcloud iam service-accounts add-iam-policy-binding $SA --project=$PROJECT `
    --member=$ME --role="roles/iam.serviceAccountTokenCreator"
gcloud projects add-iam-policy-binding $PROJECT --member=$ME `
    --role="roles/consumerprocurement.entitlementManager"

# 6. Budget alert (alerts only -- GCP budgets do not stop spending)
gcloud billing budgets create --billing-account=$BILLING --display-name="djinnite-e2e" `
    --budget-amount=10USD --filter-projects="projects/$PROJECT" `
    --threshold-rule=percent=0.5 --threshold-rule=percent=1.0
```

| Grant | To | Why |
|---|---|---|
| `roles/aiplatform.user` | runner SA, test project | `aiplatform.endpoints.predict` (generate, count tokens) and model listing |
| `roles/serviceusage.serviceUsageConsumer` | runner SA, test project | lets the runner name the test project as its quota project (`x-goog-user-project`) |
| *(nothing)* | runner SA, decoy project | deliberately absent: naming the decoy as quota project must fail with 403 |
| `roles/iam.serviceAccountTokenCreator` | you, on the runner SA | local runs impersonate the runner -- no key files |
| `roles/consumerprocurement.entitlementManager` | you, test project | required to enable partner (Anthropic) models in Model Garden |

**Console steps** (no CLI equivalent worth scripting):

7. **Model Garden** (test project): open *Claude Haiku 4.5* -> *Enable*, accept
   Anthropic's terms. Do the same for *Claude Sonnet 5.5* (the canary).
8. **Quota** (IAM & Admin -> Quotas): request
   `global_online_prediction_requests_per_base_model` (and the input/output
   token-per-minute quotas) for `anthropic-claude-haiku-4-5` at a small value,
   e.g. 10 QPM. **Do not** request quota for Sonnet 5.5 -- it is the 429 canary.
   Expect friction: new projects are often refused Claude quota for lack of
   usage history ("NOT_ENOUGH_USAGE_HISTORY"). The Gemini tier builds that
   history on the same project; re-request after some weeks of nightly runs.
   Until then the Claude functional tests skip by name and the Claude
   plumbing tests still run (they need no quota).
9. **Org policy:** if the organization restricts Model Garden models or
   partner-model features by org policy, allow Haiku 4.5 and Sonnet 5.5 on
   the test project. Munin's org denies web search; web-search tests are in
   the extended tier only.

**No CI.** The e2e tier is run manually, by a maintainer or an agent
(see "Who runs it"). Nothing runs it on push or on a schedule.

## Credentials on a developer machine

Keep the e2e identity separate from any consumer's ADC (Munin's lives in the
default gcloud config) by giving it its own gcloud config directory:

```powershell
$env:CLOUDSDK_CONFIG = "$HOME\.gcloud-djinnite-e2e"
gcloud auth application-default login --impersonate-service-account=$SA
Remove-Item Env:CLOUDSDK_CONFIG
# -> $HOME\.gcloud-djinnite-e2e\application_default_credentials.json
#    (type "impersonated_service_account"; google-auth reads it natively)
```

The harness points `GOOGLE_APPLICATION_CREDENTIALS` at that file for the
test session only (from `DJINNITE_E2E_CREDENTIALS`), so the default ADC is
never touched.

## Harness

### Invocation

```powershell
uv run pytest tests/ --e2e-platform -rA -s 2>&1 | tee $env:TEMP\e2e.log           # default tier
uv run pytest tests/ --e2e-platform --e2e-extended -rA -s 2>&1 | tee $env:TEMP\e2e.log  # + extended
```

Without `--e2e-platform` the e2e tests are skipped ("rerun with
--e2e-platform"), exactly like `--live`, so a plain `uv run pytest tests/`
stays offline and free.

### Configuration (environment; CI-friendly)

| Variable | Default | Meaning |
|---|---|---|
| `DJINNITE_E2E_PROJECT` | **required** | test project |
| `DJINNITE_E2E_CREDENTIALS` | ADC default | path to the impersonated ADC file |
| `DJINNITE_E2E_DECOY_PROJECT` | unset -> decoy tests fail | project the runner has no rights on |
| `DJINNITE_E2E_LOCATIONS` | `global,us` | locations to discover |
| `DJINNITE_E2E_GEMINI_MODEL` | `gemini-3.5-flash` | |
| `DJINNITE_E2E_GEMINI_LEGACY_MODEL` | `gemini-2.5-flash` | for the `us-central1` default test |
| `DJINNITE_E2E_CLAUDE_MODEL` | `claude-haiku-4-5-20251001` | |
| `DJINNITE_E2E_CLAUDE_CANARY` | `claude-sonnet-5-5` | must have zero quota |
| `DJINNITE_E2E_MAX_COST` | `0.50` | session hard cap, dollars |

Opted in with `DJINNITE_E2E_PROJECT` unset, or no ADC resolvable: the session
**fails** at the first e2e test with the exact missing piece named. Never a skip.

### Session fixtures (`tests/_e2e.py`)

1. **Preflight.** Resolve ADC; print credential type, principal (SA email when
   impersonating), project, quota project, locations. ASCII only.
2. **Location discovery.** `probe_availability()` for every (provider, model,
   location) -- token counts, unbilled. Printed as a table; tests take their
   locations from it. Required: Gemini primary `available` at `global`, else
   the session fails (misconfiguration, not environment).
3. **Cost ledger.** Every billed test records its `AIResponse`. At session end
   the ledger prints per-test tokens and `total_cost`, and the session fails
   if the total exceeds `DJINNITE_E2E_MAX_COST`. This also cross-checks
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
| G6 | legacy: `backend="vertexai"`, no location | client location is `us-central1`; call succeeds with the legacy model | yes |
| G7 | `quota_project` = test project | succeeds (credentials carry the quota project) | yes (tiny) |
| G8 | `quota_project` = decoy | `AIAuthenticationError` whose message names the decoy -- proves the quota project actually travels | no |
| G9 | unknown model ID | `AIModelNotFoundError` | no |
| G10 | missing ADC (`GOOGLE_APPLICATION_CREDENTIALS` -> nonexistent file, in a subprocess) | `AIAuthenticationError` | no (no network) |
| G11 | `is_available()`, `list_models()` with no key | True; list contains the primary model | no |

### Claude on Vertex -- plumbing (default tier, needs no quota)

| # | Test | Asserts | Billed |
|---|---|---|---|
| C1 | keyless construct + `generate_json` with Haiku @ global | **either** success **or** `AIRateLimitError` naming `anthropic-claude-haiku-4-5` -- **never** auth / not-found. Either outcome proves ADC, routing, location, the `@` model ID and Model Garden enablement. | only if quota |
| C2 | canary (Sonnet 5.5, zero quota) | `AIRateLimitError`, `RESOURCE_EXHAUSTED` in message | no |
| C3 | `quota_project` = decoy (`default_headers` path) | `AIAuthenticationError` naming the decoy | no |
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
`"no Claude quota in project djinnite-e2e (haiku: no_quota @ global,us)"`,
listed in the summary. C1-C6 still run.

### Scripts (default tier)

| # | Test | Asserts | Billed |
|---|---|---|---|
| S1 | `probe_platform --write` against a **temp copy** of the catalog (subprocess, `-u`) | exit 0; ASCII output; written `platforms.vertexai.locations` equals the discovery table; real catalog untouched | no |

### Extended tier (`--e2e-extended`, needs a go-ahead per run)

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
| Gemini default location unchanged | G6 |
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

**Not covered until Claude quota exists:** C7-C12 and X4. C1-C6 still
prove the Claude platform plumbing.

## Who runs it

AGENTS.md rule (adopted): **a change to platform-mode code is not finished
until `uv run pytest tests/ --e2e-platform` passes**, run by the agent doing
the work, not handed to the user. The default tier needs no go-ahead: it is
capped at $0.50 and runs only in the dedicated project. The extended tier,
and any run against another project, needs the user's go-ahead.

## Implementation

| File | Contents |
|---|---|
| `tests/conftest.py` | `--e2e-platform` / `--e2e-extended`; gating; opt-in without config is a `UsageError`; session fixtures `e2e_session` (credentials + preflight), `e2e_availability`, `e2e_ledger` (cap) |
| `tests/_e2e.py` | `E2EConfig` (DJINNITE_E2E_*), credential description, `CostLedger`, Gemini 429 retry, missing-ADC subprocess probe |
| `tests/_contract.py` | the call-shape contract, shared by direct live and platform e2e |
| `tests/test_e2e_vertexai_gemini.py` | G1-G11, X2 |
| `tests/test_e2e_vertexai_claude.py` | C1-C12, X3, X4 |
| `tests/test_e2e_probe_platform.py` | S1, X1 |
| `tests/test_live_contract.py` | the contract in direct mode (`--live`) |
| `tests/test_live_claude_thinking.py` | 5.x thinking semantics in direct mode (`--live`) |
| `scripts/smoke_platform.py` | removed |

`DJINNITE_E2E_CLAUDE_5X_MODEL` (extended X4) should be `claude-sonnet-5-5`
once it has quota -- and then the canary must move to another enabled,
zero-quota model, since one model cannot be both.

## Decisions (2026-10-04)

1. **Dedicated project** `djinnite-e2e` -- adopted. It keeps tests off
   Munin's quota and budget; it starts with no Claude quota or usage history.
2. **Agents run the default tier** as part of finishing platform work, under
   the $0.50 cap -- adopted.
3. **`smoke_platform.py` retired** -- adopted, with the direct-live layer so
   verification does not rest on Vertex alone.
4. **No CI** -- testing is manual.
