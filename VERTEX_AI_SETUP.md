# Using Djinnite on Google Vertex AI

This guide sets up **platform mode**: Djinnite calls Gemini and Claude models
through Google Cloud's Vertex AI, authenticated with your Google Cloud
identity instead of provider API keys. Follow it top to bottom; each step says
what to run, why, and how to check it worked.

You need platform mode when your code runs on Google Cloud (Cloud Run, GKE,
Compute Engine) with no API keys, or when billing and data handling must go
through your Google Cloud project. If you have a Gemini or Anthropic API key
and none of that applies, direct mode (USE.md) is simpler.

## What you will end up with

* A Google Cloud project with the Vertex AI API enabled.
* For Claude: the models enabled in Model Garden, allowed by your
  organization's policy, and with quota.
* An identity your code runs as, with permission to call the models.
* `config/ai_config.json` (or code) telling Djinnite to use the platform.

Throughout, replace `my-project` and the other example values with your own.

## Before you start

* A Google Cloud project with billing enabled (step 1 checks it).
* The Google Cloud CLI (`gcloud`) installed.
* Enough permission on the project to enable APIs and grant roles (Owner
  works). Two one-time steps need roles that Owner may not include -- they
  are called out where they come up.

Run everything in one PowerShell window. Step 1 sets `$PROJECT`, an ordinary
PowerShell variable, and every command after it names the project with
`$PROJECT`. Nothing changes gcloud's saved default project, the environment,
or any other shell. If you open a new window partway through, set `$PROJECT`
there first.

## Step 1. Sign in and check the project

See which account gcloud will use:

```powershell
gcloud auth list
```

The account marked `*` runs the commands in this guide. If the account you
need is not listed, add it with `gcloud auth login` (this also makes it the
active account; `gcloud config set account <email>` switches back later).

Then set `$PROJECT` and check the project:

```powershell
$PROJECT = "my-project"
gcloud projects describe $PROJECT --format="value(projectId,lifecycleState)"
gcloud billing projects describe $PROJECT --format="value(billingEnabled)"
```

Use the project ID (lowercase), not the display name the console shows.
Every gcloud command below names the project with `$PROJECT`. A command that
named no project would charge its API call to your saved default project,
and might offer to enable APIs there. If gcloud ever asks to enable an API
on a project other than yours, answer **N** and check the command. If
`$PROJECT` is not set, gcloud stops with "The project property is set to the
empty string" rather than using your default project.

**Check:** the first command prints your project ID and `ACTIVE`; the second
prints `True`. An error from the first means the ID is wrong or your account
has no access to the project. `False` from the second means the project has
no billing account yet: link one in the console under **Billing**.

## Step 2. Enable the Vertex AI API (if it is not already)

```powershell
gcloud services list --enabled --project=$PROJECT
```

If `aiplatform.googleapis.com` is not in the list, enable it:

```powershell
gcloud services enable aiplatform.googleapis.com --project=$PROJECT
```

**Check:** run the list command again; `aiplatform.googleapis.com` appears.

Gemini models need nothing more on Google's side. Claude models also need
steps 3-5.

## Step 3. Enable the Claude models in Model Garden (Claude only)

Anthropic's models are partner models: someone must enable each one in the
project and accept Anthropic's terms, once.

1. Give the person doing this the **Consumer Procurement Entitlement Manager**
   role on the project (Owner does not necessarily include it):

   ```powershell
   gcloud projects add-iam-policy-binding $PROJECT `
       --member="user:<their email>" --role="roles/consumerprocurement.entitlementManager"
   ```

2. In the console, open **Vertex AI -> Model Garden**, search for each Claude
   model your code calls (for example *Claude Sonnet 5.5*), open it and click
   **Enable**. Accept the terms.

**Check:** the model's Model Garden page no longer offers "Enable".

## Step 4. Allow Claude's features in your organization (Claude only)

If your project belongs to a Google Cloud **organization**, the org policy
`constraints/vertexai.allowedPartnerModelFeatures` denies **structured
outputs** and **web search** for Claude by default. Djinnite's
`generate_json()` uses structured outputs on every call, so without this step
Claude's `generate_json()` cannot work. (Projects with no organization skip
this step.)

Setting it needs the **Organization Policy Administrator** role
(`roles/orgpolicy.policyAdmin`), usually held at the organization level, and
the Organization Policy API enabled on the project. See what is enforced now:

```powershell
gcloud services enable orgpolicy.googleapis.com --project=$PROJECT
gcloud org-policies describe vertexai.allowedPartnerModelFeatures --project=$PROJECT --effective
```

`denyAll: true` in the output means Claude's features are blocked for this
project -- the default in an organization.

Create `claude-features.yaml` that contain the features we'll need, 
and then you'll apply it.  The policy allows every feature
Djinnite uses -- structured outputs and web search -- for each model, so the
project can do everything the models can. Keep one pair of lines per model
you enabled in step 3, and delete the `web_search` lines only if you want web
search blocked. 

```yaml
name: projects/[my-project here]/policies/vertexai.allowedPartnerModelFeatures
spec:
  rules:
  - values:
      allowedValues:
      - publishers/anthropic/models/claude-sonnet-5-5:structured_outputs
      - publishers/anthropic/models/claude-sonnet-5-5:web_search
      - publishers/anthropic/models/claude-opus-5-5:structured_outputs
      - publishers/anthropic/models/claude-opus-5-5:web_search
      - publishers/anthropic/models/claude-haiku-4-5:structured_outputs
      - publishers/anthropic/models/claude-haiku-4-5:web_search
```

Or to create this file with a command line in powershell:

```powershell
@"
name: projects/$PROJECT/policies/vertexai.allowedPartnerModelFeatures
spec:
  rules:
  - values:
      allowedValues:
      - publishers/anthropic/models/claude-sonnet-5-5:structured_outputs
      - publishers/anthropic/models/claude-sonnet-5-5:web_search
      - publishers/anthropic/models/claude-opus-5-5:structured_outputs
      - publishers/anthropic/models/claude-opus-5-5:web_search
      - publishers/anthropic/models/claude-haiku-4-5:structured_outputs
      - publishers/anthropic/models/claude-haiku-4-5:web_search
"@ | Set-Content -Encoding ascii claude-features.yaml

Get-Content claude-features.yaml | Select-Object -First 1
```

**Before applying, check the name line.** It must print
`name: projects/<your project ID>/...`. The policy lands on the project that
line names, whatever `--project` says. If it shows `$PROJECT` or
`my-project`, the file was not filled in -- write your project ID into that
line by hand.

Then apply it using the org-policies set-policy command.

 `--billing-project` charges the call itself to your project;
without it, gcloud falls back to your default project and may offer to
enable the Organization Policy API in the wrong project:

```powershell
gcloud org-policies set-policy claude-features.yaml --billing-project=$PROJECT
```

**Check:** `gcloud org-policies describe vertexai.allowedPartnerModelFeatures --project=$PROJECT`
shows your values.

If your organization also restricts which Model Garden models may be used at
all, the allowed values there take the form
`publishers/anthropic/models/claude-sonnet-5-5:predict`; ask whoever manages
your organization's policies.

## Step 5. Get Claude quota (Claude only)

A new project often has **zero** quota for Claude models. Calls then fail
immediately with a quota error, even though everything else is right.

This quota doesn't just take minutes: it currently requires a discussion
with a Google Cloud Enterprise Account Manager. Try enabling the quota, but 
be prepared to be rejected within a few minutes. The issue isn't a new project,
these quotas are gate-kept by a discussion with an Enterprise Account Manager. 
Open a chat window with Google Enterprise Sales, and be prepared for the pain
of a sales call.

In the console: **IAM & Admin -> Quotas & System Limits**, filter by the
metric below and the model's name, and request an increase for each location
your code calls it at (see "Where models are served"). The metrics for a
model like Haiku 4.5 are:

| Location | Requests per minute metric |
|---|---|
| `global` | `global_online_prediction_requests_per_base_model` |
| `us` multi-region | `us_multi_region_online_prediction_requests_per_base_model` |
| a single region (e.g. `us-east5`) | `online_prediction_requests_per_base_model` |

Each also has input- and output-token-per-minute metrics with the same prefix
(`..._input_tokens_per_minute_per_base_model`,
`..._output_tokens_per_minute_per_base_model`). Claude models released after
May 2026 use shared quotas at `global` and multi-region endpoints; the quota
page shows which metric applies.

Expect friction: Google sometimes refuses increases on new projects for lack
of usage history. Gemini needs no quota request.

## Step 6. Give your code an identity

Djinnite uses Google's **Application Default Credentials (ADC)**: whatever
identity the environment provides. No key file, no API key.

First you will create a user, but then you will follow the necessary path for
execution either in Cloud Run, or from a local machine.

That identity needs **Vertex AI User** (`roles/aiplatform.user`) on the
project. A service account is the recommended identity, in production and on
your own machine. Create one and grant it the role:

```powershell
$SA_NAME = "my-app"
$SA = "$SA_NAME@$PROJECT.iam.gserviceaccount.com"
gcloud iam service-accounts create $SA_NAME --project=$PROJECT
gcloud projects add-iam-policy-binding $PROJECT `
    --member="serviceAccount:$SA" --role="roles/aiplatform.user"
```

`my-app` is any name you choose: 6 to 30 lowercase letters, digits and
hyphens. (Running as yourself on your own machine needs no service account:
skip to **As yourself** below.)

### On Cloud Run (or GKE, Compute Engine)

Run your service as that service account. On Cloud Run, add
`--service-account=$SA` to your `gcloud run deploy` command. Nothing else:
ADC picks up the service account automatically.

### On your localmachine

Nothing here needs a machine-wide change. Your normal `gcloud` login and any
credentials other programs use stay as they are.

Credentials for Djinnite come from a file that Google's libraries look for in
this order:

1. The file named by the `GOOGLE_APPLICATION_CREDENTIALS` environment variable.
2. `application_default_credentials.json` in gcloud's config directory --
   `%APPDATA%\gcloud` by default, or the directory named by
   `CLOUDSDK_CONFIG`.

So you can keep Djinnite's credentials in a directory of their own and point
only the processes that need them at it. Choose an empty directory, outside
any code repository:

```powershell
$CREDS_DIR = "<an empty directory for these credentials>"
New-Item -ItemType Directory -Force $CREDS_DIR | Out-Null
```

Then write the credentials there, as the service account or as yourself. In
both, `CLOUDSDK_CONFIG` points gcloud at that directory for this window only
and is removed straight after the login. (Run without it, the login replaces
the machine's default credentials file instead.)

**As a service account** (recommended; matches production, no key file).
This uses the service account created above. You need **Service Account
Token Creator** on it:

```powershell
gcloud iam service-accounts add-iam-policy-binding $SA --project=$PROJECT `
    --member="user:<your email>" --role="roles/iam.serviceAccountTokenCreator"

$env:CLOUDSDK_CONFIG = $CREDS_DIR
gcloud auth application-default login --impersonate-service-account=$SA
Remove-Item Env:CLOUDSDK_CONFIG
```

A browser opens: sign in as `<your email>`. A new grant can take a few
minutes to take effect.

**As yourself.** Your Google account needs `roles/aiplatform.user` on the
project (Owner has it):

```powershell
$env:CLOUDSDK_CONFIG = $CREDS_DIR
gcloud auth application-default login
Remove-Item Env:CLOUDSDK_CONFIG
```

If Google's errors say a quota project is required, tell Djinnite which
project to bill (`quota_project`, step 7); your account then needs
`roles/serviceusage.serviceUsageConsumer` on that project (Owner and Editor
include it).

**Either way**, point the program that uses Djinnite at the file, in its
window (or its service definition). An environment variable set with `$env:`
lasts for that PowerShell window only:

```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS = "$CREDS_DIR\application_default_credentials.json"
```

**Check**, in the window that has `GOOGLE_APPLICATION_CREDENTIALS` set:

```powershell
uv run python -c "import google.auth; c, _ = google.auth.default(); print(type(c).__name__)"
```

It prints the credential type (`Credentials` for impersonated or user
credentials) rather than an error.

## Step 7. Configure Djinnite

Either in `config/ai_config.json` (your project's config directory -- see
USE.md), or directly in code.

### In ai_config.json

```json
{
  "platforms": {
    "vertexai": {
      "project_id": "my-project",
      "quota_project": "my-project",
      "locations": ["global", "us"]
    }
  },
  "providers": {
    "claude": {
      "mode": "platform", "platform": "vertexai", "location": "global",
      "enabled": true, "default_model": "claude-sonnet-5-5"
    },
    "gemini": {
      "mode": "platform", "platform": "vertexai", "location": "global",
      "enabled": true, "default_model": "gemini-3.5-flash"
    }
  },
  "default_provider": "claude"
}
```

* `platforms.vertexai` holds settings shared by every provider on Vertex.
  A provider entry may override `project_id`, `location` or `quota_project`.
* `location`: one where the model is served (see "Where models are
  served"). If you leave it out, Djinnite uses `global`.
* `quota_project`: omit it unless step 6 told you to set it.
* `locations`: the locations `probe_platform` checks (step 8); not used for
  calls.
* No `api_key` anywhere. A provider can be in platform mode while another
  stays in direct mode with its key.

Then:

```python
from djinnite import get_provider, load_ai_config

cfg = load_ai_config()
claude = get_provider("claude", model=cfg.providers["claude"].default_model,
                      **cfg.provider_kwargs("claude"))
```

### In code

```python
from djinnite import get_provider

claude = get_provider("claude", model="claude-sonnet-5-5",
                      platform="vertexai", project_id="my-project",
                      location="global")
gemini = get_provider("gemini", model="gemini-3.5-flash",
                      platform="vertexai", project_id="my-project",
                      location="global", quota_project="my-project")
```

Everything after construction -- `generate()`, `generate_json()`,
`history=`, `thinking=`, costs, errors -- works as in direct mode.

## Step 8. Verify

**1. Which models answer, where.** This makes only token-count calls, which
Google does not charge for:

```powershell
uv run python -u -m djinnite.scripts.probe_platform --platform vertexai
```

It reads `platforms.vertexai` from your ai_config.json and prints one line
per model and location:

```
  [OK] claude claude-sonnet-5-5 @global available
  [WARN] claude claude-sonnet-5-5 @us no_quota (...)
```

`available` means the model answered there; `no_quota` points at step 5,
`no_access` at step 6, and `not_found` at step 3 or "Where models are
served". Nothing is written to Djinnite's
catalog unless you add `--write`. (Claude token counting is only offered at
`global` and `us`; rows for single regions such as `us-east5` cannot be
checked this way.)

**2. A real call** (costs a fraction of a cent):

```python
import json
from djinnite import get_provider

p = get_provider("claude", model="claude-sonnet-5-5", platform="vertexai",
                 project_id="my-project", location="global")
r = p.generate_json("Name the capital of France as JSON.",
                    {"type": "object", "properties": {"city": {"type": "string"}},
                     "required": ["city"]})
print(json.loads(r.content), r.usage["total_cost"])
```

## Where models are served

Your code chooses the model and location -- in each `get_provider()` call or
in `ai_config.json` (step 7) -- and changes them as models come and go. A
location is where the request is served; each model is offered only at some
locations, and the one you configure must be one of them. Steps 3-5 are per
model, and quota (step 5) is per location as well, so repeat them when your
code starts calling a new Claude model or location.

| Model (Djinnite id) | Vertex locations | Notes |
|---|---|---|
| `claude-sonnet-5-5`, `claude-opus-5-5` | `global`, `us`, `eu` | not at single regions such as `us-east5` |
| `claude-haiku-4-5-20251001` | `global`, `us-east5`, `europe-west1` | not at `us` |
| `gemini-3.5-flash` | `global`, `us`, `eu`, and some single regions | not at `us-central1` |

* **Use `global` unless you have a reason not to.** It is the most available
  endpoint and, for Claude and for Gemini 3 and later, the cheapest:
  `us`, `eu` and single regions cost 10% more (see "Costs").
* Use `us` or `eu` when data must be processed in that geography.
* Djinnite's default location is `global` for both Claude and Gemini. If
  you set another, confirm on the model's Google page that it is served
  there.
* Model ids: use Djinnite's ids (the left column). Djinnite translates dated
  ids for Vertex (`claude-haiku-4-5-20251001` is sent as
  `claude-haiku-4-5@20251001`). Google's own pages and org-policy values use
  the short id (`claude-haiku-4-5`).

## Costs

Djinnite reports `token_cost` / `total_cost` from its model catalog, which
holds each model's standard price -- the same as Vertex's `global` price.

| | `global` | `us`, `eu`, single regions |
|---|---|---|
| Claude (Sonnet 4.5 and later) | catalog price | **+10%** -- Djinnite applies it (`usage["price_multiplier"] == 1.10`) |
| Gemini 3 and later (GA) | catalog price | **+10%** -- Djinnite does **not** yet apply it; costs are under-reported by 10% (`us-central1` counts as non-global) |
| Older Gemini models | catalog price | same price |

Token counting is free. Requests Google rejects (quota, permission, not
found) are not charged.

## When something fails

Djinnite raises its usual errors; on Vertex, the message carries Google's own
explanation.

| Error | Message mentions | Cause and fix |
|---|---|---|
| `AIAuthenticationError` | "Application Default Credentials unavailable" | No ADC on this machine or service: step 6. |
| `AIAuthenticationError` | "refused the request", `PERMISSION_DENIED` | The identity lacks `roles/aiplatform.user` (step 6); or the Vertex AI API is off (step 2); or `quota_project` names a project the identity can't use for quota. |
| `AIRateLimitError` | `RESOURCE_EXHAUSTED`, quota | Zero or exhausted quota for that model at that location: step 5. Common for Claude on new projects. |
| `AIModelNotFoundError` | "not served at this location, or not enabled in Model Garden" | Wrong location for the model (see "Where models are served"), or the Claude model isn't enabled (step 3). |
| Claude `generate()` works but `generate_json()` fails | structured outputs, or an organization policy | Structured outputs are not allowed for that model: step 4. |
| `AIProviderError` | "project_id is required" | Set `project_id` (ai_config `platforms.vertexai` or the keyword). |

For the full error contract, see USE.md "Error Handling Contract".

## Limits of platform mode

* **Claude model listing:** Vertex has no "list models" endpoint for Claude.
  `list_models()` on a Claude platform provider returns the models
  `probe_platform --write` recorded as available at that location, or nothing.
* **Availability checks at single regions:** `is_available()` and
  `probe_platform` use token counting, which Claude offers only at `global`
  and `us`.
* **Catalog maintenance** (`update_models`, `update_model_costs`) uses
  provider API keys; platform-mode entries are skipped. The catalog that
  ships with Djinnite works for platform mode as-is.
* **Claude web search on Vertex** is billed at the catalog's first-party
  price ($10 per 1,000 searches); Vertex's own price is not yet verified.
