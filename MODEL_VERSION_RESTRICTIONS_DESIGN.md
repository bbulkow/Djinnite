# Model availability by SDK and API version -- Design / Open Task

**Status:** OPEN -- recorded 2026-11-06, not started. Raised by Brian during
the anthropic 2.x / google-genai 3.x migration (DEVELOPMENT.md "Breaking
Changes Log", 0.7.0).

**Why it exists:** a model can be in the catalog and still be unusable in a
particular consumer's process, because the consumer's process runs an SDK
version, or talks to an API version, that does not serve it. Today Djinnite
finds that out at call time, as a provider error, and `list_models()`
cannot say which catalog models the running process can actually reach.

## What was observed on 2026-11-06

* **Gemini API version.** With google-genai 3.x's default API version
  (`v1`), 28 of the 46 catalog Gemini models -- every `-preview` and
  `-latest` model -- are unlisted by `models.list` **and** uncallable
  (`404 NOT_FOUND: models/gemini-3.1-pro-preview is not found for API
  version v1`). Structured output and the Google Search grounding tool are
  rejected on `v1` as well (`400 INVALID_ARGUMENT: JSON mode is not enabled
  for api version v1` / `Search as tool is not enabled for api version v1`),
  although Google's feature table lists both under `v1`. On
  `v1beta` all 46 are listed and callable. Djinnite now pins `v1beta` for
  direct access (`GeminiProvider.DIRECT_API_VERSION`), but an operator who
  sets `GOOGLE_GENAI_API_VERSION=v1` is back in that situation with no
  warning from Djinnite.
* **Claude API version.** anthropic 2.x sends API version `2026-10-21`
  (header on direct, `anthropic_version` body field on Vertex), under which
  `temperature` is rejected on every model. anthropic 1.x sends
  `2023-06-01`, under which the 4.5/4.6 family accepts it. A capability can
  therefore depend on the SDK major, not only on the model. Djinnite
  resolved this one by never sending the parameter, but the next such
  difference may not have a "never" answer.
* **SDK vocabulary.** A new model may need a value the installed SDK does
  not have (a `ThinkingLevel` member, a tool type string, a new
  `FinishReason`), so a catalog refreshed with a newer SDK can describe
  models the consumer's older SDK cannot drive.

The 0.7.0 rule "floors, never ceilings" makes this more likely, not less:
Djinnite deliberately runs on whatever SDK major the consumer resolved.

## Requirements (as stated, to be refined)

1. The catalog can record, per model, a **restriction on the running
   environment**: a minimum (or exact) SDK version per provider type, and
   the API version(s) the model is served on. Discovery should fill in
   what it can (for Gemini, which API version listed the model; for
   Claude, the API version the probe ran under), and `model_overrides.json`
   can pin the rest.
2. At **call time**, a provider checks the running SDK version and the API
   version it is configured to use against the model's restriction, and
   fails before any HTTP call with a Djinnite error that names the model,
   the restriction and what is running -- not a provider 404 after the
   fact.
3. **Iteration over models** (`list_models()`, `AIConfig`-level helpers,
   `validate_ai`) distinguishes "in the catalog" from "usable here", and
   flags the models the calling project cannot use and why, instead of
   listing them as available.

## Open questions

* Where the running SDK version comes from (`importlib.metadata.version`)
  and how the API version in use is read back from each SDK client
  (google-genai exposes `_api_client._http_options.api_version`; anthropic
  exposes nothing public for the version it sends).
* Whether the restriction is a new `ModelCapabilities` field or a sibling
  block on the model entry (it is not a capability of the model; it is a
  fact about the serving environment), and how `update_models` discovers
  and preserves it. See AGENTS.md "The catalog is generated".
* What `probe_platform` records for Vertex, where the API version is
  `v1beta1` and the SDK major differs again.
* Whether a consumer can declare "I run SDK X" up front (an `AIConfig`
  field) so the check works without importing the SDK, for example in
  `validate_ai`.

## Sequencing

Design first (this document, reviewed), then catalog schema, then the
call-time check, then the listing. Nothing here changes the 0.7.0 public
API; the error raised at call time should be a `DjinniteCapabilityDeniedError`
subclass or sibling so existing handlers keep working.
