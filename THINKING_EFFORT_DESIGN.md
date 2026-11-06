# Thinking effort: discovery and portable request semantics

Status, 2026-11-03: implemented in the working tree. The current policy
below supersedes the earlier draft and implementation that injected medium
whenever a model advertised effort. Brian explicitly approved the corrected
principles and the fixes in this review.

## Governing rule

Naming a model and provider must preserve native defaults. Supporting an
optional effort parameter is not a reason to send it.

1. Omitted effort stays omitted unless an explicit value is necessary to
   honor the request.
2. Explicit effort is a preference: map it to the nearest supported level
   compatible with the whole request, breaking ties toward the lower level.
3. Explicit thinking settings and requested features such as web search
   remain requirements. Do not discard them to make an effort fit.
4. Unknown support stays unknown. No automatic retry or runtime catalog
   mutation is introduced by this change.

Effort labels are ordinal, not equivalent token budgets or equivalent
reasoning depth across vendors. Numeric thinking budgets remain a separate
control and are never manufactured from effort labels.

## Discovery repair

The original defects were confirmed. The fixes remain:

* `effort` joins adaptive and budget as evidence of thinking on. The
  between_tools setting alone is not such evidence.
* Responses API discovery distinguishes unsupported levels from unsupported
  reasoning parameters. Unrelated errors are inconclusive and preserve
  cached capabilities; a 404 is not evidence of no reasoning.
* Accepted OpenAI levels come from the supported-values list in a recognized
  rejection. The combination probe uses a supported level near medium,
  rather than an unsupported low or a feature-restricted minimal.
* Confirmed no-thinking styles persist as `[]`; unknown remains `None`.
* Invalid artificial probe caps were removed. Claude still supplies its
  required max_tokens parameter.

The prior session's live evidence is retained below. No discovery reprobe
was performed during this correction. Grok's OpenAI-shaped error assumptions
remain unverified against xAI.

## Runtime behavior

The vocabulary is `minimal < low < medium < high < xhigh < max`.
The keyword-only `effort=` accepts these levels. `thinking="<level>"`
remains a deprecated alias for effort with thinking=None. Supplying both
that alias and effort raises ValueError.

| Request | Behavior |
|---|---|
| No thinking or effort arguments | Omit both controls; preserve native defaults |
| Explicit effort | Vendor vocabulary, model ladder, then request constraints |
| OpenAI/Grok thinking=True, toggleable or unknown model | Effort is the on control; send the mapped required-effort default |
| OpenAI/Grok thinking=True, known always-on model | No effort is necessary; preserve native default |
| Claude thinking=True | Adaptive or enabled budget block; no invented effort |
| Gemini thinking=True | Native on default for level-based or always-on models; dynamic budget for other budget models |
| thinking=False | Preserve the explicit off requirement; reject if the model cannot honor it |
| Effort on a model/state without that control | Log and disregard effort; do not fabricate a budget |

`request_policy.json` supplies a default target only when an effort is
required. The shipped target is medium. A null configured target cannot
cancel an explicit thinking=True: medium is then the fallback selection
target, still constrained to supported levels. It is never inserted into
an ordinary call. Provider-specific defaults have the same restricted scope.

Temperature behavior is unchanged by this correction: omitted temperature
uses the optional configured default (null as shipped); unsupported explicit
temperature is logged and dropped before incompatibility checks.

## Model and request constraints

The shared resolver receives thinking and web_search before SDK request
construction. It maps the preference to the model ladder, excluding levels
that violate the provider's known request restrictions before selecting the
nearest value. An empty candidate set raises AIProviderError locally.

* GPT-5, mini, nano and dated snapshots: minimal cannot accompany search;
  select low (or the nearest other allowed catalog level), retaining search.
  Scope this rule to that family rather than assuming it applies to all
  OpenAI or Grok models. Evidence: the previous session's six-model probes
  and [OpenAI web-search limitations](https://developers.openai.com/api/docs/guides/tools-web-search#limitations).
* Claude between_tools: allow only low, medium and high. An explicit xhigh
  or max maps to high while retaining between_tools. Omitted effort remains
  omitted; Sonnet 5.5's native default is high. Evidence:
  [Anthropic's Sonnet 5.5 guide](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-sonnet-5-5#running-without-up-front-thinking).
* Gemini: eleven named text models have documented levels pinned in
  `config/model_overrides.json`, applied with apply_overrides. In particular,
  3.7 Flash, 3.8 Flash, 3.1 Pro and 4 Argon have low as their minimum;
  minimal maps to low. Evidence: [Google's model-level table](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/thinking#supported-thinking-levels-by-model)
  (checked 2026-11-03) and the [explicit 3.8 validation rule](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/guides/gemini-3-8-flash#mandatory_api_rules_and_behavioral_conventions).
  The customtools endpoint uses 3.1 Pro's ladder, as it is a variant of that
  model: [Google model documentation](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-pro-preview.md).

The generated catalog is never hand-edited. Overrides retain source notes
and `_overridden` provenance. Gemini's floating latest aliases retain unknown
ladders; their current target was not established by these sources. Unknown
model ladders use the vendor vocabulary as a best effort, not as verified
per-model support. Unrecognized vendor failures still surface normally.
Recognized effort failures on OpenAI/Grok and Claude raise
DjinniteEffortMappingError. Mapping changes are logged at INFO; all native
settings are available through DJINNITE_DEBUG_REQUESTS.

## Compatibility and validation

Ordinary calls retain native effort defaults. OpenAI/Grok thinking=True on
toggleable models uses the configured target (medium as shipped), where the
old implementation sent high. To request high, pass effort="high". Explicit
levels may move up or down to preserve the other requested behavior.

Offline validation after the correction: 774 passed, 66 live/platform tests
skipped, zero failures/errors, 840 collected. Tests cover generate and
generate_json, null and non-null policy defaults, model-specific Gemini
levels, the GPT-5 search combination, the Claude between_tools combination,
and an unsatisfiable combination failing before the SDK call. Gemini and
Claude request-shape tests also exercise both direct and platform branches
with stubs. Nine inherited non-environmental budget skips now assert the
expected rejection.

Direct live validation: `uv run pytest tests/test_live_effort.py --live -rA -s`
passed all four cases using the existing ai_config direct entries:

| Model and request | Accepted | Catalog-estimated cost |
|---|---|---|
| Gemini 3.8 Flash, effort=minimal -> LOW, JSON | yes | $0.000046 |
| Sonnet 5.5, between_tools + max -> high, JSON | yes | $0.000484 |
| GPT-5, minimal + search -> low + search | yes | $0.006031 |
| GPT-5.4, thinking=True with null effort default -> medium | yes | $0.000362 |

These prove vendor acceptance of the corrected requests; they do not
measure equivalent reasoning depth or prove that an optional search tool
was used. The reported total is about $0.006923 from catalog pricing,
not a billing-statement reconciliation.

**Platform verification remains pending.** DJINNITE_E2E_* was unset in this
session. ai_config has direct API keys, but no platform entries. The
separate GCP project is an existing repository test-policy requirement,
not a missing provider API key. PLATFORM_E2E_TEST_DESIGN.md still labels
its project setup pending; Brian said this requirement was unknown to him.
Per AGENTS.md, the platform changes are not finished until the e2e tier
passes. No GCP project was created or configured in this session.
*(2026-11-06: that requirement was removed. The tier is now
`--live-platform`, configured by `ai_config.json`'s `platforms` block; see
PLATFORM_LIVE_TEST_DESIGN.md.)*

## Appendix A — live evidence, 2026-11-03

OpenAI Responses API, prompt "Say hi.". Some early effort requests carried a
16- or 100-token output cap; every rejection came before generation, so the
cap did not affect them.

| Request | Response |
|---|---|
| `gpt-5-pro`, effort `low`, and effort `none` | 400: "Supported values are: 'high'." |
| `gpt-5.2-pro`, `gpt-5.4-pro`, `gpt-5.5-pro`, effort `low` | 400: "Supported values are: 'medium', 'high', and 'xhigh'." |
| `gpt-5.5`, effort `minimal` | 400: "Supported values are: 'none', 'low', 'medium', 'high', and 'xhigh'." |
| `gpt-5`, effort `xhigh` | 400: "Supported values are: 'minimal', 'low', 'medium', and 'high'." |
| `gpt-6.1-terra`, effort `minimal` | 400: "Supported values are: 'none', 'low', 'medium', 'high', 'xhigh', and 'max'." |
| `gpt-5.5`, effort `bogus` | 400: "Supported values are: 'none', 'minimal', 'low', 'medium', 'high', 'xhigh', and 'max'." |
| `gpt-4.1`, effort `low` or `none` | 400, `"param": "reasoning.effort"`, `"code": "unsupported_parameter"` |
| `gpt-4.1`, temperature 0.5, `max_output_tokens=10` | 400: "Expected a value >= 16, but got 10 instead." |
| `gpt-4.1`, temperature 0.5, no cap | accepted |
| `gpt-5-codex`, `gpt-5.1-codex`, `-codex-max`, `-codex-mini`, `gpt-5.2-codex`, `gpt-4.1-nano` | 404 `model_not_found`; `shutdown_date` passed |
| `models.retrieve("gpt-5.5-pro")` | fields `id`, `created`, `object`, `owned_by`, `shutdown_date` only |

Error body for an unsupported level:

```json
{"message": "Unsupported value: 'low' is not supported with the 'gpt-5.5-pro' model. Supported values are: 'medium', 'high', and 'xhigh'.", "type": "invalid_request_error", "param": "reasoning.effort", "code": "unsupported_value"}
```

