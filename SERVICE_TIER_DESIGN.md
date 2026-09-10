# Service Tiers and Context-Length Pricing — Design / Open Task

**Status:** OPEN — recorded, not started. Recorded 2026-09-10.

**Depends on:** multi-request contexts (see "Sequencing" below). The service-tier
work should land *with* that, not before it.

**Why it exists:** `gpt-5.4-pro` oscillated between $30/$180 and $15/$90 across
consecutive `update_model_costs` runs. Neither number was wrong — they are the
Standard and Flex rates for the same model. The catalog holds one scalar pair,
so the estimator had to pick, and picked differently each time. The immediate
contamination is fixed (see "Already done"); this document records the
structural work that is not.

---

## The problem: several published prices, one field

A vendor publishes multiple rates for one model on one page. `gpt-5.4-pro`:

| Rate | Input / 1M | Output / 1M | Selected by |
|---|---|---|---|
| Standard, base context | $30 | $180 | default |
| Standard, long context | $60 | $270 | **input tokens > 272,000** |
| Flex (off-peak) | $15 | $90 | request `service_tier` |
| Batch | discounted | discounted | async submission |
| Regional data residency | +10% | +10% | endpoint / account |

`ModelCosting` has exactly one `input_per_1m` and one `output_per_1m`. Every
row above is "the price".

These are **two independent axes**, and conflating them is what made the
original bug hard to see:

* **Context length** — not a choice. It is a property of the request, applied
  after the fact by the vendor. Djinnite *is* exposed to this today.
* **Service tier** — a choice, made at request time. Djinnite is *not* exposed
  today, because it never sends `service_tier` (verified: zero occurrences in
  the codebase). Flex prices are currently unreachable, which is precisely why
  a Flex price in the catalog is pure contamination rather than a useful
  number.

## Already done (Phase 1, 2026-09-10)

* Estimator prompt pins the **Standard** service tier by name and explicitly
  rejects Flex / Batch / Priority / regional rows, and asks for the base
  context tier.
* `ModelCosting.published_figure` persists the vendor's price text verbatim,
  including which tier it came from. It was already being requested from the
  estimator and discarded.
* The DIVERGENT audit lines print that quoted figure, so a tier swap is
  distinguishable from a real price change at a glance.
* `gpt-5.4-pro` corrected to $30/$180 (Standard), verified by two independent
  estimation passes.
* Limitation documented in DEVELOPMENT.md § "Known limitation: context-length
  pricing tiers".

## Open work

### 1. Context-length tiers (correctness bug, independent of everything else)

`AIResponse.token_cost` under-reports for a request above the threshold. It is
reachable today: `generate()` accepts a single prompt string, and a 300k-token
prompt is legal against a 1,050,000-token window. It just reports roughly half
the true cost.

```json
"costing": {
  "input_per_1m": 30.0,
  "output_per_1m": 180.0,
  "context_tiers": [
    { "above_input_tokens": 272000, "input_per_1m": 60.0, "output_per_1m": 270.0 }
  ]
}
```

`BaseAIProvider._compute_token_cost` selects the bracket from actual
`input_tokens` before multiplying.

**The subtlety worth a dedicated test:** the threshold is measured on *input*
tokens but changes the *output* rate too. A short answer to a very long prompt
is billed at the long-context output rate.

Absent `context_tiers` must mean exactly today's behaviour, so existing
catalogs and consumers stay correct.

### 2. Service tiers (feature, not a bug)

Djinnite **should** offer flex/batch — many consumer workloads are
deferred-time and would take the discount. That makes service tier a
request-time parameter, not merely a catalog fact:

```json
"service_tiers": {
  "flex":  { "input_per_1m": 15.0, "output_per_1m": 90.0 },
  "batch": { "input_per_1m": 15.0, "output_per_1m": 90.0 }
}
```

Requires, roughly:

* A `service_tier` request parameter threaded through `generate()` /
  `generate_json()` to each provider SDK, with per-provider capability
  detection — not every model offers every tier, and this belongs in
  `ModelCapabilities` as a list-of-states like the existing entries.
* Cost computation selecting the tier's rates.
* Deciding what Batch means for a synchronous API. Batch is asynchronous with a
  turnaround window; it is not a flag on a normal call. It may need a distinct
  submission path rather than a parameter, which is a much larger change than
  Flex.

## Sequencing: why this waits on multi-request contexts

Both halves get materially easier, and one of them only becomes worth doing,
once Djinnite exposes multi-request contexts:

* **Context tiers become common rather than rare.** Today input size is bounded
  by one caller's single prompt. With accumulating context, crossing 272k stops
  being an edge case and becomes the normal end state of a long session — which
  is exactly when correct tier selection starts to matter for real money.
* **Service tier is a session-level choice.** "This whole conversation is
  deferred-time, bill it at Flex" is a property of the context, not of each
  call. Adding a per-call parameter first would mean threading it through every
  call site and then rethreading it when contexts land.
* **Batch needs somewhere to live.** An asynchronous submit/collect cycle has no
  natural home in a stateless single-shot API. A context object gives it one.

Doing the context work first means the tier work is designed against the API it
will actually live in.

**Recommended order:** context-length tiers (item 1) may be pulled forward
independently if a consumer starts sending very large single prompts — it is a
genuine under-billing today and does not depend on contexts. Service tiers
(item 2) should wait.

## Guards to add with the implementation

* `context_tiers` sorted, non-overlapping, thresholds strictly below
  `context_window` — a tier above the window is unreachable and signals a bad
  estimate.
* A model whose `context_window` exceeds a known vendor threshold but declares
  no `context_tiers` gets flagged for review. That is how the next
  `gpt-5.4-pro` gets found before it silently under-bills.
* Overrides need no change: `model_overrides.json` can already pin
  `costing.context_tiers` wholesale, because `flatten_entry` descends dicts but
  assigns lists atomically.
