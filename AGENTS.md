# AGENTS.md — Djinnite

Working notes for AI agents (and humans) contributing to this repo. The
[README](README.md) covers what Djinnite is and how to use it as a library;
[DEVELOPMENT.md](DEVELOPMENT.md) is the authoritative reference for the
public API, the model catalog schema, and breaking-change history. Read
those first.

This file is the **single home for project-specific agent guidance.** Do
not duplicate this content into hidden tool memory.

## Directive: no hidden memory for this project

Anything an agent learns about this project that another contributor (human
or AI) would also benefit from — conventions, gotchas, known drift, user
preferences specific to this repo — goes into a versioned file in this
repo:

* This file (`AGENTS.md`) for repo-wide agent rules and short notes.
* [DEVELOPMENT.md](DEVELOPMENT.md) for API / schema / implementation detail.
* The relevant code's docstrings and comments for narrow technical points.

Do **not** write project-relevant content into hidden tool memory
directories (e.g. `~/.claude/projects/.../memory/`). They are invisible to
everyone else and lead to two contributors making different decisions
based on different knowledge.

User-only preferences (e.g. how to address the user, personal aliases) are
fine to keep in tool memory. Project facts are not.

## Repo conventions

### Always run Python via `uv run`

This is a `uv`-managed project. The SDK dependencies (`anthropic`,
`google-genai`, `openai`) only resolve inside the project venv. Bare
`python` will pick up a different interpreter and fail with "package not
installed." This applies to scripts, one-liners, and ad-hoc smoke tests.

```
✅  uv run python scripts/update_models.py --reprobe all
✅  uv run python -m djinnite.scripts.update_models
✅  uv run python -c "from djinnite.config_loader import load_model_catalog"
✅  uv run pytest tests/ -v
❌  python scripts/update_models.py …
```

DEVELOPMENT.md has the longer-form rationale.

### No emoji or other non-ASCII glyphs in Python output

The Windows console default codepage is `cp1252`. Emoji and box-drawing
characters (`━`, `╮`, `…`, etc.) raise `UnicodeEncodeError` at print time
and have, in practice, blocked entire scripts from running (`update_model_costs.py`).

Use plain ASCII labels: `[OK]`, `[FAIL]`, `[WARN]`, `[SKIP]`. Plain dashes
instead of box-drawing. This applies to `print` statements, log lines,
and any string written to stdout/stderr from code in this repo.

(Markdown docs and JSON catalog values are unaffected — the rule is
specifically about runtime Python output.)

### Fix "pre-existing" test failures before starting new work

When a test fails, errors during collection, or is skipped for a reason that
isn't purely environmental, treat it as a real defect — fix it *before*
embarking on the feature or upgrade you came here to do. Do not work around it,
do not route around it with flags or `--ignore`, and do not write it off as
"pre-existing" and move on. A red suite you inherit is still a red suite you
ship.

"Error collecting" is a failure, not a warning. It means pytest never ran those
tests at all, so their result is unknown rather than passing.

Two failure modes this repo has actually hit, both of which report green:

* **A test that `return`s instead of asserting.** pytest ignores the return
  value, so a function that returns a count of 99 failures still reports PASS.
  `filterwarnings = ["error::pytest.PytestReturnNotNoneWarning"]` in
  `pyproject.toml` now turns this into a hard error — leave it enabled.
* **A module that fails to import.** The whole file is skipped with a
  collection error while the summary line still says "N passed".

The bar for finishing: `uv run pytest tests/` reports zero failures and zero
errors, and the count of collected tests is the count you expect.

### Anthropic's `models.list()` reports capabilities for free

`client.models.list()` now returns a `capabilities` block per model —
`thinking.types.{adaptive,enabled}.supported`, `effort.{low,medium,high,xhigh,max}`,
`max_input_tokens`, `max_tokens`, `structured_outputs`, `image_input`, and more.
Verified against live probing: it matched on all 10 Claude models.

Prefer it over `update_models --reprobe` for anything it covers. Reprobing
spends real tokens across three providers to rediscover facts one free,
unauthenticated-cost list call already states. Probing is still needed for
what the API does not report — notably cross-capability `incompatible`
combinations.

This is also how the catalog drifted: `context_window` sat at 200000 for
eight Claude models the API reports as 1000000, and the `effort` capability
was absent entirely, so `thinking="high"` was rejected on every Claude
despite eight models supporting it.

### Running a model catalog update, observably

`update_models` makes live API calls across every configured provider and can
run for ten minutes or more. Run it so you can see what it did and prove what
it changed.

**Back up first, then run unbuffered, teed to a log:**

```powershell
cp config/model_catalog.json /tmp/catalog.before.json
uv run python -u -m djinnite.scripts.update_models 2>&1 | tee /tmp/update.log
```

Three details that matter more than they look:

* **`-u` is required.** Python block-buffers stdout when it is not a terminal,
  so a redirected run shows *nothing* until it exits. A ten-minute run looks
  identical to a hung one.
* **Never pipe through `tail`/`head`.** `tail` buffers the whole stream and
  discards everything but the end, so the estimation and probe lines — the
  only record of what the run actually decided — are gone. `tee` keeps them.
* **Relay the script's raw output, verbatim, as it runs.** The `tee` is for
  the agent's own diffing; the user wants to *watch the run*. Do not start
  the update and then go quiet for ten minutes.

  Paste the actual log lines. Do not summarize them, do not reword them,
  do not replace a run of probe lines with a count of them, and do not
  append an interpretation of what a `[WARN]` means. The output format was
  designed in this repo by the people reading it — they already know what
  `json=[OK] temp=[FAIL] think=[OK](adaptive) incompat=2` means, and a
  paraphrase is strictly less information than the line it replaced.
  Analysis is welcome *later*, if asked; it is not a substitute for the
  transcript.

  A silent ten-minute run is indistinguishable from a hung one *for the
  person watching*, and this pipeline spends real money on live API calls
  across three providers. Frequent verbatim excerpts (the new lines since
  last time) beat one large dump at the end.

  **Give the user the log path when you start, not when you finish.** A
  background task's output pane does not stream, so relaying excerpts is the
  agent's half of the job and the user still has no window of their own.
  Hand them the tail command up front so they are never dependent on the
  agent's polling cadence:

  ```powershell
  Get-Content -Wait -Tail 60 <path to the tee'd log>
  ```

  `-Wait` is PowerShell's `tail -f`. Say this in the same message that starts
  the run.

**Afterwards, diff against the backup.** The summary line is not sufficient:
it reports counts, not which fields moved. A refresh rebuilds each model's
`capabilities` dict wholesale, so a field the writer does not know about is
dropped silently and the summary still says "Unchanged".

```powershell
uv run python -c "import json; a=json.load(open('/tmp/catalog.before.json')); b=json.load(open('config/model_catalog.json')); ..."
```

Check specifically that `context_window`, `max_output_tokens`, `thinking_style`
and `effort_levels` survived. `uv run pytest tests/test_catalog_schema.py`
covers the known cases.

**Is it hung, or just slow?** Estimation blocks on web-search-backed API calls
that take tens of seconds each, so low CPU is normal — a healthy run may use
only ~6s of CPU in 9 minutes. Do not judge by CPU alone, and do not judge by
the wrong process: `python.exe -m djinnite.scripts.update_models` is a launcher
shim that sits at ~0.02s CPU and 4MB. The real worker is the `uv` python at
100MB+. Sample it:

```powershell
$w = Get-CimInstance Win32_Process -Filter "Name like '%python%'" |
     Sort-Object WorkingSetSize -Descending | Select-Object -First 1
Get-NetTCPConnection | Where-Object { $_.OwningProcess -eq $w.ProcessId -and $_.State -eq 'Established' } |
     Select-Object RemoteAddress, RemotePort, CreationTime
```

Established connections with `CreationTime` newer than the last thing you saw
in the log mean it is still working. This is also how to tell whether a run
survived the machine sleeping: new connections after the wake means yes.

**Batch size is load-bearing.** The AI estimator degrades silently on large
batches — it answers `0` ("unsure") for every model rather than erroring. A
run that logs `Got limits for 0 models` did not fail; it gave up. Compare the
model count in that line against a batch that succeeded before assuming the
data was unavailable.

Degradation has a second, nastier signature: **plausible-looking wrong
numbers, not zeros.** In the 2026-09-10 run a 10-model batch returned
`gpt-6-astra` at `max_output_tokens: 128000, context_window: 128000` — two
real numbers, both wrong together. Re-asking for that *one* model returned
`context_window: 1050000`, matching the rest of the GPT-5.x/6 family. So
`Got limits for N/N models` is not evidence the values are right; a batch can
answer confidently and badly. When a new model's limits look like a round
default, or the two numbers match each other, re-ask for it alone before
believing them:

```powershell
uv run python -c "from djinnite.config_loader import load_ai_config; from djinnite.scripts.update_models import estimate_output_limits_with_ai; print(estimate_output_limits_with_ai([{'id':'MODEL_ID'}],'chatgpt',load_ai_config()))"
```

`ctx == max_output_tokens` specifically is always wrong — no model can spend
its entire context on output with nothing left for a prompt.
`sanitize_estimated_limits` now drops that case, and
`test_context_window_exceeds_output_cap` catches any that reach the catalog.

### The catalog is generated. Humans edit `model_overrides.json`.

Four config files, each with a distinct **role**. The role is what keeps this
from becoming a file per parameter:

| file | role | edited by |
|---|---|---|
| `ai_config.json` | which providers, which keys | human |
| `known_model_defaults.json` | **inputs to** discovery: estimator choice, provider vision defaults | human |
| `model_overrides.json` | **decisions on top of** discovery: any field, any model | human |
| `model_catalog.json` | generated output of the three above plus the provider APIs | **nobody** |

The line to hold onto: *defaults feed into discovery; overrides sit on top of
it.* Anything a human wants to pin about a specific model goes in
`model_overrides.json` regardless of which field it is — disabled state, a
corrected context window, a hand-verified price. It is named for the
relationship, not the parameter, so it never needs a sibling.

**Never hand-edit `model_catalog.json`.** It is regenerated, and an edit there
is lost with no error. To make the generated file still readable, every
overridden model carries an `_overridden` block recording which fields a human
set and what discovery had said:

```json
"_overridden": {
  "context_window": { "was": 128000 }
}
```

So you *read* the catalog and *edit* the overrides. `apply_overrides --list`
prints that view directly.

**One write path.** `model_overrides.save_catalog()` applies the overrides and
then writes; `update_models`, `update_model_costs` and `apply_overrides` all go
through it. `test_every_write_path_routes_through_save_catalog` fails if
anything calls `json.dump(catalog, ...)` directly — that bypass is how a
refresh would land with human decisions silently missing.

**Removing an override reverts immediately**, without waiting for a refresh:
the `was` value in the provenance block is restored. That is what the `was`
record is for.

To change something:

```powershell
# 1. edit config/model_overrides.json
# 2. preview -- read the [OVERRIDE] and [REVERT] lines
uv run python -m djinnite.scripts.apply_overrides --dry-run
# 3. apply
uv run python -m djinnite.scripts.apply_overrides
```

#### Why this replaced `disabled_models.json`

Disable state used to live in its own file *and* in the catalog. Runtime read
only the catalog; the maintenance command re-enabled anything absent from the
file. Seven models — `gpt-4o`, `gpt-4o-mini`, four `*-search-preview` variants
and `gemini-3.1-flash-live-preview` — carried disable reasons recorded **only**
in the catalog, one command away from being silently re-enabled. The mechanism
that hid it: `merge_model_data` copied `disabled` forward on every refresh, so
the state renewed itself indefinitely while the file knew nothing.

The first fix attempt — make the catalog value a projection of the disable
file — was not a fix. The catalog is still a readable, editable JSON file, so
that change only converted "your edit drifts" into "your edit vanishes
silently," and it left three different override mechanisms in place
(`disabled_models.json`, `known_model_defaults.json`, and an in-catalog
`costing.source: "manual"` sentinel that had zero users). A file per parameter
does not scale: "disabled models" has no sensible sibling for a pinned context
window or a verified price.

`scripts/disable_models.py` is now a shim that exits non-zero with a pointer,
so old habits fail loudly instead of quietly doing nothing.

#### Stale override entries are fine

16 of the migrated entries reference models the providers have delisted. That
is not an error and is not a test failure — providers retire dated preview
snapshots constantly, and a defensive entry for one that may return is
legitimate. `apply_overrides` reports them as `[INFO] ... match no catalog
model` so the file can be pruned deliberately rather than automatically.

### Risky actions still need confirmation

`uv run python -m djinnite.scripts.update_models --reprobe all` makes live
API calls against three providers and costs real tokens. Don't run it
unprompted to "verify" something — scope down to one or two model IDs
first. The user has paid for surprise probes more than once.

## Pointers

* **Public API contract:** [DEVELOPMENT.md § THE CONTRACT](DEVELOPMENT.md).
* **Capability schema** (the list-of-states pattern, vocabularies,
  pre-flight rules): [DEVELOPMENT.md § ModelCapabilities](DEVELOPMENT.md).
* **Token budgets** (request param vs catalog field vs response field,
  and the per-SDK native mapping for output / thinking / total / input /
  search): [DEVELOPMENT.md § Token Budgets](DEVELOPMENT.md). Read this
  before touching anything that mentions tokens — names like
  `max_output_tokens`, `context_window`, and `thinking` each control a
  different budget and the per-provider semantics differ.
* **Breaking change log:** [DEVELOPMENT.md § Breaking Changes Log](DEVELOPMENT.md).
* **Pricing has more than one number per model.** Vendors publish Standard,
  Flex, Batch and long-context rates on the same page; the catalog stores
  **Standard at the base context tier**, and `ModelCosting.published_figure`
  records which row the number came from. A large DIVERGENT swing is more often
  a tier mix-up than a real price change — check the quoted figure before
  believing it. Open work (service tiers, context-length tiers, and why they
  wait on multi-request contexts) is recorded in
  [SERVICE_TIER_DESIGN.md](SERVICE_TIER_DESIGN.md); the current limitation is
  in [DEVELOPMENT.md § Known limitation: context-length pricing
  tiers](DEVELOPMENT.md).
