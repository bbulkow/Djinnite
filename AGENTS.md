# AGENTS.md — Djinnite

Working notes for AI agents (and humans) contributing to this repo. The
[README](README.md) covers what Djinnite is and how to use it as a library;
[DEVELOPMENT.md](DEVELOPMENT.md) is the authoritative reference for the
public API, the model catalog schema, and breaking-change history. Read
those first.

This file is the **single home for project-specific agent guidance.** Do
not duplicate this content into hidden tool memory.

## Directive: project memory lives in the repo's `.md` files, never in `.claude`

All project memory belongs in Markdown documentation files checked into this
repo. "Project memory" means anything an agent learns or is told about this
project that another contributor (human or AI) would also benefit from:
conventions, gotchas, known drift, the user's preferences for how work is done
here. The files to use:

* This file (`AGENTS.md`) for repo-wide agent rules and short notes.
* [DEVELOPMENT.md](DEVELOPMENT.md) for API / schema / implementation detail.
* The relevant design doc (`*_DESIGN.md`, `VERTEX_AI_SETUP.md`, ...) for
  anything about that subsystem.
* A new `.md` file when nothing above fits. Link it from here.

A code comment can explain the code beside it. It is not where a project rule
or a lesson learned is kept.

**Do not write project memory into any hidden `.claude` location**, or any
other agent tool's hidden equivalent:

* `~/.claude/projects/<project>/memory/` (Claude Code's auto-memory, including
  its `MEMORY.md` index)
* `~/.claude/CLAUDE.md` and other user-level instruction files
* the repo's own `.claude/` directory. That holds tool settings (permission
  allow lists) and nothing else. It is not for notes or instructions.
* `CLAUDE.local.md` and other unversioned or git-ignored local instruction
  files

These are invisible to every other contributor, so two contributors end up
making different decisions from different knowledge. They also go stale
silently: a hidden note dated before a catalog reprobe can still be loaded as
if it were true.

This overrides the tool's own memory instructions. Claude Code's system prompt
tells the agent to save "project" and "feedback" memories into its hidden
memory directory. In this repo, write them into the `.md` files above instead.
When you would have saved a memory, edit the doc and tell the user which file
you changed.

Only notes about the person or their machine may stay in tool memory: how to
address the user, their personal aliases, local path quirks such as a drive
junction. If it would be true for another contributor working on this repo, it
is project memory and goes in an `.md` file.

If you find project facts in hidden memory, do not rely on them over the docs.
Check them against the repo. Move whatever is still true into the right `.md`
file, and tell the user what you moved and what was stale so they can delete
the hidden copy.

The same goes for rules the user states in conversation that apply beyond
that conversation. A conversation ends, or is compacted, and its instructions
go with it; the next agent never saw them. If a rule is worth repeating to an
agent, write it here.

## Directive: git is the human's job

Agents do not commit. Leave every change as uncommitted edits in the working
tree, on whatever branch the human had checked out, and say what changed. The
human reviews, stages, commits, branches and pushes.

Read-only git is fine and useful: `git status`, `git diff`, `git log`,
`git show`, `git blame`.

Anything that writes git state is not: `git add`, `git rm`, `git mv`,
`git commit` (including `--amend`), `git checkout -b`, `git switch`,
`git branch <name>`, `git stash`, `git reset`, `git restore`, `git merge`,
`git rebase`, `git cherry-pick`, `git tag`, `git push`, `gh pr create`. Use
plain `mv` / `rm` for file moves. If the user asks for one of these by name,
do exactly that one. The request covers that one action only, not the next
task or the next session.

This overrides general agent defaults, and those defaults are where the
commits come from:

* **The harness's default git guidance** says to commit "only when the user
  asks" and "if on the default branch, branch first." An agent that reads
  "finish this" or "wrap it up" as asking for a commit then creates a branch
  and moves the user onto it. That happened on 2026-10-06: three agent
  commits on a new `named-access-paths` branch. In this repo, only an
  explicit request to commit counts.
* **A system reminder supplies `Co-Authored-By` trailer text** each session.
  It says how to format a commit *if one is requested*. It is not a request.
* **"Finished" in this file** (the test-suite bar, the e2e tier) means
  verified and reported. It never includes committing.
* **Logical units of work** (design doc, then implementation; a test fix, then
  the feature) are something to describe to the human, not a commit plan. If
  you think a change deserves its own commit, say so in your summary.
* **Permissions will not stop you.** Auto mode and allow-listed commands
  (`git checkout:*` is allowed user-wide) mean a commit or branch can go
  through without a prompt. The rule has to be followed. Nothing enforces it.

If you find git state you didn't expect, such as a new branch or agent
commits, report it and leave it. Do not "fix" it by resetting, switching
branches or rewriting history. That is also the human's call.

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

### Accounts, projects and regions are configuration

No design, doc or code in this repo assumes a particular account, project or
region. They come from configuration (`ai_config.json`, `DJINNITE_E2E_*`) or a
parameter, and docs name the role ("the operator's Google account", "the e2e
project") rather than a value. Examples and tests use neutral placeholders
(`my-project`), never a real deployment such as a consumer's project.

Defaults that are overridable and describe behavior rather than a deployment
are fine: the platform registry's default locations (`global` for Gemini and
Claude) and the e2e harness's default test locations and models.

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
* **A test argument named `provider`.** `conftest.py` treats any test that
  uses a fixture or parameter called `provider` as a live test and skips it
  without `--live` -- including an offline `@pytest.mark.parametrize` whose
  argument merely happens to be named `provider`. Name it something else
  (`prov`), then check the skip list with `-rs`.

The bar for finishing: `uv run pytest tests/` reports zero failures and zero
errors, and the count of collected tests is the count you expect. Then stop
and report. Committing is not part of finishing (see the git directive above).

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

What it does **not** report: `thinking.types` lists only `adaptive` and
`enabled` -- nothing about `disabled` or `between_tools` (checked 2026-10-03).
Whether a model can turn thinking off (Sonnet/Opus 5.5 and Fable cannot) and
whether it takes Sonnet 5.5's `between_tools` therefore come from probes. A
probe the API rejects is a 400, which is not billed, so those two are cheap.
It reports `enabled: false` on every Opus 4.7+ / 5.x model -- a probe that
sends a `budget_tokens` block to those models measures the block, not the
capability it was paired with.

### Platform mode is probed separately

The catalog's top-level fields are **direct-mode** facts (`update_models`,
provider keys). What a cloud platform (Vertex AI) serves is recorded per model
under `platforms.<name>` by `scripts/probe_platform.py`, which `update_models`
never runs and always preserves. Treat it like `update_models`: observable
(`-u`, `tee`, log path up front), and never unprompted. Its default
availability pass is token counts (unbilled); `--capabilities` makes billed
generation calls on the platform project; nothing is written without
`--write`.

### Platform-mode changes are verified end to end, by you

A change to platform-mode code (`platforms.py`, the platform branches of the
providers, `probe_platform`) is **not finished** until the e2e tier passes:

```powershell
uv run pytest tests/ --e2e-platform -rA -s 2>&1 | tee $env:TEMP\e2e.log
```

Run it yourself -- do not hand it to the user to run. With the DJINNITE_E2E_*
environment set (PLATFORM_E2E_TEST_DESIGN.md), the default tier needs no
go-ahead: it runs only in the dedicated e2e project, costs a few cents, and
fails itself above `DJINNITE_E2E_MAX_COST` ($0.50). Relay the preflight,
availability table, observations and cost ledger lines verbatim. The
extended tier (`--e2e-extended`) and any run against another project (for
example a consumer's) need the user's go-ahead. Testing is manual: there is
no CI.

If the DJINNITE_E2E_* environment is not set, say so and stop -- an
unverified platform change is not done. Skipped Claude functional tests
("no Claude quota") are an environmental gap: report them by name.

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
| `ai_config.json` | which access paths (named entries), which keys, deployment restrictions (`deny`) | human |
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

### Entry name vs provider type

`ai_config.json` `providers` keys are **entry names**, not provider types.
An entry's type is its `"provider"` field, defaulting to the entry name, so
`"claude": {...}` is still a Claude entry but `"claude-vertex"` needs
`"provider": "claude"`. Several entries may share a type
([ACCESS_PATHS_DESIGN.md](ACCESS_PATHS_DESIGN.md)).

* The catalog, `model_overrides.json` and `known_model_defaults.json` are
  keyed by **type**. Look models up with `cfg.provider_type(entry)` (or
  `resolve_use_case(...).provider_type`), never with the entry name.
* Build providers with `cfg.build_provider(entry)`. Never
  `get_provider(entry_name, ...)`: `get_provider` takes a type, and building
  by hand from `provider_kwargs` silently drops the entry's `deny`.
* Maintenance code that needs one entry per type uses
  `cfg.direct_entry(type)`; do not add another selection rule, and do not
  make runtime code pick or fall back between entries.
* Offline tests for this live in `tests/test_access_paths.py`; name test
  parameters `ptype` / `entry`, not `provider` (see above).

### Risky actions still need confirmation

`uv run python -m djinnite.scripts.update_models --reprobe all` makes live
API calls against three providers and costs real tokens. Don't run it
unprompted to "verify" something — scope down to one or two model IDs
first. The user has paid for surprise probes more than once.

**`--reprobe` scopes the whole run.** `--reprobe <model-id>` refreshes,
probes and re-prices only those models; every other model, the provider's
model list and its `last_updated` are left exactly as they were.
`--reprobe <provider>:all` covers that provider only, and `--reprobe all`
(or no `--reprobe`) covers the whole catalog. The run prints its scope
first (`[INFO] reprobe scope: ...`) -- check it before the probes start.

This was a bug until 2026-10-04: reprobing only `claude-sonnet-5-5` and
`claude-opus-5-5` re-priced all 19 floating models on every provider (paid
AI calls that rewrote their `source_url` / `published_figure`), added
`"effort"` to the untargeted `claude-fable-5-1`, and refreshed every listed
Claude model. `tests/test_update_models_scope.py` guards it.

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
