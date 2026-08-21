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

Two details that matter more than they look:

* **`-u` is required.** Python block-buffers stdout when it is not a terminal,
  so a redirected run shows *nothing* until it exits. A ten-minute run looks
  identical to a hung one.
* **Never pipe through `tail`/`head`.** `tail` buffers the whole stream and
  discards everything but the end, so the estimation and probe lines — the
  only record of what the run actually decided — are gone. `tee` keeps them.

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
