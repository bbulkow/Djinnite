"""
Model Overrides

The single human-editable record of decisions that sit ON TOP of discovery.

Djinnite has two kinds of human input, and the distinction is what keeps this
from becoming a file per parameter:

  * ``known_model_defaults.json``  -- inputs TO discovery. Which model does the
    estimating, what a provider's vision limits default to. Read before the
    provider APIs are called.
  * ``model_overrides.json``       -- decisions ON TOP OF discovery. Any field,
    any model, applied last and unconditionally.

``model_catalog.json`` is the generated output of those two plus the provider
APIs. Nobody hand-edits it. An edit there is lost on the next refresh, which is
why the catalog carries an ``_overridden`` block recording exactly which fields
a human set and what discovery had said -- so the catalog stays readable even
though it is not writable.

Naming note: this file is named for the RELATIONSHIP (override), not for a
parameter. That is deliberate. ``disabled_models.json`` could not absorb a
pinned context window or a verified price without spawning siblings; this can.
"""

import json
import sys
from pathlib import Path
from typing import Any, Optional

try:
    from djinnite.config_loader import _resolve_config_file
except ImportError:
    _project_root = str(Path(__file__).resolve().parent.parent)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from config_loader import _resolve_config_file


OVERRIDES_FILENAME = "model_overrides.json"

# Bookkeeping block written into each overridden model in the catalog. Maps a
# dotted field path to what discovery produced before the human value replaced
# it, which is what makes removing an override reversible without a full
# refresh. Never hand-edited -- it is regenerated on every apply.
PROVENANCE_KEY = "_overridden"

# Marker for "discovery did not produce this field at all", so restoring means
# deleting the key rather than writing a value.
_WAS_ABSENT = "__absent__"

# Keys inside an override entry that are documentation, not data.
_META_PREFIX = "_"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_overrides(path: Optional[Path] = None) -> dict:
    """Load model_overrides.json. Returns {} when the file is absent.

    An absent file is normal (a project may override nothing); a malformed one
    is not, and raises rather than silently dropping every human decision.
    """
    p = path or _resolve_config_file(OVERRIDES_FILENAME)
    if not Path(p).exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("models", {})


def lookup(overrides: dict, provider: str, model_id: str) -> dict:
    """Find a model's override entry.

    Accepts ``provider/model-id`` (preferred -- unambiguous once local model
    backends land) and falls back to a bare ``model-id``.
    """
    entry = overrides.get(f"{provider}/{model_id}")
    if entry is None:
        entry = overrides.get(model_id)
    return entry or {}


# ---------------------------------------------------------------------------
# Dotted-path access, so a nested field like costing.input_per_1m can be
# overridden without restating the whole costing block.
# ---------------------------------------------------------------------------

def _get_path(obj: dict, path: str) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return _WAS_ABSENT
        cur = cur[part]
    return cur


def _set_path(obj: dict, path: str, value: Any) -> None:
    parts = path.split(".")
    cur = obj
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _del_path(obj: dict, path: str) -> None:
    parts = path.split(".")
    cur = obj
    for part in parts[:-1]:
        cur = cur.get(part)
        if not isinstance(cur, dict):
            return
    cur.pop(parts[-1], None)


def flatten_entry(entry: dict, prefix: str = "") -> dict:
    """Flatten an override entry to {dotted_path: value}, dropping _meta keys.

    A nested dict is descended into rather than replacing the catalog's whole
    block, so overriding costing.input_per_1m leaves the discovered source_url
    and updated date intact.
    """
    out: dict = {}
    for k, v in entry.items():
        if k.startswith(_META_PREFIX):
            continue
        path = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten_entry(v, prefix=f"{path}."))
        else:
            out[path] = v
    return out


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

def apply_to_model(model: dict, entry: dict) -> dict:
    """Apply one override entry to one catalog model. Mutates in place.

    Returns a report dict describing what happened, so callers can print a
    divergence summary. Reverting is handled here too: a field recorded in the
    model's provenance block but absent from the current entry is restored to
    the value discovery produced, which is what lets removing a line from the
    overrides file take effect immediately rather than at the next refresh.
    """
    report = {"applied": [], "noop": [], "reverted": []}

    wanted = flatten_entry(entry)
    prior = model.get(PROVENANCE_KEY) or {}

    # Restore anything that used to be overridden and no longer is.
    for path, record in prior.items():
        if path in wanted:
            continue
        was = record.get("was", _WAS_ABSENT)
        current = _get_path(model, path)
        if was == _WAS_ABSENT:
            _del_path(model, path)
        else:
            _set_path(model, path, was)
        report["reverted"].append((path, current, was))

    provenance: dict = {}
    for path, value in wanted.items():
        # Two different comparisons, and conflating them made every run
        # reprint all 106 unchanged disable fields:
        #
        #   current    -- what the catalog holds right now. Decides whether
        #                 THIS run changed anything, which is what gets
        #                 reported. A field already at the override value is a
        #                 no-op no matter how it got there.
        #   discovered -- what discovery produced before any human touched it.
        #                 Recorded as provenance so the override stays
        #                 reversible, and shown for context. NOT a change.
        current = _get_path(model, path)
        if path in prior:
            discovered = prior[path].get("was", _WAS_ABSENT)
        else:
            discovered = current

        _set_path(model, path, value)
        provenance[path] = {"was": discovered}

        if current == value:
            report["noop"].append((path, value))
        else:
            report["applied"].append((path, current, value))

    if provenance:
        model[PROVENANCE_KEY] = provenance
    else:
        model.pop(PROVENANCE_KEY, None)

    return report


def apply_overrides(catalog: dict, overrides: dict, verbose: bool = True) -> dict:
    """Apply the override file to a whole catalog. Mutates in place.

    This is the ONE place human decisions enter the catalog. Every script that
    writes the catalog must route through save_catalog() below so that it
    cannot be bypassed by accident.
    """
    stats = {"models": 0, "fields": 0, "noop": 0, "reverted": 0, "stale": 0}
    matched: set = set()

    for provider_name, provider_data in catalog.items():
        if not isinstance(provider_data, dict):
            continue
        for model in provider_data.get("models", []):
            model_id = model.get("id")
            entry = lookup(overrides, provider_name, model_id)
            if entry:
                matched.add(f"{provider_name}/{model_id}")
                matched.add(model_id)
            elif PROVENANCE_KEY not in model:
                continue

            rep = apply_to_model(model, entry)
            if rep["applied"] or rep["reverted"]:
                stats["models"] += 1
            stats["fields"] += len(rep["applied"])
            stats["noop"] += len(rep["noop"])
            stats["reverted"] += len(rep["reverted"])

            if not verbose:
                continue
            for path, was, now in rep["applied"]:
                shown = "(absent)" if was == _WAS_ABSENT else repr(was)
                print(f"  [OVERRIDE] {provider_name}/{model_id}.{path}: "
                      f"{shown} -> {now!r}")
            for path, was, now in rep["reverted"]:
                shown = "(absent)" if now == _WAS_ABSENT else repr(now)
                print(f"  [REVERT]   {provider_name}/{model_id}.{path}: "
                      f"override removed, restored discovered={shown}")

    # Entries that match nothing. Not an error: providers delist dated preview
    # snapshots all the time and a defensive entry for a model that may come
    # back is legitimate. Worth surfacing so the file can be pruned on purpose.
    stale = sorted(k for k in overrides if k not in matched)
    stats["stale"] = len(stale)
    if verbose and stale:
        print(f"  [INFO] {len(stale)} override entries match no catalog model "
              f"(delisted, or a typo): {stale}")

    return stats


def save_catalog(catalog: dict, catalog_path: Path,
                 overrides: Optional[dict] = None,
                 verbose: bool = True) -> dict:
    """Apply overrides, then write the catalog. The only sanctioned write path.

    Every script that persists the catalog calls this. Writing the file
    directly would let a refresh land without human decisions applied, which is
    the whole failure this design exists to prevent.
    """
    if overrides is None:
        overrides = load_overrides()
    if verbose:
        print("\n[TOOL] Applying human overrides (model_overrides.json)...")
    stats = apply_overrides(catalog, overrides, verbose=verbose)
    if verbose:
        print(f"  Models touched: {stats['models']}  Fields set: {stats['fields']}  "
              f"Already matching: {stats['noop']}  Reverted: {stats['reverted']}")
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)
    if verbose:
        print(f"\n[SAVE] Saved to {catalog_path}")
    return stats
