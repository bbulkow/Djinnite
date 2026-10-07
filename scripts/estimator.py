"""
Which model estimates prices and limits for the maintenance scripts.

One resolver shared by ``update_models`` (modality and output-limit
estimation) and ``update_model_costs`` (price estimation), so the two cannot
choose differently (ACCESS_PATHS_DESIGN.md "Maintenance scripts").

Precedence for the provider type and model:

1. ``--estimator MODEL`` on the command line. Its type is the catalog section
   that lists that model id, else the type of the ``default_provider`` entry.
2. ``known_model_defaults.json`` -> ``estimator.{provider, model}``.
   ``provider`` there is a provider **type**, not an ai_config entry name.
3. The ``default_provider`` entry's type and ``default_model``.

The entry the estimator runs through is ``AIConfig.direct_entry(type)`` --
the same rule ``update_models`` uses to refresh that type, so there is one
selection rule. Estimation needs a provider API key, so a platform-mode entry
never serves. The resolver raises ``EstimatorUnavailable`` when there is no
usable direct entry, when several direct entries of the type exist and none
is named after it, or when that entry's ``deny`` blocks a capability the
caller ``needs``; the caller decides what that means (``update_model_costs``
stops before writing anything).
"""

import json
from typing import NamedTuple, Optional

try:
    from djinnite.config_loader import _resolve_config_file, PROVIDER_TYPES
except ImportError:
    import sys
    from pathlib import Path
    _project_root = str(Path(__file__).resolve().parent.parent)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from config_loader import _resolve_config_file, PROVIDER_TYPES


class EstimatorUnavailable(Exception):
    """No usable estimator: the message says why (shown to the operator)."""


class Estimator(NamedTuple):
    """The estimator model and the direct-mode ai_config entry it runs through.

    Build it with ``ai_config.build_provider(est.entry, est.model, ...)``.
    """
    provider_type: str
    model: str
    entry: str
    api_key: str


def load_estimator_defaults() -> dict:
    """The ``estimator`` block of known_model_defaults.json, or ``{}``."""
    path = _resolve_config_file("known_model_defaults.json")
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    block = data.get("estimator") or {}
    return block if isinstance(block, dict) else {}


def _catalog_type_of(model_id: str, catalog: Optional[dict]) -> Optional[str]:
    """The catalog section (provider type) that lists ``model_id``, if any."""
    for ptype, block in (catalog or {}).items():
        if not isinstance(block, dict):
            continue
        if any(m.get("id") == model_id for m in block.get("models", []) or []):
            return ptype
    return None


def _default_type(ai_config) -> str:
    """Provider type of the ``default_provider`` entry (its name if unconfigured)."""
    name = ai_config.default_provider
    if name in ai_config.providers:
        return ai_config.provider_type(name)
    return name


def _describe_entries(ai_config, ptype: str) -> str:
    """``claude-vertex [platform], claude-old [direct, disabled]`` for messages."""
    parts = []
    for name, pc in ai_config.providers.items():
        if ai_config.provider_type(name) != ptype:
            continue
        notes = [pc.mode]
        if not pc.enabled:
            notes.append("disabled")
        elif pc.mode == "direct" and not ai_config.is_usable(name):
            notes.append("no usable api_key")
        parts.append(f"{name} [{', '.join(notes)}]")
    return ", ".join(parts) or "none"


def _entry_for(ai_config, ptype: str) -> str:
    """The direct-mode entry the estimator of type ``ptype`` runs through."""
    try:
        entry = ai_config.direct_entry(ptype)
    except ValueError as e:
        raise EstimatorUnavailable(str(e)) from e
    if entry is None:
        raise EstimatorUnavailable(
            f"no usable direct-mode entry for estimator type {ptype!r} "
            f"(entries: {_describe_entries(ai_config, ptype)}) -- estimation "
            f"needs a provider API key"
        )
    return entry


def resolve_estimator(ai_config, *, cli_model: Optional[str] = None,
                      catalog: Optional[dict] = None,
                      known_defaults: Optional[dict] = None,
                      needs: tuple[str, ...] = ()) -> Estimator:
    """Resolve the estimator type, model and entry (see the module docstring).

    Args:
        ai_config: The loaded ``AIConfig``.
        cli_model: ``--estimator MODEL`` from the command line, if given.
        catalog: The raw catalog dict (``{type: {"models": [...]}}``), used to
            find the type of ``cli_model``.
        known_defaults: The ``estimator`` block of known_model_defaults.json;
            read from the file when ``None``.
        needs: Capabilities the caller's estimation requests use (e.g.
            ``("web_search",)`` for prices). If the entry denies one for the
            estimator model, every request would be refused before it is
            sent, so the estimator is unavailable.

    Raises:
        EstimatorUnavailable: no model to use, an unknown provider type, no
            usable direct-mode entry of that type, several direct entries
            of that type and none named after it, or an entry ``deny`` that
            blocks something in ``needs``.
    """
    if known_defaults is None:
        known_defaults = load_estimator_defaults()

    if cli_model:
        ptype = _catalog_type_of(cli_model, catalog) or _default_type(ai_config)
        model = cli_model
        source = f"--estimator {cli_model}"
    elif known_defaults.get("model"):
        ptype = known_defaults.get("provider") or "gemini"
        model = known_defaults["model"]
        source = "known_model_defaults.json estimator"
    else:
        default = ai_config.default_provider
        pc = ai_config.get_provider(default)
        if pc is None:
            raise EstimatorUnavailable(
                f"no estimator in known_model_defaults.json and default_provider "
                f"{default!r} is not a configured, enabled entry"
            )
        ptype = ai_config.provider_type(default)
        model = pc.default_model
        source = f"default_provider entry {default!r}"

    if ptype not in PROVIDER_TYPES:
        raise EstimatorUnavailable(
            f"estimator type {ptype!r} (from {source}) is not a provider type "
            f"{PROVIDER_TYPES}"
        )
    if not model:
        raise EstimatorUnavailable(f"no estimator model (from {source})")

    entry = _entry_for(ai_config, ptype)
    pc = ai_config.providers[entry]
    blocked = [cap for cap in needs if cap in pc.denied_for(model)]
    if blocked:
        reason = f" (deny_reason: {pc.deny_reason})" if pc.deny_reason else ""
        raise EstimatorUnavailable(
            f"estimator entry {entry!r} denies {', '.join(blocked)} for model "
            f"{model!r}{reason}, which estimation needs"
        )
    return Estimator(ptype, model, entry, pc.api_key)


__all__ = ["Estimator", "EstimatorUnavailable", "resolve_estimator", "load_estimator_defaults"]
