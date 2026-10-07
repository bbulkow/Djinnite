"""
Djinnite Configuration Loader

Handles loading and validation of AI-related configuration files:
- AI provider configuration (ai_config.json)
- Model catalog (model_catalog.json)

Host project config files are expected at:
    <project_root>/config/ai_config.json
    <project_root>/config/model_catalog.json

Where <project_root> is the parent directory of the djinnite package.
"""

import json
from pathlib import Path
from typing import Any, Final, NamedTuple, Optional
from dataclasses import dataclass, field


# Capability vocabularies — fixed by the Djinnite API contract.
# Each per-model `ModelCapabilities` field is `list[str] | None`, where the
# list is the subset of these tokens that the model accepts.
#
#   None  = unknown (not yet probed / inconclusive) — runtime skips pre-flight.
#   list  = the closed set of states the model supports.
#
# Empty lists are not emitted by probes; if a capability is fully unsupported,
# the explicit "off" token is in the list (e.g. a non-thinking model is ["off"]).

ON_OFF_STATES: Final[tuple[str, ...]] = ("on", "off")
THINKING_STATES: Final[tuple[str, ...]] = ON_OFF_STATES
STRUCTURED_JSON_STATES: Final[tuple[str, ...]] = ON_OFF_STATES
WEB_SEARCH_STATES: Final[tuple[str, ...]] = ON_OFF_STATES
JSON_WITH_SEARCH_STATES: Final[tuple[str, ...]] = ON_OFF_STATES
TEMPERATURE_STATES: Final[tuple[str, ...]] = ("any", "default")
# "between_tools" is Claude Sonnet 5.5's lowest thinking setting
# (``thinking={"type": "between_tools"}``); callers select it with the
# sentinel ``thinking="between_tools"``.
THINKING_STYLE_VALUES: Final[tuple[str, ...]] = ("adaptive", "budget", "effort", "between_tools")
# Union of every provider's effort vocabulary. A model's `effort_levels`
# is the subset it actually accepts -- these differ *within* a provider
# (Claude Opus 4.5 stops at "high"; Opus 5 goes to "max"), which is why
# the provider-wide set alone cannot pre-flight a request.
EFFORT_LEVEL_VALUES: Final[tuple[str, ...]] = (
    "minimal", "low", "medium", "high", "xhigh", "max",
)

# Access modes (see ai_providers/platforms.py): "direct" is the provider's
# own API; "platform" is a cloud platform serving the provider's models.
ACCESS_MODES: Final[tuple[str, ...]] = ("direct", "platform")

# Provider types: the keys of ``ai_providers.PROVIDERS`` (a test pins the
# two together). Kept here so reading ai_config.json imports no provider SDK.
# An ai_config entry's type is its ``provider`` field, defaulting to the
# entry's name (ACCESS_PATHS_DESIGN.md).
PROVIDER_TYPES: Final[tuple[str, ...]] = ("gemini", "claude", "chatgpt", "grok")

# Capabilities an ai_config entry may ``deny``; each removes the "on" state.
# These are request features a deployment's policy can switch off (Vertex
# org policy gates structured outputs and web search), not model facts.
# ``thinking`` and ``temperature`` are not deniable: no platform policy gates
# them (ACCESS_PATHS_DESIGN.md "Restrictions").
DENIABLE_CAPABILITIES: Final[tuple[str, ...]] = (
    "structured_json", "web_search", "json_with_search",
)

# Key of a ``deny`` map that applies to every model the entry serves.
DENY_ALL_MODELS: Final[str] = "*"

# Per-location availability of a model on a platform, as recorded by
# scripts/probe_platform.py in the catalog's ``platforms.<name>.locations``.
#   available -- the probe call succeeded
#   not_found -- 404 (not served there, or not enabled for the project)
#   no_access -- 401/403 (credentials or IAM refused)
#   no_quota  -- 429 / RESOURCE_EXHAUSTED (served, but no quota granted)
#   unknown   -- anything else (transient or unclassified)
LOCATION_STATUS_VALUES: Final[tuple[str, ...]] = (
    "available", "not_found", "no_access", "no_quota", "unknown",
)


# Maps each "activatable" capability to the state token that means "the
# caller is using this feature". The probe orchestrator iterates these
# pairs to build combinations to test; the runtime validator uses the
# same map to convert call arguments into the capability-state vocabulary
# the catalog records under `capabilities.incompatible`.
#
# `json_with_search` is intentionally excluded — it is itself the answer
# to the pair `structured_json="on" + web_search="on"` and gets its own
# capability field. Re-probing it here would be redundant.
ACTIVATABLE_CAPABILITIES: Final[dict[str, str]] = {
    "temperature":     "any",
    "thinking":        "on",
    "structured_json": "on",
    "web_search":      "on",
}


# Vocabulary lookup for incompatible-list validation. Maps each
# capability name to its allowed state tokens, so a catalog entry like
# {"temperature": "any", "thinking": "on"} can be checked field-by-field.
_INCOMPATIBLE_VOCABS: Final[dict[str, tuple[str, ...]]] = {
    "temperature":      TEMPERATURE_STATES,
    "thinking":         THINKING_STATES,
    "structured_json":  STRUCTURED_JSON_STATES,
    "web_search":       WEB_SEARCH_STATES,
    "json_with_search": JSON_WITH_SEARCH_STATES,
}


def _coerce_states(raw: Any, vocab: tuple[str, ...]) -> Optional[list[str]]:
    """Normalize a raw capability value into ``list[str] | None``.

    Accepts the new list shape and the legacy bool / single-string shapes
    so older catalog files keep loading. Migration rules:

    * ``None``      → ``None`` (unknown).
    * ``True``      → both states for two-token vocabularies (``["on","off"]``
                      and ``["any","default"]``); the full vocabulary
                      otherwise. This is the toggleable assumption — a
                      re-probe pass tightens always-on cases.
    * ``False``     → ``["off"]`` for ``ON_OFF`` vocabularies, ``["default"]``
                      for ``TEMPERATURE_STATES``; ``None`` otherwise.
    * ``str``       → ``[<value>]`` if the value is in the vocabulary, else
                      ``None``.
    * ``list[str]`` → kept; unknown tokens are filtered out. An empty result
                      after filtering becomes ``None``.
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        if raw:
            if vocab == ON_OFF_STATES:
                return ["on", "off"]
            if vocab == TEMPERATURE_STATES:
                return ["any", "default"]
            return list(vocab)
        if vocab == ON_OFF_STATES:
            return ["off"]
        if vocab == TEMPERATURE_STATES:
            return ["default"]
        return None
    if isinstance(raw, str):
        return [raw] if raw in vocab else None
    if isinstance(raw, list):
        kept = [v for v in raw if isinstance(v, str) and v in vocab]
        return kept or None
    return None


def _coerce_incompatible(raw: Any) -> Optional[list[dict[str, str]]]:
    """Normalize the raw `incompatible` value into ``list[dict] | None``.

    Each entry must be a dict mapping a known capability name (a key in
    ``_INCOMPATIBLE_VOCABS``) to a state token in that capability's
    vocabulary. Invalid keys/values are dropped; if an entry has any
    invalid pairs after filtering, the whole entry is dropped.

    Returns:
        * ``None`` — unknown / not yet probed. Runtime treats this as
          "no constraints recorded; do not pre-flight."
        * ``[]`` — probed; no incompatibilities found.
        * non-empty list — confirmed forbidden combinations.
    """
    if raw is None:
        return None
    if not isinstance(raw, list):
        return None
    out: list[dict[str, str]] = []
    for entry in raw:
        if not isinstance(entry, dict) or not entry:
            continue
        cleaned: dict[str, str] = {}
        ok = True
        for cap, state in entry.items():
            if not isinstance(cap, str) or not isinstance(state, str):
                ok = False
                break
            vocab = _INCOMPATIBLE_VOCABS.get(cap)
            if vocab is None or state not in vocab:
                ok = False
                break
            cleaned[cap] = state
        if ok and len(cleaned) >= 2:
            out.append(cleaned)
    return out


def _parse_deny_list(raw: Any, where: str) -> tuple[str, ...]:
    """Validate a list of deniable capability names; de-duplicated, in order.

    Shared by ``load_ai_config`` (each list in an entry's ``deny``) and
    ``get_provider(deny=...)``, so the vocabulary has one definition.

    Raises:
        ValueError: ``raw`` is not a list of names in ``DENIABLE_CAPABILITIES``.
    """
    if not isinstance(raw, (list, tuple)):
        raise ValueError(
            f"{where} must be a list of capability names from "
            f"{DENIABLE_CAPABILITIES}, got {type(raw).__name__}"
        )
    out: list[str] = []
    for cap in raw:
        if cap not in DENIABLE_CAPABILITIES:
            raise ValueError(
                f"{where}: {cap!r} is not a deniable capability; "
                f"use one of {DENIABLE_CAPABILITIES}"
            )
        if cap not in out:
            out.append(cap)
    return tuple(out)


def _parse_deny(raw: Any, where: str) -> dict[str, tuple[str, ...]]:
    """Normalize an entry's ``deny`` into ``{model_id | "*": capabilities}``.

    A list applies to every model (it becomes the ``"*"`` key); a map names
    models by Djinnite model ID, with ``"*"`` for every model. Model IDs are
    not checked against the catalog here: the config loads without it, and a
    model the catalog later drops must not stop the config from loading
    (``validate_ai`` reports unknown IDs).

    Raises:
        ValueError: wrong shape, an empty model key, or a capability outside
            ``DENIABLE_CAPABILITIES``.
    """
    if raw is None:
        return {}
    if isinstance(raw, list):
        caps = _parse_deny_list(raw, where)
        return {DENY_ALL_MODELS: caps} if caps else {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"{where} must be a list of capabilities from {DENIABLE_CAPABILITIES}, "
            f"or a map of model ID (or \"{DENY_ALL_MODELS}\") to such a list; "
            f"got {type(raw).__name__}"
        )
    out: dict[str, tuple[str, ...]] = {}
    for model_id, caps_raw in raw.items():
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError(
                f"{where}: model keys must be non-empty model IDs or \"{DENY_ALL_MODELS}\", "
                f"each mapped to capabilities from {DENIABLE_CAPABILITIES}"
            )
        caps = _parse_deny_list(caps_raw, f"{where}.{model_id}")
        if caps:
            out[model_id] = caps
    return out


# Configuration discovery
#
# Local project config with fallback to package defaults:
#   - PACKAGE_CONFIG_DIR: Djinnite's own config/ (ships with the distribution)
#   - PROJECT_CONFIG_DIR: Host project's config/ (user overrides + secrets)
#
# When reading a config file, the project-local copy takes priority.
# If not found there, the package default is used.  This means users
# only need ai_config.json in their project -- model_catalog.json and
# known_model_defaults.json are inherited from the package unless
# explicitly overridden.

# Package's own config (always exists -- ships with Djinnite)
PACKAGE_CONFIG_DIR = Path(__file__).parent / "config"


def _discover_project_config_dir() -> Optional[Path]:
    """Find the host project's config directory, if any."""
    # 1. Current Working Directory (standalone projects & CLI use)
    cwd_config = Path.cwd() / "config"
    if cwd_config.exists() and cwd_config.is_dir():
        if cwd_config.resolve() != PACKAGE_CONFIG_DIR.resolve():
            return cwd_config

    # 2. Parent of the package (submodule/integrated use)
    pkg_parent_config = Path(__file__).parent.parent / "config"
    if pkg_parent_config.exists() and pkg_parent_config.is_dir():
        if pkg_parent_config.resolve() != PACKAGE_CONFIG_DIR.resolve():
            return pkg_parent_config

    return None


PROJECT_CONFIG_DIR = _discover_project_config_dir()


def _resolve_config_file(filename: str) -> Path:
    """Resolve a config file: local project config with package fallback.

    Checks the project's config directory first.  If the file is not
    found there (or no project dir exists), falls back to the package's
    own config directory.
    """
    if PROJECT_CONFIG_DIR:
        project_file = PROJECT_CONFIG_DIR / filename
        if project_file.exists():
            return project_file
    return PACKAGE_CONFIG_DIR / filename


# Backward compatibility -- points to the project dir when available,
# otherwise the package dir.  Scripts use this for writes.
CONFIG_DIR = PROJECT_CONFIG_DIR or PACKAGE_CONFIG_DIR


@dataclass
class ProviderConfig:
    """One ai_config ``providers`` entry: a provider type reached one way.

    The entry's key in ``providers`` is its ``name``, chosen by the user;
    ``provider`` is the provider type (a ``PROVIDER_TYPES`` value) and
    defaults to the name, so ``"claude": {...}`` is a Claude entry. Several
    entries may share a type -- e.g. ``"claude"`` direct and
    ``"claude-vertex"`` on Vertex AI (ACCESS_PATHS_DESIGN.md).

    ``mode`` is ``"direct"`` (the provider's own API, authenticated with
    ``api_key``) or ``"platform"`` (a cloud platform serving the provider's
    models, named by ``platform`` and authenticated with the platform's own
    credentials -- no ``api_key`` needed). Platform-mode settings
    (``project_id``, ``location``, ``quota_project``) are resolved from the
    entry first, then from the top-level ``platforms.<name>`` block.

    ``deny`` maps a model ID, or ``"*"`` for every model, to capabilities
    this deployment does not allow (see ``denied_for``); ``deny_reason`` is
    shown in the error a denied request raises.
    """
    api_key: str = ""
    enabled: bool = True
    default_model: str = ""
    use_cases: dict[str, str] = field(default_factory=dict)
    # Legacy Gemini switch: 'gemini' (AI Studio) or 'vertexai'. A value of
    # 'vertexai' is read as mode="platform", platform="vertexai".
    backend: str = "gemini"
    project_id: Optional[str] = None
    modality_policy: dict[str, bool] = field(default_factory=dict)  # Policy for allowing/disabling modalities
    mode: str = "direct"
    platform: Optional[str] = None
    location: Optional[str] = None
    quota_project: Optional[str] = None
    name: str = ""
    provider: str = ""
    deny: dict[str, tuple[str, ...]] = field(default_factory=dict)
    deny_reason: str = ""

    def denied_for(self, model: Optional[str]) -> tuple[str, ...]:
        """Capabilities denied for ``model``: the ``"*"`` list plus the model's own."""
        out = list(self.deny.get(DENY_ALL_MODELS, ()))
        for cap in self.deny.get(model, ()) if model else ():
            if cap not in out:
                out.append(cap)
        return tuple(out)


class ModelChoice(NamedTuple):
    """Which entry, provider type and model a use case resolves to."""
    entry: str
    provider_type: str
    model: str


# Keyword arguments build_provider() accepts on top of an entry. Anything
# that would change what the entry *is* (its platform, mode or restrictions)
# is not among them: configure another entry instead.
_BUILD_OVERRIDES: Final[tuple[str, ...]] = (
    "api_key", "location", "project_id", "quota_project", "require_pricing",
)
_PLATFORM_OVERRIDES: Final[tuple[str, ...]] = ("location", "project_id", "quota_project")


def _factory():
    """``ai_providers.get_provider``, imported on first use.

    Imported relative to this module so the providers and their error classes
    come from the same package copy as the caller's (this file is imported
    both as ``djinnite.config_loader`` and as top-level ``config_loader``).
    """
    try:
        from .ai_providers import get_provider
    except ImportError:
        from ai_providers import get_provider  # type: ignore
    return get_provider


@dataclass
class PlatformConfig:
    """Settings for one cloud platform (top-level ``platforms`` block).

    Platforms rarely offer a model-discovery endpoint, so the config says
    where to look: ``locations`` is the set ``probe_platform`` checks.
    ``project_id`` / ``quota_project`` are the defaults for every provider
    entry that uses this platform.
    """
    project_id: Optional[str] = None
    quota_project: Optional[str] = None
    locations: list[str] = field(default_factory=list)
    enabled: bool = True
    # Default location for entries on this platform that set none (else the
    # platform registry's default, e.g. "global" on Vertex AI).
    location: Optional[str] = None


@dataclass
class AIConfig:
    """Full AI configuration: named provider entries and platform settings.

    Keys of ``providers`` are **entry names**. An entry's provider type is
    ``provider_type(name)``; it equals the name unless the entry sets
    ``"provider"``. ``default_provider`` and the ``name`` argument of the
    methods below are entry names.
    """
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    default_provider: str = "gemini"
    modality_policy: dict[str, bool] = field(default_factory=dict)  # Global policy
    platforms: dict[str, PlatformConfig] = field(default_factory=dict)

    def provider_kwargs(self, name: str) -> dict:
        """Constructor kwargs for entry ``name``: ``get_provider(type, model=..., **kwargs)``.

        The one place the config -> provider mapping lives. Direct mode
        yields only ``api_key``; platform mode adds ``platform`` and its
        settings. ``api_key`` is ``None`` when the entry has none. The
        provider type is not included -- it is ``provider_type(name)`` -- and
        neither are the entry's ``deny`` restrictions. ``build_provider(name)``
        applies both.

        Raises:
            KeyError: ``name`` is not a configured entry.
        """
        pc = self.providers[name]
        kwargs: dict = {"api_key": pc.api_key or None}
        if pc.mode == "platform":
            kwargs["platform"] = pc.platform
            for key in ("project_id", "location", "quota_project"):
                value = getattr(pc, key)
                if value:
                    kwargs[key] = value
        return kwargs

    def provider_type(self, name: str) -> str:
        """The provider type of entry ``name`` (its ``provider``, else its name).

        Raises:
            KeyError: ``name`` is not a configured entry.
        """
        return self.providers[name].provider or name

    def is_usable(self, name: str) -> bool:
        """True if the entry has a known provider type and the credentials its mode needs.

        Direct mode needs a real API key (placeholders like
        ``"your-...-key-here"`` do not count); platform mode authenticates
        with the platform's own credentials and needs none.
        """
        pc = self.providers.get(name)
        if pc is None or self.provider_type(name) not in PROVIDER_TYPES:
            return False
        if pc.mode == "platform":
            return True
        return bool(pc.api_key) and "your" not in pc.api_key.lower()

    def entries_of_type(self, provider_type: str, *, mode: Optional[str] = None,
                        usable_only: bool = False) -> list[str]:
        """Names of the enabled entries of ``provider_type``, in config order.

        ``mode`` keeps only ``"direct"`` or ``"platform"`` entries;
        ``usable_only`` keeps only entries ``is_usable`` accepts.
        """
        return [
            name for name, pc in self.providers.items()
            if pc.enabled
            and self.provider_type(name) == provider_type
            and (mode is None or pc.mode == mode)
            and (not usable_only or self.is_usable(name))
        ]

    def direct_entry(self, provider_type: str) -> Optional[str]:
        """The direct-mode entry maintenance scripts use for ``provider_type``.

        Among enabled, usable, direct-mode entries of that type: the one named
        after the type; otherwise the only one; ``None`` if there are none.
        This is the rule for catalog maintenance and the live test harness,
        not runtime selection -- applications name their entry.

        Raises:
            ValueError: several such entries, none named after the type.
        """
        candidates = self.entries_of_type(provider_type, mode="direct", usable_only=True)
        if provider_type in candidates:
            return provider_type
        if len(candidates) <= 1:
            return candidates[0] if candidates else None
        raise ValueError(
            f"ai_config has several direct-mode entries of type {provider_type!r} "
            f"({', '.join(candidates)}) and none is named {provider_type!r}. "
            f"Catalog maintenance needs one: name one entry {provider_type!r}, "
            f"or disable the others."
        )

    def build_provider(self, name: str, model: Optional[str] = None, **overrides):
        """Build a ready provider from entry ``name``.

        Uses the entry's provider type, credentials, platform settings and
        ``deny`` restrictions (resolved for ``model``). ``model`` defaults to
        the entry's ``default_model``.

        Args:
            name: Entry name (a key of ``providers``).
            model: Model ID; defaults to the entry's ``default_model``.
            **overrides: Per-call settings: ``api_key``, ``require_pricing``,
                and, for platform entries, ``location``, ``project_id``,
                ``quota_project``. Settings that would change what the entry
                is -- its platform, mode or restrictions -- are refused:
                configure another entry instead.

        Returns:
            A ``BaseAIProvider`` whose ``entry`` is ``name``.

        Raises:
            ValueError: unknown or disabled entry, an entry whose type is not a
                provider type, a direct entry without a usable API key, or a
                refused override.
            AIProviderError: as ``get_provider`` (catalog, model, platform).
        """
        pc = self.providers.get(name)
        if pc is None:
            raise ValueError(
                f"No ai_config entry {name!r}. Entries: {', '.join(self.providers) or '(none)'}"
            )
        if not pc.enabled:
            raise ValueError(f"ai_config entry {name!r} is disabled")
        ptype = self.provider_type(name)
        if ptype not in PROVIDER_TYPES:
            raise ValueError(
                f"ai_config entry {name!r}: {ptype!r} is not a provider type "
                f"{PROVIDER_TYPES}. Add \"provider\": \"<type>\" to the entry."
            )
        refused = sorted(k for k in overrides if k not in _BUILD_OVERRIDES)
        if refused:
            raise ValueError(
                f"build_provider({name!r}): {', '.join(refused)} cannot be overridden "
                f"(allowed: {', '.join(_BUILD_OVERRIDES)}). To reach the provider "
                f"another way, configure another entry."
            )
        if pc.mode == "direct":
            platform_only = sorted(k for k in overrides if k in _PLATFORM_OVERRIDES)
            if platform_only:
                raise ValueError(
                    f"build_provider({name!r}): {', '.join(platform_only)} applies "
                    f"to platform entries only; {name!r} is direct mode"
                )
            if not overrides.get("api_key") and not self.is_usable(name):
                raise ValueError(
                    f"ai_config entry {name!r} has no usable api_key (direct mode needs one)"
                )
        model_id = model or pc.default_model or None
        kwargs = {**self.provider_kwargs(name), **overrides}
        return _factory()(
            ptype, model=model_id, entry=name,
            deny=list(pc.denied_for(model_id)), deny_reason=pc.deny_reason or None,
            **kwargs,
        )

    def get_provider(self, name: str) -> Optional[ProviderConfig]:
        """The entry ``name``, or ``None`` if it is not configured or disabled."""
        provider = self.providers.get(name)
        if provider and provider.enabled:
            return provider
        return None

    def get_default_provider(self) -> Optional[ProviderConfig]:
        """The ``default_provider`` entry (see ``get_provider``)."""
        return self.get_provider(self.default_provider)

    def get_model_for_use_case(self, use_case: str, provider_name: Optional[str] = None) -> tuple[str, str]:
        """
        Get the appropriate model for a use case.

        Returns:
            tuple of (entry_name, model_id). The entry name is not necessarily
            a provider type; ``resolve_use_case`` also returns the type.
        """
        choice = self.resolve_use_case(use_case, provider_name)
        return (choice.entry, choice.model)

    def resolve_use_case(self, use_case: str, entry: Optional[str] = None) -> ModelChoice:
        """The entry, provider type and model for ``use_case``.

        ``entry`` defaults to ``default_provider``. The model is the entry's
        ``use_cases[use_case]``, else its ``default_model``. Build it with
        ``build_provider(choice.entry, choice.model)``; look it up with
        ``catalog.get_model(choice.provider_type, choice.model)``.

        Raises:
            ValueError: the entry is not configured or is disabled.
        """
        entry = entry or self.default_provider
        pc = self.get_provider(entry)
        if not pc:
            raise ValueError(f"Provider '{entry}' not found or disabled")
        return ModelChoice(entry, self.provider_type(entry),
                           pc.use_cases.get(use_case, pc.default_model))

    def capabilities_for(self, name: str, model: str,
                         catalog: Optional["ModelCatalog"] = None) -> "ModelCapabilities":
        """What a request through entry ``name`` to ``model`` may use.

        The catalog's direct-mode capabilities, overlaid with those probed on
        the entry's platform (``ModelInfo.for_platform``), narrowed by the
        entry's ``deny`` (``ModelInfo.with_denied``). For inspection: requests
        are checked by the same layers in pre-flight.

        Raises:
            KeyError: ``name`` is not a configured entry, or ``model`` is not
                in the catalog for the entry's type.
        """
        pc = self.providers[name]
        ptype = self.provider_type(name)
        catalog = catalog or load_model_catalog()
        info = catalog.get_model(ptype, model)
        if info is None:
            raise KeyError(f"model {model!r} is not in the catalog for provider type {ptype!r}")
        if pc.mode == "platform":
            info = info.for_platform(pc.platform)
        return info.with_denied(pc.denied_for(model)).capabilities


def _parse_vision_limit(value) -> Optional[float]:
    """Parse a vision limit value from JSON.

    Returns:
        None       -- unknown / not yet discovered
        float('inf') -- confirmed unlimited
        positive float -- actual limit
    """
    if value is None:
        return None
    if value == "inf" or value == float('inf'):
        return float('inf')
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    return None


def _serialize_vision_limit(value: Optional[float]):
    """Serialize a vision limit value for JSON.

    float('inf') -> "inf", None -> None, otherwise the numeric value.
    """
    if value is None:
        return None
    if value == float('inf'):
        return "inf"
    if value == int(value):
        return int(value)
    return value


@dataclass
class VisionLimits:
    """Image input constraints for vision-capable models.

    Limit semantics:
        None         -- unknown / not yet discovered (fail-open)
        float('inf') -- confirmed unlimited (no constraint)
        positive number -- actual limit

    In JSON, float('inf') is stored as the string "inf".
    """
    max_image_bytes: Optional[float] = None       # Max bytes per image (e.g., 5242880 for 5 MB)
    max_dimension_px: Optional[float] = None      # Max width or height in pixels (e.g., 8000)
    max_images_per_request: Optional[float] = None # Max images in a single request
    supported_formats: list[str] = field(default_factory=list)  # e.g., ["jpeg", "png", "gif", "webp"]


@dataclass
class ModelCosting:
    """Dollar-based pricing for an AI model.

    Attributes:
        input_per_1m: Dollar cost per 1 million input tokens.
        output_per_1m: Dollar cost per 1 million output tokens.
            Thinking/reasoning tokens are billed at this rate.
        source: How the pricing was determined. One of:
            ``default``   placeholder, never priced.
            ``manual``    human-fixed; never touched by the updater.
            ``published`` priced from an official page with a ``source_url``.
            ``estimated`` priced via AI web search, weak/no source URL.
            ``unknown``   no public per-1M-token price exists (specialty model);
                          stable terminal state that requires human review.
            ``failed``    estimation errored/malformed; transient, retried.
        updated: ISO date (YYYY-MM-DD) when the pricing was last updated.
        search_cost_per_unit: Dollar cost per billable web search event.
            Varies by model.  None means unknown or web search not supported.
        pricing_class: ``fixed`` or ``floating`` (see ``pricing_class.py``).
            ``floating`` ids are re-priced every run; ``fixed`` ids only when
            missing.  None means not yet classified.
        pricing_class_source: ``auto`` (classifier) or ``manual`` (human pinned).
        source_url: Official pricing page the figure was taken from, if any.
        published_figure: The vendor's price text, quoted verbatim, naming the
            tier it came from (e.g. ``"$30 / $180 per 1M (Standard)"``). Human
            cross-check only -- never used for computation.

            Vendors publish several rates for one model: a Standard service
            tier, a discounted Flex/Batch tier, and a higher context-length
            tier above some input-token threshold. All three are "the price"
            on the same page. This field records WHICH one the stored number
            is, so a tier mix-up is visible instead of looking like a price
            change. gpt-5.4-pro oscillated $30/$180 <-> $15/$90 across two
            consecutive runs -- Standard vs Flex -- and nothing in the catalog
            could tell them apart.

            NOTE: ``input_per_1m`` / ``output_per_1m`` are ALWAYS the Standard
            service tier at the BASE context tier. See DEVELOPMENT.md
            "Known limitation: context-length pricing tiers".
    """
    input_per_1m: Optional[float] = None
    output_per_1m: Optional[float] = None
    source: str = "default"
    updated: str = ""
    search_cost_per_unit: Optional[float] = None
    pricing_class: Optional[str] = None
    pricing_class_source: str = "auto"
    source_url: Optional[str] = None
    published_figure: Optional[str] = None

    def is_unverified(self) -> bool:
        """True if this price was never checked against a published source.

        ``source="estimated"`` means a model guessed the number. That is not
        the same as a price read off the vendor's pricing page.  Treating the
        two alike is what let ``gemini-flash-latest`` sit at an estimated
        $0.50/$3.00 while the real price was $1.50/$7.50.
        """
        if self.input_per_1m is None and self.output_per_1m is None:
            return False  # no price to verify
        if self.source in ("manual", "published"):
            return False
        return True


@dataclass
class Modalities:
    """Input and output modality capabilities."""
    input: list[str] = field(default_factory=lambda: ["text"])
    output: list[str] = field(default_factory=lambda: ["text"])


@dataclass
class ModelCapabilities:
    """
    Per-model lists of supported Djinnite-API states.

    Every field is ``list[str] | None``. ``None`` means unknown / not yet
    probed (runtime pre-flight is skipped). A non-null list is the subset
    of the capability's fixed Djinnite vocabulary that this model accepts.

    Attributes:
        structured_json: Subset of ``STRUCTURED_JSON_STATES``. ``"on"`` means
            the model accepts a schema; ``"off"`` is plain text.
        temperature: Subset of ``TEMPERATURE_STATES``. ``"any"`` means the
            model accepts a caller-specified temperature; ``"default"`` means
            the model's default is used (caller's value is stripped).
        thinking: Subset of ``THINKING_STATES``. ``"on"`` means the model
            accepts a thinking-enabled request; ``"off"`` means it accepts
            an explicit thinking-disabled request. ``["on"]`` is an
            always-on reasoning model; ``["off"]`` is a non-thinking model.
        web_search: Subset of ``WEB_SEARCH_STATES``.
        json_with_search: Subset of ``JSON_WITH_SEARCH_STATES`` — whether the
            structured-JSON + web-search combination is accepted.
        thinking_style: Subset of ``THINKING_STYLE_VALUES``. Identifies the
            provider-native thinking param style(s) the model accepts; a
            single model may support multiple (Claude 4.7: adaptive+budget).
        effort_levels: The effort strings this model accepts, when
            ``thinking_style`` contains ``"effort"``. Levels vary per model
            within a provider -- Claude Opus 4.5 takes low/medium/high while
            Opus 5 also takes xhigh/max -- so the provider-wide vocabulary
            is not sufficient to pre-flight a request. ``None`` means
            unknown (no pre-flight); the provider vocabulary is used as the
            only check.
        incompatible: List of forbidden capability-state combinations.
            Each entry is a ``dict[str, str]`` mapping capability name to
            state token (from that capability's vocabulary). Semantics:
            if a request simultaneously selects every state in the dict,
            the request is invalid and Djinnite raises before any HTTP
            call. ``None`` means "not yet probed"; ``[]`` means "probed,
            no incompatibilities found." Discovered by the combinatorial
            probe orchestrator; enforced by
            ``BaseAIProvider._validate_incompatible_combinations``.
    """
    structured_json: Optional[list[str]] = None
    temperature: Optional[list[str]] = None
    thinking: Optional[list[str]] = None
    web_search: Optional[list[str]] = None
    json_with_search: Optional[list[str]] = None
    thinking_style: Optional[list[str]] = None
    effort_levels: Optional[list[str]] = None
    incompatible: Optional[list[dict[str, str]]] = None


@dataclass
class PlatformModelInfo:
    """What a cloud platform offers for one model (catalog ``platforms.<name>``).

    Generated by ``scripts/probe_platform.py``; never hand-edited (pin a
    value through ``model_overrides.json`` instead).

    Attributes:
        locations: Location -> status from ``LOCATION_STATUS_VALUES``.
            Informational: runtime never blocks a call on it, because a stale
            ``not_found`` would hide a model the platform has since added.
        capabilities: Capabilities probed *on the platform*, or ``None`` if
            only availability was probed. A field left ``None`` falls back
            to the direct-mode value (see ``ModelInfo.for_platform``).
        probed: ISO date of the probe run.
    """
    locations: dict[str, str] = field(default_factory=dict)
    capabilities: Optional[ModelCapabilities] = None
    probed: str = ""


@dataclass
class ModelInfo:
    """Information about a single AI model.
    
    Attributes:
        id: The model ID (e.g. "gemini-2.5-flash", "claude-sonnet-4-20250514")
        name: Human-readable display name
        context_window: Total token budget for a request — sum of input,
            output, thinking, and tool-call round-trips. This is the
            provider's published "context window" for the model (e.g.
            Claude Sonnet 4 = 200,000; Gemini 2.5 Flash = 1,048,576).
            It is **not** an input-only cap; the provider enforces it
            server-side and exceeding it raises ``AIContextLengthError``.
        max_output_tokens: Maximum tokens the model may emit in a single
            response. Callers should use this to set ``max_output_tokens``
            on ``generate()`` / ``generate_json()`` and avoid truncation.
            A value of 0 means unknown. Per-provider semantic note: Claude
            and Gemini cap visible output only (thinking is counted under
            a separate budget); OpenAI's ``max_output_tokens`` caps
            visible output and reasoning combined.
        capabilities: Per-model lists of supported Djinnite-API states
            (see ``ModelCapabilities``).
        modalities: Input/output modality capabilities
        costing: Dollar-based pricing (input/output per 1M tokens, search per unit)
        disabled: Whether this model is disabled (blocked from use at runtime)
        disabled_reason: Human-readable explanation for why the model is disabled
        platforms: Per-platform facts (``PlatformModelInfo``) keyed by
            platform name. The other fields are direct-mode facts.
    """
    id: str
    name: str
    context_window: int
    max_output_tokens: int = 0
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    modalities: Modalities = field(default_factory=Modalities)
    costing: ModelCosting = field(default_factory=ModelCosting)
    vision_limits: Optional[VisionLimits] = None
    disabled: bool = False
    disabled_reason: str = ""
    platforms: dict[str, PlatformModelInfo] = field(default_factory=dict)

    def for_platform(self, name: Optional[str]) -> "ModelInfo":
        """This model as seen through platform ``name``.

        Capabilities probed on the platform replace the direct-mode values
        field by field; a platform field that is ``None`` (not probed, or
        inconclusive) keeps the direct-mode value. Everything else --
        costing, limits, modalities -- is shared. Returns ``self`` when the
        platform has no recorded capabilities (or ``name`` is ``None``).
        """
        pm = self.platforms.get(name) if name else None
        if pm is None or pm.capabilities is None:
            return self
        from dataclasses import fields, replace
        merged = replace(self.capabilities)
        for f in fields(ModelCapabilities):
            value = getattr(pm.capabilities, f.name)
            if value is not None:
                setattr(merged, f.name, value)
        return replace(self, capabilities=merged)

    def with_denied(self, deny) -> "ModelInfo":
        """This model with the capabilities in ``deny`` unavailable.

        Each denied capability loses its ``"on"`` state; an unknown (``None``)
        or emptied list becomes ``["off"]``. ``json_with_search`` is
        structured JSON and web search together, so denying either part
        takes it off too -- as pre-flight does, since such a request carries
        both. Used to *show* an ai_config entry's effective view
        (``AIConfig.capabilities_for``); pre-flight enforces ``deny``
        separately so its error names the entry.
        """
        if not deny:
            return self
        from dataclasses import replace
        merged = replace(self.capabilities)
        caps = list(deny)
        if ("structured_json" in caps or "web_search" in caps) and "json_with_search" not in caps:
            caps.append("json_with_search")
        for cap in caps:
            current = getattr(merged, cap)
            kept = [s for s in (current or []) if s != "on"]
            setattr(merged, cap, kept or ["off"])
        return replace(self, capabilities=merged)

    @property
    def supports_structured_json(self) -> Optional[bool]:
        """Convenience accessor: True iff "on" is in the structured_json list."""
        ssj = self.capabilities.structured_json
        if ssj is None:
            return None
        return "on" in ssj


@dataclass
class ModelCatalog:
    """Catalog of available models across all providers."""
    providers: dict[str, list[ModelInfo]] = field(default_factory=dict)
    
    def get_model(self, provider: str, model_id: str) -> Optional[ModelInfo]:
        """Get model info by provider and model ID."""
        models = self.providers.get(provider, [])
        for model in models:
            if model.id == model_id:
                return model
        return None
    
    def list_models(self, provider: str) -> list[ModelInfo]:
        """List all models for a provider."""
        return self.providers.get(provider, [])

    def find_models(self, 
                    input_modality: Optional[str] = None, 
                    output_modality: Optional[str] = None,
                    provider: Optional[str] = None) -> list[tuple[str, ModelInfo]]:
        """
        Find models across providers that support specific input and/or output modalities.
        
        Args:
            input_modality: Filter by input capability (e.g. 'vision', 'audio')
            output_modality: Filter by output capability (e.g. 'audio', 'text')
            provider: Limit search to a specific provider
            
        Returns:
            List of (provider_name, ModelInfo) tuples
        """
        results = []
        providers_to_search = [provider] if provider else self.providers.keys()
        
        for p_name in providers_to_search:
            for model in self.providers.get(p_name, []):
                match = True
                if input_modality and input_modality not in model.modalities.input:
                    match = False
                if output_modality and output_modality not in model.modalities.output:
                    match = False
                
                if match:
                    results.append((p_name, model))
        return results


def _no_duplicate_keys(path: Path):
    """``object_pairs_hook`` that raises on a key repeated in one JSON object.

    Plain ``json.load`` keeps the last of two equal keys and drops the other
    without a word -- in ai_config.json, a whole provider entry.
    """
    def hook(pairs):
        obj: dict = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError(
                    f"{path}: duplicate key {key!r} in one JSON object (its keys: "
                    f"{', '.join(k for k, _ in pairs)}). JSON keeps only the last "
                    f"one and silently drops the other, so every key must be "
                    f"unique. To configure one provider type twice, give the "
                    f"second entry another name and set \"provider\": \"<type>\"."
                )
            obj[key] = value
        return obj
    return hook


def load_json_file(path: Path, default: Any = None, *, reject_duplicate_keys: bool = False) -> Any:
    """
    Load a JSON file, returning default if file doesn't exist.

    Args:
        path: Path to the JSON file
        default: Default value to return if file doesn't exist
        reject_duplicate_keys: Raise ``ValueError`` when an object repeats a
            key, instead of keeping the last one (``load_ai_config`` sets it).

    Returns:
        Parsed JSON data or default value
    """
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        if reject_duplicate_keys:
            return json.load(f, object_pairs_hook=_no_duplicate_keys(path))
        return json.load(f)


def load_ai_config(config_path: Optional[Path] = None) -> AIConfig:
    """
    Load AI provider configuration.
    
    Args:
        config_path: Optional custom path to ai_config.json
        
    Returns:
        AIConfig object with provider settings

    Raises:
        ValueError: a duplicate key anywhere in the file, an entry whose
            ``provider``, ``mode``, ``platform`` or ``deny`` is invalid.
    """
    path = config_path or _resolve_config_file("ai_config.json")

    data = load_json_file(path, reject_duplicate_keys=True)

    platforms: dict[str, PlatformConfig] = {}
    for pname, pdata in (data.get("platforms") or {}).items():
        if pname.startswith("_") or not isinstance(pdata, dict):
            continue  # notes / malformed entries
        platforms[pname] = PlatformConfig(
            project_id=pdata.get("project_id"),
            quota_project=pdata.get("quota_project"),
            locations=list(pdata.get("locations") or []),
            enabled=pdata.get("enabled", True),
            location=pdata.get("location"),
        )

    providers = {}
    for name, provider_data in data.get("providers", {}).items():
        if name.startswith("_"):
            continue  # notes, as under "platforms"
        if not isinstance(provider_data, dict):
            raise ValueError(
                f"ai_config providers.{name} must be an object, got {type(provider_data).__name__}"
            )
        # Provider type: explicit "provider", else the entry's name. An
        # explicit unknown type is an error; an implicit one (an old entry
        # such as "openai") still loads and is reported unusable.
        ptype = provider_data.get("provider")
        if "provider" in provider_data and ptype not in PROVIDER_TYPES:
            raise ValueError(
                f"ai_config providers.{name}.provider must be one of "
                f"{PROVIDER_TYPES}, got {ptype!r}"
            )
        ptype = ptype or name
        deny_reason = provider_data.get("deny_reason", "") or ""
        if not isinstance(deny_reason, str):
            raise ValueError(f"ai_config providers.{name}.deny_reason must be a string")

        backend = provider_data.get("backend", "gemini")
        platform = provider_data.get("platform")
        # Legacy: backend="vertexai" predates platform mode.
        if platform is None and backend == "vertexai":
            platform = "vertexai"
        mode = provider_data.get("mode") or ("platform" if platform else "direct")
        if mode not in ACCESS_MODES:
            raise ValueError(
                f"ai_config providers.{name}.mode must be one of {ACCESS_MODES}, got {mode!r}"
            )
        if mode == "platform" and not platform:
            raise ValueError(
                f"ai_config providers.{name}: mode 'platform' requires a 'platform' name"
            )
        if mode == "direct" and provider_data.get("platform"):
            raise ValueError(
                f"ai_config providers.{name}: 'platform' is set but mode is 'direct'"
            )

        # Platform settings: the entry wins, then the platform block.
        pconf = platforms.get(platform) if mode == "platform" else None
        def _setting(key):
            value = provider_data.get(key)
            if value is None and pconf is not None:
                value = getattr(pconf, key, None)
            return value

        providers[name] = ProviderConfig(
            api_key=provider_data.get("api_key", ""),
            enabled=provider_data.get("enabled", True),
            default_model=provider_data.get("default_model", ""),
            use_cases=provider_data.get("use_cases", {}),
            backend=backend,
            project_id=_setting("project_id"),
            modality_policy=provider_data.get("modality_policy", {}),
            mode=mode,
            platform=platform if mode == "platform" else None,
            location=_setting("location"),
            quota_project=_setting("quota_project"),
            name=name,
            provider=ptype,
            deny=_parse_deny(provider_data.get("deny"), f"ai_config providers.{name}.deny"),
            deny_reason=deny_reason,
        )

    return AIConfig(
        providers=providers,
        default_provider=data.get("default_provider", "gemini"),
        modality_policy=data.get("modality_policy", {}),
        platforms=platforms,
    )


def load_model_catalog(catalog_path: Optional[Path] = None) -> ModelCatalog:
    """
    Load the model catalog.
    
    Args:
        catalog_path: Optional custom path to model_catalog.json
        
    Returns:
        ModelCatalog object with available models
    """
    path = catalog_path or _resolve_config_file("model_catalog.json")

    data = load_json_file(path)

    providers = {}
    for provider_name, provider_data in data.items():
        models = []
        for model_data in provider_data.get("models", []):
            costing_data = model_data.get("costing", {})

            costing = ModelCosting(
                input_per_1m=costing_data.get("input_per_1m"),
                output_per_1m=costing_data.get("output_per_1m"),
                source=costing_data.get("source", "default"),
                updated=costing_data.get("updated", ""),
                search_cost_per_unit=costing_data.get("search_cost_per_unit"),
                pricing_class=costing_data.get("pricing_class"),
                pricing_class_source=costing_data.get("pricing_class_source", "auto"),
                source_url=costing_data.get("source_url"),
                published_figure=costing_data.get("published_figure"),
            )

            # Handle modalities schema evolution
            raw_modalities = model_data.get("modalities")
            if isinstance(raw_modalities, dict):
                modalities = Modalities(
                    input=raw_modalities.get("input", ["text"]),
                    output=raw_modalities.get("output", ["text"])
                )
            elif isinstance(raw_modalities, list):
                # Fallback: assume list means input capabilities, output is text
                modalities = Modalities(input=raw_modalities, output=["text"])
            else:
                # Default for old models
                caps = model_data.get("capabilities", ["text"])
                modalities = Modalities(input=caps, output=["text"])
            
            # Load capabilities — support new list format, old tri-state-bool
            # format, and the original flat supports_structured_json field.
            raw_caps = model_data.get("capabilities")
            if isinstance(raw_caps, dict):
                caps = _parse_capabilities(raw_caps)
            else:
                # Old format: migrate from flat supports_structured_json field
                raw_ssj = model_data.get("supports_structured_json")
                caps = ModelCapabilities(
                    structured_json=_coerce_states(raw_ssj, ON_OFF_STATES),
                )

            # Load vision limits if present
            raw_vl = model_data.get("vision_limits")
            vision_limits = None
            if isinstance(raw_vl, dict):
                vision_limits = VisionLimits(
                    max_image_bytes=_parse_vision_limit(raw_vl.get("max_image_bytes")),
                    max_dimension_px=_parse_vision_limit(raw_vl.get("max_dimension_px")),
                    max_images_per_request=_parse_vision_limit(raw_vl.get("max_images_per_request")),
                    supported_formats=raw_vl.get("supported_formats", []),
                )

            models.append(ModelInfo(
                id=model_data["id"],
                name=model_data["name"],
                context_window=model_data.get("context_window", 0),
                max_output_tokens=model_data.get("max_output_tokens", 0),
                capabilities=caps,
                modalities=modalities,
                costing=costing,
                vision_limits=vision_limits,
                disabled=model_data.get("disabled", False),
                disabled_reason=model_data.get("disabled_reason", ""),
                platforms=_parse_platforms(model_data.get("platforms")),
            ))
        providers[provider_name] = models

    return ModelCatalog(providers=providers)


def _parse_capabilities(raw_caps: dict) -> ModelCapabilities:
    """Build ``ModelCapabilities`` from a catalog ``capabilities`` dict."""
    return ModelCapabilities(
        structured_json=_coerce_states(raw_caps.get("structured_json"), ON_OFF_STATES),
        temperature=_coerce_states(raw_caps.get("temperature"), TEMPERATURE_STATES),
        thinking=_coerce_states(raw_caps.get("thinking"), ON_OFF_STATES),
        web_search=_coerce_states(raw_caps.get("web_search"), ON_OFF_STATES),
        json_with_search=_coerce_states(raw_caps.get("json_with_search"), ON_OFF_STATES),
        thinking_style=_coerce_states(raw_caps.get("thinking_style"), THINKING_STYLE_VALUES),
        effort_levels=_coerce_states(raw_caps.get("effort_levels"), EFFORT_LEVEL_VALUES),
        incompatible=_coerce_incompatible(raw_caps.get("incompatible")),
    )


def _parse_platforms(raw: Any) -> dict[str, PlatformModelInfo]:
    """Parse a model's catalog ``platforms`` block. Unknown statuses are dropped."""
    out: dict[str, PlatformModelInfo] = {}
    if not isinstance(raw, dict):
        return out
    for pname, pdata in raw.items():
        if not isinstance(pdata, dict):
            continue
        locations = {
            str(loc): status
            for loc, status in (pdata.get("locations") or {}).items()
            if status in LOCATION_STATUS_VALUES
        }
        raw_caps = pdata.get("capabilities")
        out[pname] = PlatformModelInfo(
            locations=locations,
            capabilities=_parse_capabilities(raw_caps) if isinstance(raw_caps, dict) else None,
            probed=pdata.get("probed", "") or "",
        )
    return out


if __name__ == "__main__":
    # Test loading configs
    print("Testing Djinnite configuration loader...")
    
    # Test AI config
    ai_config = load_ai_config()
    print(f"AI Config loaded. Default provider: {ai_config.default_provider}")
    print(f"Available providers: {list(ai_config.providers.keys())}")
    
    # Test model catalog
    catalog = load_model_catalog()
    print(f"\nModel Catalog loaded. Providers: {list(catalog.providers.keys())}")
    for provider, models in catalog.providers.items():
        print(f"  {provider}: {[m.id for m in models]}")
    
    print("\nDjinnite configuration loader test complete.")


__all__ = [
    "AIConfig",
    "ProviderConfig",
    "PlatformConfig",
    "PlatformModelInfo",
    "ModelInfo",
    "ModelCapabilities",
    "ModelCatalog",
    "Modalities",
    "VisionLimits",
    "load_ai_config",
    "load_model_catalog",
    "CONFIG_DIR",
    "PACKAGE_CONFIG_DIR",
    "PROJECT_CONFIG_DIR",
    "_resolve_config_file",
    "ON_OFF_STATES",
    "THINKING_STATES",
    "STRUCTURED_JSON_STATES",
    "WEB_SEARCH_STATES",
    "JSON_WITH_SEARCH_STATES",
    "TEMPERATURE_STATES",
    "THINKING_STYLE_VALUES",
    "ACTIVATABLE_CAPABILITIES",
    "ACCESS_MODES",
    "LOCATION_STATUS_VALUES",
    "PROVIDER_TYPES",
    "DENIABLE_CAPABILITIES",
    "DENY_ALL_MODELS",
    "ModelChoice",
]
