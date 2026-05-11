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
from typing import Any, Final, Optional
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
THINKING_STYLE_VALUES: Final[tuple[str, ...]] = ("adaptive", "budget", "effort")


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
    """Configuration for a single AI provider."""
    api_key: str
    enabled: bool = True
    default_model: str = ""
    use_cases: dict[str, str] = field(default_factory=dict)
    # Backend-specific fields
    backend: str = "gemini"  # For Google: 'gemini' (AI Studio) or 'vertexai'
    project_id: Optional[str] = None  # Required for Vertex AI
    modality_policy: dict[str, bool] = field(default_factory=dict)  # Policy for allowing/disabling modalities


@dataclass
class AIConfig:
    """Full AI configuration with all providers."""
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    default_provider: str = "gemini"
    modality_policy: dict[str, bool] = field(default_factory=dict)  # Global policy
    
    def get_provider(self, name: str) -> Optional[ProviderConfig]:
        """Get provider config by name, returns None if not found or disabled."""
        provider = self.providers.get(name)
        if provider and provider.enabled:
            return provider
        return None
    
    def get_default_provider(self) -> Optional[ProviderConfig]:
        """Get the default provider configuration."""
        return self.get_provider(self.default_provider)
    
    def get_model_for_use_case(self, use_case: str, provider_name: Optional[str] = None) -> tuple[str, str]:
        """
        Get the appropriate model for a use case.
        
        Returns:
            tuple of (provider_name, model_id)
        """
        provider_name = provider_name or self.default_provider
        provider = self.get_provider(provider_name)
        
        if not provider:
            raise ValueError(f"Provider '{provider_name}' not found or disabled")
        
        # Check if there's a specific model for this use case
        model = provider.use_cases.get(use_case, provider.default_model)
        return (provider_name, model)


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
        source: How the pricing was determined
            (estimated/manual/failed).
        updated: ISO date when the pricing was last updated.
        search_cost_per_unit: Dollar cost per billable web search event.
            Varies by model.  None means unknown or web search not supported.
    """
    input_per_1m: Optional[float] = None
    output_per_1m: Optional[float] = None
    source: str = "default"
    updated: str = ""
    search_cost_per_unit: Optional[float] = None


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
    incompatible: Optional[list[dict[str, str]]] = None


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


def load_json_file(path: Path, default: Any = None) -> Any:
    """
    Load a JSON file, returning default if file doesn't exist.
    
    Args:
        path: Path to the JSON file
        default: Default value to return if file doesn't exist
        
    Returns:
        Parsed JSON data or default value
    """
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_ai_config(config_path: Optional[Path] = None) -> AIConfig:
    """
    Load AI provider configuration.
    
    Args:
        config_path: Optional custom path to ai_config.json
        
    Returns:
        AIConfig object with provider settings
    """
    path = config_path or _resolve_config_file("ai_config.json")
    
    data = load_json_file(path)

    providers = {}
    for name, provider_data in data.get("providers", {}).items():
        providers[name] = ProviderConfig(
            api_key=provider_data.get("api_key", ""),
            enabled=provider_data.get("enabled", True),
            default_model=provider_data.get("default_model", ""),
            use_cases=provider_data.get("use_cases", {}),
            backend=provider_data.get("backend", "gemini"),
            project_id=provider_data.get("project_id"),
            modality_policy=provider_data.get("modality_policy", {})
        )
    
    return AIConfig(
        providers=providers,
        default_provider=data.get("default_provider", "gemini"),
        modality_policy=data.get("modality_policy", {})
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
                caps = ModelCapabilities(
                    structured_json=_coerce_states(raw_caps.get("structured_json"), ON_OFF_STATES),
                    temperature=_coerce_states(raw_caps.get("temperature"), TEMPERATURE_STATES),
                    thinking=_coerce_states(raw_caps.get("thinking"), ON_OFF_STATES),
                    web_search=_coerce_states(raw_caps.get("web_search"), ON_OFF_STATES),
                    json_with_search=_coerce_states(raw_caps.get("json_with_search"), ON_OFF_STATES),
                    thinking_style=_coerce_states(raw_caps.get("thinking_style"), THINKING_STYLE_VALUES),
                    incompatible=_coerce_incompatible(raw_caps.get("incompatible")),
                )
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
            ))
        providers[provider_name] = models
    
    return ModelCatalog(providers=providers)


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
]
