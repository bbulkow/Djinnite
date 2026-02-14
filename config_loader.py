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
from typing import Any, Optional
from dataclasses import dataclass, field


# Configuration discovery
# We check two locations for the 'config/' directory:
# 1. Current Working Directory (handles standalone projects & CLI use)
# 2. Parent of the package (handles submodule/integrated use)

def _discover_config_dir() -> Path:
    """Find the best configuration directory."""
    cwd_config = Path.cwd() / "config"
    if cwd_config.exists() and cwd_config.is_dir():
        return cwd_config
    
    # Fallback to parent of package (submodule case)
    pkg_parent_config = Path(__file__).parent.parent / "config"
    return pkg_parent_config

CONFIG_DIR = _discover_config_dir()


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


@dataclass
class ModelCosting:
    """Cost-related information for an AI model."""
    score: float = 1.0
    source: str = "default"
    updated: str = ""
    tier: str = "standard"


@dataclass
class Modalities:
    """Input and output modality capabilities."""
    input: list[str] = field(default_factory=lambda: ["text"])
    output: list[str] = field(default_factory=lambda: ["text"])


@dataclass
class ModelInfo:
    """Information about a single AI model."""
    id: str
    name: str
    context_window: int
    capabilities: list[str] = field(default_factory=list)
    modalities: Modalities = field(default_factory=Modalities)
    costing: ModelCosting = field(default_factory=ModelCosting)


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
    path = config_path or CONFIG_DIR / "ai_config.json"
    
    try:
        data = load_json_file(path)
    except FileNotFoundError:
        # Return empty config if file doesn't exist
        print(f"Warning: AI config not found at {path}. Using empty configuration.")
        return AIConfig()
    
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
    path = catalog_path or CONFIG_DIR / "model_catalog.json"
    
    try:
        data = load_json_file(path)
    except FileNotFoundError:
        print(f"Warning: Model catalog not found at {path}. Using empty catalog.")
        return ModelCatalog()
    
    providers = {}
    for provider_name, provider_data in data.items():
        models = []
        for model_data in provider_data.get("models", []):
            # Support both old and new costing schema
            costing_data = model_data.get("costing", {})
            costing = ModelCosting(
                score=costing_data.get("score", model_data.get("cost_score", 1.0)),
                source=costing_data.get("source", model_data.get("cost_source", "default")),
                updated=costing_data.get("updated", model_data.get("cost_updated", "")),
                tier=costing_data.get("tier", model_data.get("cost_tier", "standard"))
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
            
            models.append(ModelInfo(
                id=model_data["id"],
                name=model_data["name"],
                context_window=model_data.get("context_window", 0),
                capabilities=model_data.get("capabilities", []),
                modalities=modalities,
                costing=costing
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
