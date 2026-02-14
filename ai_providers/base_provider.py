"""
Base AI Provider

Abstract base class defining the interface for all AI providers.
Each concrete provider wraps its native SDK directly.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any, Union, List, Dict


@dataclass
class AIResponse:
    """
    Standardized response from any AI provider.
    
    Provides a consistent interface regardless of which provider
    generated the response.
    """
    content: str
    model: str
    provider: str
    usage: dict[str, int] = field(default_factory=dict)
    parts: List[Dict] = field(default_factory=list)  # Multimodal output parts
    raw_response: Any = None
    
    @property
    def input_tokens(self) -> int:
        """Number of input tokens used."""
        return self.usage.get("input_tokens", 0)
    
    @property
    def output_tokens(self) -> int:
        """Number of output tokens generated."""
        return self.usage.get("output_tokens", 0)
    
    @property
    def total_tokens(self) -> int:
        """Total tokens used (input + output)."""
        return self.input_tokens + self.output_tokens


class AIProviderError(Exception):
    """Base exception for AI provider errors."""
    
    def __init__(self, message: str, provider: str, original_error: Optional[Exception] = None):
        self.provider = provider
        self.original_error = original_error
        super().__init__(f"[{provider}] {message}")


class AIRateLimitError(AIProviderError):
    """Raised when rate limit is exceeded."""
    pass


class AIAuthenticationError(AIProviderError):
    """Raised when authentication fails."""
    pass


class AIModelNotFoundError(AIProviderError):
    """Raised when the requested model is not available."""
    pass


class DjinniteModalityError(AIProviderError):
    """Raised when a model is asked to handle an unsupported modality."""
    def __init__(self, message: str, provider: str, model: str, requested_modalities: list[str], supported_modalities: list[str]):
        self.model = model
        self.requested_modalities = requested_modalities
        self.supported_modalities = supported_modalities
        full_message = f"{message} (Model: {model}, Requested: {requested_modalities}, Supported: {supported_modalities})"
        super().__init__(full_message, provider=provider)


class BaseAIProvider(ABC):
    """
    Abstract base class for AI providers.
    
    Each provider implementation should:
    1. Initialize with an API key and optional model
    2. Implement the generate() method for text generation
    3. Handle provider-specific errors gracefully
    """
    
    PROVIDER_NAME: str = "base"
    
    def __init__(self, api_key: str, model: str):
        """
        Initialize the provider.
        
        Args:
            api_key: API key for authentication
            model: Model ID to use (Required)
        """
        if not model:
            raise ValueError(f"Model must be specified for {self.PROVIDER_NAME} provider. Check your ai_config.json.")
            
        self.api_key = api_key
        self.model = model
        self._client = None
        self._initialize_client()
    
    @abstractmethod
    def _initialize_client(self) -> None:
        """
        Initialize the provider's client/SDK.
        
        This is called during __init__ and should set up self._client.
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: Union[str, List[Dict]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AIResponse:
        """
        Generate a response from the AI model.
        
        Args:
            prompt: The user prompt/message (str or list of multimodal parts)
            system_prompt: Optional system instruction
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate (provider default if None)
            
        Returns:
            AIResponse with the generated content
            
        Raises:
            AIProviderError: If generation fails
            DjinniteModalityError: If prompt contains unsupported modalities
        """
        pass
    
    def _normalize_input(self, prompt: Union[str, List[Dict]]) -> List[Dict]:
        """
        Standardize the input prompt into a list of multimodal parts.
        
        Args:
            prompt: Either a plain string or a list of part dictionaries
            
        Returns:
            List of dicts in the form [{"type": "text", "text": "..."}]
        """
        if isinstance(prompt, str):
            return [{"type": "text", "text": prompt}]
        elif isinstance(prompt, list):
            # Basic validation of parts
            for i, part in enumerate(prompt):
                if not isinstance(part, dict) or "type" not in part:
                    raise ValueError(f"Invalid multimodal part at index {i}: {part}")
            return prompt
        else:
            raise ValueError(f"Prompt must be a string or a list of dicts, got {type(prompt)}")

    def _validate_modalities(self, parts: List[Dict], supported_modalities: list[str]):
        """
        Validate that all modalities in parts are supported by the model.
        
        Args:
            parts: The normalized input parts
            supported_modalities: List of supported input modalities (e.g. ['text', 'vision'])
            
        Raises:
            DjinniteModalityError: If an unsupported modality is requested
        """
        requested = list(set(p["type"] for p in parts))
        unsupported = [m for m in requested if m not in supported_modalities]
        
        if unsupported:
            raise DjinniteModalityError(
                f"Model does not support requested modalities: {unsupported}",
                provider=self.PROVIDER_NAME,
                model=self.model,
                requested_modalities=requested,
                supported_modalities=supported_modalities
            )

    def generate_json(
        self,
        prompt: Union[str, List[Dict]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        web_search: bool = False,
    ) -> AIResponse:
        """
        Generate a JSON response from the AI model.
        
        Uses a lower temperature by default for more deterministic output.
        The prompt should instruct the model to output valid JSON.
        
        Args:
            prompt: The user prompt (should request JSON output)
            system_prompt: Optional system instruction
            temperature: Sampling temperature (default 0.3 for consistency)
            max_tokens: Maximum tokens to generate
            web_search: If True, enable web search for current info (if supported)
            
        Returns:
            AIResponse with JSON content
        """
        # Default implementation just calls generate with lower temperature
        # Subclasses can override to use provider-specific JSON modes
        # Note: web_search is ignored in base implementation
        return self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the provider is available and properly configured.
        
        Returns:
            True if provider can be used, False otherwise
        """
        pass
    
    @abstractmethod
    def list_models(self) -> list[dict]:
        """
        List available models from the provider.
        
        Returns:
            List of dictionaries with keys: id, name, context_window, etc.
        """
        pass

    def discover_modalities(self, model_id: str) -> Dict[str, List[str]]:
        """
        Discover input/output modalities for a model based on its ID.
        
        Default implementation assumes text-only.
        
        Returns:
            Dict with 'input' and 'output' keys.
        """
        return {"input": ["text"], "output": ["text"]}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model})"
