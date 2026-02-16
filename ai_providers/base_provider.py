"""
Base AI Provider

Abstract base class defining the interface for all AI providers.
Each concrete provider wraps its native SDK directly.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any, Union, List, Dict, Type


@dataclass
class AIResponse:
    """
    Standardized response from any AI provider.
    
    Provides a consistent interface regardless of which provider
    generated the response.
    
    Attributes:
        content: The generated text content.
        model: The model ID that produced the response.
        provider: The provider name (e.g. "gemini", "claude", "chatgpt").
        usage: Token usage dict with keys "input_tokens" and "output_tokens".
        parts: Multimodal output parts (interleaved text, images, etc.).
        raw_response: The original provider SDK response object.
        truncated: True if the output was cut short due to the max output
            token limit. When True, ``content`` contains only partial output
            and an ``AIOutputTruncatedError`` will normally be raised so
            callers cannot accidentally act on incomplete data.
        finish_reason: The provider-native finish/stop reason string
            (e.g. "stop", "length", "max_tokens", "MAX_TOKENS").
    """
    content: str
    model: str
    provider: str
    usage: dict[str, int] = field(default_factory=dict)
    parts: List[Dict] = field(default_factory=list)  # Multimodal output parts
    raw_response: Any = None
    truncated: bool = False
    finish_reason: Optional[str] = None
    
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


class AIOutputTruncatedError(AIProviderError):
    """
    Raised when the API returned HTTP 200 OK but the model's output was
    truncated because it hit the maximum output token limit.

    The partial response is attached so callers can inspect what was
    generated and the token-usage metadata.

    Provider-specific truncation indicators:
        - OpenAI:    finish_reason == "length"
        - Anthropic: stop_reason  == "max_tokens"
        - Gemini:    finishReason == "MAX_TOKENS"

    Attributes:
        partial_response: The incomplete AIResponse containing whatever
            content the model produced before hitting the limit.
    """
    def __init__(
        self,
        message: str,
        provider: str,
        partial_response: 'AIResponse',
        original_error: Optional[Exception] = None,
    ):
        self.partial_response = partial_response
        super().__init__(message, provider, original_error)


class AIContextLengthError(AIProviderError):
    """
    Raised when the API returned HTTP 400 Bad Request because the input
    prompt (plus any expected output reservation) exceeds the model's
    context window.

    Provider-specific error indicators:
        - OpenAI:    400 Bad Request, code: "context_length_exceeded"
        - Anthropic: 400 Bad Request, type: "invalid_request_error"
        - Gemini:    400 INVALID_ARGUMENT

    The consumer should shorten the prompt, reduce max_tokens, or switch
    to a model with a larger context window.
    """
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
    
    Providers are catalog-aware: when initialized via ``get_provider()``,
    they receive ``model_info`` from the model catalog.  This enables
    pre-flight capability checks (e.g., ``supports_structured_json``)
    before making API calls.  Pass ``force=True`` on methods like
    ``generate_json()`` to bypass these checks (used by probes).
    """
    
    PROVIDER_NAME: str = "base"
    
    def __init__(self, api_key: str, model: str, model_info=None):
        """
        Initialize the provider.
        
        Args:
            api_key: API key for authentication
            model: Model ID to use (Required)
            model_info: Optional ModelInfo from the catalog for pre-flight
                        capability checks. Passed automatically by
                        ``get_provider()``. None means no catalog validation.
        """
        if not model:
            raise ValueError(f"Model must be specified for {self.PROVIDER_NAME} provider. Check your ai_config.json.")
            
        self.api_key = api_key
        self.model = model
        self._model_info = model_info  # From catalog — may be None
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
        web_search: bool = False,
    ) -> AIResponse:
        """
        Generate a freeform text response from the AI model.
        
        Args:
            prompt: The user prompt/message (str or list of multimodal parts)
            system_prompt: Optional system instruction
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate (provider default if None)
            web_search: If True, enable web/grounding search for current info
                        (provider support varies).
            
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

    def _normalize_schema(self, schema: Union[Dict, Type]) -> Dict:
        """
        Normalize a schema into a standard JSON Schema dictionary.

        Accepts either a raw JSON Schema dict or a Pydantic BaseModel class
        and returns a plain ``dict`` suitable for passing to any provider's
        structured-output API.

        Args:
            schema: A JSON Schema dictionary **or** a Pydantic ``BaseModel``
                    subclass (the *class itself*, not an instance).

        Returns:
            A JSON Schema dictionary.

        Raises:
            TypeError: If ``schema`` is not a dict or a Pydantic BaseModel class.
        """
        if isinstance(schema, dict):
            return schema

        # Check for Pydantic BaseModel class (not instance)
        # We do a lazy check so pydantic is not a hard dependency.
        try:
            import pydantic
            if isinstance(schema, type) and issubclass(schema, pydantic.BaseModel):
                return schema.model_json_schema()
        except ImportError:
            pass

        raise TypeError(
            f"schema must be a dict (JSON Schema) or a Pydantic BaseModel class, "
            f"got {type(schema).__name__}. "
            f"Use generate() for freeform text responses."
        )

    @staticmethod
    def _is_pydantic_generated(schema: Dict) -> bool:
        """
        Heuristic: detect whether a schema dict was auto-generated by
        Pydantic's ``model_json_schema()``.

        Pydantic-generated schemas typically contain ``"title"`` at the
        top level and may contain ``"$defs"`` for nested models.  Hand-written
        schemas rarely include ``"title"``.

        Returns:
            True if the schema appears to be Pydantic-generated.
        """
        return "title" in schema

    @staticmethod
    def _schema_contains_additional_properties(schema: Dict) -> bool:
        """
        Recursively check whether any node in the JSON Schema tree
        contains the ``additionalProperties`` key.

        Returns:
            True if ``additionalProperties`` is found anywhere.
        """
        if not isinstance(schema, dict):
            return False
        if "additionalProperties" in schema:
            return True
        # Recurse into object properties
        for prop_schema in (schema.get("properties") or {}).values():
            if BaseAIProvider._schema_contains_additional_properties(prop_schema):
                return True
        # Recurse into array items
        items = schema.get("items")
        if isinstance(items, dict):
            if BaseAIProvider._schema_contains_additional_properties(items):
                return True
        # Recurse into combinators (anyOf, oneOf, allOf)
        for combinator in ("anyOf", "oneOf", "allOf"):
            for branch in (schema.get(combinator) or []):
                if BaseAIProvider._schema_contains_additional_properties(branch):
                    return True
        # Recurse into $defs
        for def_schema in (schema.get("$defs") or {}).values():
            if BaseAIProvider._schema_contains_additional_properties(def_schema):
                return True
        return False

    @staticmethod
    def _strip_additional_properties(schema: Dict) -> Dict:
        """
        Return a deep copy of *schema* with every ``additionalProperties``
        key removed, recursively.

        This is used both as a defensive cleanup for Gemini and to sanitize
        Pydantic-generated schemas before caller-contract validation.
        """
        import copy
        schema = copy.deepcopy(schema)
        BaseAIProvider._strip_additional_properties_inplace(schema)
        return schema

    @staticmethod
    def _strip_additional_properties_inplace(schema: Dict) -> None:
        """In-place recursive removal of ``additionalProperties``."""
        if not isinstance(schema, dict):
            return
        schema.pop("additionalProperties", None)
        for prop_schema in (schema.get("properties") or {}).values():
            BaseAIProvider._strip_additional_properties_inplace(prop_schema)
        items = schema.get("items")
        if isinstance(items, dict):
            BaseAIProvider._strip_additional_properties_inplace(items)
        for combinator in ("anyOf", "oneOf", "allOf"):
            for branch in (schema.get(combinator) or []):
                BaseAIProvider._strip_additional_properties_inplace(branch)
        for def_schema in (schema.get("$defs") or {}).values():
            BaseAIProvider._strip_additional_properties_inplace(def_schema)

    def _validate_caller_schema(self, schema: Dict) -> Dict:
        """
        Validate the caller's schema and enforce the Djinnite contract:

        **Callers must NOT include ``additionalProperties`` in their schemas.**
        Djinnite adds or removes it per-provider automatically.

        - For **hand-written dict schemas**: raises ``ValueError`` if
          ``additionalProperties`` is found anywhere in the tree.
        - For **Pydantic-generated schemas** (detected by heuristic): silently
          strips ``additionalProperties`` since Pydantic's
          ``model_json_schema()`` auto-generates it.

        Args:
            schema: The JSON Schema dict (already normalized from dict/Pydantic).

        Returns:
            The (possibly cleaned) schema dict.

        Raises:
            ValueError: If a hand-written schema contains ``additionalProperties``.
        """
        if not self._schema_contains_additional_properties(schema):
            return schema

        # Pydantic auto-generates additionalProperties — silently strip it
        if self._is_pydantic_generated(schema):
            return self._strip_additional_properties(schema)

        # Hand-written schema with additionalProperties → hard error
        raise ValueError(
            "Caller schema must not contain 'additionalProperties'. "
            "Djinnite manages this field per-provider for cross-provider "
            "compatibility. Remove 'additionalProperties' from your schema; "
            "Djinnite implicitly enforces strict mode (no extra properties)."
        )

    def _prepare_schema_for_provider(self, schema: Dict) -> Dict:
        """
        Provider-specific schema transformation.

        The base implementation returns the schema unchanged (suitable for
        Claude which is neutral on ``additionalProperties``).

        Subclasses override this to add provider-specific fields:
        - **OpenAI**: adds ``additionalProperties: false`` to every object,
          wraps top-level arrays in an object envelope.
        - **Gemini**: strips ``additionalProperties`` defensively.

        Args:
            schema: A validated, clean JSON Schema dict (no ``additionalProperties``).

        Returns:
            A provider-ready schema dict.
        """
        return schema

    def _check_capability(self, capability: str) -> None:
        """
        Pre-flight check: raise if the catalog says the model does NOT
        support the requested capability.

        Skipped when ``self._model_info`` is None (no catalog loaded,
        e.g. during probing or testing).

        Args:
            capability: One of ``"structured_json"`` (more may be added).

        Raises:
            AIProviderError: If the catalog explicitly says False for
                this capability.
        """
        if self._model_info is None:
            return  # No catalog → allow (could be a new/unknown model)

        if capability == "structured_json":
            ssj = self._model_info.supports_structured_json
            if ssj is False:
                raise AIProviderError(
                    f"Model '{self.model}' does not support structured JSON "
                    f"(supports_structured_json=false in catalog). "
                    f"Use a different model, or pass force=True to bypass.",
                    provider=self.PROVIDER_NAME,
                )
            # True or None → allow

    def generate_json(
        self,
        prompt: Union[str, List[Dict]],
        schema: Union[Dict, Type],
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        web_search: bool = False,
        force: bool = False,
    ) -> AIResponse:
        """
        Generates structured JSON **strictly** adhering to the provided ``schema``.

        [AGENT NOTE — PREFERRED PATH FOR STRUCTURED DATA]:
        This method activates provider-native **Strict Mode / Constraint Decoding**
        (OpenAI ``json_schema`` with ``strict=True``, Anthropic ``output_config``,
        Gemini ``response_schema``).  The output is **guaranteed** to validate
        against the supplied schema — it is not "best-effort" JSON.

        Use this method for **all programmatic tasks** — routing decisions,
        entity extraction, SQL generation, tool-call argument construction,
        structured reporting — where schema adherence is critical.
        Do **not** use ``generate()`` for structured data; that method returns
        freeform text with no structural guarantees.

        Args:
            prompt: The user prompt / message (str or list of multimodal parts).
            schema: **Required.** A Pydantic ``BaseModel`` class or a JSON Schema
                    dictionary describing the exact structure the response must
                    conform to.  The output is guaranteed to validate against
                    this structure via provider-native Constraint Decoding.
            system_prompt: Optional system instruction prepended to the request.
            temperature: Sampling temperature (default 0.3 for deterministic,
                         schema-conforming output).
            max_tokens: Maximum tokens to generate (provider default if None).
            web_search: If True, enable web/grounding search for current info
                        (provider support varies).
            force: If True, skip catalog pre-flight checks. Used by probes
                   and testing. Default False.

        Returns:
            AIResponse whose ``content`` is a JSON string that conforms to
            ``schema``.

        Raises:
            ValueError: If ``schema`` is None (fail-fast — use ``generate()``
                        for freeform text).
            TypeError: If ``schema`` is not a dict or Pydantic BaseModel class.
            AIProviderError: If model doesn't support structured JSON (per catalog).
            AIOutputTruncatedError: If the JSON output was truncated.
        """
        if schema is None:
            raise ValueError(
                "schema is required for generate_json(). "
                "Pass a Pydantic BaseModel class or a JSON Schema dict. "
                "Use generate() for freeform text responses."
            )

        # Pre-flight: check catalog before burning an API call
        if not force:
            self._check_capability("structured_json")

        # Normalize once; providers use self._normalize_schema() in their
        # overrides, but the base implementation validates eagerly.
        self._normalize_schema(schema)

        # Default implementation — subclasses override with provider-native
        # strict modes.  The base fallback just calls generate() with a
        # JSON-requesting system prompt (no structural guarantee).
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

    def probe_temperature(self) -> Optional[bool]:
        """
        Probe whether the current model accepts the temperature parameter.
        
        Returns:
            True  – model accepts temperature
            False – model rejects temperature (reasoning/o3 models)
            None  – inconclusive (rate limit, timeout)
        """
        return None  # Base: unknown — subclasses override

    def probe_thinking(self) -> Optional[bool]:
        """
        Probe whether the current model supports extended thinking/reasoning.
        
        Returns:
            True  – model supports thinking mode
            False – model does not support thinking
            None  – inconclusive
        """
        return None  # Base: unknown — subclasses override

    def probe_structured_json(self) -> Optional[bool]:
        """
        Probe whether the current model supports schema-enforced structured
        JSON output (Constraint Decoding).

        Sends a minimal request using the provider-native strict JSON schema
        mechanism.  If the model accepts the request → ``True``.  If the
        provider returns a 400/unsupported error → ``False``.  If the result
        is ambiguous (rate limit, auth error, etc.) → ``None``.

        This method is used by ``update_models.py`` during catalog refresh.
        Subclasses should override with provider-specific probe logic.

        Returns:
            True  – confirmed supported
            False – confirmed NOT supported
            None  – inconclusive (rate limit, auth error, etc.)
        """
        return None  # Base: unknown

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
