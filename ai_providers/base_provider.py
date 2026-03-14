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
    def thinking_tokens(self) -> Optional[int]:
        """
        Number of tokens used for internal reasoning/thinking.

        Returns ``None`` if the provider did not report thinking tokens
        (unknown — distinct from 0 which means "confirmed no thinking").
        When ``None``, ``total_tokens`` is computed from input + output
        only and may be an undercount.
        """
        return self.usage.get("thinking_tokens")
    
    @property
    def total_tokens(self) -> int:
        """
        Total tokens used.

        If ``thinking_tokens`` is available, includes it.  If ``None``
        (unknown), computed as ``input_tokens + output_tokens`` only.
        Check ``thinking_tokens is None`` to know if the total is
        potentially incomplete.
        """
        total = self.usage.get("total_tokens")
        if total is not None:
            return total
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
        thinking: Union[bool, int, str, None] = None,
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
            thinking: Optional thinking/reasoning control.
                If ``True``: enable thinking at maximum budget (recommended).
                If ``False``: explicitly disable thinking.
                If ``int``: a specific token budget for internal reasoning.
                If ``str``: an effort level (``"low"``, ``"medium"``, ``"high"``).
                If ``None`` (default): no thinking requested.
                The provider translates this into its native format
                (Claude ``thinking`` block, OpenAI ``reasoning_effort``,
                Gemini ``thinking_config``).  Temperature conflicts are
                handled automatically.

                **Budget guidance:** Token budgets are highly unpredictable —
                they depend on prompt complexity, model version, and task type.
                The recommended default is ``thinking=True`` (maximum budget).
                Only use explicit ``int`` budgets after profiling specific
                workloads.  Low budgets cause partial/useless reasoning that
                is still charged.
            
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
    def _add_additional_properties_false(schema: Dict) -> None:
        """
        Recursively add ``additionalProperties: false`` to every object
        node in the JSON Schema tree.  Mutates *schema* in-place.

        Used by OpenAI and Claude providers which both require explicit
        ``additionalProperties: false`` for strict/constrained JSON mode.
        """
        if not isinstance(schema, dict):
            return
        # If this node is (or could be) an object, inject the field
        if schema.get("type") == "object" or "properties" in schema:
            schema["additionalProperties"] = False
        # Recurse into properties
        for prop_schema in (schema.get("properties") or {}).values():
            BaseAIProvider._add_additional_properties_false(prop_schema)
        # Recurse into array items
        items = schema.get("items")
        if isinstance(items, dict):
            BaseAIProvider._add_additional_properties_false(items)
        # Recurse into combinators
        for combinator in ("anyOf", "oneOf", "allOf"):
            for branch in (schema.get(combinator) or []):
                BaseAIProvider._add_additional_properties_false(branch)
        # Recurse into $defs
        for def_schema in (schema.get("$defs") or {}).values():
            BaseAIProvider._add_additional_properties_false(def_schema)

    @staticmethod
    def _ensure_required_arrays(schema: Dict) -> None:
        """
        Recursively ensure every object node with ``properties`` has a
        ``required`` array listing ALL property keys.  Mutates *schema*
        in-place.

        OpenAI strict mode requires ``required`` to include every key in
        ``properties``.  Claude's grammar compiler has a limit on optional
        parameters (those NOT in ``required``) — adding all properties to
        ``required`` avoids hitting that limit.
        """
        if not isinstance(schema, dict):
            return
        # If this node has properties, ensure required lists all of them
        props = schema.get("properties")
        if props:
            schema["required"] = list(props.keys())
        # Recurse into properties
        for prop_schema in (props or {}).values():
            BaseAIProvider._ensure_required_arrays(prop_schema)
        # Recurse into array items
        items = schema.get("items")
        if isinstance(items, dict):
            BaseAIProvider._ensure_required_arrays(items)
        # Recurse into combinators
        for combinator in ("anyOf", "oneOf", "allOf"):
            for branch in (schema.get(combinator) or []):
                BaseAIProvider._ensure_required_arrays(branch)
        # Recurse into $defs
        for def_schema in (schema.get("$defs") or {}).values():
            BaseAIProvider._ensure_required_arrays(def_schema)

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

    # ------------------------------------------------------------------
    # Thinking & Temperature resolution helpers
    # ------------------------------------------------------------------

    # Mapping from effort strings to approximate fractions of max_tokens,
    # used when translating a string effort level to a token budget.
    _EFFORT_FRACTIONS: Dict[str, float] = {
        "low": 0.25,
        "medium": 0.50,
        "high": 0.80,
    }

    # Generous fallback when max_output_tokens is unknown from catalog.
    _DEFAULT_THINKING_BUDGET: int = 32768

    def _resolve_thinking(
        self,
        thinking: Union[bool, int, str, None],
    ) -> Union[bool, int, str, None]:
        """
        Validate and normalize the caller's ``thinking`` parameter.

        - ``None``  → passthrough (no thinking).
        - ``False`` → explicitly disable thinking.
        - ``True``  → enable thinking at maximum budget.
        - ``int``   → validated positive token budget.
        - ``str``   → validated effort level (``"low"``, ``"medium"``, ``"high"``).

        Raises ``AIProviderError`` if the catalog says the model does not
        support thinking (``capabilities.thinking is False``) and the
        caller requested thinking (``True``, ``int``, or ``str``).
        Skipped when ``self._model_info`` is ``None`` (no catalog).

        Returns:
            The validated thinking value, or ``None``/``False``.
        """
        if thinking is None or thinking is False:
            return thinking

        # Pre-flight: catalog says model can't think → reject
        if self._model_info is not None:
            cap = self._model_info.capabilities.thinking
            if cap is False:
                raise AIProviderError(
                    f"Model '{self.model}' does not support thinking/reasoning "
                    f"(capabilities.thinking=false in catalog). "
                    f"Use a thinking-capable model, or pass thinking=None.",
                    provider=self.PROVIDER_NAME,
                )

        # bool True → enable at maximum budget (provider handles the details)
        if thinking is True:
            return True

        if isinstance(thinking, int):
            if thinking <= 0:
                raise ValueError("thinking token budget must be a positive integer.")
            return thinking
        if isinstance(thinking, str):
            low = thinking.lower()
            if low not in self._EFFORT_FRACTIONS:
                raise ValueError(
                    f"thinking effort must be one of 'low', 'medium', 'high' — "
                    f"got '{thinking}'."
                )
            return low
        raise TypeError(
            f"thinking must be bool, int (token budget), str (effort level), or None — "
            f"got {type(thinking).__name__}."
        )

    def _resolve_max_tokens(self, max_tokens: Optional[int]) -> Optional[int]:
        """
        Resolve the effective ``max_tokens`` for a request.

        If the caller passed ``None``, auto-fill from the model catalog's
        ``max_output_tokens``.  This prevents expensive incomplete responses
        and ensures the model has its full output capacity available.

        Resolution order:
        1. Caller's explicit value (if provided and > 0)
        2. Model catalog ``max_output_tokens`` (if available and > 0)
        3. ``None`` (let the provider SDK use its own default)

        Returns:
            An integer token limit, or ``None`` if unknown.
        """
        if max_tokens is not None and max_tokens > 0:
            return max_tokens
        if self._model_info and self._model_info.max_output_tokens > 0:
            return self._model_info.max_output_tokens
        return None

    def _get_max_thinking_budget(self, max_tokens: Optional[int]) -> int:
        """
        Determine the maximum thinking budget for ``thinking=True``.

        Resolution order:
        1. Model catalog ``max_output_tokens`` (if available and > 0)
        2. Caller's ``max_tokens`` (if provided)
        3. ``_DEFAULT_THINKING_BUDGET`` fallback

        Returns:
            An integer token budget.
        """
        if self._model_info and self._model_info.max_output_tokens > 0:
            return self._model_info.max_output_tokens
        if max_tokens and max_tokens > 0:
            return max_tokens
        return self._DEFAULT_THINKING_BUDGET

    def _resolve_temperature(
        self,
        temperature: float,
        thinking_active: bool,
    ) -> Optional[float]:
        """
        Decide the effective temperature to send to the provider.

        * If the catalog says ``capabilities.temperature is False`` → ``None``
          (omit temperature entirely to avoid 400 errors).
        * If ``thinking_active`` is True the provider subclass is responsible
          for any further override (e.g., Claude forces temperature=1).
          The base implementation still strips it when the catalog forbids it.
        * Otherwise → return the caller's value unchanged.

        Returns:
            The temperature to use, or ``None`` meaning "omit".
        """
        if self._model_info is not None:
            if self._model_info.capabilities.temperature is False:
                return None
        return temperature

    def _effort_to_budget(self, effort: str, max_tokens: int) -> int:
        """
        Convert a string effort level to a token budget.

        Uses ``_EFFORT_FRACTIONS`` to compute a fraction of *max_tokens*.
        Ensures a minimum of 1024 tokens.

        Args:
            effort: ``"low"``, ``"medium"``, or ``"high"``.
            max_tokens: The max output token limit to base the fraction on.

        Returns:
            An integer token budget.
        """
        frac = self._EFFORT_FRACTIONS.get(effort, 0.50)
        return max(1024, int(max_tokens * frac))

    @staticmethod
    def _budget_to_effort(budget: int) -> str:
        """
        Convert a token budget integer to an effort level string.

        Thresholds:
        - ≤ 2048 → ``"low"``
        - ≤ 16384 → ``"medium"``
        - > 16384 → ``"high"``

        Returns:
            ``"low"``, ``"medium"``, or ``"high"``.
        """
        if budget <= 2048:
            return "low"
        if budget <= 16384:
            return "medium"
        return "high"

    # ------------------------------------------------------------------

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
        thinking: Union[bool, int, str, None] = None,
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
            thinking: Optional thinking/reasoning control (same semantics
                      as ``generate()``).

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

    def probe_thinking_style(self) -> Optional[str]:
        """
        Probe which thinking style the current model supports.

        Performs a multi-tier probe (provider-specific) to determine the
        most capable thinking mode the model accepts.  Subclasses override
        with provider-native logic.

        Returns:
            ``"adaptive"`` – model supports adaptive/self-regulating thinking
            ``"budget"``   – model supports fixed token-budget thinking
            ``"effort"``   – model supports effort-level thinking (OpenAI)
            ``None``       – model does not support thinking, or inconclusive
        """
        # Default: fall back to probe_thinking() for a simple yes/no.
        # Subclasses override with richer detection.
        return None

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
