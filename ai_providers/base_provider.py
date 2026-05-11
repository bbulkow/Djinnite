"""
Base AI Provider

Abstract base class defining the interface for all AI providers.
Each concrete provider wraps its native SDK directly.
"""

import json as _json
import os
import struct
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any, Union, List, Dict, Tuple, Type


def _get_image_dimensions(data: bytes) -> Optional[Tuple[int, int]]:
    """
    Extract (width, height) from image header bytes without PIL.

    Supports JPEG, PNG, GIF, and WebP.  Returns None if the format
    is unrecognized or the header is too short (fail-open).
    """
    if len(data) < 24:
        return None

    # PNG: 8-byte signature then IHDR chunk with width/height at bytes 16-24
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        w, h = struct.unpack('>II', data[16:24])
        return (w, h)

    # GIF: "GIF87a" or "GIF89a", width/height at bytes 6-10 (little-endian)
    if data[:6] in (b'GIF87a', b'GIF89a'):
        w, h = struct.unpack('<HH', data[6:10])
        return (w, h)

    # JPEG: scan for SOF markers (0xFF 0xC0..0xCF, excluding 0xC4 and 0xCC)
    if data[:2] == b'\xff\xd8':
        i = 2
        while i < len(data) - 9:
            if data[i] != 0xFF:
                break
            marker = data[i + 1]
            # SOF markers: 0xC0-0xCF except 0xC4 (DHT) and 0xCC (DAC)
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xCC):
                h, w = struct.unpack('>HH', data[i + 5:i + 9])
                return (w, h)
            # Skip to next marker using segment length
            seg_len = struct.unpack('>H', data[i + 2:i + 4])[0]
            i += 2 + seg_len
        return None

    # WebP: "RIFF....WEBP"
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        # VP8 lossy
        if data[12:16] == b'VP8 ' and len(data) >= 30:
            # Width/height at bytes 26-30 (little-endian, 14-bit values)
            w = struct.unpack('<H', data[26:28])[0] & 0x3FFF
            h = struct.unpack('<H', data[28:30])[0] & 0x3FFF
            return (w, h)
        # VP8L lossless
        if data[12:16] == b'VP8L' and len(data) >= 25:
            bits = struct.unpack('<I', data[21:25])[0]
            w = (bits & 0x3FFF) + 1
            h = ((bits >> 14) & 0x3FFF) + 1
            return (w, h)
        # VP8X extended
        if data[12:16] == b'VP8X' and len(data) >= 30:
            w = struct.unpack('<I', data[24:27] + b'\x00')[0] + 1
            h = struct.unpack('<I', data[27:30] + b'\x00')[0] + 1
            return (w, h)
        return None

    return None


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
        usage: Usage and cost dict.  Token counts: ``input_tokens``,
            ``output_tokens``, ``total_tokens``, ``thinking_tokens``.
            Search: ``search_units``, ``search_result_tokens``.
            Costs (dollars): ``token_cost``, ``search_cost``,
            ``search_cost_per_unit``, ``total_cost``.
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

    @property
    def search_units(self) -> int:
        """Number of billable web search events (0 if no search was used)."""
        return self.usage.get("search_units", 0)

    @property
    def search_result_tokens(self) -> Optional[int]:
        """
        Tokens consumed by search result content injected into context.

        Relevant for Anthropic (billed at input rate) and legacy OpenAI
        (GPT-4o).  Returns ``None`` if not reported by the provider.
        """
        return self.usage.get("search_result_tokens")

    @property
    def search_cost(self) -> Optional[float]:
        """Total dollar cost of web search events for this response.

        Computed as ``search_units * search_cost_per_unit`` by the provider
        (using the rate from the model catalog).  Returns ``None`` if the
        per-unit rate is unknown or no search was performed.
        """
        return self.usage.get("search_cost")

    @property
    def token_cost(self) -> Optional[float]:
        """Dollar cost of token usage (input + output + thinking).

        Computed from per-model rates in the catalog.  Returns ``None``
        if the model's pricing is unknown.
        """
        return self.usage.get("token_cost")

    @property
    def total_cost(self) -> Optional[float]:
        """Total dollar cost: token_cost + search_cost.

        Returns ``None`` if no cost information is available.
        """
        return self.usage.get("total_cost")


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

    The consumer should shorten the prompt, reduce max_output_tokens, or
    switch to a model with a larger context window.
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
            model_info: ModelInfo from the catalog for pre-flight capability
                        checks. Passed automatically by ``get_provider()``,
                        which requires the catalog to exist.  May be None
                        only for direct construction (tests/probes).
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

    def _compute_search_cost(self, usage: dict) -> None:
        """Compute dollar cost of web search events and store in usage dict."""
        units = usage.get("search_units", 0)
        if not units:
            return
        cost_per_unit = None
        if self._model_info and self._model_info.costing:
            cost_per_unit = self._model_info.costing.search_cost_per_unit
        if cost_per_unit is not None:
            usage["search_cost_per_unit"] = cost_per_unit
            usage["search_cost"] = round(units * cost_per_unit, 6)

    def _compute_token_cost(self, usage: dict) -> None:
        """Compute dollar cost of token usage and store in usage dict.

        Reads ``input_per_1m`` / ``output_per_1m`` from the model catalog.
        Thinking tokens are billed at the output rate.  Anthropic reports
        them separately (``_thinking_billed_separately=True``), while
        OpenAI/Google include them in ``output_tokens`` already.
        """
        if not self._model_info or not self._model_info.costing:
            return
        costing = self._model_info.costing
        if costing.input_per_1m is None or costing.output_per_1m is None:
            return

        input_t = usage.get("input_tokens", 0)
        output_t = usage.get("output_tokens", 0)

        if usage.get("_thinking_billed_separately"):
            output_t += usage.get("thinking_tokens") or 0

        input_cost = input_t * costing.input_per_1m / 1_000_000
        output_cost = output_t * costing.output_per_1m / 1_000_000
        usage["token_cost"] = round(input_cost + output_cost, 8)

    def _compute_costs(self, usage: dict) -> None:
        """Compute all dollar costs (token + search) and store in usage dict.

        Single entry point called by providers after populating token counts
        and search units.  Produces ``token_cost``, ``search_cost``, and
        ``total_cost`` in the usage dict.
        """
        self._compute_token_cost(usage)
        self._compute_search_cost(usage)
        token_cost = usage.get("token_cost")
        search_cost = usage.get("search_cost")
        if token_cost is not None or search_cost is not None:
            usage["total_cost"] = round((token_cost or 0) + (search_cost or 0), 8)

    @abstractmethod
    def generate(
        self,
        prompt: Union[str, List[Dict]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: Optional[int] = None,
        web_search: bool = False,
        thinking: Union[bool, int, str, None] = None,
    ) -> AIResponse:
        """
        Generate a freeform text response from the AI model.

        Args:
            prompt: The user prompt/message (str or list of multimodal parts)
            system_prompt: Optional system instruction
            temperature: Sampling temperature (0.0-1.0)
            max_output_tokens: Cap on output tokens the model may emit.
                Auto-fills from the catalog's ``ModelInfo.max_output_tokens``
                when ``None``. Note: per-provider semantics differ — see the
                "Token Budgets" section in DEVELOPMENT.md. Claude/Gemini cap
                visible output only; OpenAI caps visible+reasoning combined.
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

    def _validate_vision_limits(self, parts: List[Dict]) -> None:
        """
        Pre-flight: reject oversized images before making API calls.

        Checks byte size, pixel dimensions, and image count against the
        model's ``vision_limits`` from the catalog.  Fails open (skips
        validation) when limits are unknown.
        """
        if not self._model_info or not self._model_info.vision_limits:
            return

        limits = self._model_info.vision_limits
        image_parts = [p for p in parts if p["type"] == "image"]
        if not image_parts:
            return

        # Check image count
        if limits.max_images_per_request and len(image_parts) > limits.max_images_per_request:
            raise AIProviderError(
                f"Too many images: {len(image_parts)} exceeds limit of "
                f"{limits.max_images_per_request} for model {self.model}",
                provider=self.PROVIDER_NAME,
            )

        import base64

        for i, part in enumerate(image_parts):
            image_data = part.get("image_data")
            if image_data is None:
                continue  # URL-based image, can't pre-validate

            # Get raw bytes for size/dimension checks
            if isinstance(image_data, bytes):
                raw = image_data
            else:
                raw = base64.b64decode(image_data)

            # Check byte size
            if limits.max_image_bytes and len(raw) > limits.max_image_bytes:
                mb_limit = limits.max_image_bytes / (1024 * 1024)
                mb_actual = len(raw) / (1024 * 1024)
                raise AIProviderError(
                    f"Image {i + 1} is {mb_actual:.1f} MB, exceeds "
                    f"{mb_limit:.0f} MB limit for model {self.model}. "
                    f"Resize before sending.",
                    provider=self.PROVIDER_NAME,
                )

            # Check pixel dimensions
            if limits.max_dimension_px:
                dims = _get_image_dimensions(raw)
                if dims:
                    w, h = dims
                    if w > limits.max_dimension_px or h > limits.max_dimension_px:
                        raise AIProviderError(
                            f"Image {i + 1} is {w}x{h} px, exceeds "
                            f"{limits.max_dimension_px} px max dimension "
                            f"for model {self.model}.",
                            provider=self.PROVIDER_NAME,
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

    # Caller-facing canonical effort levels (the union of OpenAI's
    # reasoning_effort vocabulary and Gemini's ThinkingLevel enum).
    _EFFORT_LEVELS: frozenset[str] = frozenset({"minimal", "low", "medium", "high"})

    def _resolve_thinking(
        self,
        thinking: Union[bool, int, str, None],
    ) -> Union[bool, int, str, None]:
        """
        Validate and normalize the caller's ``thinking`` parameter.

        - ``None``  → passthrough (no thinking, no pre-flight).
        - ``False`` → explicitly disable thinking; requires ``"off"`` in
                      ``capabilities.thinking``.
        - ``True``  → enable thinking at maximum budget; requires ``"on"``.
        - ``int``   → validated positive token budget; requires ``"on"``.
        - ``str``   → validated effort level; requires ``"on"``.

        ``capabilities.thinking_style`` is informational only at this layer —
        providers translate between budget/effort/adaptive shapes
        transparently. This method enforces only on/off membership.

        Raises ``AIProviderError`` when the catalog says the requested
        on/off state is not supported. The two failure modes — "model
        doesn't think at all" vs "model can't disable thinking" — produce
        distinct messages. Pre-flight is skipped when ``self._model_info``
        is ``None`` (no catalog).

        Returns:
            The validated thinking value, or ``None``/``False``.
        """
        caps = self._model_info.capabilities if self._model_info is not None else None

        if thinking is None:
            return None

        if thinking is False:
            if caps is not None and caps.thinking is not None and "off" not in caps.thinking:
                raise AIProviderError(
                    f"Model '{self.model}' does not support disabling thinking "
                    f"(capabilities.thinking={caps.thinking} in catalog — "
                    f"reasoning is always on). Pass thinking=None to leave "
                    f"the provider default in place, or use thinking=True/int/str.",
                    provider=self.PROVIDER_NAME,
                )
            return False

        # thinking is True | int | str → caller wants thinking enabled.
        if caps is not None and caps.thinking is not None and "on" not in caps.thinking:
            raise AIProviderError(
                f"Model '{self.model}' does not support thinking/reasoning "
                f"(capabilities.thinking={caps.thinking} in catalog). "
                f"Use a thinking-capable model, or pass thinking=None.",
                provider=self.PROVIDER_NAME,
            )

        if thinking is True:
            return True

        # ``capabilities.thinking_style`` IS enforced here. Each native
        # provider field (Claude budget_tokens, OpenAI reasoning.effort,
        # Gemini thinking_budget / thinking_level) accepts a specific
        # caller shape; mismatches fail fast locally with a clear message
        # rather than being silently translated into an arbitrary value.
        styles = caps.thinking_style if caps is not None else None

        # bool True is also an int subclass — handle bool before int.
        # (Already handled above; arriving here means thinking is not bool.)
        if isinstance(thinking, int):
            if thinking <= 0:
                raise ValueError("thinking token budget must be a positive integer.")
            if styles is not None and "budget" not in styles:
                raise ValueError(
                    f"thinking=int (token budget) is not supported by model "
                    f"'{self.model}'. Model accepts thinking_style={styles}. "
                    f"Pass a string ({sorted(self._EFFORT_LEVELS)}) "
                    f"or True/False/None instead."
                )
            return thinking
        if isinstance(thinking, str):
            low = thinking.lower()
            if low not in self._EFFORT_LEVELS:
                raise ValueError(
                    f"thinking effort must be one of "
                    f"{sorted(self._EFFORT_LEVELS)} — got '{thinking}'."
                )
            if styles is not None and "effort" not in styles:
                raise ValueError(
                    f"thinking=str (effort level) is not supported by model "
                    f"'{self.model}'. Model accepts thinking_style={styles}. "
                    f"Pass an int token budget or True/False/None instead."
                )
            return low
        raise TypeError(
            f"thinking must be bool, int (token budget), str (effort level), or None — "
            f"got {type(thinking).__name__}."
        )

    def _resolve_max_output_tokens(self, max_output_tokens: Optional[int]) -> Optional[int]:
        """
        Resolve the effective ``max_output_tokens`` for a request.

        If the caller passed ``None``, auto-fill from the model catalog's
        ``max_output_tokens``. This prevents expensive incomplete responses
        and ensures the model has its full output capacity available.

        Resolution order:
        1. Caller's explicit value (if provided and > 0)
        2. Model catalog ``max_output_tokens`` (if available and > 0)
        3. ``None`` (let the provider SDK use its own default)

        Returns:
            An integer token limit, or ``None`` if unknown.
        """
        if max_output_tokens is not None and max_output_tokens > 0:
            return max_output_tokens
        if self._model_info and self._model_info.max_output_tokens > 0:
            return self._model_info.max_output_tokens
        return None

    def _get_max_thinking_budget(self, max_output_tokens: Optional[int]) -> int:
        """
        Determine the maximum thinking budget for ``thinking=True`` on
        budget-style providers (Claude).

        Resolution order:
        1. Model catalog ``max_output_tokens`` (if available and > 0)
        2. Caller's ``max_output_tokens`` (if provided and > 0)
        3. Raise — refuse to invent a budget.

        Raises:
            ValueError: When neither the catalog nor the caller supplied a
                positive ``max_output_tokens`` to size the budget against.
        """
        if self._model_info and self._model_info.max_output_tokens > 0:
            return self._model_info.max_output_tokens
        if max_output_tokens and max_output_tokens > 0:
            return max_output_tokens
        raise ValueError(
            f"thinking=True on model '{self.model}' requires a model "
            f"catalog entry with max_output_tokens, or an explicit "
            f"max_output_tokens argument; cannot synthesize a thinking "
            f"budget. Pass an explicit int token budget instead."
        )

    def _resolve_temperature(
        self,
        temperature: float,
        thinking_active: bool,
    ) -> Optional[float]:
        """
        Decide the effective temperature to send to the provider.

        Consults ``capabilities.temperature``:

        * ``"any"`` in list → caller's value is sent as-is.
        * ``"any"`` absent (e.g. ``["default"]``) → caller's value is stripped
          (returns ``None``) so the model uses its own default. This avoids
          the 400 error reasoning-only models raise when given an explicit
          temperature.
        * ``None`` (unknown) → passthrough.

        ``thinking_active`` is informational; the provider subclass is
        responsible for any provider-specific override (e.g., Claude forces
        temperature=1 when thinking).

        Returns:
            The temperature to use, or ``None`` meaning "omit".
        """
        if self._model_info is not None:
            cap = self._model_info.capabilities.temperature
            if cap is not None and "any" not in cap:
                return None
        return temperature

    # ------------------------------------------------------------------
    # Probe-error classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_probe_error(exc: Exception) -> str:
        """Classify an exception raised by a tier-probe as 'not_supported' or 'inconclusive'.

        Used by multi-tier probes (e.g. ``probe_thinking_style``) to decide
        whether a tier failure is a confirmed signal that the model rejects
        the feature, or a transient/unknown failure that should leave the
        catalog unchanged. Conservative bias: anything ambiguous returns
        ``"inconclusive"`` so the probe returns ``None`` and the orchestrator
        keeps any cached value (or leaves it null) rather than committing a
        partial truth.

        Returns:
            ``"not_supported"`` — the API clearly rejected the feature
                (HTTP 400, "invalid argument", "not supported", "unknown
                field", etc.). Recording the tier as a confirmed-no is
                safe.
            ``"inconclusive"`` — rate limit, timeout, network error, 5xx,
                or unknown error class. Caller should refuse to commit
                this probe run.
        """
        err = str(exc).lower()
        status = (
            getattr(exc, "status_code", None)
            or getattr(exc, "http_status", None)
            or getattr(exc, "code", None)
        )

        # Inconclusive: transient, retryable, or environmental
        if status in (408, 409, 425, 429, 500, 502, 503, 504):
            return "inconclusive"
        if any(s in err for s in (
            "rate", "quota", "429",
            "timeout", "deadline", "deadline_exceeded",
            "503", "504", "500",
            "internal", "unavailable", "resource_exhausted",
            "connection", "network",
        )):
            return "inconclusive"

        # Confirmed not supported: API explicitly rejected the request shape
        if status == 400:
            return "not_supported"
        if any(s in err for s in (
            "invalid_argument", "invalid argument",
            "not support", "not_supported", "unsupported",
            "unknown field", "unknown_field",
            "bad request", "400",
        )):
            return "not_supported"

        # Unknown error → conservative: don't commit a wrong negative
        return "inconclusive"

    # ------------------------------------------------------------------
    # Opt-in request inspection (DJINNITE_DEBUG_REQUESTS=1)
    # ------------------------------------------------------------------

    _DEBUG_REQUEST_ENV: str = "DJINNITE_DEBUG_REQUESTS"

    @staticmethod
    def _debug_requests_enabled() -> bool:
        """Return True if DJINNITE_DEBUG_REQUESTS is set to a truthy value.

        Read on every call so the toggle works without restart. The cost
        of a single ``os.environ.get`` is negligible (~microseconds) and
        keeps the off-path branch-predictable.
        """
        val = os.environ.get(BaseAIProvider._DEBUG_REQUEST_ENV, "")
        return val.strip().lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _elide_for_debug(obj: Any, max_str: int = 200, max_list: int = 8) -> Any:
        """Recursively shorten long strings, bytes, and large lists for one-line debug output."""
        if isinstance(obj, str):
            if len(obj) > max_str:
                return f"<str len={len(obj)} head={obj[:max_str]!r}>"
            return obj
        if isinstance(obj, (bytes, bytearray)):
            return f"<bytes len={len(obj)}>"
        if isinstance(obj, dict):
            return {k: BaseAIProvider._elide_for_debug(v, max_str, max_list) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            head = [BaseAIProvider._elide_for_debug(x, max_str, max_list) for x in obj[:max_list]]
            if len(obj) > max_list:
                head.append(f"... (+{len(obj) - max_list} more)")
            return head
        return obj

    def _debug_dump_request(
        self,
        *,
        method: str,
        caller_args: Dict[str, Any],
        native_config: Dict[str, Any],
    ) -> None:
        """Dump one debug line per SDK request when DJINNITE_DEBUG_REQUESTS=1.

        The line is written to stderr with no logging-library dependency.
        Both the caller's pre-resolve args and the resolved provider-native
        config dict are included; large fields (prompts, system_instruction,
        message lists) are elided. Zero overhead when the env var is unset.
        """
        if not self._debug_requests_enabled():
            return
        try:
            elided_caller = self._elide_for_debug(caller_args)
            elided_native = self._elide_for_debug(native_config)
            line = (
                f"[DJINNITE_REQUEST] {self.PROVIDER_NAME}/{self.model} {method} "
                f"caller={_json.dumps(elided_caller, default=str, ensure_ascii=False)} "
                f"native={_json.dumps(elided_native, default=str, ensure_ascii=False)}"
            )
            print(line, file=sys.stderr, flush=True)
        except Exception as e:
            # Diagnostics must never break the request path.
            print(
                f"[DJINNITE_REQUEST] {self.PROVIDER_NAME}/{self.model} {method} "
                f"<dump failed: {e!r}>",
                file=sys.stderr, flush=True,
            )

    # ------------------------------------------------------------------

    def _check_capability(self, capability: str) -> None:
        """
        Pre-flight check: raise if the catalog says the model does NOT
        support the requested capability ("on" not in the relevant list).

        Skipped when ``self._model_info`` is None (no catalog loaded,
        e.g. during probing or testing) or when the field is ``None``
        (unknown).

        Args:
            capability: ``"structured_json"``, ``"json_with_search"``,
                or ``"web_search"``.

        Raises:
            AIProviderError: If the catalog's list does not contain ``"on"``.
        """
        if self._model_info is None:
            return  # No catalog → allow (could be a new/unknown model)

        caps = self._model_info.capabilities

        if capability == "structured_json":
            ssj = caps.structured_json
            if ssj is not None and "on" not in ssj:
                raise AIProviderError(
                    f"Model '{self.model}' does not support structured JSON "
                    f"(capabilities.structured_json={ssj} in catalog). "
                    f"Use a different model, or pass force=True to bypass.",
                    provider=self.PROVIDER_NAME,
                )

        elif capability == "json_with_search":
            jws = caps.json_with_search
            if jws is not None and "on" not in jws:
                raise AIProviderError(
                    f"Model '{self.model}' does not support combining structured "
                    f"JSON output with web search (capabilities.json_with_search={jws} "
                    f"in catalog). Use a model that supports this combination "
                    f"(e.g. Gemini 3.x), or pass force=True to bypass.",
                    provider=self.PROVIDER_NAME,
                )

        elif capability == "web_search":
            ws = caps.web_search
            if ws is not None and "on" not in ws:
                raise AIProviderError(
                    f"Model '{self.model}' does not support native web search "
                    f"(capabilities.web_search={ws} in catalog). Use a model "
                    f"that supports it, or pass force=True to bypass.",
                    provider=self.PROVIDER_NAME,
                )

    def generate_json(
        self,
        prompt: Union[str, List[Dict]],
        schema: Union[Dict, Type],
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_output_tokens: Optional[int] = None,
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
            max_output_tokens: Cap on output tokens the model may emit
                (auto-fills from catalog when ``None``). See ``generate()``
                for the per-provider semantic note.
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
            max_output_tokens=max_output_tokens,
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

    def probe_thinking_style(self) -> Optional[list[str]]:
        """
        Probe which thinking styles the current model supports.

        Performs a multi-tier probe (provider-specific) and returns the
        list of styles the model accepts. A model may support more than
        one — Claude 4.7 supports both ``"adaptive"`` and ``"budget"``.

        Returns:
            ``list[str]`` drawn from ``("adaptive", "budget", "effort")``:

            * non-empty list → confirmed: those styles work.
            * empty list ``[]`` → confirmed: model does not support thinking.
            * ``None`` → inconclusive (rate limit, transient error). The
              orchestrator treats this as unknown.
        """
        return None  # Base: unknown — subclasses override

    def probe_thinking_disable(self) -> Optional[bool]:
        """
        Probe whether the model accepts an explicit "thinking disabled"
        request — i.e. whether the caller can pass ``thinking=False``
        without the vendor rejecting the call.

        Returns:
            True  – disable is accepted (toggleable model)
            False – disable is rejected (always-on reasoning model)
            None  – inconclusive, or the model does not support thinking
                    at all (in which case ``"off"`` is the only valid state)
        """
        return None  # Base: unknown — subclasses override

    def probe_web_search(self) -> Optional[bool]:
        """
        Probe whether the model accepts a web-search / grounding tool.

        Returns:
            True  – web search accepted
            False – vendor rejects web-search request
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

    def probe_json_with_search(self) -> Optional[bool]:
        """
        Probe whether the current model supports **both** structured JSON
        output (Constraint Decoding) and web search/grounding combined in
        a single request.

        Some providers/models support each feature individually but reject
        the combination (e.g. Gemini 2.x).  This probe tests the combo.

        Returns:
            True  – both features work together in one request
            False – the combination is rejected by the provider
            None  – inconclusive (rate limit, auth error, etc.)
        """
        return None  # Base: unknown — subclasses override

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
