"""
Anthropic Claude AI Provider

Wraps the Anthropic SDK for Claude models.
Supports native web search for Claude 4.5+/4.6+ models.
"""

import json
import base64
from typing import Optional, Union, List, Dict, Type

from .base_provider import (
    BaseAIProvider,
    AIResponse,
    AIProviderError,
    AIRateLimitError,
    AIAuthenticationError,
    AIModelNotFoundError,
    AIOutputTruncatedError,
    AIContextLengthError,
)


# Web search tool configuration (GA since Claude 4.6, Feb 2026)
_WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search"}


def _supports_native_web_search(model_id: str) -> bool:
    """
    Check if a Claude model supports native web search.
    Supported on Claude 4.5+ and 4.6+ models.
    """
    low = model_id.lower()
    if "latest" in low:
        return True
    # Match both dash and dot formats: 4-5, 4.5, 4-6, 4.6
    for gen in ("4-5", "4.5", "4-6", "4.6"):
        if gen in low:
            return True
    return False


class ClaudeProvider(BaseAIProvider):
    """
    Anthropic Claude AI provider implementation.
    """
    
    PROVIDER_NAME = "claude"
    
    def _initialize_client(self) -> None:
        """Initialize the Anthropic client."""
        try:
            import anthropic
            self._anthropic = anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise AIProviderError(
                "anthropic package not installed. "
                "Install with: pip install anthropic",
                provider=self.PROVIDER_NAME
            )
        except Exception as e:
            raise AIProviderError(
                f"Failed to initialize Claude client: {e}",
                provider=self.PROVIDER_NAME,
                original_error=e
            )

    def _map_parts(self, parts: List[Dict]) -> List:
        """Map internal parts to Anthropic SDK content blocks."""
        claude_content = []
        for part in parts:
            if part["type"] == "text":
                claude_content.append({"type": "text", "text": part["text"]})
            elif part["type"] == "image":
                if "image_data" in part:
                    data = part["image_data"]
                    if isinstance(data, bytes):
                        data = base64.b64encode(data).decode("utf-8")
                    claude_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": part.get("mime_type", "image/jpeg"),
                            "data": data
                        }
                    })
                # Claude doesn't support file_uri directly in the same way as Gemini
        return claude_content
    
    @staticmethod
    def _count_search_units(response) -> tuple[int, Optional[int]]:
        """Count billable web search invocations from an Anthropic response.

        Returns:
            (search_units, search_result_tokens) -- search_result_tokens is
            None when no search occurred or the SDK doesn't report it.
        """
        units = 0
        if not hasattr(response, 'content') or not response.content:
            return 0, None
        for block in response.content:
            btype = getattr(block, 'type', None)
            if btype == 'tool_use' and getattr(block, 'name', None) == 'web_search':
                units += 1
            elif btype == 'server_tool_use' and getattr(block, 'name', None) == 'web_search':
                units += 1
        # Anthropic bills search_result tokens at the model's input rate.
        # The SDK may expose these via usage; extract if available.
        result_tokens = None
        if units and hasattr(response, 'usage') and response.usage:
            result_tokens = getattr(response.usage, 'server_tool_use_input_tokens', None)
        return units, result_tokens

    def _build_claude_thinking(
        self,
        thinking: Union[bool, int, str, None],
        max_tokens: int,
    ) -> tuple[Optional[dict], int]:
        """
        Translate the unified ``thinking`` parameter into Claude's native
        thinking block format and adjust ``max_tokens`` to be valid.

        Claude supports two thinking types:
        - ``"adaptive"``: model decides when/how much to think, with a
          budget cap.  Preferred for newest models.
        - ``"enabled"``: fixed-budget explicit thinking.  For models that
          support thinking but not adaptive mode.

        The ``thinking_style`` from the model catalog determines which type
        to use.  If no catalog is available, defaults to ``"adaptive"``
        (the newer, more capable mode).

        **Invariant enforced:** Claude requires ``max_tokens > budget_tokens``.
        If the caller's ``max_tokens`` is not large enough, this method
        automatically adjusts it upward to leave room for output.

        Args:
            thinking: The caller's thinking parameter (already validated
                      by ``_resolve_thinking``).
            max_tokens: The effective max output tokens for the request.

        Returns:
            A tuple of ``(thinking_block, adjusted_max_tokens)``.
            ``thinking_block`` is ``None`` if thinking is not requested.
        """
        if thinking is None or thinking is False:
            return None, max_tokens

        # Determine thinking style from catalog, default to adaptive
        style = "adaptive"
        if self._model_info and self._model_info.capabilities.thinking_style:
            style = self._model_info.capabilities.thinking_style

        # Compute budget_tokens
        if thinking is True:
            budget = self._get_max_thinking_budget(max_tokens)
        elif isinstance(thinking, int):
            budget = thinking
        else:
            # str effort level → token budget as fraction of max_tokens
            budget = self._effort_to_budget(thinking, max_tokens)

        # Invariant: max_tokens must exceed budget_tokens to leave room
        # for the actual response output.
        if budget >= max_tokens:
            max_tokens = budget + max(1024, budget // 4)

        thinking_type = "adaptive" if style == "adaptive" else "enabled"
        return {"type": thinking_type, "budget_tokens": budget}, max_tokens

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
        Generate a response using Claude.
        """
        # Validate & normalize the thinking parameter
        thinking = self._resolve_thinking(thinking)

        try:
            parts = self._normalize_input(prompt)
            self._validate_vision_limits(parts)
            claude_content = self._map_parts(parts)

            # Claude requires max_tokens to be specified.
            # Auto-fill from catalog if caller didn't provide one.
            max_tokens = self._resolve_max_tokens(max_tokens) or 8192

            # Build thinking block + adjust max_tokens (invariant: max_tokens > budget)
            thinking_active = thinking is not None and thinking is not False
            thinking_block, max_tokens = self._build_claude_thinking(thinking, max_tokens)
            
            # Resolve temperature: strip if catalog says not supported,
            # force to 1 when thinking is active (Claude requirement).
            effective_temp = self._resolve_temperature(temperature, thinking_active)
            if thinking_active and effective_temp is not None:
                effective_temp = 1  # Claude requires temperature=1 when thinking is on

            # Build request kwargs
            kwargs = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": claude_content}],
            }

            # Temperature: only include if resolved (not stripped)
            if effective_temp is not None:
                kwargs["temperature"] = effective_temp
            
            if system_prompt:
                kwargs["system"] = system_prompt

            # Thinking: add the provider-native thinking block
            if thinking_block is not None:
                kwargs["thinking"] = thinking_block

            # Web search: add tool + beta header via SDK
            if web_search:
                if not _supports_native_web_search(self.model):
                    raise AIProviderError(
                        f"Web search not supported for model '{self.model}'.",
                        provider=self.PROVIDER_NAME,
                    )
                kwargs["tools"] = [_WEB_SEARCH_TOOL]

            # Generate response.
            # Always use streaming — Claude requires it for large max_tokens
            # values (>~64K) and for long-running thinking requests.
            # stream.get_final_message() returns the same response object as
            # messages.create(), so all downstream parsing works unchanged.
            with self._client.messages.stream(**kwargs) as stream:
                response = stream.get_final_message()
            
            # Extract content
            content = ""
            output_parts = []
            if response.content:
                for block in response.content:
                    if hasattr(block, 'text'):
                        content += block.text
                        output_parts.append({"type": "text", "text": block.text})
            
            # Extract usage info
            usage = {}
            if response.usage:
                input_t = getattr(response.usage, 'input_tokens', 0) or 0
                output_t = getattr(response.usage, 'output_tokens', 0) or 0
                thinking_t = getattr(response.usage, 'thinking_tokens', None)
                usage = {
                    "input_tokens": input_t,
                    "output_tokens": output_t,
                    "total_tokens": input_t + output_t,
                    "thinking_tokens": thinking_t,
                    "_thinking_billed_separately": True,
                }

            # Count billable search events
            s_units, s_result_tokens = self._count_search_units(response)
            if s_units:
                usage["search_units"] = s_units
            if s_result_tokens is not None:
                usage["search_result_tokens"] = s_result_tokens
            self._compute_costs(usage)

            # Detect output truncation: Anthropic returns stop_reason="max_tokens"
            # when the output was cut short due to the max_tokens limit.
            # This is an HTTP 200 response — the SDK does NOT raise an exception.
            stop_reason = getattr(response, 'stop_reason', None)
            is_truncated = (stop_reason == "max_tokens")
            
            ai_response = AIResponse(
                content=content,
                model=self.model,
                provider=self.PROVIDER_NAME,
                usage=usage,
                parts=output_parts,
                raw_response=response,
                truncated=is_truncated,
                finish_reason=stop_reason,
            )
            
            if is_truncated:
                raise AIOutputTruncatedError(
                    f"Output truncated: model hit max output token limit "
                    f"(stop_reason='max_tokens', output_tokens={usage.get('output_tokens', '?')})",
                    provider=self.PROVIDER_NAME,
                    partial_response=ai_response,
                )
            
            return ai_response
            
        except (AIOutputTruncatedError, AIContextLengthError):
            raise  # Never swallow our own semantic errors
        except Exception as e:
            error_message = str(e).lower()
            error_type = type(e).__name__
            
            # Detect context length exceeded: Anthropic SDK raises
            # anthropic.BadRequestError (HTTP 400) with type="invalid_request_error"
            # when the input exceeds the model's context window.
            if ("too many" in error_message and "token" in error_message) or \
               ("context" in error_message and "length" in error_message) or \
               ("prompt is too long" in error_message):
                raise AIContextLengthError(
                    f"Input context too long for model '{self.model}': {e}",
                    provider=self.PROVIDER_NAME,
                    original_error=e
                )
            elif "authentication" in error_message or "api_key" in error_message or "AuthenticationError" in error_type:
                raise AIAuthenticationError(
                    "Invalid API key or authentication failed",
                    provider=self.PROVIDER_NAME,
                    original_error=e
                )
            elif "rate" in error_message or "RateLimitError" in error_type:
                raise AIRateLimitError(
                    "Rate limit exceeded",
                    provider=self.PROVIDER_NAME,
                    original_error=e
                )
            elif "model" in error_message and ("not found" in error_message or "NotFoundError" in error_type):
                raise AIModelNotFoundError(
                    f"Model '{self.model}' not found",
                    provider=self.PROVIDER_NAME,
                    original_error=e
                )
            else:
                raise AIProviderError(
                    f"Generation failed: {e}",
                    provider=self.PROVIDER_NAME,
                    original_error=e
                )
    
    # ------------------------------------------------------------------
    # Schema normalization for Claude strict mode
    # ------------------------------------------------------------------

    def _prepare_schema_for_provider(self, schema: Dict) -> Dict:
        """
        Claude-specific schema transformation.

        Claude's ``output_config`` JSON schema mode requires explicit
        ``additionalProperties: false`` on all object nodes (same as OpenAI).
        Unlike OpenAI, Claude accepts top-level arrays — no wrapping needed.

        1. Deep-copies the schema to avoid mutating the caller's dict.
        2. Recursively adds ``additionalProperties: false`` to every object.
        """
        import copy
        schema = copy.deepcopy(schema)
        self._ensure_required_arrays(schema)
        self._add_additional_properties_false(schema)
        return schema

    # ------------------------------------------------------------------

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
        Generates structured JSON using Anthropic's **Constraint Decoding** (``output_config``).

        [AGENT NOTE]: Uses ``output_config`` with ``json_schema`` to enforce Guaranteed
        Structure at the API level.  The output is mathematically constrained to the
        supplied schema — no post-hoc parsing or validation needed.

        Args:
            prompt: The user prompt (str or list of multimodal parts).
            schema: **Required.** A Pydantic BaseModel class or JSON Schema dict.
            system_prompt: Optional system instruction.
            temperature: Sampling temperature (default 0.3).
            max_tokens: Maximum tokens to generate.
            web_search: If True, enable native Claude web search (4.5+/4.6+ models).
            thinking: Optional thinking/reasoning control (same as generate()).

        Returns:
            AIResponse whose ``content`` is schema-conforming JSON.
        """
        if schema is None:
            raise ValueError(
                "schema is required for generate_json(). "
                "Use generate() for freeform text responses."
            )
        if not force:
            self._check_capability("structured_json")
        json_schema = self._normalize_schema(schema)
        json_schema = self._validate_caller_schema(json_schema)
        json_schema = self._prepare_schema_for_provider(json_schema)

        # Claude requires max_tokens. Auto-fill from catalog, fallback to 8192.
        max_tokens = self._resolve_max_tokens(max_tokens) or 8192
        
        if web_search:
            if not _supports_native_web_search(self.model):
                raise AIProviderError(
                    f"Web search not supported for model '{self.model}'.",
                    provider=self.PROVIDER_NAME,
                )
            if not force:
                self._check_capability("json_with_search")

        # Validate & normalize thinking
        thinking = self._resolve_thinking(thinking)

        try:
            parts = self._normalize_input(prompt)
            self._validate_vision_limits(parts)
            claude_content = self._map_parts(parts)

            # Build thinking block + adjust max_tokens (invariant: max_tokens > budget)
            thinking_active = thinking is not None and thinking is not False
            thinking_block, max_tokens = self._build_claude_thinking(thinking, max_tokens)

            # Resolve temperature: strip if catalog forbids, force=1 for thinking
            effective_temp = self._resolve_temperature(temperature, thinking_active)
            if thinking_active and effective_temp is not None:
                effective_temp = 1
            
            kwargs = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": claude_content}],
                # Anthropic Constraint Decoding via output_config.format
                "output_config": {
                    "format": {
                        "type": "json_schema",
                        "schema": json_schema,
                    }
                },
            }

            if effective_temp is not None:
                kwargs["temperature"] = effective_temp
            
            if system_prompt:
                kwargs["system"] = system_prompt

            # Thinking block
            if thinking_block is not None:
                kwargs["thinking"] = thinking_block

            # Web search: combine output_config (constraint decoding) with
            # web_search tool in the same request — native JSON + search.
            if web_search:
                kwargs["tools"] = [_WEB_SEARCH_TOOL]

            # Always use streaming — same reason as generate().
            with self._client.messages.stream(**kwargs) as stream:
                response = stream.get_final_message()
            
            # Extract content
            content = ""
            output_parts = []
            if response.content:
                for block in response.content:
                    if hasattr(block, 'text'):
                        content += block.text
                        output_parts.append({"type": "text", "text": block.text})
            
            # Extract usage info
            usage = {}
            if response.usage:
                input_t = getattr(response.usage, 'input_tokens', 0) or 0
                output_t = getattr(response.usage, 'output_tokens', 0) or 0
                thinking_t = getattr(response.usage, 'thinking_tokens', None)
                usage = {
                    "input_tokens": input_t,
                    "output_tokens": output_t,
                    "total_tokens": input_t + output_t,
                    "thinking_tokens": thinking_t,
                    "_thinking_billed_separately": True,
                }

            # Count billable search events
            s_units, s_result_tokens = self._count_search_units(response)
            if s_units:
                usage["search_units"] = s_units
            if s_result_tokens is not None:
                usage["search_result_tokens"] = s_result_tokens
            self._compute_costs(usage)

            # Detect output truncation
            stop_reason = getattr(response, 'stop_reason', None)
            is_truncated = (stop_reason == "max_tokens")
            
            ai_response = AIResponse(
                content=content,
                model=self.model,
                provider=self.PROVIDER_NAME,
                usage=usage,
                parts=output_parts,
                raw_response=response,
                truncated=is_truncated,
                finish_reason=stop_reason,
            )
            
            if is_truncated:
                raise AIOutputTruncatedError(
                    f"JSON output truncated: model hit max output token limit "
                    f"(stop_reason='max_tokens', output_tokens={usage.get('output_tokens', '?')})",
                    provider=self.PROVIDER_NAME,
                    partial_response=ai_response,
                )
            
            return ai_response
            
        except (AIOutputTruncatedError, AIContextLengthError):
            raise
        except AIProviderError:
            raise
        except Exception as e:
            error_message = str(e).lower()
            error_type = type(e).__name__
            
            if ("too many" in error_message and "token" in error_message) or \
               ("context" in error_message and "length" in error_message) or \
               ("prompt is too long" in error_message):
                raise AIContextLengthError(
                    f"Input context too long for model '{self.model}': {e}",
                    provider=self.PROVIDER_NAME,
                    original_error=e
                )
            
            raise AIProviderError(
                f"JSON generation failed: {e}",
                provider=self.PROVIDER_NAME,
                original_error=e
            )
    
    def is_available(self) -> bool:
        """Check if Claude is available."""
        if not self.api_key:
            return False
        
        try:
            self._client.count_tokens(
                model=self.model,
                messages=[{"role": "user", "content": "test"}]
            )
            return True
        except Exception:
            return self._client is not None

    def list_models(self) -> list[dict]:
        """List available models from Claude."""
        if not self.api_key:
            return []
            
        try:
            models = self._client.models.list(limit=100)
            
            models_list = []
            for model in models:
                model_id = model.id
                name = getattr(model, "display_name", model_id)
                context = 200000
                cost = "standard"
                
                if "opus" in model_id:
                    cost = "premium"
                elif "haiku" in model_id:
                    cost = "economical"
                
                modalities = ["text", "vision"]
                
                models_list.append({
                    "id": model_id,
                    "name": name,
                    "context_window": context,
                    "modalities": modalities,
                    "cost_tier": cost
                })
            
            return models_list
        except Exception as e:
            print(f"Error listing Claude models: {e}")
            return []

    def probe_temperature(self) -> Optional[bool]:
        """Probe whether this Claude model accepts temperature. (All Claude models do.)"""
        try:
            self._client.messages.create(
                model=self.model, max_tokens=10, temperature=0.5,
                messages=[{"role": "user", "content": "Say hi."}],
            )
            return True
        except Exception as e:
            err = str(e).lower()
            if "rate" in err or "timeout" in err:
                return None
            return False

    def probe_thinking(self) -> Optional[bool]:
        """Probe whether this Claude model supports extended thinking."""
        style = self.probe_thinking_style()
        if style is None:
            return False
        if style == "_inconclusive":
            return None
        return True

    def probe_thinking_style(self) -> Optional[str]:
        """
        Multi-tier probe to determine which thinking style Claude supports.

        Uses the same invariants as ``_build_claude_thinking()``:
        - ``temperature=1`` (Claude requires this when thinking is on).
        - ``max_tokens > budget_tokens`` (room for output after thinking).

        1. Try adaptive thinking (newest models).
        2. Fall back to enabled/budget thinking (older thinking models).
        3. Both fail → model doesn't support thinking.

        Returns:
            ``"adaptive"`` – model supports adaptive thinking
            ``"budget"``   – model supports fixed-budget thinking only
            ``None``       – model does not support thinking
            ``"_inconclusive"`` – transient error (rate limit, timeout)
        """
        # Probe budget: small enough to be cheap, but we need max_tokens > budget.
        _PROBE_BUDGET = 1024
        _PROBE_MAX_TOKENS = 2048  # Must exceed _PROBE_BUDGET

        # Tier 1: Try adaptive (newest, preferred)
        try:
            self._client.messages.create(
                model=self.model,
                max_tokens=_PROBE_MAX_TOKENS,
                temperature=1,  # Claude requires temp=1 with thinking
                messages=[{"role": "user", "content": "Say hi."}],
                thinking={"type": "adaptive", "budget_tokens": _PROBE_BUDGET},
            )
            return "adaptive"
        except Exception as e:
            err = str(e).lower()
            if "rate" in err or "timeout" in err:
                return "_inconclusive"
            # Adaptive not supported — try budget/enabled

        # Tier 2: Try enabled (fixed budget)
        try:
            self._client.messages.create(
                model=self.model,
                max_tokens=_PROBE_MAX_TOKENS,
                temperature=1,  # Claude requires temp=1 with thinking
                messages=[{"role": "user", "content": "Say hi."}],
                thinking={"type": "enabled", "budget_tokens": _PROBE_BUDGET},
            )
            return "budget"
        except Exception as e:
            err = str(e).lower()
            if "rate" in err or "timeout" in err:
                return "_inconclusive"
            return None

    def probe_structured_json(self) -> Optional[bool]:
        """Probe whether this Claude model supports output_config JSON schema mode."""
        # NOTE: Probe schemas are internal (bypass the caller validation
        # pipeline) and talk directly to the provider API.  Claude requires
        # additionalProperties: false for strict mode.
        _PROBE_SCHEMA = {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
            "additionalProperties": False,
        }
        try:
            self._client.messages.create(
                model=self.model,
                max_tokens=50,
                temperature=0,
                messages=[{"role": "user", "content": "Return the number 1."}],
                output_config={
                    "format": {
                        "type": "json_schema",
                        "schema": _PROBE_SCHEMA,
                    }
                },
            )
            return True
        except Exception as e:
            err = str(e).lower()
            status = getattr(e, 'status_code', None) or getattr(e, 'http_status', None)
            if status == 400 or "not supported" in err or "invalid" in err or "output_config" in err:
                return False
            if status in (401, 403, 429) or "rate" in err or "quota" in err:
                return None
            return None

    def probe_json_with_search(self) -> Optional[bool]:
        """Probe whether this Claude model supports output_config + web_search combined."""
        if not _supports_native_web_search(self.model):
            return False

        _PROBE_SCHEMA = {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
            "additionalProperties": False,
        }
        try:
            self._client.messages.create(
                model=self.model,
                max_tokens=50,
                temperature=0,
                messages=[{"role": "user", "content": "Return the number 1."}],
                output_config={
                    "format": {
                        "type": "json_schema",
                        "schema": _PROBE_SCHEMA,
                    }
                },
                tools=[_WEB_SEARCH_TOOL],
            )
            return True
        except Exception as e:
            err = str(e).lower()
            status = getattr(e, 'status_code', None) or getattr(e, 'http_status', None)
            if status == 400 or "not supported" in err or "invalid" in err or "incompatible" in err:
                return False
            if status in (401, 403, 429) or "rate" in err or "quota" in err:
                return None
            return None

    def discover_modalities(self, model_id: str) -> Dict[str, List[str]]:
        """Discover modalities for Claude models."""
        input_modalities = ["text"]
        output_modalities = ["text"]
        
        low_id = model_id.lower()
        if any(x in low_id for x in ["claude-3", "claude-4"]):
            input_modalities.append("vision")
            
        return {"input": input_modalities, "output": output_modalities}
