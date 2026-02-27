"""
OpenAI Provider

Wraps the OpenAI SDK using the **Responses API** (the successor to
Chat Completions).  Supports native web search, structured JSON output,
and reasoning/thinking — all through a single API surface.
"""

import copy
import json as _json

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


class OpenAIProvider(BaseAIProvider):
    """
    OpenAI provider implementation using the **Responses API**.

    The Responses API is OpenAI's recommended API for all new development
    (successor to Chat Completions).  It provides native web search,
    structured output, and reasoning as first-class features.
    """

    PROVIDER_NAME = "chatgpt"

    def __init__(self, api_key: str, model: str, gemini_api_key: Optional[str] = None, model_info=None):
        """
        Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key
            model: Model ID to use
            gemini_api_key: Deprecated — ignored.  Web search now uses
                OpenAI's native Responses API.  Kept for backward
                compatibility with existing ``get_provider()`` calls.
            model_info: Optional ModelInfo from catalog for pre-flight checks
        """
        # gemini_api_key accepted but ignored — native web search now
        super().__init__(api_key, model, model_info=model_info)

    def _initialize_client(self) -> None:
        """Initialize the OpenAI client."""
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise AIProviderError(
                "openai package not installed. "
                "Install with: pip install openai",
                provider=self.PROVIDER_NAME
            )
        except Exception as e:
            raise AIProviderError(
                f"Failed to initialize OpenAI client: {e}",
                provider=self.PROVIDER_NAME,
                original_error=e
            )

    # ------------------------------------------------------------------
    # Input mapping
    # ------------------------------------------------------------------

    def _map_parts(self, parts: List[Dict]) -> List:
        """Map internal parts to OpenAI Responses API input content."""
        openai_content = []
        for part in parts:
            if part["type"] == "text":
                openai_content.append({"type": "input_text", "text": part["text"]})
            elif part["type"] == "image":
                if "image_data" in part:
                    import base64
                    data = part["image_data"]
                    if isinstance(data, bytes):
                        data = base64.b64encode(data).decode("utf-8")
                    mime = part.get("mime_type", "image/jpeg")
                    openai_content.append({
                        "type": "input_image",
                        "image_url": f"data:{mime};base64,{data}",
                    })
                elif "file_uri" in part:
                    openai_content.append({
                        "type": "input_image",
                        "image_url": part["file_uri"],
                    })
        return openai_content

    # ------------------------------------------------------------------
    # Thinking translation
    # ------------------------------------------------------------------

    def _build_openai_thinking(
        self,
        thinking: Union[bool, int, str, None],
    ) -> Optional[dict]:
        """
        Translate the unified ``thinking`` parameter into the Responses API
        ``reasoning`` parameter.

        Returns:
            A dict for the ``reasoning`` kwarg, or ``None``.
        """
        if thinking is None or thinking is False:
            return None
        if thinking is True:
            effort = "high"
        elif isinstance(thinking, str):
            effort = thinking
        else:
            effort = self._budget_to_effort(thinking)
        return {"effort": effort}

    # ------------------------------------------------------------------
    # Response parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text_from_output(output) -> tuple[str, List[Dict]]:
        """
        Extract text content and parts from a Responses API output list.

        Returns:
            (text_content, output_parts)
        """
        text_content = ""
        output_parts = []
        if output:
            for item in output:
                item_type = getattr(item, "type", "")
                if item_type == "message":
                    # Message output items contain content blocks
                    for block in getattr(item, "content", []):
                        block_type = getattr(block, "type", "")
                        if block_type == "output_text":
                            t = getattr(block, "text", "")
                            text_content += t
                            output_parts.append({"type": "text", "text": t})
                elif item_type == "web_search_call":
                    # Web search tool call — metadata only, not content
                    pass
                elif item_type == "reasoning":
                    # Reasoning output — internal, not returned as content
                    pass
        return text_content, output_parts

    @staticmethod
    def _extract_usage(response) -> dict:
        """Extract token usage from a Responses API response."""
        usage = {}
        if hasattr(response, "usage") and response.usage:
            input_t = getattr(response.usage, "input_tokens", 0) or 0
            output_t = getattr(response.usage, "output_tokens", 0) or 0
            total_t = getattr(response.usage, "total_tokens", None)
            # Reasoning/thinking tokens — nested in output_tokens_details
            thinking_t = None
            details = getattr(response.usage, "output_tokens_details", None)
            if details:
                thinking_t = getattr(details, "reasoning_tokens", None)
            usage = {
                "input_tokens": input_t,
                "output_tokens": output_t,
                "total_tokens": total_t if total_t is not None else input_t + output_t,
                "thinking_tokens": thinking_t,  # None if not reported
            }
        return usage

    @staticmethod
    def _is_truncated(response) -> tuple[bool, Optional[str]]:
        """
        Check if a Responses API response was truncated.

        Returns:
            (is_truncated, finish_reason)
        """
        status = getattr(response, "status", "completed")
        if status == "incomplete":
            details = getattr(response, "incomplete_details", None)
            reason = getattr(details, "reason", "unknown") if details else "unknown"
            return True, reason
        # Normal completion
        return False, status

    # ------------------------------------------------------------------
    # generate()
    # ------------------------------------------------------------------

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
        Generate a response using OpenAI's Responses API.
        """
        # Validate & normalize thinking
        thinking = self._resolve_thinking(thinking)
        thinking_active = thinking is not None and thinking is not False

        try:
            parts = self._normalize_input(prompt)
            openai_content = self._map_parts(parts)

            # Build input: either simple string or structured with role
            if len(openai_content) == 1 and openai_content[0].get("type") == "input_text":
                # Simple text prompt — can use string input
                api_input = openai_content[0]["text"]
            else:
                # Multimodal — use structured input
                api_input = [{"role": "user", "content": openai_content}]

            kwargs = {
                "model": self.model,
                "input": api_input,
            }

            # System prompt → instructions
            if system_prompt:
                kwargs["instructions"] = system_prompt

            # Temperature: catalog-aware + thinking-aware stripping
            effective_temp = self._resolve_temperature(temperature, thinking_active)
            if thinking_active:
                effective_temp = None  # Reasoning models reject temperature
            if effective_temp is not None:
                kwargs["temperature"] = effective_temp

            # Max tokens
            resolved_max = self._resolve_max_tokens(max_tokens)
            if resolved_max:
                kwargs["max_output_tokens"] = resolved_max

            # Thinking → reasoning parameter
            reasoning = self._build_openai_thinking(thinking)
            if reasoning is not None:
                kwargs["reasoning"] = reasoning

            # Web search → native tool
            if web_search:
                kwargs["tools"] = [{"type": "web_search_preview"}]

            # Make the API call
            response = self._client.responses.create(**kwargs)

            # Parse response
            content, output_parts = self._extract_text_from_output(response.output)
            # Fallback: try output_text if our parser didn't find content
            if not content and hasattr(response, "output_text"):
                content = response.output_text or ""
                if content:
                    output_parts = [{"type": "text", "text": content}]

            usage = self._extract_usage(response)
            is_truncated, finish_reason = self._is_truncated(response)

            ai_response = AIResponse(
                content=content,
                model=self.model,
                provider=self.PROVIDER_NAME,
                usage=usage,
                parts=output_parts,
                raw_response=response,
                truncated=is_truncated,
                finish_reason=finish_reason,
            )

            if is_truncated:
                raise AIOutputTruncatedError(
                    f"Output truncated: model hit max output token limit "
                    f"(status='incomplete', reason='{finish_reason}', "
                    f"output_tokens={usage.get('output_tokens', '?')})",
                    provider=self.PROVIDER_NAME,
                    partial_response=ai_response,
                )

            return ai_response

        except (AIOutputTruncatedError, AIContextLengthError):
            raise
        except Exception as e:
            error_message = str(e).lower()

            error_code = getattr(e, "code", None) or ""
            if str(error_code) == "context_length_exceeded" or "context_length_exceeded" in error_message:
                raise AIContextLengthError(
                    f"Input context too long for model '{self.model}': {e}",
                    provider=self.PROVIDER_NAME,
                    original_error=e,
                )
            elif "api_key" in error_message or "auth" in error_message:
                raise AIAuthenticationError(
                    "Invalid API key or authentication failed",
                    provider=self.PROVIDER_NAME,
                    original_error=e,
                )
            elif "rate" in error_message or "quota" in error_message:
                raise AIRateLimitError(
                    "Rate limit exceeded",
                    provider=self.PROVIDER_NAME,
                    original_error=e,
                )
            elif "model" in error_message and "not found" in error_message:
                raise AIModelNotFoundError(
                    f"Model '{self.model}' not found",
                    provider=self.PROVIDER_NAME,
                    original_error=e,
                )
            else:
                raise AIProviderError(
                    f"Generation failed: {e}",
                    provider=self.PROVIDER_NAME,
                    original_error=e,
                )

    # ------------------------------------------------------------------
    # Schema normalization for OpenAI strict mode
    # ------------------------------------------------------------------

    def _prepare_schema_for_provider(self, schema: Dict) -> Dict:
        """
        OpenAI-specific schema transformation.

        1. Deep-copies the schema to avoid mutating the caller's dict.
        2. Recursively adds ``additionalProperties: false`` to every object.
        3. If the top-level type is ``"array"``, wraps it in an object
           envelope because OpenAI strict mode requires top-level ``type: "object"``.
        """
        schema = copy.deepcopy(schema)

        self._openai_array_wrapped = False
        if schema.get("type") == "array":
            schema = {
                "type": "object",
                "properties": {"items": schema},
                "required": ["items"],
            }
            self._openai_array_wrapped = True

        self._add_additional_properties_false(schema)
        return schema

    # ------------------------------------------------------------------
    # generate_json()
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
        Generates structured JSON using OpenAI's Responses API with
        schema-enforced output (``text.format.json_schema``).

        Args:
            prompt: The user prompt (str or list of multimodal parts).
            schema: **Required.** A Pydantic BaseModel class or JSON Schema dict.
            system_prompt: Optional system instruction.
            temperature: Sampling temperature (default 0.3).
            max_tokens: Maximum tokens to generate.
            web_search: If True, enable native OpenAI web search.
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

        # Validate & normalize thinking
        thinking = self._resolve_thinking(thinking)
        thinking_active = thinking is not None and thinking is not False

        try:
            parts = self._normalize_input(prompt)
            openai_content = self._map_parts(parts)

            if len(openai_content) == 1 and openai_content[0].get("type") == "input_text":
                api_input = openai_content[0]["text"]
            else:
                api_input = [{"role": "user", "content": openai_content}]

            kwargs = {
                "model": self.model,
                "input": api_input,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "structured_response",
                        "strict": True,
                        "schema": json_schema,
                    }
                },
            }

            if system_prompt:
                kwargs["instructions"] = system_prompt

            # Temperature
            effective_temp = self._resolve_temperature(temperature, thinking_active)
            if thinking_active:
                effective_temp = None
            if effective_temp is not None:
                kwargs["temperature"] = effective_temp

            # Max tokens
            resolved_max = self._resolve_max_tokens(max_tokens)
            if resolved_max:
                kwargs["max_output_tokens"] = resolved_max

            # Thinking
            reasoning = self._build_openai_thinking(thinking)
            if reasoning is not None:
                kwargs["reasoning"] = reasoning

            # Web search
            if web_search:
                kwargs["tools"] = [{"type": "web_search_preview"}]

            response = self._client.responses.create(**kwargs)

            # Parse response
            content, output_parts = self._extract_text_from_output(response.output)
            if not content and hasattr(response, "output_text"):
                content = response.output_text or ""
                if content:
                    output_parts = [{"type": "text", "text": content}]

            # Transparently unwrap array envelope if we wrapped it
            if getattr(self, "_openai_array_wrapped", False) and content:
                try:
                    parsed = _json.loads(content)
                    content = _json.dumps(parsed["items"])
                except (KeyError, _json.JSONDecodeError):
                    pass

            usage = self._extract_usage(response)
            is_truncated, finish_reason = self._is_truncated(response)

            ai_response = AIResponse(
                content=content,
                model=self.model,
                provider=self.PROVIDER_NAME,
                usage=usage,
                parts=output_parts,
                raw_response=response,
                truncated=is_truncated,
                finish_reason=finish_reason,
            )

            if is_truncated:
                raise AIOutputTruncatedError(
                    f"JSON output truncated: model hit max output token limit "
                    f"(status='incomplete', reason='{finish_reason}', "
                    f"output_tokens={usage.get('output_tokens', '?')})",
                    provider=self.PROVIDER_NAME,
                    partial_response=ai_response,
                )

            return ai_response

        except AIProviderError:
            raise
        except Exception as e:
            error_message = str(e).lower()
            error_code = getattr(e, "code", None) or ""
            if str(error_code) == "context_length_exceeded" or "context_length_exceeded" in error_message:
                raise AIContextLengthError(
                    f"Input context too long for model '{self.model}': {e}",
                    provider=self.PROVIDER_NAME,
                    original_error=e,
                )
            raise AIProviderError(
                f"JSON generation failed: {e}",
                provider=self.PROVIDER_NAME,
                original_error=e,
            )

    # ------------------------------------------------------------------
    # Probes
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Check if OpenAI is available."""
        if not self.api_key:
            return False
        try:
            self._client.models.list()
            return True
        except Exception:
            return False

    def list_models(self) -> list[dict]:
        """List available models from OpenAI."""
        if not self.api_key:
            return []
        try:
            models = self._client.models.list()
            models_list = []
            for model in models.data:
                model_id = model.id
                if "gpt" not in model_id:
                    continue
                context = 128000
                if "gpt-3.5" in model_id:
                    context = 16000
                elif "mini" in model_id:
                    context = 256000
                modalities = ["text"]
                if any(x in model_id.lower() for x in ["vision", "gpt-4", "4o"]):
                    modalities.append("vision")
                models_list.append({
                    "id": model_id,
                    "name": model_id,
                    "context_window": context,
                    "modalities": modalities,
                    "cost_tier": "standard",
                })
            return models_list
        except Exception as e:
            print(f"Error listing OpenAI models: {e}")
            return []

    def probe_temperature(self) -> Optional[bool]:
        """Probe whether this OpenAI model accepts temperature."""
        try:
            self._client.responses.create(
                model=self.model,
                input="Say hi.",
                temperature=0.5,
                max_output_tokens=10,
            )
            return True
        except Exception as e:
            err = str(e).lower()
            status = getattr(e, "status_code", None)
            if status == 429 or "rate" in err or "timeout" in err:
                return None
            if "temperature" in err and ("unsupported" in err or "not support" in err):
                return False
            return False

    def probe_thinking(self) -> Optional[bool]:
        """Probe whether this OpenAI model supports reasoning."""
        style = self.probe_thinking_style()
        if style is None:
            return False
        if style == "_inconclusive":
            return None
        return True

    def probe_thinking_style(self) -> Optional[str]:
        """
        Probe which thinking style this OpenAI model supports.

        Returns:
            ``"effort"``   – model supports reasoning effort
            ``None``       – model does not support thinking
            ``"_inconclusive"`` – transient error
        """
        try:
            self._client.responses.create(
                model=self.model,
                input="Say hi.",
                reasoning={"effort": "low"},
                max_output_tokens=100,
            )
            return "effort"
        except Exception as e:
            err = str(e).lower()
            status = getattr(e, "status_code", None)
            if status == 429 or "rate" in err or "timeout" in err:
                return "_inconclusive"
            return None

    def probe_structured_json(self) -> Optional[bool]:
        """Probe whether this OpenAI model supports structured JSON output."""
        _PROBE_SCHEMA = {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
            "additionalProperties": False,
        }
        try:
            self._client.responses.create(
                model=self.model,
                input="Return the number 1.",
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "probe",
                        "strict": True,
                        "schema": _PROBE_SCHEMA,
                    }
                },
                max_output_tokens=50,
            )
            return True
        except Exception as e:
            err = str(e).lower()
            status = getattr(e, "status_code", None) or getattr(e, "http_status", None)
            if status == 429 or "rate" in err or "quota" in err or "timeout" in err:
                return None
            return False

    def discover_modalities(self, model_id: str) -> Dict[str, List[str]]:
        """Discover modalities for OpenAI models."""
        input_modalities = ["text"]
        output_modalities = ["text"]

        low_id = model_id.lower()
        if "gpt-4o" in low_id or "vision" in low_id:
            input_modalities.append("vision")
        if "tts" in low_id:
            input_modalities = ["text"]
            output_modalities = ["audio"]
        if "whisper" in low_id or "audio" in low_id:
            input_modalities.append("audio")
            if "preview" in low_id:
                output_modalities.append("audio")

        return {"input": input_modalities, "output": output_modalities}
