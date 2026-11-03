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
from .responses_effort import ResponsesEffortProbes, effort_error_kind


class OpenAIProvider(ResponsesEffortProbes, BaseAIProvider):
    """
    OpenAI provider implementation using the **Responses API**.

    The Responses API is OpenAI's recommended API for all new development
    (successor to Chat Completions).  It provides native web search,
    structured output, and reasoning as first-class features.
    """

    PROVIDER_NAME = "chatgpt"
    # Effort is this provider's only reasoning control, so thinking=True
    # without a catalog entry still sends the configured default.
    _EFFORT_EXPRESSES_THINKING = True

    def __init__(self, api_key: str, model: str, gemini_api_key: Optional[str] = None, model_info=None, require_pricing: bool = True):
        """
        Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key
            model: Model ID to use
            gemini_api_key: Deprecated — ignored.  Web search now uses
                OpenAI's native Responses API.  Kept for backward
                compatibility with existing ``get_provider()`` calls.
            model_info: Optional ModelInfo from catalog for pre-flight checks
            require_pricing: Fast-fail on missing/unknown price (see base).
        """
        # gemini_api_key accepted but ignored — native web search now
        super().__init__(api_key, model, model_info=model_info, require_pricing=require_pricing)

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

    def _responses_input(self, turns, openai_content: List[Dict]) -> List[Dict]:
        """Role-tagged ``input`` items: history turns, then the prompt.

        Assistant content is a plain string -- the Responses API rejects
        ``input_text`` parts in the assistant role.
        """
        items: List[Dict] = []
        for role, parts in turns:
            if role == "assistant":
                items.append({"role": "assistant", "content": self._history_text(parts)})
            else:
                items.append({"role": "user", "content": self._map_parts(parts)})
        items.append({"role": "user", "content": openai_content})
        return items

    # ------------------------------------------------------------------
    # Thinking translation
    # ------------------------------------------------------------------

    def _effort_allowed(self, level: str, *, thinking, web_search: bool) -> bool:
        # GPT-5, mini, nano and their dated snapshots reject search at
        # minimal. Do not extend that observation to other model families.
        import re
        gpt5 = re.fullmatch(r"gpt-5(?:-(?:mini|nano))?(?:-\d{4}-\d{2}-\d{2})?", self.model)
        return not (gpt5 and web_search and level == "minimal")

    def _build_openai_thinking(
        self,
        thinking: Union[bool, int, str, None],
        effort_sent: Optional[str],
    ) -> Optional[dict]:
        """
        Build the Responses API ``reasoning`` parameter.

        ``thinking=False`` sends ``effort: "none"``. Otherwise the effort
        ``_resolve_effort`` chose (the caller's Djinnite level or the
        configured default, mapped to a level the model accepts) is sent;
        with none chosen, nothing is sent and the model's default applies.
        There is no token-budget field, so an int is refused.

        Returns:
            A dict for the ``reasoning`` kwarg, or ``None``.
        """
        if thinking is False:
            return {"effort": "none"}
        if isinstance(thinking, int) and not isinstance(thinking, bool):
            raise ValueError(
                f"thinking=int (token budget) is not supported on model "
                f"'{self.model}': this provider's reasoning control is an "
                f"effort level. Pass effort=, True, False, or None."
            )
        if effort_sent is not None:
            return {"effort": effort_sent}
        return None

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
    def _extract_refusal(output) -> str:
        """Concatenate any ``refusal`` content blocks from a Responses API output list."""
        refusal = ""
        for item in output or []:
            if getattr(item, "type", "") == "message":
                for block in getattr(item, "content", None) or []:
                    if getattr(block, "type", "") == "refusal":
                        refusal += getattr(block, "refusal", "") or ""
        return refusal

    @staticmethod
    def _count_search_units(response) -> int:
        """Count billable web search actions from an OpenAI Responses API response.

        OpenAI bills per ``web_search_call`` item in ``response.output``
        where ``action.type == "search"``.  The ``action`` field is a
        complex object containing ``type``, ``query``, and optional
        ``domain_filters``.
        """
        count = 0
        output = getattr(response, "output", None)
        if not output:
            return 0
        for item in output:
            item_type = getattr(item, "type", None)
            if item_type == "web_search_call":
                action = getattr(item, "action", None)
                if action is not None:
                    action_type = getattr(action, "type", None)
                    if action_type == "search":
                        count += 1
                    elif action_type is None:
                        # Fallback: action exists but has no type -> count it
                        count += 1
        return count

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
                "_thinking_billed_separately": False,
            }
        # Count billable search events
        s_units = OpenAIProvider._count_search_units(response)
        if s_units:
            usage["search_units"] = s_units
        return usage

    @staticmethod
    def _is_truncated(response) -> tuple[bool, Optional[str]]:
        """
        Check if a Responses API response was truncated.

        ``incomplete`` with reason ``content_filter`` is a provider block,
        not truncation; callers raise ``AIEmptyResponseError`` for it.

        Returns:
            (is_truncated, finish_reason)
        """
        status = getattr(response, "status", "completed")
        if status == "incomplete":
            details = getattr(response, "incomplete_details", None)
            reason = getattr(details, "reason", "unknown") if details else "unknown"
            return reason != "content_filter", reason
        # Normal completion
        return False, status

    # ------------------------------------------------------------------
    # generate()
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: Union[str, List[Dict]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        web_search: bool = False,
        thinking: Union[bool, int, str, None] = None,
        *,
        effort: Optional[str] = None,
        history: Optional[List[Dict]] = None,
    ) -> AIResponse:
        """
        Generate a response using OpenAI's Responses API.
        """
        _orig_caller = {
            "thinking": thinking,
            "effort": effort,
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
            "web_search": web_search,
            "system_prompt": system_prompt,
            "history_turns": len(history or []),
        }
        # Validate & normalize thinking
        thinking, effort_sent = self._resolve_effort(thinking, effort, web_search=web_search)
        turns = self._normalize_history(history)
        # An effort sent is reasoning switched on (OpenAI's off is "none").
        thinking_active = self._thinking_active(thinking, effort_sent)

        try:
            parts = self._normalize_input(prompt)
            self._validate_vision_limits(self._all_parts(turns, parts))
            openai_content = self._map_parts(parts)

            # Cross-capability pre-flight from catalog.
            state = {
                "temperature":     "default",
                "thinking":        "on"  if thinking_active        else "off",
                "structured_json": "off",
                "web_search":      "on"  if web_search             else "off",
            }
            # Temperature first: a value the model cannot take is logged and
            # dropped, so the pre-flight below checks what is really sent.
            effective_temp = self._resolve_temperature(temperature, state)
            state["temperature"] = "any" if effective_temp is not None else "default"
            self._validate_incompatible_combinations(state)

            # Build input: either simple string or structured with role.
            # With history, always role-tagged items ending in the prompt.
            if turns:
                api_input = self._responses_input(turns, openai_content)
            elif len(openai_content) == 1 and openai_content[0].get("type") == "input_text":
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
            if effective_temp is not None:
                kwargs["temperature"] = effective_temp

            # Max output tokens
            resolved_max = self._resolve_max_output_tokens(max_output_tokens)
            if resolved_max:
                kwargs["max_output_tokens"] = resolved_max

            # Thinking → reasoning parameter
            reasoning = self._build_openai_thinking(thinking, effort_sent)
            if reasoning is not None:
                kwargs["reasoning"] = reasoning

            # Web search → native tool
            if web_search:
                kwargs["tools"] = [{"type": "web_search_preview"}]

            self._debug_dump_request(
                method="generate", caller_args=_orig_caller, native_config=kwargs,
            )

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
            self._compute_costs(usage)
            is_truncated, finish_reason = self._is_truncated(response)
            block_reason = "content_filter" if finish_reason == "content_filter" else None

            # A refusal is the model's own answer: return it as content.
            refusal = self._extract_refusal(response.output)
            if refusal and not content:
                content = refusal
                output_parts = [{"type": "text", "text": refusal}]
                finish_reason = "refusal"

            ai_response = AIResponse(
                content=content,
                model=self.model,
                provider=self.PROVIDER_NAME,
                usage=usage,
                parts=output_parts,
                raw_response=response,
                truncated=is_truncated,
                finish_reason=finish_reason,
                block_reason=block_reason,
            )

            if is_truncated:
                raise AIOutputTruncatedError(
                    f"Output truncated: model hit max output token limit "
                    f"(status='incomplete', reason='{finish_reason}', "
                    f"output_tokens={usage.get('output_tokens', '?')})",
                    provider=self.PROVIDER_NAME,
                    partial_response=ai_response,
                )

            # Provider filtered the output: not the model's answer.
            if block_reason:
                self._raise_empty(
                    ai_response, reason=block_reason,
                    details={"status": getattr(response, "status", None),
                             "partial_chars": len(content)},
                    web_search=web_search, thinking=thinking,
                )

            return ai_response

        except AIProviderError:
            raise
        except Exception as e:
            if effort_sent is not None and effort_error_kind(e) == "level":
                raise self._effort_mapping_error(effort_sent, e)
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

        self._ensure_required_arrays(schema)
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
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        web_search: bool = False,
        force: bool = False,
        thinking: Union[bool, int, str, None] = None,
        *,
        effort: Optional[str] = None,
        history: Optional[List[Dict]] = None,
    ) -> AIResponse:
        """
        Generates structured JSON using OpenAI's Responses API with
        schema-enforced output (``text.format.json_schema``).

        Args:
            prompt: The user prompt (str or list of multimodal parts).
            schema: **Required.** A Pydantic BaseModel class or JSON Schema dict.
            system_prompt: Optional system instruction.
            temperature: Sampling temperature (default 0.3).
            max_output_tokens: Cap on output tokens (auto-fills from catalog).
            web_search: If True, enable native OpenAI web search.
            thinking: Optional thinking/reasoning control (same as generate()).

        Returns:
            AIResponse whose ``content`` is schema-conforming JSON.
        """
        _orig_caller = {
            "thinking": thinking,
            "effort": effort,
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
            "web_search": web_search,
            "system_prompt": system_prompt,
            "force": force,
            "history_turns": len(history or []),
        }
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
        thinking, effort_sent = self._resolve_effort(thinking, effort, web_search=web_search)
        turns = self._normalize_history(history)
        # An effort sent is reasoning switched on (OpenAI's off is "none").
        thinking_active = self._thinking_active(thinking, effort_sent)

        try:
            parts = self._normalize_input(prompt)
            self._validate_vision_limits(self._all_parts(turns, parts))
            openai_content = self._map_parts(parts)

            # Cross-capability pre-flight from catalog.
            state = {
                "temperature":     "default",
                "thinking":        "on"  if thinking_active        else "off",
                "structured_json": "on",
                "web_search":      "on"  if web_search             else "off",
                "json_with_search": "on" if web_search             else "off",
            }
            # Temperature first: a value the model cannot take is logged and
            # dropped, so the pre-flight below checks what is really sent.
            effective_temp = self._resolve_temperature(temperature, state)
            state["temperature"] = "any" if effective_temp is not None else "default"
            self._validate_incompatible_combinations(state)

            if turns:
                api_input = self._responses_input(turns, openai_content)
            elif len(openai_content) == 1 and openai_content[0].get("type") == "input_text":
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
            if effective_temp is not None:
                kwargs["temperature"] = effective_temp

            # Max output tokens
            resolved_max = self._resolve_max_output_tokens(max_output_tokens)
            if resolved_max:
                kwargs["max_output_tokens"] = resolved_max

            # Thinking
            reasoning = self._build_openai_thinking(thinking, effort_sent)
            if reasoning is not None:
                kwargs["reasoning"] = reasoning

            # Web search
            if web_search:
                kwargs["tools"] = [{"type": "web_search_preview"}]

            self._debug_dump_request(
                method="generate_json", caller_args=_orig_caller, native_config=kwargs,
            )

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
            self._compute_costs(usage)
            is_truncated, finish_reason = self._is_truncated(response)
            block_reason = "content_filter" if finish_reason == "content_filter" else None
            refusal = self._extract_refusal(response.output)
            if refusal and not content:
                finish_reason = "refusal"

            ai_response = AIResponse(
                content=content,
                model=self.model,
                provider=self.PROVIDER_NAME,
                usage=usage,
                parts=output_parts,
                raw_response=response,
                truncated=is_truncated,
                finish_reason=finish_reason,
                block_reason=block_reason,
            )

            if is_truncated:
                raise AIOutputTruncatedError(
                    f"JSON output truncated: model hit max output token limit "
                    f"(status='incomplete', reason='{finish_reason}', "
                    f"output_tokens={usage.get('output_tokens', '?')})",
                    provider=self.PROVIDER_NAME,
                    partial_response=ai_response,
                )

            # generate_json promises schema-conforming JSON: a block, a
            # refusal, or an empty reply cannot meet that.
            if block_reason or finish_reason == "refusal" or not content.strip():
                self._raise_empty(
                    ai_response,
                    reason=block_reason or ("refusal" if finish_reason == "refusal" else "empty"),
                    details={"status": getattr(response, "status", None),
                             "refusal": refusal[:200]},
                    web_search=web_search, thinking=thinking,
                )

            return ai_response

        except AIProviderError:
            raise
        except Exception as e:
            if effort_sent is not None and effort_error_kind(e) == "level":
                raise self._effort_mapping_error(effort_sent, e)
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
                # OpenAI's models endpoint does not report context windows,
                # so there is nothing here to read. The previous guesses
                # ("mini" implies 256000, everything else 128000) wrote
                # invented numbers into the catalog that were
                # indistinguishable from measured ones -- gpt-5 landed at
                # context_window == max_output_tokens == 128000, leaving no
                # room for input. Omit the key instead: update_models
                # preserves the existing catalog value and falls back to
                # web-search-backed estimation, per this repo's
                # "no static model data in Python" policy.
                modalities = ["text"]
                if any(x in model_id.lower() for x in ["vision", "gpt-4", "4o"]):
                    modalities.append("vision")
                models_list.append({
                    "id": model_id,
                    "name": model_id,
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

    def _build_combination_probe_request(self, active_states: Dict[str, str]) -> dict:
        """Map activated states to OpenAI Responses API kwargs."""
        kwargs: dict = {
            "model": self.model,
            "input": "Say hi.",
        }
        if active_states.get("temperature") == "any":
            kwargs["temperature"] = 0.5
        if active_states.get("thinking") == "on":
            kwargs["reasoning"] = {"effort": self.probed_effort_level or "low"}
        if active_states.get("structured_json") == "on":
            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "probe_response",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"v": {"type": "integer"}},
                        "required": ["v"],
                        "additionalProperties": False,
                    },
                }
            }
        if active_states.get("web_search") == "on":
            kwargs["tools"] = [{"type": "web_search_preview"}]
        return kwargs

    def _run_combination_probe(self, kwargs: dict) -> None:
        """Send the probe via OpenAI's responses.create."""
        self._client.responses.create(**kwargs)

    def probe_thinking(self) -> Optional[bool]:
        """Probe whether this OpenAI model supports reasoning."""
        styles = self.probe_thinking_style()
        if styles is None:
            return None
        return bool(styles)

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
            )
            return True
        except Exception as e:
            err = str(e).lower()
            status = getattr(e, "status_code", None) or getattr(e, "http_status", None)
            if status == 429 or "rate" in err or "quota" in err or "timeout" in err:
                return None
            return False

    def probe_web_search(self) -> Optional[bool]:
        """Probe whether this OpenAI model accepts the web_search_preview tool."""
        try:
            self._client.responses.create(
                model=self.model,
                input="Say hi.",
                tools=[{"type": "web_search_preview"}],
            )
            return True
        except Exception as e:
            err = str(e).lower()
            status = getattr(e, "status_code", None)
            if status == 429 or "rate" in err or "quota" in err or "timeout" in err:
                return None
            return False

    def probe_json_with_search(self) -> Optional[bool]:
        """Probe whether this OpenAI model supports structured JSON + web search combined."""
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
                tools=[{"type": "web_search_preview"}],
            )
            return True
        except Exception as e:
            err = str(e).lower()
            status = getattr(e, "status_code", None)
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
