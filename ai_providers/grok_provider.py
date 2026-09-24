"""
Grok (xAI) Provider

Wraps the xAI Grok API, which implements the **OpenAI Responses API**
(``POST /v1/responses``) at ``https://api.x.ai/v1``.  Because the wire format
is the Responses API, this provider uses the ``openai`` SDK pointed at xAI's
base URL.

This is a **standalone** provider (parallel stack): it deliberately shares no
code with ``OpenAIProvider``.  The two dialects already diverge — xAI uses
**Live Search** (billed per *source*) rather than OpenAI's ``web_search_preview``
tool (billed per *search*) — and are expected to keep diverging.  The genuinely
cross-provider logic (schema normalization, cost computation, modality/vision
validation, thinking resolution, probe orchestration) lives in
``BaseAIProvider`` and is reused, not duplicated.
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
    AIPricingError,
)


class GrokProvider(BaseAIProvider):
    """
    xAI Grok provider implementation using the **Responses API**.

    xAI exposes an OpenAI-compatible surface; ``/v1/responses`` accepts the
    same request shape as OpenAI's Responses API (``input``, ``instructions``,
    ``reasoning``, ``text.format`` for structured output).  The two
    provider-specific divergences handled here are web search (xAI Live Search)
    and model listing (``grok-*`` ids).
    """

    PROVIDER_NAME = "grok"
    BASE_URL = "https://api.x.ai/v1"

    def __init__(self, api_key: str, model: str, model_info=None, require_pricing: bool = True):
        """
        Initialize the Grok provider.

        Args:
            api_key: xAI API key.
            model: Model ID to use (e.g. ``grok-4.5``).
            model_info: Optional ModelInfo from catalog for pre-flight checks.
            require_pricing: Fast-fail on missing/unknown price (see base).
        """
        super().__init__(api_key, model, model_info=model_info, require_pricing=require_pricing)

    def _initialize_client(self) -> None:
        """Initialize the OpenAI SDK client pointed at xAI's base URL."""
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.BASE_URL)
        except ImportError:
            raise AIProviderError(
                "openai package not installed. "
                "Install with: pip install openai",
                provider=self.PROVIDER_NAME
            )
        except Exception as e:
            raise AIProviderError(
                f"Failed to initialize xAI client: {e}",
                provider=self.PROVIDER_NAME,
                original_error=e
            )

    # ------------------------------------------------------------------
    # Input mapping
    # ------------------------------------------------------------------

    def _map_parts(self, parts: List[Dict]) -> List:
        """Map internal parts to Responses API input content."""
        content = []
        for part in parts:
            if part["type"] == "text":
                content.append({"type": "input_text", "text": part["text"]})
            elif part["type"] == "image":
                if "image_data" in part:
                    import base64
                    data = part["image_data"]
                    if isinstance(data, bytes):
                        data = base64.b64encode(data).decode("utf-8")
                    mime = part.get("mime_type", "image/jpeg")
                    content.append({
                        "type": "input_image",
                        "image_url": f"data:{mime};base64,{data}",
                    })
                elif "file_uri" in part:
                    content.append({
                        "type": "input_image",
                        "image_url": part["file_uri"],
                    })
        return content

    def _build_input(self, openai_content: List[Dict]):
        """Collapse a single text part to a plain string, else structured input."""
        if len(openai_content) == 1 and openai_content[0].get("type") == "input_text":
            return openai_content[0]["text"]
        return [{"role": "user", "content": openai_content}]

    # ------------------------------------------------------------------
    # Thinking translation
    # ------------------------------------------------------------------

    def _build_grok_thinking(
        self,
        thinking: Union[bool, int, str, None],
    ) -> Optional[dict]:
        """
        Translate the unified ``thinking`` parameter into the Responses API
        ``reasoning`` parameter.

        xAI's ``reasoning.effort`` accepts string levels only (there is no
        integer-budget field).  Note that some Grok models (e.g. the flagship
        reasoning models) reason unconditionally and reject an explicit effort
        or ``"none"`` — this is discovered by ``probe_thinking_*`` and reflected
        in the catalog, so the runtime pre-flight rejects unsupported requests
        before they reach the vendor.

        Returns:
            A dict for the ``reasoning`` kwarg, or ``None``.
        """
        if thinking is None:
            return None
        if thinking is False:
            return {"effort": "none"}
        if thinking is True:
            return {"effort": "high"}
        if isinstance(thinking, str):
            return {"effort": thinking}
        raise ValueError(
            f"thinking=int (token budget) is not supported on xAI Grok "
            f"model '{self.model}'. xAI's reasoning parameter accepts only "
            f"string effort levels ('low'/'high', and 'none' to disable on "
            f"models that allow it). Pass a string, True, False, or None."
        )

    # ------------------------------------------------------------------
    # Web search (xAI Live Search)
    # ------------------------------------------------------------------
    #
    # Live Search is xAI's analogue of Gemini grounding / Claude web search.
    # On the Responses API it is enabled with a server-side ``web_search`` tool.
    # xAI bills per *source* returned (~$25/1k), not per search call — so the
    # search-unit counter below counts returned source citations, not tool
    # invocations.
    #
    # NOTE (open item, see GROK_PROVIDER_DESIGN.md): the exact Live Search
    # enablement shape on ``/v1/responses`` (server tool vs. ``search_parameters``)
    # and the field carrying returned sources are pinned down against the live
    # API during rollout.  Both are isolated to the two methods below so an
    # adjustment is a one-line change.

    def _web_search_tools(self) -> List[Dict]:
        """Return the tool list that enables xAI Live Search."""
        return [{"type": "web_search"}]

    @staticmethod
    def _count_search_units(response) -> int:
        """Count billable Live Search sources from a Responses API response.

        xAI bills per source returned.  Sources surface as ``url_citation``
        annotations attached to output-message text blocks; we count the
        distinct URLs cited.  Falls back to counting ``web_search`` tool-call
        items when no annotations are present.
        """
        output = getattr(response, "output", None)
        if not output:
            return 0

        cited_urls = set()
        search_calls = 0
        for item in output:
            item_type = getattr(item, "type", None)
            if item_type in ("web_search_call", "web_search"):
                search_calls += 1
            if item_type == "message":
                for block in getattr(item, "content", []):
                    for ann in getattr(block, "annotations", []) or []:
                        if getattr(ann, "type", None) in ("url_citation", "citation"):
                            url = getattr(ann, "url", None)
                            if url:
                                cited_urls.add(url)
        if cited_urls:
            return len(cited_urls)
        return search_calls

    # ------------------------------------------------------------------
    # Response parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text_from_output(output) -> tuple[str, List[Dict]]:
        """Extract text content and parts from a Responses API output list."""
        text_content = ""
        output_parts = []
        if output:
            for item in output:
                item_type = getattr(item, "type", "")
                if item_type == "message":
                    for block in getattr(item, "content", []):
                        block_type = getattr(block, "type", "")
                        if block_type == "output_text":
                            t = getattr(block, "text", "")
                            text_content += t
                            output_parts.append({"type": "text", "text": t})
                # web_search / reasoning items carry metadata only, not content
        return text_content, output_parts

    def _extract_usage(self, response) -> dict:
        """Extract token usage from a Responses API response."""
        usage = {}
        if hasattr(response, "usage") and response.usage:
            input_t = getattr(response.usage, "input_tokens", 0) or 0
            output_t = getattr(response.usage, "output_tokens", 0) or 0
            total_t = getattr(response.usage, "total_tokens", None)
            thinking_t = None
            details = getattr(response.usage, "output_tokens_details", None)
            if details:
                thinking_t = getattr(details, "reasoning_tokens", None)
            usage = {
                "input_tokens": input_t,
                "output_tokens": output_t,
                "total_tokens": total_t if total_t is not None else input_t + output_t,
                "thinking_tokens": thinking_t,  # None if not reported
                # xAI, like OpenAI, folds reasoning tokens into output_tokens.
                "_thinking_billed_separately": False,
            }
        s_units = self._count_search_units(response)
        if s_units:
            usage["search_units"] = s_units
        return usage

    @staticmethod
    def _is_truncated(response) -> tuple[bool, Optional[str]]:
        """Check if a Responses API response was truncated."""
        status = getattr(response, "status", "completed")
        if status == "incomplete":
            details = getattr(response, "incomplete_details", None)
            reason = getattr(details, "reason", "unknown") if details else "unknown"
            return True, reason
        return False, status

    def _map_error(self, e: Exception) -> AIProviderError:
        """Map an SDK exception to the Djinnite exception hierarchy."""
        error_message = str(e).lower()
        error_code = getattr(e, "code", None) or ""
        if str(error_code) == "context_length_exceeded" or "context_length_exceeded" in error_message \
                or "maximum context" in error_message:
            return AIContextLengthError(
                f"Input context too long for model '{self.model}': {e}",
                provider=self.PROVIDER_NAME,
                original_error=e,
            )
        if "api_key" in error_message or "auth" in error_message or "invalid key" in error_message:
            return AIAuthenticationError(
                "Invalid API key or authentication failed",
                provider=self.PROVIDER_NAME,
                original_error=e,
            )
        if "rate" in error_message or "quota" in error_message:
            return AIRateLimitError(
                "Rate limit exceeded",
                provider=self.PROVIDER_NAME,
                original_error=e,
            )
        if "model" in error_message and "not found" in error_message:
            return AIModelNotFoundError(
                f"Model '{self.model}' not found",
                provider=self.PROVIDER_NAME,
                original_error=e,
            )
        return AIProviderError(
            f"Generation failed: {e}",
            provider=self.PROVIDER_NAME,
            original_error=e,
        )

    # ------------------------------------------------------------------
    # generate()
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: Union[str, List[Dict]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: Optional[int] = None,
        web_search: bool = False,
        thinking: Union[bool, int, str, None] = None,
    ) -> AIResponse:
        """Generate a response using xAI's Responses API."""
        _orig_caller = {
            "thinking": thinking,
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
            "web_search": web_search,
            "system_prompt": system_prompt,
        }
        thinking = self._resolve_thinking(thinking)
        thinking_active = thinking is not None and thinking is not False

        try:
            parts = self._normalize_input(prompt)
            self._validate_vision_limits(parts)
            content = self._map_parts(parts)

            self._validate_incompatible_combinations({
                "temperature":     "any" if temperature is not None else "default",
                "thinking":        "on"  if thinking_active        else "off",
                "structured_json": "off",
                "web_search":      "on"  if web_search             else "off",
            })

            kwargs = {
                "model": self.model,
                "input": self._build_input(content),
            }

            if system_prompt:
                kwargs["instructions"] = system_prompt

            effective_temp = self._resolve_temperature(temperature, thinking_active)
            if thinking_active:
                effective_temp = None
            if effective_temp is not None:
                kwargs["temperature"] = effective_temp

            resolved_max = self._resolve_max_output_tokens(max_output_tokens)
            if resolved_max:
                kwargs["max_output_tokens"] = resolved_max

            reasoning = self._build_grok_thinking(thinking)
            if reasoning is not None:
                kwargs["reasoning"] = reasoning

            if web_search:
                kwargs["tools"] = self._web_search_tools()

            self._debug_dump_request(
                method="generate", caller_args=_orig_caller, native_config=kwargs,
            )

            response = self._client.responses.create(**kwargs)

            content_text, output_parts = self._extract_text_from_output(response.output)
            if not content_text and hasattr(response, "output_text"):
                content_text = response.output_text or ""
                if content_text:
                    output_parts = [{"type": "text", "text": content_text}]

            usage = self._extract_usage(response)
            self._compute_costs(usage)
            is_truncated, finish_reason = self._is_truncated(response)

            ai_response = AIResponse(
                content=content_text,
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

        except (AIOutputTruncatedError, AIContextLengthError, AIPricingError):
            raise
        except Exception as e:
            raise self._map_error(e)

    # ------------------------------------------------------------------
    # Schema normalization for strict mode
    # ------------------------------------------------------------------

    def _prepare_schema_for_provider(self, schema: Dict) -> Dict:
        """
        Prepare a schema for the Responses API ``text.format`` strict mode.

        1. Deep-copies to avoid mutating the caller's dict.
        2. Adds ``additionalProperties: false`` to every object and fills
           ``required`` arrays (strict mode requires both).
        3. Wraps a top-level ``array`` in an object envelope (strict mode
           requires a top-level object); unwrapped transparently after the call.
        """
        schema = copy.deepcopy(schema)

        self._grok_array_wrapped = False
        if schema.get("type") == "array":
            schema = {
                "type": "object",
                "properties": {"items": schema},
                "required": ["items"],
            }
            self._grok_array_wrapped = True

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
        temperature: float = 0.3,
        max_output_tokens: Optional[int] = None,
        web_search: bool = False,
        force: bool = False,
        thinking: Union[bool, int, str, None] = None,
    ) -> AIResponse:
        """
        Generate structured JSON using xAI's Responses API with schema-enforced
        output (``text.format.json_schema``).
        """
        _orig_caller = {
            "thinking": thinking,
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
            "web_search": web_search,
            "system_prompt": system_prompt,
            "force": force,
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

        thinking = self._resolve_thinking(thinking)
        thinking_active = thinking is not None and thinking is not False

        try:
            parts = self._normalize_input(prompt)
            self._validate_vision_limits(parts)
            content = self._map_parts(parts)

            self._validate_incompatible_combinations({
                "temperature":      "any" if temperature is not None else "default",
                "thinking":         "on"  if thinking_active        else "off",
                "structured_json":  "on",
                "web_search":       "on"  if web_search             else "off",
                "json_with_search": "on"  if web_search             else "off",
            })

            kwargs = {
                "model": self.model,
                "input": self._build_input(content),
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

            effective_temp = self._resolve_temperature(temperature, thinking_active)
            if thinking_active:
                effective_temp = None
            if effective_temp is not None:
                kwargs["temperature"] = effective_temp

            resolved_max = self._resolve_max_output_tokens(max_output_tokens)
            if resolved_max:
                kwargs["max_output_tokens"] = resolved_max

            reasoning = self._build_grok_thinking(thinking)
            if reasoning is not None:
                kwargs["reasoning"] = reasoning

            if web_search:
                kwargs["tools"] = self._web_search_tools()

            self._debug_dump_request(
                method="generate_json", caller_args=_orig_caller, native_config=kwargs,
            )

            response = self._client.responses.create(**kwargs)

            content_text, output_parts = self._extract_text_from_output(response.output)
            if not content_text and hasattr(response, "output_text"):
                content_text = response.output_text or ""
                if content_text:
                    output_parts = [{"type": "text", "text": content_text}]

            # Transparently unwrap array envelope if we wrapped it
            if getattr(self, "_grok_array_wrapped", False) and content_text:
                try:
                    parsed = _json.loads(content_text)
                    content_text = _json.dumps(parsed["items"])
                except (KeyError, _json.JSONDecodeError):
                    pass

            usage = self._extract_usage(response)
            self._compute_costs(usage)
            is_truncated, finish_reason = self._is_truncated(response)

            ai_response = AIResponse(
                content=content_text,
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
    # Availability & model listing
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Check if the xAI API is reachable with this key."""
        if not self.api_key:
            return False
        try:
            self._client.models.list()
            return True
        except Exception:
            return False

    # Substrings marking non-text / specialized models to skip in list_models.
    _SPECIALIZED_MARKERS = ("image", "imagine", "vision-beta", "tts", "voice", "embed", "whisper")

    def list_models(self) -> list[dict]:
        """List available ``grok-*`` language models from xAI."""
        if not self.api_key:
            return []
        try:
            models = self._client.models.list()
            models_list = []
            for model in models.data:
                model_id = model.id
                low = model_id.lower()
                if not low.startswith("grok"):
                    continue
                if any(m in low for m in self._SPECIALIZED_MARKERS):
                    continue
                # Rough default context; real limits are resolved by the
                # catalog updater (API metadata / AI estimation / probes).
                context = 131072
                modalities = ["text"]
                if "vision" in low or "grok-4" in low or "grok-2-vision" in low:
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
            print(f"Error listing xAI Grok models: {e}")
            return []

    # ------------------------------------------------------------------
    # Probes (drive live capability discovery in scripts/update_models.py)
    # ------------------------------------------------------------------

    def probe_temperature(self) -> Optional[bool]:
        """Probe whether this Grok model accepts temperature."""
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
            return False

    def _build_combination_probe_request(self, active_states: Dict[str, str]) -> dict:
        """Map activated states to Responses API kwargs."""
        kwargs: dict = {
            "model": self.model,
            "input": "Say hi.",
            "max_output_tokens": 2048,
        }
        if active_states.get("temperature") == "any":
            kwargs["temperature"] = 0.5
        if active_states.get("thinking") == "on":
            kwargs["reasoning"] = {"effort": "low"}
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
            kwargs["tools"] = self._web_search_tools()
        return kwargs

    def _run_combination_probe(self, kwargs: dict) -> None:
        """Send the probe via xAI's responses.create."""
        self._client.responses.create(**kwargs)

    def probe_thinking(self) -> Optional[bool]:
        """Probe whether this Grok model supports reasoning."""
        styles = self.probe_thinking_style()
        if styles is None:
            return None
        return bool(styles)

    def probe_thinking_disable(self) -> Optional[bool]:
        """
        Whether xAI accepts an explicit thinking-disabled request
        (``reasoning={"effort": "none"}``).

        Returns:
            True  – ``effort: "none"`` accepted (toggleable model).
            False – vendor rejects it (always-on reasoning model).
            None  – inconclusive (rate limit / timeout).
        """
        try:
            self._client.responses.create(
                model=self.model,
                input="Say hi.",
                reasoning={"effort": "none"},
                max_output_tokens=20,
            )
            return True
        except Exception as e:
            err = str(e).lower()
            status = getattr(e, "status_code", None)
            if status == 429 or "rate" in err or "timeout" in err:
                return None
            return False

    def probe_thinking_style(self) -> Optional[list[str]]:
        """
        Probe which thinking styles this Grok model supports.

        xAI exposes ``reasoning_effort`` (no token-budget analogue), so the
        result is ``["effort"]`` or ``[]``.
        """
        try:
            self._client.responses.create(
                model=self.model,
                input="Say hi.",
                reasoning={"effort": "low"},
                max_output_tokens=100,
            )
            return ["effort"]
        except Exception as e:
            err = str(e).lower()
            status = getattr(e, "status_code", None)
            if status == 429 or "rate" in err or "timeout" in err:
                return None
            return []

    def probe_structured_json(self) -> Optional[bool]:
        """Probe whether this Grok model supports structured JSON output."""
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

    def probe_web_search(self) -> Optional[bool]:
        """Probe whether this Grok model accepts the Live Search tool."""
        try:
            self._client.responses.create(
                model=self.model,
                input="Say hi.",
                tools=self._web_search_tools(),
                max_output_tokens=50,
            )
            return True
        except Exception as e:
            err = str(e).lower()
            status = getattr(e, "status_code", None)
            if status == 429 or "rate" in err or "quota" in err or "timeout" in err:
                return None
            return False

    def probe_json_with_search(self) -> Optional[bool]:
        """Probe whether this Grok model supports structured JSON + Live Search combined."""
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
                tools=self._web_search_tools(),
                max_output_tokens=50,
            )
            return True
        except Exception as e:
            err = str(e).lower()
            status = getattr(e, "status_code", None)
            if status == 429 or "rate" in err or "quota" in err or "timeout" in err:
                return None
            return False

    def discover_modalities(self, model_id: str) -> Dict[str, List[str]]:
        """Discover modalities for Grok models."""
        input_modalities = ["text"]
        output_modalities = ["text"]

        low_id = model_id.lower()
        if "vision" in low_id or "grok-4" in low_id or "grok-2-vision" in low_id:
            input_modalities.append("vision")

        return {"input": input_modalities, "output": output_modalities}
