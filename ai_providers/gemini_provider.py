"""
Google Gemini AI Provider

Wraps the Google Gen AI SDK (google-genai) for Gemini models.
"""

import copy

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
from .platforms import get_platform, resolve_platform


# google-genai >=2.18 logs a warning on every direct ``models.generate_content``
# call that leaves automatic function calling (AFC) enabled, steering callers
# toward ``Chat.send_message``. AFC is a client-side loop that auto-executes
# Python callables across multiple round trips; Djinnite is a single-shot
# wrapper and registers no callable tools (``google_search`` is a server-side
# tool), so AFC is inapplicable. Disabling it takes the SDK's early-return path
# -- skipping machinery we never use, and the warning with it.
_DISABLE_AFC = {"disable": True}

# Candidate finish reasons that mean the model finished on its own (or hit
# the cap). Anything else (SAFETY, RECITATION, PROHIBITED_CONTENT, BLOCKLIST,
# SPII, ...) means the provider filtered the output.
_MODEL_FINISHES = ("STOP", "MAX_TOKENS", "FINISH_REASON_UNSPECIFIED")


def _enum_name(value) -> Optional[str]:
    """Bare name of an SDK enum (``FinishReason.SAFETY`` -> ``"SAFETY"``) or str."""
    if value is None:
        return None
    name = getattr(value, "name", None)
    return name if isinstance(name, str) else str(value).split(".")[-1]


class GeminiProvider(BaseAIProvider):
    """
    Google Gemini AI provider implementation.

    Access modes (see ``platforms.py``):

    * direct (default, ``backend="gemini"``): Google AI Studio with an API key.
    * ``platform="vertexai"`` (or the legacy ``backend="vertexai"``): Google
      Vertex AI. With no ``api_key`` the client authenticates with
      Application Default Credentials; no key is required or sent.

    Uses the google-genai SDK.
    """

    PROVIDER_NAME = "gemini"

    # Class-level defaults: direct mode (also read by tests' _bare() helper).
    backend: str = "gemini"
    project_id: Optional[str] = None
    quota_project: Optional[str] = None

    def __init__(
        self,
        api_key: Optional[str],
        model: str,
        backend: str = "gemini",
        project_id: Optional[str] = None,
        model_info=None,
        require_pricing: bool = True,
        *,
        platform: Optional[str] = None,
        location: Optional[str] = None,
        quota_project: Optional[str] = None,
    ):
        """
        Initialize the Gemini provider.

        Args:
            api_key: The Google API key. Optional on Vertex AI: without one
                the client uses Application Default Credentials.
            model: The model ID to use
            backend: Legacy switch -- 'gemini' (AI Studio) or 'vertexai'
                (same as ``platform="vertexai"``).
            project_id: The Google Cloud project ID (required for Vertex AI)
            require_pricing: Fast-fail on missing/unknown price (see base).
            platform: ``"vertexai"`` for Google Vertex AI; ``None`` = direct.
            location: Vertex location (default ``"us-central1"``, unchanged
                from before platform mode). ``"global"`` serves the newest
                models, e.g. gemini-3.5-flash.
            quota_project: Project billed for quota (for user ADC created
                with ``--disable-quota-project``). Applied to the ADC
                credentials, because google-genai overwrites an
                ``x-goog-user-project`` header with the credentials' own
                quota project whenever they carry one. ADC only: rejected
                together with ``api_key``.
        """
        self.platform = resolve_platform(platform, backend, provider=self.PROVIDER_NAME)
        self.backend = "vertexai" if self.platform == "vertexai" else backend
        self.project_id = project_id
        self.quota_project = quota_project
        self.location = location.lower() if location else None
        if self.platform:
            spec = get_platform(self.platform, self.PROVIDER_NAME)
            self.location = self.location or spec.default_location.get(self.PROVIDER_NAME)
            self._price_multiplier = spec.price_multiplier(self.PROVIDER_NAME, self.location)
        super().__init__(api_key, model, model_info=model_info, require_pricing=require_pricing)

    def _initialize_client(self) -> None:
        """Initialize the Gemini client for this access mode."""
        try:
            from google import genai

            if self.platform == "vertexai":
                if not self.project_id:
                    raise AIProviderError(
                        "project_id is required for Vertex AI backend",
                        provider=self.PROVIDER_NAME
                    )
                client_kwargs = {}
                if self.api_key:
                    if self.quota_project:
                        raise AIProviderError(
                            "quota_project applies to Application Default "
                            "Credentials; omit api_key to use it",
                            provider=self.PROVIDER_NAME,
                        )
                    # Keyed Vertex: exactly the client arguments used before
                    # platform mode.
                    client_kwargs["api_key"] = self.api_key
                elif self.quota_project:
                    import google.auth
                    credentials, _ = google.auth.default(
                        scopes=["https://www.googleapis.com/auth/cloud-platform"],
                        quota_project_id=self.quota_project,
                    )
                    client_kwargs["credentials"] = credentials
                # No api_key and no credentials: google-genai resolves ADC
                # lazily, at the first request.
                self._client = genai.Client(
                    vertexai=True,
                    project=self.project_id,
                    location=self.location,
                    **client_kwargs,
                )
            elif self.platform:
                raise AIProviderError(
                    f"platform '{self.platform}' is not implemented for Gemini",
                    provider=self.PROVIDER_NAME,
                )
            else:
                # Default to Google AI Studio
                self._client = genai.Client(api_key=self.api_key)

        except ImportError:
            raise AIProviderError(
                "google-genai package not installed. "
                "Install with: pip install google-genai",
                provider=self.PROVIDER_NAME
            )
        except AIProviderError:
            raise
        except Exception as e:
            if type(e).__name__ in ("DefaultCredentialsError", "RefreshError"):
                raise AIAuthenticationError(
                    f"Google Application Default Credentials unavailable: {e}",
                    provider=self.PROVIDER_NAME,
                    original_error=e,
                )
            raise AIProviderError(
                f"Failed to initialize Gemini client: {e}",
                provider=self.PROVIDER_NAME,
                original_error=e
            )

    def _map_platform_error(self, e: Exception) -> Optional[AIProviderError]:
        """Map a Vertex AI failure to a Djinnite error, or ``None`` to fall through.

        Platform mode only; the AI Studio mapping is unchanged. Keyed on the
        google-genai ``APIError`` ``code`` / ``status`` rather than message
        text, so a 403 that mentions "quota project" is not misread as a
        rate limit.
        """
        code = getattr(e, "code", None)
        status = str(getattr(e, "status", None) or "")
        text = getattr(e, "message", None) or str(e)
        where = f"project '{self.project_id}', location '{self.location}'"
        type_name = type(e).__name__
        if code == 429 or status == "RESOURCE_EXHAUSTED":
            return AIRateLimitError(
                f"Vertex AI quota or rate limit for '{self.model}' ({where}): {status} {text}".strip(),
                provider=self.PROVIDER_NAME, original_error=e,
            )
        if code in (401, 403) or status in ("PERMISSION_DENIED", "UNAUTHENTICATED"):
            return AIAuthenticationError(
                f"Vertex AI refused the request ({where}): {status} {text}".strip(),
                provider=self.PROVIDER_NAME, original_error=e,
            )
        if code == 404 or status == "NOT_FOUND":
            return AIModelNotFoundError(
                f"Model '{self.model}' not found on Vertex AI ({where}) -- not "
                f"served at this location, or not enabled for the project: {text}",
                provider=self.PROVIDER_NAME, original_error=e,
            )
        if type_name in ("DefaultCredentialsError", "RefreshError"):
            return AIAuthenticationError(
                f"Google Application Default Credentials unavailable ({type_name}): {e}. "
                f"Run 'gcloud auth application-default login' or attach a service account.",
                provider=self.PROVIDER_NAME, original_error=e,
            )
        return None

    def _gemini_contents(self, turns, gemini_parts):
        """Native ``contents``: today's flat parts list, or role-tagged turns.

        Without history the request is unchanged (a flat list of Parts).
        With history every turn becomes ``types.Content(role=...)`` --
        ``"user"`` or ``"model"`` -- and the prompt is the final user turn.
        """
        if not turns:
            return gemini_parts
        from google.genai import types
        contents = [
            types.Content(
                role="model" if role == "assistant" else "user",
                parts=self._map_parts(parts),
            )
            for role, parts in turns
        ]
        contents.append(types.Content(role="user", parts=gemini_parts))
        return contents
    
    def _map_parts(self, parts: List[Dict]) -> List:
        """Map internal parts to Gemini SDK parts."""
        from google.genai import types
        gemini_parts = []
        
        for part in parts:
            if part["type"] == "text":
                gemini_parts.append(types.Part.from_text(text=part["text"]))
            elif part["type"] == "image":
                if "image_data" in part:
                    gemini_parts.append(types.Part.from_bytes(
                        data=part["image_data"],
                        mime_type=part.get("mime_type", "image/jpeg")
                    ))
                elif "file_uri" in part:
                    gemini_parts.append(types.Part.from_uri(
                        file_uri=part["file_uri"],
                        mime_type=part.get("mime_type", "image/jpeg")
                    ))
            elif part["type"] == "audio":
                if "audio_data" in part:
                    gemini_parts.append(types.Part.from_bytes(
                        data=part["audio_data"],
                        mime_type=part.get("mime_type", "audio/mp3")
                    ))
                elif "file_uri" in part:
                    gemini_parts.append(types.Part.from_uri(
                        file_uri=part["file_uri"],
                        mime_type=part.get("mime_type", "audio/mp3")
                    ))
            elif part["type"] == "video":
                 if "file_uri" in part:
                    gemini_parts.append(types.Part.from_uri(
                        file_uri=part["file_uri"],
                        mime_type=part.get("mime_type", "video/mp4")
                    ))
            # Add other types as needed
            
        return gemini_parts

    @staticmethod
    def _usage_from_metadata(metadata) -> dict:
        """Token usage from a response's ``usage_metadata``.

        Gemini reports thinking separately: ``candidates_token_count``
        excludes ``thoughts_token_count``, and Google bills thoughts at the
        output rate. ``_thinking_billed_separately`` is therefore True, so
        ``token_cost`` counts them (it used to be False, under-reporting the
        cost of every thinking call). ``total_token_count`` already
        includes thoughts.
        """
        input_t = getattr(metadata, 'prompt_token_count', 0) or 0
        output_t = getattr(metadata, 'candidates_token_count', 0) or 0
        total_t = getattr(metadata, 'total_token_count', None)
        thinking_t = getattr(metadata, 'thoughts_token_count', None)
        return {
            "input_tokens": input_t,
            "output_tokens": output_t,
            "total_tokens": total_t if total_t is not None else input_t + output_t,
            "thinking_tokens": thinking_t,  # None if not reported
            "_thinking_billed_separately": True,
        }

    @staticmethod
    def _count_search_units(response) -> int:
        """Count billable web search queries from a Gemini response.

        Google bills per individual search query in
        ``candidates[0].grounding_metadata.web_search_queries``.
        """
        try:
            candidate = response.candidates[0] if response.candidates else None
            if candidate is None:
                return 0
            metadata = getattr(candidate, 'grounding_metadata', None)
            if metadata is None:
                return 0
            queries = getattr(metadata, 'web_search_queries', None)
            if queries:
                return len(queries)
        except (IndexError, AttributeError):
            pass
        return 0

    @staticmethod
    def _response_diagnostics(response) -> dict:
        """Collect why a Gemini response may carry no usable content.

        Reads ``prompt_feedback`` (prompt-level block) and the first
        candidate's finish reason, finish message, and safety ratings.
        Every read is defensive: any field may be absent.
        """
        def _ratings(items) -> Optional[str]:
            out = []
            for r in items or []:
                s = f"{_enum_name(getattr(r, 'category', None))}={_enum_name(getattr(r, 'probability', None))}"
                if getattr(r, "blocked", None):
                    s += "[blocked]"
                out.append(s)
            return ", ".join(out) or None

        feedback = getattr(response, "prompt_feedback", None)
        candidates = getattr(response, "candidates", None) or []
        details = {
            "block_reason": _enum_name(getattr(feedback, "block_reason", None)),
            "block_reason_message": getattr(feedback, "block_reason_message", None),
            "prompt_safety_ratings": _ratings(getattr(feedback, "safety_ratings", None)),
            "candidates": len(candidates),
        }
        if candidates:
            cand = candidates[0]
            details["finish_reason"] = _enum_name(getattr(cand, "finish_reason", None))
            details["finish_message"] = getattr(cand, "finish_message", None)
            details["safety_ratings"] = _ratings(getattr(cand, "safety_ratings", None))
        return details

    @staticmethod
    def _provider_block_reason(details: dict) -> Optional[str]:
        """Prompt block reason, or the candidate's filter finish reason, else None."""
        if details.get("block_reason"):
            return details["block_reason"]
        finish = details.get("finish_reason")
        if finish is not None and finish not in _MODEL_FINISHES:
            return finish
        return None

    def _check_empty(
        self,
        ai_response: AIResponse,
        details: dict,
        *,
        json_mode: bool,
        web_search: bool,
        thinking,
    ) -> None:
        """Apply the empty/blocked contract (see AIEmptyResponseError).

        * Provider blocked/filtered (even with partial text, which is
          incomplete like truncated output), or no candidates at all ->
          raise in both methods.
        * ``generate_json``: also raise on empty text.
        * ``generate``: an honest empty STOP is returned.
        """
        block = ai_response.block_reason
        no_output = not ai_response.content and not ai_response.parts
        if json_mode:
            should_raise = bool(block) or not ai_response.content.strip()
        else:
            should_raise = bool(block) or (no_output and details.get("candidates", 0) == 0)
        if should_raise:
            self._raise_empty(
                ai_response,
                reason=block or "empty",
                details=details,
                web_search=web_search,
                thinking=thinking,
            )

    def _build_gemini_thinking(
        self,
        thinking: Union[bool, int, str, None],
    ) -> Optional[dict]:
        """
        Translate the unified ``thinking`` parameter into Gemini's native
        ``thinking_config`` format.

        Gemini's ``ThinkingConfig`` exposes two alternative fields:
        ``thinking_budget`` (int token count, with ``0`` = disabled and
        ``-1`` = automatic) and ``thinking_level`` (the ``ThinkingLevel``
        enum: MINIMAL/LOW/MEDIUM/HIGH). Exactly one is set per request —
        which one depends on the caller's shape:

        * ``int`` / ``True`` / ``False`` → ``thinking_budget``
        * ``str`` (effort level) → ``thinking_level``

        Returns:
            A dict for ``thinking_config``, or ``None`` if not requested.
        """
        from google.genai.types import ThinkingLevel

        if thinking is None:
            return None  # No opinion — let model use its default

        if thinking is False:
            # Explicitly disable thinking. Gemini requires thinking_budget=0
            # to suppress thinking on models that default to thinking ON.
            return {"thinking_budget": 0}

        if thinking is True:
            # ``True`` is "let the model decide" — Gemini's dynamic-budget
            # signal is ``thinking_budget=-1`` (the model chooses how much
            # to think based on the prompt). Aligns with Claude's adaptive
            # mode and OpenAI's high-effort default for the same intent.
            return {"thinking_budget": -1}

        if isinstance(thinking, int):
            # bool is a subclass of int, but True/False were handled above.
            return {"thinking_budget": thinking}

        # str effort: dispatch to Gemini's native ThinkingLevel enum.
        # _resolve_thinking has already lowercased the value and validated
        # it against {"minimal","low","medium","high"} and the model's
        # thinking_style list.
        return {"thinking_level": ThinkingLevel[thinking.upper()]}

    def generate(
        self,
        prompt: Union[str, List[Dict]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: Optional[int] = None,
        web_search: bool = False,
        thinking: Union[bool, int, str, None] = None,
        *,
        history: Optional[List[Dict]] = None,
    ) -> AIResponse:
        """
        Generate a response using Gemini.

        ``history`` (earlier turns) becomes role-tagged ``contents``
        (``user`` / ``model``); ``prompt`` is the final user turn.
        """
        _orig_caller = {
            "thinking": thinking,
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
            "web_search": web_search,
            "system_prompt": system_prompt,
            "history_turns": len(history or []),
        }
        # Validate & normalize thinking
        thinking = self._resolve_thinking(thinking)
        turns = self._normalize_history(history)

        try:
            from google.genai import types

            parts = self._normalize_input(prompt)
            self._validate_vision_limits(self._all_parts(turns, parts))
            gemini_parts = self._map_parts(parts)

            # Cross-capability pre-flight from catalog.
            self._validate_incompatible_combinations({
                "temperature":     "any" if temperature is not None else "default",
                "thinking":        "on"  if thinking is not None    else "off",
                "structured_json": "off",
                "web_search":      "on"  if web_search              else "off",
            })

            # Resolve temperature: strip if catalog says not supported
            effective_temp = self._resolve_temperature(temperature, thinking is not None)

            # Auto-fill max_output_tokens from catalog if caller didn't provide one.
            max_output_tokens = self._resolve_max_output_tokens(max_output_tokens)

            # Build configuration
            config = {"automatic_function_calling": _DISABLE_AFC}
            if effective_temp is not None:
                config["temperature"] = effective_temp
            if max_output_tokens:
                config["max_output_tokens"] = max_output_tokens
            if system_prompt:
                config["system_instruction"] = system_prompt

            # Thinking: add thinking_config if requested
            thinking_config = self._build_gemini_thinking(thinking)
            if thinking_config is not None:
                config["thinking_config"] = thinking_config

            # Enable Google Search grounding for current information
            if web_search:
                config["tools"] = [types.Tool(google_search=types.GoogleSearch())]

            self._debug_dump_request(
                method="generate", caller_args=_orig_caller, native_config=config,
            )

            # Generate response
            response = self._client.models.generate_content(
                model=self.model,
                contents=self._gemini_contents(turns, gemini_parts),
                config=config
            )

            # Extract usage info if available
            usage = {}
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = self._usage_from_metadata(response.usage_metadata)

            # Count billable search events
            s_units = self._count_search_units(response)
            if s_units:
                usage["search_units"] = s_units
            self._compute_costs(usage)

            # Extract parts, text, and finish reason from the candidate
            output_parts = []
            text_content = ""
            finish_reason = None
            if response.candidates:
                candidate = response.candidates[0]
                # Gemini returns finish_reason as an enum or string.
                # The value "MAX_TOKENS" indicates output truncation.
                raw_finish = getattr(candidate, 'finish_reason', None)
                # Normalize: the SDK may return an enum (e.g. FinishReason.MAX_TOKENS)
                # or a string. Convert to string for consistent comparison.
                finish_reason = str(raw_finish) if raw_finish is not None else None
                
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            text_content += part.text
                            output_parts.append({"type": "text", "text": part.text})
                        elif hasattr(part, 'inline_data') and part.inline_data:
                            output_parts.append({
                                "type": "inline_data",
                                "mime_type": part.inline_data.mime_type,
                                "data": part.inline_data.data
                            })

            if not text_content:
                # SDK convenience accessor; may be None or raise when there
                # are no candidates. Never let it put None into content.
                try:
                    text_content = response.text or ""
                except Exception:
                    text_content = ""

            details = self._response_diagnostics(response)
            block_reason = self._provider_block_reason(details)
            if finish_reason is None and block_reason:
                finish_reason = block_reason

            # Detect output truncation: Gemini returns finishReason=MAX_TOKENS
            # when the output was cut short due to maxOutputTokens.
            # This is an HTTP 200 response — the SDK does NOT raise an exception.
            is_truncated = (finish_reason is not None and "MAX_TOKENS" in finish_reason.upper())

            ai_response = AIResponse(
                content=text_content,
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
                    f"(finishReason='{finish_reason}', output_tokens={usage.get('output_tokens', '?')})",
                    provider=self.PROVIDER_NAME,
                    partial_response=ai_response,
                )

            self._check_empty(ai_response, details, json_mode=False,
                              web_search=web_search, thinking=thinking)

            return ai_response

        except AIProviderError:
            raise  # Never swallow our own semantic errors (incl. fast-fail pricing)
        except Exception as e:
            if self.platform:
                mapped = self._map_platform_error(e)
                if mapped is not None:
                    raise mapped
            error_message = str(e).lower()

            # Detect context length exceeded: Gemini SDK raises exceptions
            # with HTTP 400 INVALID_ARGUMENT when input exceeds context window.
            if "invalid_argument" in error_message and \
               ("token" in error_message or "context" in error_message or "too long" in error_message):
                raise AIContextLengthError(
                    f"Input context too long for model '{self.model}': {e}",
                    provider=self.PROVIDER_NAME,
                    original_error=e
                )
            # Check for specific error types
            elif "api_key" in error_message or "authentication" in error_message:
                raise AIAuthenticationError(
                    "Invalid API key or authentication failed",
                    provider=self.PROVIDER_NAME,
                    original_error=e
                )
            elif "rate" in error_message or "quota" in error_message:
                raise AIRateLimitError(
                    "Rate limit or quota exceeded",
                    provider=self.PROVIDER_NAME,
                    original_error=e
                )
            elif "model" in error_message and "not found" in error_message:
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
    # Schema normalization for Gemini
    # ------------------------------------------------------------------

    def _prepare_schema_for_provider(self, schema: Dict) -> Dict:
        """
        Gemini-specific schema transformation.

        Gemini **rejects** ``additionalProperties`` entirely (HTTP 400).
        This method defensively strips it from the entire schema tree.
        (The caller-contract validation should have already ensured it's
        absent, but defense-in-depth is prudent.)
        """
        return self._strip_additional_properties(schema)

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
        *,
        history: Optional[List[Dict]] = None,
    ) -> AIResponse:
        """
        Generates structured JSON using Gemini's **Constraint Decoding** (``response_schema``).

        [AGENT NOTE]: Uses ``response_mime_type="application/json"`` combined with
        ``response_schema`` for Guaranteed Structure.  The output is constrained at the
        decoding level to conform to the supplied schema.

        Args:
            prompt: The user prompt (str or list of multimodal parts).
            schema: **Required.** A Pydantic BaseModel class or JSON Schema dict.
            system_prompt: Optional system instruction.
            temperature: Sampling temperature (default 0.3).
            max_output_tokens: Cap on output tokens (auto-fills from catalog).
            web_search: If True, enable Google Search grounding for current info.
            thinking: Optional thinking/reasoning control (same as generate()).
            history: Optional earlier turns (same as generate()). The schema
                constrains only the new turn.

        Returns:
            AIResponse whose ``content`` is schema-conforming JSON.
        """
        _orig_caller = {
            "thinking": thinking,
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
            if web_search:
                self._check_capability("json_with_search")
        json_schema = self._normalize_schema(schema)
        json_schema = self._validate_caller_schema(json_schema)
        json_schema = self._prepare_schema_for_provider(json_schema)

        # Validate & normalize thinking
        thinking = self._resolve_thinking(thinking)
        turns = self._normalize_history(history)

        try:
            from google.genai import types

            parts = self._normalize_input(prompt)
            self._validate_vision_limits(self._all_parts(turns, parts))
            gemini_parts = self._map_parts(parts)

            # Cross-capability pre-flight from catalog.
            self._validate_incompatible_combinations({
                "temperature":     "any" if temperature is not None else "default",
                "thinking":        "on"  if thinking is not None    else "off",
                "structured_json": "on",
                "web_search":      "on"  if web_search              else "off",
                "json_with_search": "on" if web_search              else "off",
            })

            # Resolve temperature: catalog-aware stripping
            effective_temp = self._resolve_temperature(temperature, thinking is not None)

            # Build configuration with schema-enforced JSON output
            config = {
                "response_mime_type": "application/json",
                "response_schema": json_schema,
                "automatic_function_calling": _DISABLE_AFC,
            }

            # Auto-fill max_output_tokens from catalog if caller didn't provide one.
            max_output_tokens = self._resolve_max_output_tokens(max_output_tokens)

            if effective_temp is not None:
                config["temperature"] = effective_temp

            if max_output_tokens:
                config["max_output_tokens"] = max_output_tokens
            if system_prompt:
                config["system_instruction"] = system_prompt

            # Thinking config
            thinking_config = self._build_gemini_thinking(thinking)
            if thinking_config is not None:
                config["thinking_config"] = thinking_config
            
            # Enable Google Search grounding for current information
            if web_search:
                config["tools"] = [types.Tool(google_search=types.GoogleSearch())]

            self._debug_dump_request(
                method="generate_json", caller_args=_orig_caller, native_config=config,
            )

            # Generate response
            response = self._client.models.generate_content(
                model=self.model,
                contents=self._gemini_contents(turns, gemini_parts),
                config=config
            )

            # Extract text, parts, and finish reason from candidates
            text_content = ""
            output_parts = []
            finish_reason = None
            if response.candidates:
                candidate = response.candidates[0]
                raw_finish = getattr(candidate, 'finish_reason', None)
                finish_reason = str(raw_finish) if raw_finish is not None else None
                
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            text_content += part.text
                            output_parts.append({"type": "text", "text": part.text})
            
            # Fall back to response.text if candidates extraction failed.
            # It may be None or raise when there are no candidates.
            if not text_content:
                try:
                    text_content = response.text or ""
                except Exception:
                    text_content = ""

            details = self._response_diagnostics(response)
            block_reason = self._provider_block_reason(details)
            if finish_reason is None and block_reason:
                finish_reason = block_reason

            # Extract usage info if available
            usage = {}
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = self._usage_from_metadata(response.usage_metadata)

            # Count billable search events
            s_units = self._count_search_units(response)
            if s_units:
                usage["search_units"] = s_units
            self._compute_costs(usage)

            # Detect output truncation — same check as generate()
            is_truncated = (finish_reason is not None and "MAX_TOKENS" in finish_reason.upper())
            
            ai_response = AIResponse(
                content=text_content,
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
                    f"(finishReason='{finish_reason}', output_tokens={usage.get('output_tokens', '?')})",
                    provider=self.PROVIDER_NAME,
                    partial_response=ai_response,
                )

            self._check_empty(ai_response, details, json_mode=True,
                              web_search=web_search, thinking=thinking)

            return ai_response
            
        except AIProviderError:
            raise  # Re-raise all our own errors (including truncation/context)
        except Exception as e:
            if self.platform:
                mapped = self._map_platform_error(e)
                if mapped is not None:
                    raise mapped
            error_message = str(e).lower()

            if "invalid_argument" in error_message and \
               ("token" in error_message or "context" in error_message or "too long" in error_message):
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
        """Check if Gemini is available and configured.

        **Makes a live network call** (lists one model). Direct mode needs
        an API key; platform mode authenticates with the platform's
        credentials. Any failure -- missing credentials, IAM refusal, or a
        zero quota -- returns ``False``.
        """
        if not self.platform and not self.api_key:
            return False

        try:
            # Try to list models as a connectivity check
            self._client.models.list(config={"page_size": 1})
            return True
        except Exception:
            return False

    def _availability_call(self) -> None:
        """Token count for this model: unbilled, and exercises the model's own endpoint."""
        self._client.models.count_tokens(model=self.model, contents="test")

    def list_models(self) -> list[dict]:
        """List available models from Gemini.

        Extracts both input_token_limit (context_window) and
        output_token_limit (max_output_tokens) from the API when available.
        Works in platform mode without an API key (Vertex lists its
        publisher models).
        """
        if not self.platform and not self.api_key:
            return []
            
        try:
            models_list = []
            pager = self._client.models.list()
            
            for model in pager:
                name = getattr(model, "name", "")
                if "gemini" not in name.lower():
                    continue
                
                model_id = name.split("/")[-1] if "/" in name else name
                
                # Determine capabilities
                modalities = ["text"]
                if any(x in model_id.lower() for x in ["vision", "flash", "pro"]):
                    modalities.extend(["vision", "audio", "video"])
                
                # Extract output token limit from API (Gemini exposes this)
                max_output = getattr(model, "output_token_limit", 0) or 0
                
                models_list.append({
                    "id": model_id,
                    "name": getattr(model, "display_name", model_id),
                    "context_window": getattr(model, "input_token_limit", 0),
                    "max_output_tokens": max_output,
                    "modalities": modalities,
                    "cost_tier": "standard"
                })
            
            return models_list
        except Exception as e:
            print(f"Error listing Gemini models: {e}")
            return []

    def probe_temperature(self) -> Optional[bool]:
        """Probe whether this Gemini model accepts temperature. (All Gemini text models do.)"""
        try:
            self._client.models.generate_content(
                model=self.model, contents="Say hi.",
                config={"automatic_function_calling": _DISABLE_AFC, "temperature": 0.5, "max_output_tokens": 10},
            )
            return True
        except Exception as e:
            err = str(e).lower()
            if "rate" in err or "quota" in err or "429" in err:
                return None
            return False

    def _build_combination_probe_request(self, active_states: Dict[str, str]) -> dict:
        """Map activated states to Gemini ``generate_content`` kwargs.

        The orchestrator's ``_run_combination_probe`` forwards
        ``contents`` and ``config`` to ``client.models.generate_content``.
        """
        from google.genai import types  # local import: SDK presence checked at init

        config: dict = {"max_output_tokens": 2048,
                        "automatic_function_calling": _DISABLE_AFC}
        if active_states.get("temperature") == "any":
            config["temperature"] = 0.5
        if active_states.get("thinking") == "on":
            config["thinking_config"] = {"thinking_budget": 1024}
        if active_states.get("structured_json") == "on":
            config["response_mime_type"] = "application/json"
            config["response_schema"] = {
                "type": "object",
                "properties": {"v": {"type": "integer"}},
                "required": ["v"],
            }
        if active_states.get("web_search") == "on":
            config["tools"] = [types.Tool(google_search=types.GoogleSearch())]
        return {
            "model": self.model,
            "contents": "Say hi.",
            "config": config,
        }

    def _run_combination_probe(self, kwargs: dict) -> None:
        """Send the probe via Gemini's generate_content."""
        self._client.models.generate_content(**kwargs)

    def probe_thinking(self) -> Optional[bool]:
        """Probe whether this Gemini model supports thinking mode."""
        styles = self.probe_thinking_style()
        if styles is None:
            return None
        return bool(styles)

    def probe_thinking_disable(self) -> Optional[bool]:
        """
        Whether Gemini accepts an explicit thinking-disabled request.

        Gemini's ``thinking_config`` is opt-in; omitting it always works.
        Some Gemini variants (e.g. 2.5 Pro thinking-only) accept
        ``thinking_budget=0`` as the explicit disable. We test that
        path — if the API rejects ``thinking_budget=0`` outright, the
        model is always-on and disable is unsupported.
        """
        try:
            self._client.models.generate_content(
                model=self.model, contents="Say hi.",
                config={
                    "automatic_function_calling": _DISABLE_AFC,
                    "max_output_tokens": 100,
                    "thinking_config": {"thinking_budget": 0},
                },
            )
            return True
        except Exception as e:
            err = str(e).lower()
            if "rate" in err or "quota" in err or "429" in err:
                return None
            # 400 INVALID_ARGUMENT against thinking_budget=0 → can't disable.
            # If thinking_config isn't supported at all we fall back to True
            # (omitting the param is always valid for non-thinking models).
            if "thinking" in err and ("not support" in err or "invalid" in err or "400" in err):
                return False
            return True

    def probe_thinking_style(self) -> Optional[list[str]]:
        """
        Probe which thinking styles this Gemini model supports.

        Gemini exposes two alternative shapes on ``thinking_config``:
        ``thinking_budget`` (int) and ``thinking_level`` (enum). Each is
        probed independently so models that support one but not the other
        are recorded accurately. Newer Gemini models (3.x) accept both;
        older models may only accept ``thinking_budget``.

        **Partial-success policy:** if any tier is inconclusive (rate
        limit, timeout, 5xx, unknown error class) the entire result is
        ``None``. We do NOT commit a partial truth (e.g. ``["budget"]``
        when the effort probe was rate-limited) because the catalog
        consumer would treat that as "confirmed: only budget" and the
        wrong value would stick until someone reprobed. ``None`` lets
        the orchestrator preserve any cached value or leave it null
        until a fresh probe succeeds.

        Returns:
            * Non-empty subset of ``["budget", "effort"]`` — all tiers
              completed and these are the confirmed-supported shapes.
            * ``[]`` – all tiers cleanly rejected → no thinking support.
            * ``None`` – any tier was inconclusive.
        """
        from google.genai.types import ThinkingLevel

        styles: list[str] = []
        inconclusive = False

        tiers = [
            ("budget", {"thinking_budget": 1024}),
            ("effort", {"thinking_level": ThinkingLevel.LOW}),
        ]

        for style_name, thinking_cfg in tiers:
            try:
                self._client.models.generate_content(
                    model=self.model, contents="Say hi.",
                    config={
                        "automatic_function_calling": _DISABLE_AFC,
                        "max_output_tokens": 100,
                        "thinking_config": thinking_cfg,
                    },
                )
                styles.append(style_name)
            except Exception as e:
                if self._classify_probe_error(e) == "inconclusive":
                    inconclusive = True
                # else: confirmed not_supported → leave style_name out

        if inconclusive:
            return None
        return styles

    def probe_structured_json(self) -> Optional[bool]:
        """Probe whether this Gemini model supports response_schema JSON mode."""
        _PROBE_SCHEMA = {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        }
        try:
            self._client.models.generate_content(
                model=self.model,
                contents="Return the number 1.",
                config={
                    "automatic_function_calling": _DISABLE_AFC,
                    "temperature": 0,
                    "max_output_tokens": 50,
                    "response_mime_type": "application/json",
                    "response_schema": _PROBE_SCHEMA,
                },
            )
            return True
        except Exception as e:
            err = str(e).lower()
            if "invalid" in err or "not supported" in err or "response_schema" in err or "400" in err:
                return False
            if "rate" in err or "quota" in err or "429" in err:
                return None
            return None

    def probe_web_search(self) -> Optional[bool]:
        """Probe whether this Gemini model accepts the google_search tool."""
        from google.genai import types
        try:
            self._client.models.generate_content(
                model=self.model,
                contents="Say hi.",
                config={
                    "automatic_function_calling": _DISABLE_AFC,
                    "max_output_tokens": 50,
                    "tools": [types.Tool(google_search=types.GoogleSearch())],
                },
            )
            return True
        except Exception as e:
            err = str(e).lower()
            if "rate" in err or "quota" in err or "429" in err:
                return None
            return False

    def probe_json_with_search(self) -> Optional[bool]:
        """Probe whether this Gemini model supports structured JSON + Google Search combined.

        Gemini 2.x rejects this combination (400 INVALID_ARGUMENT:
        "controlled generation is not supported with Search tool").
        Gemini 3.x supports it natively.
        """
        from google.genai import types

        _PROBE_SCHEMA = {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        }
        try:
            self._client.models.generate_content(
                model=self.model,
                contents="Return the number 1.",
                config={
                    "automatic_function_calling": _DISABLE_AFC,
                    "temperature": 0,
                    "max_output_tokens": 50,
                    "response_mime_type": "application/json",
                    "response_schema": _PROBE_SCHEMA,
                    "tools": [types.Tool(google_search=types.GoogleSearch())],
                },
            )
            return True
        except Exception as e:
            err = str(e).lower()
            if "rate" in err or "quota" in err or "429" in err:
                return None
            # Any 400/invalid/not-supported error → confirmed incompatible
            return False

    def discover_modalities(self, model_id: str) -> Dict[str, List[str]]:
        """Discover modalities for Gemini models."""
        # Most modern Gemini models are natively multimodal for input
        input_modalities = ["text", "vision", "audio", "video"]
        output_modalities = ["text"]
        
        # Suffix/ID based overrides
        low_id = model_id.lower()
        if "tts" in low_id:
            input_modalities = ["text"]
            output_modalities = ["audio"]
        elif "embedding" in low_id:
            input_modalities = ["text"]
            output_modalities = ["embedding"]
        elif "robotics" in low_id:
             input_modalities = ["text", "vision"]
        
        return {"input": input_modalities, "output": output_modalities}
