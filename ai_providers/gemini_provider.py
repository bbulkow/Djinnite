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


class GeminiProvider(BaseAIProvider):
    """
    Google Gemini AI provider implementation.
    
    Supports both Google AI Studio (backend="gemini") and Vertex AI (backend="vertexai").
    Uses the google-genai SDK.
    """
    
    PROVIDER_NAME = "gemini"
    
    def __init__(self, api_key: str, model: str, backend: str = "gemini", project_id: Optional[str] = None, model_info=None):
        """
        Initialize the Gemini provider.
        
        Args:
            api_key: The Google API key
            model: The model ID to use
            backend: The Google backend to use ('gemini' or 'vertexai')
            project_id: The Google Cloud project ID (required for Vertex AI)
        """
        self.backend = backend
        self.project_id = project_id
        super().__init__(api_key, model, model_info=model_info)

    def _initialize_client(self) -> None:
        """Initialize the Gemini client."""
        try:
            from google import genai
            
            # Configure client based on backend
            if self.backend == "vertexai":
                if not self.project_id:
                    raise AIProviderError(
                        "project_id is required for Vertex AI backend",
                        provider=self.PROVIDER_NAME
                    )
                self._client = genai.Client(
                    api_key=self.api_key,
                    vertexai=True,
                    project=self.project_id,
                    location="us-central1" # Default location for Vertex
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
        except Exception as e:
            raise AIProviderError(
                f"Failed to initialize Gemini client: {e}",
                provider=self.PROVIDER_NAME,
                original_error=e
            )
    
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

    def _build_gemini_thinking(
        self,
        thinking: Union[bool, int, str, None],
        max_tokens: Optional[int],
    ) -> Optional[dict]:
        """
        Translate the unified ``thinking`` parameter into Gemini's native
        ``thinking_config`` format.

        Gemini uses ``thinking_config={"thinking_budget": N}`` where N is
        a token count.

        Args:
            thinking: The caller's thinking parameter (already validated).
            max_tokens: The effective max output tokens for the request.

        Returns:
            A dict for ``thinking_config``, or ``None`` if not requested.
        """
        if thinking is None:
            return None  # No opinion — let model use its default

        if thinking is False:
            # Explicitly disable thinking. Gemini requires thinking_budget=0
            # to suppress thinking on models that default to thinking ON.
            return {"thinking_budget": 0}

        if thinking is True:
            budget = self._get_max_thinking_budget(max_tokens)
        elif isinstance(thinking, int):
            budget = thinking
        else:
            # str effort → token budget
            effective_max = max_tokens or 8192
            budget = self._effort_to_budget(thinking, effective_max)

        return {"thinking_budget": budget}

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
        Generate a response using Gemini.
        """
        # Validate & normalize thinking
        thinking = self._resolve_thinking(thinking)

        try:
            from google.genai import types

            parts = self._normalize_input(prompt)
            self._validate_vision_limits(parts)
            gemini_parts = self._map_parts(parts)

            # Resolve temperature: strip if catalog says not supported
            effective_temp = self._resolve_temperature(temperature, thinking is not None)
            
            # Auto-fill max_tokens from catalog if caller didn't provide one.
            max_tokens = self._resolve_max_tokens(max_tokens)

            # Build configuration
            config = {}
            if effective_temp is not None:
                config["temperature"] = effective_temp
            if max_tokens:
                config["max_output_tokens"] = max_tokens
            if system_prompt:
                config["system_instruction"] = system_prompt

            # Thinking: add thinking_config if requested
            thinking_config = self._build_gemini_thinking(thinking, max_tokens)
            if thinking_config is not None:
                config["thinking_config"] = thinking_config
            
            # Enable Google Search grounding for current information
            if web_search:
                config["tools"] = [types.Tool(google_search=types.GoogleSearch())]
            
            # Generate response
            response = self._client.models.generate_content(
                model=self.model,
                contents=gemini_parts,
                config=config
            )
            
            # Extract usage info if available
            usage = {}
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                metadata = response.usage_metadata
                input_t = getattr(metadata, 'prompt_token_count', 0) or 0
                output_t = getattr(metadata, 'candidates_token_count', 0) or 0
                total_t = getattr(metadata, 'total_token_count', None)
                thinking_t = getattr(metadata, 'thoughts_token_count', None)
                usage = {
                    "input_tokens": input_t,
                    "output_tokens": output_t,
                    "total_tokens": total_t if total_t is not None else input_t + output_t,
                    "thinking_tokens": thinking_t,  # None if not reported
                    "_thinking_billed_separately": False,
                }

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
            
            if not text_content and hasattr(response, 'text'):
                text_content = response.text

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
            )
            
            if is_truncated:
                raise AIOutputTruncatedError(
                    f"Output truncated: model hit max output token limit "
                    f"(finishReason='{finish_reason}', output_tokens={usage.get('output_tokens', '?')})",
                    provider=self.PROVIDER_NAME,
                    partial_response=ai_response,
                )
            
            return ai_response
            
        except (AIOutputTruncatedError, AIContextLengthError):
            raise  # Never swallow our own semantic errors
        except Exception as e:
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
        max_tokens: Optional[int] = None,
        web_search: bool = False,
        force: bool = False,
        thinking: Union[bool, int, str, None] = None,
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
            max_tokens: Maximum tokens to generate.
            web_search: If True, enable Google Search grounding for current info.
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
            if web_search:
                self._check_capability("json_with_search")
        json_schema = self._normalize_schema(schema)
        json_schema = self._validate_caller_schema(json_schema)
        json_schema = self._prepare_schema_for_provider(json_schema)

        # Validate & normalize thinking
        thinking = self._resolve_thinking(thinking)

        try:
            from google.genai import types
            
            parts = self._normalize_input(prompt)
            self._validate_vision_limits(parts)
            gemini_parts = self._map_parts(parts)

            # Resolve temperature: catalog-aware stripping
            effective_temp = self._resolve_temperature(temperature, thinking is not None)

            # Build configuration with schema-enforced JSON output
            config = {
                "response_mime_type": "application/json",
                "response_schema": json_schema,
            }

            # Auto-fill max_tokens from catalog if caller didn't provide one.
            max_tokens = self._resolve_max_tokens(max_tokens)

            if effective_temp is not None:
                config["temperature"] = effective_temp
            
            if max_tokens:
                config["max_output_tokens"] = max_tokens
            if system_prompt:
                config["system_instruction"] = system_prompt

            # Thinking config
            thinking_config = self._build_gemini_thinking(thinking, max_tokens)
            if thinking_config is not None:
                config["thinking_config"] = thinking_config
            
            # Enable Google Search grounding for current information
            if web_search:
                config["tools"] = [types.Tool(google_search=types.GoogleSearch())]
            
            # Generate response
            response = self._client.models.generate_content(
                model=self.model,
                contents=gemini_parts,
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
            
            # Fall back to response.text if candidates extraction failed
            if not text_content and hasattr(response, 'text'):
                text_content = response.text
            
            # Extract usage info if available
            usage = {}
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                metadata = response.usage_metadata
                input_t = getattr(metadata, 'prompt_token_count', 0) or 0
                output_t = getattr(metadata, 'candidates_token_count', 0) or 0
                total_t = getattr(metadata, 'total_token_count', None)
                thinking_t = getattr(metadata, 'thoughts_token_count', None)
                usage = {
                    "input_tokens": input_t,
                    "output_tokens": output_t,
                    "total_tokens": total_t if total_t is not None else input_t + output_t,
                    "thinking_tokens": thinking_t,
                    "_thinking_billed_separately": False,
                }

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
            )
            
            if is_truncated:
                raise AIOutputTruncatedError(
                    f"JSON output truncated: model hit max output token limit "
                    f"(finishReason='{finish_reason}', output_tokens={usage.get('output_tokens', '?')})",
                    provider=self.PROVIDER_NAME,
                    partial_response=ai_response,
                )
            
            return ai_response
            
        except AIProviderError:
            raise  # Re-raise all our own errors (including truncation/context)
        except Exception as e:
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
        """Check if Gemini is available and configured."""
        if not self.api_key:
            return False
        
        try:
            # Try to list models as a connectivity check
            self._client.models.list(config={"page_size": 1})
            return True
        except Exception:
            return False

    def list_models(self) -> list[dict]:
        """List available models from Gemini.
        
        Extracts both input_token_limit (context_window) and
        output_token_limit (max_output_tokens) from the API when available.
        """
        if not self.api_key:
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
                config={"temperature": 0.5, "max_output_tokens": 10},
            )
            return True
        except Exception as e:
            err = str(e).lower()
            if "rate" in err or "quota" in err or "429" in err:
                return None
            return False

    def probe_thinking(self) -> Optional[bool]:
        """Probe whether this Gemini model supports thinking mode."""
        style = self.probe_thinking_style()
        if style is None:
            return False
        if style == "_inconclusive":
            return None
        return True

    def probe_thinking_style(self) -> Optional[str]:
        """
        Probe which thinking style this Gemini model supports.

        Gemini uses ``thinking_config`` with ``thinking_budget``
        (budget-based thinking).

        Returns:
            ``"budget"``   – model supports thinking_config/thinking_budget
            ``None``       – model does not support thinking
            ``"_inconclusive"`` – transient error (rate limit, quota)
        """
        try:
            self._client.models.generate_content(
                model=self.model, contents="Say hi.",
                config={
                    "max_output_tokens": 100,
                    "thinking_config": {"thinking_budget": 1024},
                },
            )
            return "budget"
        except Exception as e:
            err = str(e).lower()
            if "rate" in err or "quota" in err or "429" in err:
                return "_inconclusive"
            return None

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
