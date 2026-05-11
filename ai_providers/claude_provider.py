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


# ---------------------------------------------------------------------------
# Web search tool configuration
# ---------------------------------------------------------------------------
# Tool version: web_search_20260209 (Feb 2026, GA with Claude 4.6).
#   Older version web_search_20250305 still works but lacks dynamic filtering
#   and uses ~24% more input tokens.
#
# allowed_callers: The 20260209 tool defaults to requiring "programmatic tool
#   calling", which only Claude 4.6+ models support.  Setting
#   allowed_callers=["direct"] makes it compatible with older models like
#   Haiku 4.5 that only support direct (model-initiated) tool calls.
#
# --- Multi-turn continuation (pause_turn) ---
# Web search is a *server-side* tool: Djinnite never defines or executes it;
# the Anthropic API handles search execution internally.  However, with
# constraint decoding (generate_json) or smaller models, the API may return
# stop_reason="pause_turn" meaning the model paused mid-conversation after
# triggering a search but before producing final output.
#
# Pitfall with streaming: stream.get_final_message() on a pause_turn may
# return server_tool_use content blocks WITHOUT their matching
# server_tool_result blocks (the results haven't been delivered via the
# stream yet).  If you echo these orphaned blocks back in a continuation
# message the API returns a 400 error:
#   "web_search tool use with id ... was found without a corresponding
#    web_search_tool_result block"
#
# The fix is _sanitize_content_for_continuation() which strips orphaned
# server_tool_use blocks before building the continuation message.  The
# model will re-trigger the search on the next turn if it still needs
# results.  We stay on streaming throughout (required for large max_tokens
# and long-running thinking requests -- see commit 75691b7).
# ---------------------------------------------------------------------------
_WEB_SEARCH_TOOL = {
    "type": "web_search_20260209",
    "name": "web_search",
    "allowed_callers": ["direct"],
}


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
        max_output_tokens: int,
    ) -> Optional[dict]:
        """
        Translate the unified ``thinking`` parameter into Claude's native
        thinking block format.

        Claude supports two thinking types:
        - ``"adaptive"``: model decides when/how much to think, with a
          budget cap. Preferred for newest models.
        - ``"enabled"``: fixed-budget explicit thinking. For models that
          support thinking but not adaptive mode.

        The ``thinking_style`` from the model catalog determines which type
        to use. If no catalog is available, defaults to ``"adaptive"``.

        **Caller responsibility:** Claude requires
        ``max_output_tokens > budget_tokens`` so the model has room to
        emit visible output after thinking. This method validates the
        invariant and raises a ``ValueError`` if the caller's
        ``max_output_tokens`` doesn't leave room. No silent adjustment.

        Args:
            thinking: The caller's thinking parameter (already validated
                      by ``_resolve_thinking`` — never ``str`` for Claude;
                      ``_resolve_thinking`` rejects str when the model's
                      ``thinking_style`` lacks ``"effort"``, and Claude's
                      catalog never advertises ``"effort"``).
            max_output_tokens: The effective output cap for the request.

        Returns:
            The ``thinking`` block dict, or ``None`` if not requested.
        """
        if thinking is None or thinking is False:
            return None

        # str must not reach here for Claude (caught upstream by
        # _resolve_thinking against the model's thinking_style). Defense
        # in depth — surface a clear local error rather than emitting a
        # malformed Claude request.
        if isinstance(thinking, str):
            raise ValueError(
                f"thinking=str (effort level) is not supported on Claude "
                f"model '{self.model}'. Claude's thinking block accepts "
                f"only an integer budget_tokens. Pass an int, True, "
                f"False, or None."
            )

        # Determine thinking style from catalog. Priority: adaptive > budget.
        style = "adaptive"
        styles = self._model_info.capabilities.thinking_style if self._model_info else None
        if styles:
            if "adaptive" in styles:
                style = "adaptive"
            elif "budget" in styles:
                style = "budget"

        # Compute budget_tokens
        if thinking is True:
            budget = self._get_max_thinking_budget(max_output_tokens)
        else:
            # int passthrough (bool was handled above; isinstance(True, int)
            # is True but already returned).
            budget = thinking

        # Invariant: output cap must exceed budget_tokens to leave room
        # for the visible response. Caller mistake — raise rather than
        # silently bumping max_output_tokens upward.
        if budget >= max_output_tokens:
            raise ValueError(
                f"thinking budget ({budget}) must be less than "
                f"max_output_tokens ({max_output_tokens}); Claude requires "
                f"room for visible output after thinking. Either lower the "
                f"budget or raise max_output_tokens."
            )

        thinking_type = "adaptive" if style == "adaptive" else "enabled"
        return {"type": thinking_type, "budget_tokens": budget}

    # ------------------------------------------------------------------
    # Multi-turn continuation for server-side tools (e.g. web_search)
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_content_for_continuation(content) -> list:
        """Sanitize assistant content blocks for multi-turn continuation.

        When the model pauses mid-turn (``stop_reason=pause_turn``), the
        content typically contains paired server-side tool blocks:

          server_tool_use  (id=X, the search request)
          web_search_tool_result  (tool_use_id=X, the search results)

        These paired blocks MUST be kept — they are the model's search
        context.  Stripping them forces the model to re-search from
        scratch, causing redundant API calls and 10x cost variation.

        However, the last block is often an **orphaned**
        ``server_tool_use`` without a matching ``web_search_tool_result``
        (the search was requested but results hadn't arrived when the
        stream ended).  The API rejects orphans, so we strip them.
        The model will re-issue that specific search on the next turn.
        """
        if not content:
            return content

        # Collect tool_use IDs that have a matching result
        result_ids = set()
        for block in content:
            if getattr(block, "type", None) == "web_search_tool_result":
                result_ids.add(getattr(block, "tool_use_id", None))

        return [
            block for block in content
            if not (
                getattr(block, "type", None) == "server_tool_use"
                and getattr(block, "id", None) not in result_ids
            )
        ]

    def _run_with_continuation(self, kwargs: dict, max_continuations: int = 2):
        """Stream a Messages API call, looping on ``pause_turn`` / ``tool_use``
        until the model produces ``end_turn`` or ``max_tokens``.

        Streaming is always used (required for large ``max_tokens`` values
        and long-running thinking requests).  On ``pause_turn``, the
        assistant content is sanitized to remove any orphaned
        ``server_tool_use`` blocks before being sent back for continuation.

        Returns ``(final_response, accumulated_usage_dict)``.
        """
        acc_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "thinking_tokens": 0,
            "server_tool_use_input_tokens": 0,
            "search_units": 0,
        }

        for _turn in range(max_continuations + 1):
            with self._client.messages.stream(**kwargs) as stream:
                response = stream.get_final_message()

            # Accumulate usage across turns
            if response.usage:
                acc_usage["input_tokens"] += getattr(response.usage, "input_tokens", 0) or 0
                acc_usage["output_tokens"] += getattr(response.usage, "output_tokens", 0) or 0
                acc_usage["thinking_tokens"] += getattr(response.usage, "thinking_tokens", 0) or 0
                acc_usage["server_tool_use_input_tokens"] += (
                    getattr(response.usage, "server_tool_use_input_tokens", 0) or 0
                )

            # Accumulate search units across turns (not just the final one)
            s_units, _ = self._count_search_units(response)
            acc_usage["search_units"] += s_units

            stop = getattr(response, "stop_reason", None)
            if stop not in ("pause_turn", "tool_use"):
                # Terminal turn -- return final response + totals
                return response, acc_usage

            # Model paused for server-side tool execution.
            # Sanitize the content to remove orphaned server_tool_use
            # blocks (those without a matching server_tool_result) --
            # streaming may not have delivered the result before pausing.
            clean_content = self._sanitize_content_for_continuation(
                response.content
            )
            # Stop-gap: tell the model to produce output after the first
            # pause.  Streaming with server-side tools (web_search) exposes
            # pause_turn states that the non-streaming API handled internally.
            # TODO: research the correct Anthropic streaming pattern for
            # server-tool continuations — this nudge is a workaround.
            kwargs["messages"] = kwargs["messages"] + [
                {"role": "assistant", "content": clean_content},
                {"role": "user", "content": (
                    "Stop searching. Produce the JSON output now "
                    "using the search results you already have."
                )},
            ]

        # Exhausted continuation budget
        raise AIProviderError(
            f"Model did not finish after {max_continuations} continuation "
            f"turns (last stop_reason='{stop}')",
            provider=self.PROVIDER_NAME,
        )

    def generate(
        self,
        prompt: Union[str, List[Dict]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        web_search: bool = False,
        thinking: Union[bool, int, str, None] = None,
    ) -> AIResponse:
        """
        Generate a response using Claude.

        ``temperature`` is opt-in. When omitted (default), Claude uses its own
        default (1.0) and the request ships with no ``temperature`` field.
        This avoids 400 errors on models that reject sampling parameters
        (e.g. Opus 4.7). Callers who need determinism on older models may
        pass an explicit value; catalog-strip handles models where the
        parameter is unsupported.
        """
        _orig_caller = {
            "thinking": thinking,
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
            "web_search": web_search,
            "system_prompt": system_prompt,
        }
        # Validate & normalize the thinking parameter
        thinking = self._resolve_thinking(thinking)

        try:
            parts = self._normalize_input(prompt)
            self._validate_vision_limits(parts)
            claude_content = self._map_parts(parts)

            # Claude's SDK requires the output cap (its `max_tokens` keyword)
            # to be specified. Auto-fill from catalog if caller didn't provide
            # one.
            max_output_tokens = self._resolve_max_output_tokens(max_output_tokens) or 8192

            # Build thinking block + adjust output cap to satisfy Claude's
            # invariant (max_tokens > thinking.budget_tokens).
            thinking_active = thinking is not None and thinking is not False
            thinking_block = self._build_claude_thinking(thinking, max_output_tokens)

            # Build request kwargs. The SDK keyword is literally `max_tokens`
            # (Anthropic's terminology); we pass our `max_output_tokens` value
            # under that key.
            kwargs = {
                "model": self.model,
                "max_tokens": max_output_tokens,
                "messages": [{"role": "user", "content": claude_content}],
            }

            # Temperature: only send if caller opted in. Catalog-strip is a
            # backstop for callers who pass an explicit value on a model that
            # rejects sampling params.
            if temperature is not None:
                effective_temp = self._resolve_temperature(temperature, thinking_active)
                if effective_temp is not None:
                    kwargs["temperature"] = effective_temp
            
            if system_prompt:
                kwargs["system"] = system_prompt

            # Thinking: add the provider-native thinking block
            if thinking_block is not None:
                kwargs["thinking"] = thinking_block

            # Web search: catalog decides support.
            if web_search:
                self._check_capability("web_search")
                kwargs["tools"] = [_WEB_SEARCH_TOOL]

            self._debug_dump_request(
                method="generate", caller_args=_orig_caller, native_config=kwargs,
            )

            # Generate response.
            # Stream with automatic continuation for server-side tool use
            # (e.g. web_search).  Loops on pause_turn/tool_use until
            # the model produces end_turn or max_tokens.
            response, acc_usage = self._run_with_continuation(kwargs)

            # Extract content
            content = ""
            output_parts = []
            if response.content:
                for block in response.content:
                    if hasattr(block, 'text'):
                        content += block.text
                        output_parts.append({"type": "text", "text": block.text})

            # Build usage from accumulated totals (may span multiple turns)
            input_t = acc_usage["input_tokens"]
            output_t = acc_usage["output_tokens"]
            thinking_t = acc_usage["thinking_tokens"] or None
            usage = {
                "input_tokens": input_t,
                "output_tokens": output_t,
                "total_tokens": input_t + output_t,
                "thinking_tokens": thinking_t,
                "_thinking_billed_separately": True,
            }

            # Search units accumulated across all continuation turns
            if acc_usage["search_units"]:
                usage["search_units"] = acc_usage["search_units"]
            total_search_tokens = acc_usage["server_tool_use_input_tokens"]
            if total_search_tokens:
                usage["search_result_tokens"] = total_search_tokens
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
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
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
            temperature: Sampling temperature. Opt-in (default None).  When
                omitted, Claude uses its own default and no ``temperature``
                field is sent — avoids 400 errors on 4.7+ which reject
                sampling params.
            max_output_tokens: Cap on output tokens (auto-fills from catalog,
                fallback 8192 since Claude's SDK requires a value).
            web_search: If True, enable native Claude web search (4.5+/4.6+ models).
            thinking: Optional thinking/reasoning control (same as generate()).

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

        # Claude's SDK requires the output cap. Auto-fill from catalog,
        # fallback to 8192.
        max_output_tokens = self._resolve_max_output_tokens(max_output_tokens) or 8192

        if web_search:
            if not force:
                self._check_capability("web_search")
                self._check_capability("json_with_search")

        # Validate & normalize thinking
        thinking = self._resolve_thinking(thinking)

        try:
            parts = self._normalize_input(prompt)
            self._validate_vision_limits(parts)
            claude_content = self._map_parts(parts)

            # Build thinking block + adjust output cap (invariant: max_tokens > budget)
            thinking_active = thinking is not None and thinking is not False
            thinking_block = self._build_claude_thinking(thinking, max_output_tokens)

            kwargs = {
                "model": self.model,
                "max_tokens": max_output_tokens,
                "messages": [{"role": "user", "content": claude_content}],
                # Anthropic Constraint Decoding via output_config.format
                "output_config": {
                    "format": {
                        "type": "json_schema",
                        "schema": json_schema,
                    }
                },
            }

            # Temperature: only send if caller opted in. Catalog-strip is a
            # backstop for callers who pass an explicit value on a model that
            # rejects sampling params.
            if temperature is not None:
                effective_temp = self._resolve_temperature(temperature, thinking_active)
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

            self._debug_dump_request(
                method="generate_json", caller_args=_orig_caller, native_config=kwargs,
            )

            # Stream with automatic continuation for server-side tool use
            # (e.g. web_search).  Same loop as generate().
            response, acc_usage = self._run_with_continuation(kwargs)

            # Extract content
            content = ""
            output_parts = []
            if response.content:
                for block in response.content:
                    if hasattr(block, 'text'):
                        content += block.text
                        output_parts.append({"type": "text", "text": block.text})

            # Build usage from accumulated totals (may span multiple turns)
            input_t = acc_usage["input_tokens"]
            output_t = acc_usage["output_tokens"]
            thinking_t = acc_usage["thinking_tokens"] or None
            usage = {
                "input_tokens": input_t,
                "output_tokens": output_t,
                "total_tokens": input_t + output_t,
                "thinking_tokens": thinking_t,
                "_thinking_billed_separately": True,
            }

            # Count billable search events (from final response)
            s_units, s_result_tokens = self._count_search_units(response)
            if s_units:
                usage["search_units"] = s_units
            total_search_tokens = acc_usage["server_tool_use_input_tokens"]
            if total_search_tokens:
                usage["search_result_tokens"] = total_search_tokens
            elif s_result_tokens is not None:
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
        styles = self.probe_thinking_style()
        if styles is None:
            return None
        return bool(styles)

    def probe_thinking_disable(self) -> Optional[bool]:
        """
        Whether Claude accepts an explicit thinking-disabled request.

        Claude's thinking is opt-in: a request without a ``thinking`` block
        is the natural "disabled" state and every Claude model accepts it.
        We return True unconditionally here — there is no Claude model that
        forbids omitting the ``thinking`` parameter.
        """
        return True

    def probe_thinking_style(self) -> Optional[list[str]]:
        """
        Multi-tier probe to determine which thinking styles Claude supports.

        Invariant: ``max_tokens > budget_tokens`` (room for output after thinking).

        Temperature is omitted entirely. It's never *required* on Claude, and
        sending it poisons probes against Opus 4.7 (which 400s on any
        temperature value). Claude's own default (1.0) satisfies the legacy
        "thinking needs temp=1" constraint naturally.

        Tries both adaptive and budget; a model may support both
        (Claude 4.7 = ``["adaptive", "budget"]``) or just one. If a probe
        is inconclusive (rate limit / timeout) the whole result becomes
        ``None`` — the orchestrator treats that as unknown.

        Returns:
            * Non-empty ``list[str]`` from ``("adaptive", "budget")`` — the
              styles confirmed to work.
            * ``[]`` — both tiers cleanly rejected → no thinking support.
            * ``None`` — inconclusive (rate limit / timeout on either tier).
        """
        _PROBE_BUDGET = 1024
        _PROBE_MAX_TOKENS = 2048  # Must exceed _PROBE_BUDGET

        styles: list[str] = []

        # Tier 1: adaptive (4.6/4.7+ preferred). Bare shape — 4.7 rejects
        # budget_tokens inside adaptive; 4.6 accepts bare adaptive too.
        try:
            self._client.messages.create(
                model=self.model,
                max_tokens=_PROBE_MAX_TOKENS,
                messages=[{"role": "user", "content": "Say hi."}],
                thinking={"type": "adaptive"},
            )
            styles.append("adaptive")
        except Exception as e:
            err = str(e).lower()
            if "rate" in err or "timeout" in err:
                return None

        # Tier 2: enabled / fixed-budget (older thinking models, and many
        # current models accept both modes).
        try:
            self._client.messages.create(
                model=self.model,
                max_tokens=_PROBE_MAX_TOKENS,
                messages=[{"role": "user", "content": "Say hi."}],
                thinking={"type": "enabled", "budget_tokens": _PROBE_BUDGET},
            )
            styles.append("budget")
        except Exception as e:
            err = str(e).lower()
            if "rate" in err or "timeout" in err:
                return None

        return styles

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

    def probe_web_search(self) -> Optional[bool]:
        """
        Probe whether this Claude model supports the web_search server-side tool.

        Sends a minimal request that declares ``tools=[_WEB_SEARCH_TOOL]`` but
        asks a trivial question the model is unlikely to search for. A success
        proves the API accepts the tool schema for this model. A 400 means
        the tool version isn't supported for this model.
        """
        try:
            self._client.messages.create(
                model=self.model,
                max_tokens=50,
                messages=[{"role": "user", "content": "Say hi."}],
                tools=[_WEB_SEARCH_TOOL],
            )
            return True
        except Exception as e:
            err = str(e).lower()
            status = getattr(e, 'status_code', None) or getattr(e, 'http_status', None)
            if status == 400 or "not supported" in err or "invalid" in err or "unknown tool" in err:
                return False
            if status in (401, 403, 429) or "rate" in err or "quota" in err or "timeout" in err:
                return None
            return None

    def probe_json_with_search(self) -> Optional[bool]:
        """Probe whether this Claude model supports output_config + web_search combined.

        No short-circuit on model ID — the API is the source of truth. If the
        model doesn't support web_search at all, the underlying error will be
        a 400 and we'll record it correctly as False.
        """
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
