"""
Anthropic Claude AI Provider

Wraps the Anthropic SDK for Claude models.
Supports native web search for Claude 4.5+/4.6+ models.
"""

import json
import base64
import re
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

# Google Vertex AI serves only the basic web search tool version (no dynamic
# filtering, no programmatic callers -- so no ``allowed_callers`` field).
_WEB_SEARCH_TOOL_VERTEXAI = {
    "type": "web_search_20250305",
    "name": "web_search",
}

# Vertex addresses dated snapshots as ``claude-haiku-4-5@20251001`` where the
# Claude API uses ``claude-haiku-4-5-20251001``. Undated IDs (all 4.6+ models)
# are identical on both. A rule, not a table.
_DATED_SNAPSHOT = re.compile(r"-(\d{8})$")


class ClaudeProvider(BaseAIProvider):
    """
    Anthropic Claude AI provider implementation.

    Access modes (see ``platforms.py``):

    * direct (default): the Anthropic API, ``anthropic.Anthropic(api_key=...)``.
    * ``platform="vertexai"`` (or the legacy ``backend="vertexai"``): Google
      Vertex AI via ``anthropic.AnthropicVertex``, authenticated with
      Application Default Credentials. No API key is required or sent.
    """

    PROVIDER_NAME = "claude"

    # Claude's effort vocabulary is not the cross-provider default: it has no
    # "minimal", and adds "xhigh" and "max". The API rejects anything else
    # with "should be 'low', 'medium', 'high', 'xhigh' or 'max'".
    _EFFORT_LEVELS: frozenset = frozenset({"low", "medium", "high", "xhigh", "max"})

    # thinking="between_tools" -> {"type": "between_tools"}: the lowest
    # thinking setting on models whose thinking_style lists it (Sonnet 5.5).
    _THINKING_SENTINELS: frozenset = frozenset({"between_tools"})

    # Class-level defaults: direct mode (also read by tests' _bare() helper).
    backend: Optional[str] = None
    project_id: Optional[str] = None
    quota_project: Optional[str] = None

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        model_info=None,
        require_pricing: bool = True,
        *,
        platform: Optional[str] = None,
        backend: Optional[str] = None,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        quota_project: Optional[str] = None,
    ):
        """
        Initialize the Claude provider.

        Args:
            api_key: Anthropic API key (direct mode). Ignored -- never sent --
                in platform mode.
            model: The model ID (catalog ID; Vertex dated snapshots are
                rewritten to ``@`` form automatically).
            model_info: Catalog entry for pre-flight checks.
            require_pricing: Fast-fail on missing/unknown price (see base).
            platform: ``"vertexai"`` for Google Vertex AI; ``None`` = direct.
            backend: Legacy alias -- ``"vertexai"`` means
                ``platform="vertexai"``; any other value means direct.
            project_id: Google Cloud project (required on Vertex).
            location: Vertex location: ``"global"`` (default, no price
                premium), ``"us"``/``"eu"`` multi-regions, or a region.
            quota_project: Project billed for quota, sent as the
                ``x-goog-user-project`` header (for user ADC created with
                ``--disable-quota-project``).
        """
        self.backend = backend
        self.platform = resolve_platform(platform, backend, provider=self.PROVIDER_NAME)
        self.project_id = project_id
        self.quota_project = quota_project
        self.location = location.lower() if location else None
        if self.platform:
            spec = get_platform(self.platform, self.PROVIDER_NAME)
            self.location = self.location or spec.default_location.get(self.PROVIDER_NAME)
            self._price_multiplier = spec.price_multiplier(self.PROVIDER_NAME, self.location)
        super().__init__(api_key, model, model_info=model_info, require_pricing=require_pricing)

    def _initialize_client(self) -> None:
        """Initialize the Anthropic client for this access mode."""
        try:
            import anthropic
            self._anthropic = anthropic
            if self.platform == "vertexai":
                if not self.project_id:
                    raise AIProviderError(
                        "project_id is required for the 'vertexai' platform",
                        provider=self.PROVIDER_NAME,
                    )
                vertex_kwargs = {"project_id": self.project_id, "region": self.location}
                if self.quota_project:
                    # AnthropicVertex sets only ``Authorization`` from the
                    # credentials and never derives x-goog-user-project from
                    # them, so this header is the one place the billed
                    # project is set -- and nothing overwrites it.
                    vertex_kwargs["default_headers"] = {"x-goog-user-project": self.quota_project}
                self._client = anthropic.AnthropicVertex(**vertex_kwargs)
            elif self.platform:
                raise AIProviderError(
                    f"platform '{self.platform}' is not implemented for Claude",
                    provider=self.PROVIDER_NAME,
                )
            else:
                self._client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise AIProviderError(
                "anthropic package not installed. "
                "Install with: pip install anthropic",
                provider=self.PROVIDER_NAME
            )
        except AIProviderError:
            raise
        except Exception as e:
            raise AIProviderError(
                f"Failed to initialize Claude client: {e}",
                provider=self.PROVIDER_NAME,
                original_error=e
            )

    # ------------------------------------------------------------------
    # Platform mechanics
    # ------------------------------------------------------------------

    def _wire_model(self) -> str:
        """The model ID as this access path spells it.

        Vertex uses ``@`` before a dated snapshot suffix; the Claude API and
        the catalog use ``-``.
        """
        if self.platform == "vertexai":
            return _DATED_SNAPSHOT.sub(r"@\1", self.model)
        return self.model

    def _web_search_tool(self) -> dict:
        """The web search tool definition this access path serves."""
        if self.platform == "vertexai":
            return dict(_WEB_SEARCH_TOOL_VERTEXAI)
        return _WEB_SEARCH_TOOL

    def _claude_messages(self, turns, claude_content) -> List[Dict]:
        """Native ``messages``: history turns, then the prompt as the final user turn.

        With no history this is exactly ``[{"role": "user", "content": ...}]``,
        the single-turn request. Assistant turns are sent as plain strings.
        """
        messages: List[Dict] = []
        for role, parts in turns:
            if role == "assistant":
                messages.append({"role": "assistant", "content": self._history_text(parts)})
            else:
                messages.append({"role": "user", "content": self._map_parts(parts)})
        messages.append({"role": "user", "content": claude_content})
        return messages

    @staticmethod
    def _google_error_message(e: Exception) -> str:
        """Google's own error text from a Vertex error body, else ``str(e)``."""
        body = getattr(e, "body", None)
        if isinstance(body, list) and body:
            body = body[0]
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict) and err.get("message"):
                status = err.get("status")
                return f"{status}: {err['message']}" if status else str(err["message"])
        return getattr(e, "message", None) or str(e)

    def _map_platform_exception(self, e: Exception) -> Optional[AIProviderError]:
        """Map a Vertex AI failure to a Djinnite error, or ``None`` to fall through.

        Platform mode only; the direct-mode mapping is unchanged.
        """
        anthropic = self._anthropic
        where = f"project '{self.project_id}', location '{self.location}'"
        text = self._google_error_message(e)
        if isinstance(e, anthropic.RateLimitError) or "RESOURCE_EXHAUSTED" in str(e):
            return AIRateLimitError(
                f"Vertex AI quota or rate limit for '{self._wire_model()}' ({where}): {text}",
                provider=self.PROVIDER_NAME, original_error=e,
            )
        if isinstance(e, (anthropic.PermissionDeniedError, anthropic.AuthenticationError)):
            return AIAuthenticationError(
                f"Vertex AI refused the request ({where}): {text}",
                provider=self.PROVIDER_NAME, original_error=e,
            )
        if isinstance(e, anthropic.NotFoundError):
            return AIModelNotFoundError(
                f"Model '{self._wire_model()}' not found on Vertex AI ({where}) -- "
                f"not served at this location, or not enabled in Model Garden: {text}",
                provider=self.PROVIDER_NAME, original_error=e,
            )
        type_name = type(e).__name__
        if type_name in ("DefaultCredentialsError", "RefreshError"):
            return AIAuthenticationError(
                f"Google Application Default Credentials unavailable ({type_name}): {e}. "
                f"Run 'gcloud auth application-default login' or attach a service account.",
                provider=self.PROVIDER_NAME, original_error=e,
            )
        if type_name == "MissingDependencyError":
            return AIProviderError(
                f"{e} -- install 'anthropic[vertex]' for the vertexai platform",
                provider=self.PROVIDER_NAME, original_error=e,
            )
        return None

    def _map_claude_exception(self, e: Exception, *, json_mode: bool) -> AIProviderError:
        """Map an SDK exception to a Djinnite error.

        Platform failures are mapped first; the rest is the direct-mode
        mapping, unchanged: ``generate()`` maps auth / rate / not-found by
        message, ``generate_json()`` only context length.
        """
        if self.platform:
            mapped = self._map_platform_exception(e)
            if mapped is not None:
                return mapped

        error_message = str(e).lower()
        error_type = type(e).__name__

        # Detect context length exceeded: Anthropic SDK raises
        # anthropic.BadRequestError (HTTP 400) with type="invalid_request_error"
        # when the input exceeds the model's context window.
        if ("too many" in error_message and "token" in error_message) or \
           ("context" in error_message and "length" in error_message) or \
           ("prompt is too long" in error_message):
            return AIContextLengthError(
                f"Input context too long for model '{self.model}': {e}",
                provider=self.PROVIDER_NAME,
                original_error=e
            )
        if json_mode:
            return AIProviderError(
                f"JSON generation failed: {e}",
                provider=self.PROVIDER_NAME,
                original_error=e
            )
        if "authentication" in error_message or "api_key" in error_message or "AuthenticationError" in error_type:
            return AIAuthenticationError(
                "Invalid API key or authentication failed",
                provider=self.PROVIDER_NAME,
                original_error=e
            )
        if "rate" in error_message or "RateLimitError" in error_type:
            return AIRateLimitError(
                "Rate limit exceeded",
                provider=self.PROVIDER_NAME,
                original_error=e
            )
        if "model" in error_message and ("not found" in error_message or "NotFoundError" in error_type):
            return AIModelNotFoundError(
                f"Model '{self.model}' not found",
                provider=self.PROVIDER_NAME,
                original_error=e
            )
        return AIProviderError(
            f"Generation failed: {e}",
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
            thinking: The caller's thinking parameter, already validated by
                      ``_resolve_thinking``. A ``str`` is either an effort
                      level (no block; see ``_build_claude_effort``) or a
                      sentinel such as ``"between_tools"``.
            max_output_tokens: The effective output cap for the request.

        Returns:
            The ``thinking`` block dict, or ``None`` when nothing is sent
            (``thinking=None`` -- the provider default, which is adaptive
            thinking on the 5.x models -- or an effort level).
        """
        if thinking is None:
            return None

        # Explicit off. Omitting the block is NOT off on every model: on
        # Sonnet 5 / Opus 5 (and every 5.x) omission runs adaptive thinking.
        # Models that cannot disable thinking at all (Sonnet/Opus 5.5, Fable)
        # lack "off" in the catalog, and _resolve_thinking refuses False
        # before it gets here.
        if thinking is False:
            return {"type": "disabled"}

        # Provider settings carried in the thinking string, e.g.
        # "between_tools" (Sonnet 5.5's lowest setting). The API rejects any
        # other field alongside it.
        if isinstance(thinking, str) and thinking in self._THINKING_SENTINELS:
            return {"type": thinking}

        # An effort level is not a thinking block at all on Claude: it rides
        # in ``output_config.effort``. Return None here and let
        # _build_claude_effort() carry it. _resolve_thinking has already
        # rejected str on models whose thinking_style lacks "effort".
        if isinstance(thinking, str):
            return None

        # Pick the block shape. The two Claude thinking shapes take
        # different fields and are not interchangeable:
        #
        #   {"type": "adaptive"}                       - model sizes its own
        #                                                reasoning; accepts NO
        #                                                budget_tokens.
        #   {"type": "enabled", "budget_tokens": N}    - explicit budget,
        #                                                1024 <= N < max_tokens.
        #
        # An explicit int budget therefore *requires* the "enabled" shape;
        # sending budget_tokens on an adaptive block is rejected outright
        # ("thinking.adaptive.budget_tokens: Extra inputs are not
        # permitted"). thinking=True has no budget to honor, so it prefers
        # adaptive wherever the model offers it -- Anthropic deprecated
        # "enabled" on adaptive-capable models.
        #
        # An int reaching this point is already known to be legal:
        # _resolve_thinking rejects int on models whose thinking_style
        # lacks "budget".
        styles = self._model_info.capabilities.thinking_style if self._model_info else None
        wants_explicit_budget = thinking is not True

        if wants_explicit_budget:
            style = "budget"
        elif styles and "adaptive" in styles:
            style = "adaptive"
        elif styles and "budget" in styles:
            style = "budget"
        else:
            # No catalog entry: adaptive is the current default shape and
            # the only one that needs no budget to be well-formed.
            style = "adaptive"

        if style == "adaptive":
            # No budget_tokens, and no max_output_tokens invariant to
            # enforce -- the model allocates its own reasoning within the
            # output cap.
            return {"type": "adaptive"}

        # Budget shape: compute or accept budget_tokens.
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

        return {"type": "enabled", "budget_tokens": budget}

    def _build_claude_effort(self, thinking) -> Optional[str]:
        """Extract the ``output_config.effort`` level, if the caller asked for one.

        Claude carries reasoning effort in ``output_config``, alongside the
        structured-output ``format`` key -- not in the ``thinking`` block.
        The two compose: a request may set both an effort level and an
        adaptive thinking block.

        Returns the level string, or ``None`` when the caller did not pass
        an effort level. A sentinel such as ``"between_tools"`` is not an
        effort level: no effort is sent and the model's default applies
        (``high`` on Sonnet 5.5, the highest level between_tools accepts).
        """
        if isinstance(thinking, str) and thinking not in self._THINKING_SENTINELS:
            return thinking
        return None

    def _thinking_active(self, thinking) -> bool:
        """Whether the request selects the catalog's thinking ``"on"`` state.

        ``None`` (provider default), ``False`` and ``"between_tools"`` --
        Anthropic's documented way to turn thinking off on Sonnet 5.5 --
        are ``"off"``.
        """
        if thinking is None or thinking is False:
            return False
        return not (isinstance(thinking, str) and thinking in self._THINKING_SENTINELS)

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

        ``thinking_tokens`` is the sum of ``usage.output_tokens_details
        .thinking_tokens`` over turns -- a SUBSET of ``output_tokens``, which
        Anthropic documents as the inclusive billing total. It is ``None``
        (unknown, never 0) when any turn's response does not report it.
        """
        acc_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "thinking_tokens": 0,
            "server_tool_use_input_tokens": 0,
            "search_units": 0,
        }
        thinking_reported = True

        for _turn in range(max_continuations + 1):
            with self._client.messages.stream(**kwargs) as stream:
                response = stream.get_final_message()

            # Accumulate usage across turns
            if response.usage:
                acc_usage["input_tokens"] += getattr(response.usage, "input_tokens", 0) or 0
                acc_usage["output_tokens"] += getattr(response.usage, "output_tokens", 0) or 0
                details = getattr(response.usage, "output_tokens_details", None)
                turn_thinking = getattr(details, "thinking_tokens", None) if details is not None else None
                if turn_thinking is None:
                    thinking_reported = False
                else:
                    acc_usage["thinking_tokens"] += turn_thinking
                acc_usage["server_tool_use_input_tokens"] += (
                    getattr(response.usage, "server_tool_use_input_tokens", 0) or 0
                )
            else:
                thinking_reported = False

            # Accumulate search units across turns (not just the final one)
            s_units, _ = self._count_search_units(response)
            acc_usage["search_units"] += s_units

            stop = getattr(response, "stop_reason", None)
            if stop not in ("pause_turn", "tool_use"):
                # Terminal turn -- return final response + totals
                if not thinking_reported:
                    acc_usage["thinking_tokens"] = None
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
        *,
        history: Optional[List[Dict]] = None,
    ) -> AIResponse:
        """
        Generate a response using Claude.

        ``temperature`` is opt-in. When omitted (default), Claude uses its own
        default (1.0) and the request ships with no ``temperature`` field.
        This avoids 400 errors on models that reject sampling parameters
        (e.g. Opus 4.7). Callers who need determinism on older models may
        pass an explicit value; catalog-strip handles models where the
        parameter is unsupported.

        ``history`` (earlier turns) is sent as the leading ``messages``;
        ``prompt`` is the final user turn. See ``BaseAIProvider.generate``.
        """
        _orig_caller = {
            "thinking": thinking,
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
            "web_search": web_search,
            "system_prompt": system_prompt,
            "history_turns": len(history or []),
        }
        # Validate & normalize the thinking parameter
        thinking = self._resolve_thinking(thinking)
        turns = self._normalize_history(history)

        try:
            parts = self._normalize_input(prompt)
            self._validate_vision_limits(self._all_parts(turns, parts))
            claude_content = self._map_parts(parts)

            # Claude's SDK requires the output cap (its `max_tokens` keyword)
            # to be specified. Auto-fill from catalog if caller didn't provide
            # one.
            max_output_tokens = self._resolve_max_output_tokens(max_output_tokens) or 8192

            # Build thinking block + adjust output cap to satisfy Claude's
            # invariant (max_tokens > thinking.budget_tokens).
            thinking_active = self._thinking_active(thinking)
            thinking_block = self._build_claude_thinking(thinking, max_output_tokens)

            # Catalog web-search check first, so its message wins over an
            # ai_config deny (checked in the combination pre-flight below).
            if web_search:
                self._check_capability("web_search")

            # Cross-capability pre-flight from catalog (e.g. temp + thinking
            # is rejected on every current Claude thinking model).
            self._validate_incompatible_combinations({
                "temperature":     "any" if temperature is not None else "default",
                "thinking":        "on"  if thinking_active        else "off",
                "structured_json": "off",
                "web_search":      "on"  if web_search             else "off",
            })

            # Build request kwargs. The SDK keyword is literally `max_tokens`
            # (Anthropic's terminology); we pass our `max_output_tokens` value
            # under that key.
            kwargs = {
                "model": self._wire_model(),
                "max_tokens": max_output_tokens,
                "messages": self._claude_messages(turns, claude_content),
            }

            # Temperature: only send if caller opted in. Catalog-strip is a
            # backstop for callers who pass an explicit value on a model that
            # rejects sampling params.
            if temperature is not None:
                effective_temp = self._resolve_temperature(temperature, thinking_active)
                if effective_temp is not None:
                    # anthropic>=1.0 dropped temperature/top_p/top_k from the
                    # messages.create()/stream() signatures -- passing them is
                    # a TypeError. The API still honors the parameter, so it
                    # goes through extra_body, which the SDK merges into the
                    # request JSON verbatim. See the SDK's MIGRATION.md.
                    kwargs["extra_body"] = {"temperature": effective_temp}
            
            if system_prompt:
                kwargs["system"] = system_prompt

            # Thinking: add the provider-native thinking block
            if thinking_block is not None:
                kwargs["thinking"] = thinking_block

            # Effort rides in output_config, not the thinking block. Merge
            # rather than assign -- output_config also carries `format` for
            # constraint decoding.
            effort = self._build_claude_effort(thinking)
            if effort is not None:
                kwargs.setdefault("output_config", {})["effort"] = effort

            # Web search: catalog decides support (checked above); the access
            # path decides the tool version.
            if web_search:
                kwargs["tools"] = [self._web_search_tool()]

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

            # Build usage from accumulated totals (may span multiple turns).
            # thinking_tokens is a subset of output_tokens (None = not
            # reported), so it is not billed again.
            input_t = acc_usage["input_tokens"]
            output_t = acc_usage["output_tokens"]
            usage = {
                "input_tokens": input_t,
                "output_tokens": output_t,
                "total_tokens": input_t + output_t,
                "thinking_tokens": acc_usage["thinking_tokens"],
                "_thinking_billed_separately": False,
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

        except AIProviderError:
            raise  # Never swallow our own semantic errors (incl. fast-fail pricing)
        except Exception as e:
            raise self._map_claude_exception(e, json_mode=False)
    
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
        *,
        history: Optional[List[Dict]] = None,
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
            history: Optional earlier turns (same as generate()).
                ``output_config.format`` is a request-level setting that
                constrains only the turn being generated; earlier assistant
                turns, including replayed JSON, are sent as plain text.

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
        turns = self._normalize_history(history)

        try:
            parts = self._normalize_input(prompt)
            self._validate_vision_limits(self._all_parts(turns, parts))
            claude_content = self._map_parts(parts)

            # Build thinking block + adjust output cap (invariant: max_tokens > budget)
            thinking_active = self._thinking_active(thinking)
            thinking_block = self._build_claude_thinking(thinking, max_output_tokens)

            # Cross-capability pre-flight from catalog.
            self._validate_incompatible_combinations({
                "temperature":     "any" if temperature is not None else "default",
                "thinking":        "on"  if thinking_active        else "off",
                "structured_json": "on",
                "web_search":      "on"  if web_search             else "off",
                "json_with_search": "on" if web_search             else "off",
            })

            kwargs = {
                "model": self._wire_model(),
                "max_tokens": max_output_tokens,
                "messages": self._claude_messages(turns, claude_content),
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
                    # anthropic>=1.0 dropped temperature/top_p/top_k from the
                    # messages.create()/stream() signatures -- passing them is
                    # a TypeError. The API still honors the parameter, so it
                    # goes through extra_body, which the SDK merges into the
                    # request JSON verbatim. See the SDK's MIGRATION.md.
                    kwargs["extra_body"] = {"temperature": effective_temp}
            
            if system_prompt:
                kwargs["system"] = system_prompt

            # Thinking block
            if thinking_block is not None:
                kwargs["thinking"] = thinking_block

            # Effort shares output_config with the constraint-decoding
            # `format` key set above, so merge into it.
            effort = self._build_claude_effort(thinking)
            if effort is not None:
                kwargs.setdefault("output_config", {})["effort"] = effort

            # Web search: combine output_config (constraint decoding) with
            # web_search tool in the same request — native JSON + search.
            if web_search:
                kwargs["tools"] = [self._web_search_tool()]

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

            # Build usage from accumulated totals (may span multiple turns).
            # thinking_tokens is a subset of output_tokens (None = not
            # reported), so it is not billed again.
            input_t = acc_usage["input_tokens"]
            output_t = acc_usage["output_tokens"]
            usage = {
                "input_tokens": input_t,
                "output_tokens": output_t,
                "total_tokens": input_t + output_t,
                "thinking_tokens": acc_usage["thinking_tokens"],
                "_thinking_billed_separately": False,
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

            # generate_json promises schema-conforming JSON: a refusal or an
            # empty reply cannot meet that (see AIEmptyResponseError).
            if stop_reason == "refusal" or not content.strip():
                self._raise_empty(
                    ai_response,
                    reason="refusal" if stop_reason == "refusal" else "empty",
                    details={
                        "stop_reason": stop_reason,
                        "block_types": ",".join(
                            str(getattr(b, "type", "?")) for b in (response.content or [])
                        ),
                        "text": content.strip()[:200],
                    },
                    web_search=web_search,
                    thinking=thinking,
                )

            return ai_response
            
        except (AIOutputTruncatedError, AIContextLengthError):
            raise
        except AIProviderError:
            raise
        except Exception as e:
            raise self._map_claude_exception(e, json_mode=True)

    def is_available(self) -> bool:
        """Check if Claude is reachable for this model.

        **Makes a live network call** (a token count, which is not billed).

        * Direct mode: ``False`` without an API key; otherwise ``True`` even
          when the call fails, as long as a client exists (unchanged
          behavior).
        * Platform mode: needs no key; ``True`` only if the call succeeds.
          Any failure -- including a zero Vertex quota (429), missing ADC,
          or a model not enabled at this location -- returns ``False``.
        """
        if not self.platform and not self.api_key:
            return False

        try:
            self._availability_call()
            return True
        except Exception:
            if self.platform:
                return False
            return self._client is not None

    def _availability_call(self) -> None:
        """Token count: the cheapest live call (unbilled; works on Vertex)."""
        self._client.messages.count_tokens(
            model=self._wire_model(),
            messages=[{"role": "user", "content": "test"}]
        )

    def list_models(self) -> list[dict]:
        """List available Claude models.

        Direct mode asks the Anthropic Models API. Vertex AI has no Models
        endpoint, so platform mode returns the catalog models that
        ``scripts/probe_platform.py`` recorded as ``available`` at this
        location (``[]`` if the platform has never been probed).
        """
        if self.platform:
            return self._list_platform_models()
        if not self.api_key:
            return []

        try:
            models = self._client.models.list(limit=100)
            
            models_list = []
            for model in models:
                model_id = model.id
                name = getattr(model, "display_name", model_id)

                # The models endpoint reports these directly; do not guess.
                # A hardcoded 200000 here was writing the wrong context
                # window into the catalog for every 1M-context model, and
                # violated this repo's "no static model data in Python"
                # policy for update_models.
                context = getattr(model, "max_input_tokens", None) or 200000
                api_max_output = getattr(model, "max_tokens", None) or 0

                cost = self._cost_tier(model_id)

                modalities = ["text", "vision"]

                # Effort levels vary per model (Opus 4.5 stops at "high";
                # Opus 5 accepts "max"), and the endpoint enumerates them,
                # so no probing is needed to discover the set.
                effort_levels = None
                caps = getattr(model, "capabilities", None)
                effort = getattr(caps, "effort", None) if caps else None
                if effort is not None and getattr(effort, "supported", False):
                    effort_levels = [
                        lvl for lvl in ("low", "medium", "high", "xhigh", "max")
                        if getattr(getattr(effort, lvl, None), "supported", False)
                    ] or None

                entry = {
                    "id": model_id,
                    "name": name,
                    "context_window": context,
                    "modalities": modalities,
                    "cost_tier": cost,
                }
                if api_max_output:
                    entry["max_output_tokens"] = api_max_output
                if effort_levels:
                    entry["effort_levels"] = effort_levels
                models_list.append(entry)
            
            return models_list
        except Exception as e:
            print(f"Error listing Claude models: {e}")
            return []

    @staticmethod
    def _cost_tier(model_id: str) -> str:
        """Coarse cost tier from the model family name."""
        if "opus" in model_id:
            return "premium"
        if "haiku" in model_id:
            return "economical"
        return "standard"

    def _list_platform_models(self) -> list[dict]:
        """Catalog models recorded ``available`` on this platform and location."""
        try:
            try:
                from djinnite.config_loader import load_model_catalog
            except ImportError:
                from config_loader import load_model_catalog  # type: ignore
            catalog = load_model_catalog()
        except Exception:
            return []
        models_list = []
        for info in catalog.list_models(self.PROVIDER_NAME):
            pm = info.platforms.get(self.platform)
            if info.disabled or pm is None or pm.locations.get(self.location) != "available":
                continue
            entry = {
                "id": info.id,
                "name": info.name,
                "context_window": info.context_window,
                "modalities": list(info.modalities.input),
                "cost_tier": self._cost_tier(info.id),
            }
            if info.max_output_tokens:
                entry["max_output_tokens"] = info.max_output_tokens
            if info.capabilities.effort_levels:
                entry["effort_levels"] = list(info.capabilities.effort_levels)
            models_list.append(entry)
        return models_list

    def probe_temperature(self) -> Optional[bool]:
        """Probe whether this Claude model accepts temperature. (All Claude models do.)

        Cross-capability constraints (e.g. ``temperature`` + ``thinking``
        rejected together on every current thinking Claude) are NOT
        captured here — this probe deliberately isolates the temperature
        signal by sending no other features. The interaction is
        discovered by ``probe_incompatible_combinations`` and stored in
        ``capabilities.incompatible``.
        """
        try:
            self._client.messages.create(
                model=self._wire_model(), max_tokens=10,
                extra_body={"temperature": 0.5},
                messages=[{"role": "user", "content": "Say hi."}],
            )
            return True
        except Exception as e:
            err = str(e).lower()
            if "rate" in err or "timeout" in err:
                return None
            return False

    def _build_combination_probe_request(self, active_states: Dict[str, str]) -> dict:
        """
        Map a set of activated capability states into Anthropic SDK
        kwargs for ``messages.create``. Used by the cross-capability
        probe orchestrator on the base class.
        """
        kwargs: dict = {
            "model": self._wire_model(),
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": "Say hi."}],
        }
        if active_states.get("temperature") == "any":
            kwargs["extra_body"] = {"temperature": 0.5}
        if active_states.get("thinking") == "on":
            # Use a shape the model accepts, or the rejection of the SHAPE
            # is recorded as an incompatibility. Sending "enabled" (budget)
            # unconditionally put {thinking:on, structured_json:on} and
            # {thinking:on, web_search:on} on every Opus 4.7+ / 5.x model,
            # all of which reject budget_tokens. Budget only when the model
            # is budget-only; adaptive otherwise (or when styles are unknown).
            styles = (self._probe_supported_states or {}).get("thinking_style")
            if styles is None and self._model_info is not None:
                styles = self._model_info.capabilities.thinking_style
            if styles and "budget" in styles and "adaptive" not in styles:
                # budget_tokens (1024) < max_tokens (2048): the invariant.
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": 1024}
            else:
                kwargs["thinking"] = {"type": "adaptive"}
        if active_states.get("structured_json") == "on":
            kwargs["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": {"v": {"type": "integer"}},
                        "required": ["v"],
                        "additionalProperties": False,
                    },
                }
            }
        if active_states.get("web_search") == "on":
            kwargs["tools"] = [self._web_search_tool()]
        return kwargs

    def _run_combination_probe(self, kwargs: dict) -> None:
        """Send the probe via Anthropic's messages.create."""
        self._client.messages.create(**kwargs)

    def probe_thinking(self) -> Optional[bool]:
        """Probe whether this Claude model supports extended thinking."""
        styles = self.probe_thinking_style()
        if styles is None:
            return None
        return any(s in ("adaptive", "budget") for s in styles)

    def probe_thinking_disable(self) -> Optional[bool]:
        """
        Whether this model accepts ``thinking={"type": "disabled"}``.

        Omitting the block is NOT a reliable "off": on the 5.x models it
        runs adaptive thinking. ``thinking=False`` therefore sends an
        explicit ``disabled`` block, and this probe asks the API whether the
        model takes it. Sonnet 5.5, Opus 5.5 and the Fable models reject it
        with a 400 (always-on reasoning; Sonnet 5.5's lowest setting is
        ``between_tools`` instead).

        Returns:
            True  -- accepted.
            False -- rejected (400): thinking cannot be turned off.
            None  -- inconclusive (rate limit, 5xx, unknown error).
        """
        try:
            self._client.messages.create(
                model=self._wire_model(),
                max_tokens=16,
                messages=[{"role": "user", "content": "Say hi."}],
                thinking={"type": "disabled"},
            )
            return True
        except Exception as e:
            if self._classify_probe_error(e) == "not_supported":
                return False
            return None

    def probe_thinking_style(self) -> Optional[list[str]]:
        """
        Multi-tier probe to determine which thinking styles Claude supports.

        Invariant: ``max_tokens > budget_tokens`` (room for output after thinking).

        Temperature is omitted entirely. It's never *required* on Claude, and
        sending it poisons probes against Opus 4.7 (which 400s on any
        temperature value). Claude's own default (1.0) satisfies the legacy
        "thinking needs temp=1" constraint naturally.

        **Partial-success policy:** if any tier is inconclusive (rate
        limit, timeout, 5xx, unknown error class) the entire result is
        ``None``. We do NOT commit a partial truth (e.g. ``["adaptive"]``
        when the budget probe was rate-limited) because the catalog
        consumer would treat that as "confirmed: only adaptive" and the
        wrong value would stick until someone reprobed.

        Returns:
            * Non-empty ``list[str]`` from ``("adaptive", "budget",
              "between_tools")`` — the styles confirmed to work.
            * ``[]`` — every tier cleanly rejected → no thinking support.
            * ``None`` — any tier was inconclusive.

        ``between_tools`` is a thinking *setting* (Sonnet 5.5's lowest), not
        evidence that the model thinks; ``update_models`` does not count it
        toward the thinking "on" state.
        """
        _PROBE_BUDGET = 1024
        _PROBE_MAX_TOKENS = 2048  # Must exceed _PROBE_BUDGET

        tiers = [
            # Tier 1: adaptive (4.6/4.7+ preferred). Bare shape — 4.7 rejects
            # budget_tokens inside adaptive; 4.6 accepts bare adaptive too.
            ("adaptive", {"type": "adaptive"}),
            # Tier 2: enabled / fixed-budget (older thinking models, and many
            # current models accept both modes).
            ("budget", {"type": "enabled", "budget_tokens": _PROBE_BUDGET}),
            # Tier 3: between_tools (Sonnet 5.5 only). Must carry no other
            # field; valid at the default effort, which is <= high.
            ("between_tools", {"type": "between_tools"}),
        ]

        styles: list[str] = []
        inconclusive = False

        for style_name, thinking_cfg in tiers:
            try:
                self._client.messages.create(
                    model=self._wire_model(),
                    max_tokens=_PROBE_MAX_TOKENS,
                    messages=[{"role": "user", "content": "Say hi."}],
                    thinking=thinking_cfg,
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
                model=self._wire_model(),
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

        Sends a minimal request that declares the access path's web search tool but
        asks a trivial question the model is unlikely to search for. A success
        proves the API accepts the tool schema for this model. A 400 means
        the tool version isn't supported for this model.
        """
        try:
            self._client.messages.create(
                model=self._wire_model(),
                max_tokens=50,
                messages=[{"role": "user", "content": "Say hi."}],
                tools=[self._web_search_tool()],
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
                model=self._wire_model(),
                max_tokens=50,
                messages=[{"role": "user", "content": "Return the number 1."}],
                output_config={
                    "format": {
                        "type": "json_schema",
                        "schema": _PROBE_SCHEMA,
                    }
                },
                tools=[self._web_search_tool()],
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
