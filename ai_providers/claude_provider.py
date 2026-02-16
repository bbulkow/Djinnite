"""
Anthropic Claude AI Provider

Wraps the Anthropic SDK for Claude models.
Supports native web search for Claude 4.5+ models.
"""

import json
import urllib.request
import urllib.error
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


# Web search beta configuration
WEB_SEARCH_BETA_HEADER = "web-search-2025-03-05"
WEB_SEARCH_TOOL = {
    "type": "web_search_20250305",
    "name": "web_search"
}


def _supports_native_web_search(model_id: str) -> bool:
    """
    Check if a Claude model supports native web search.
    """
    if "4-5" in model_id or "4.5" in model_id:
        return True
    if "latest" in model_id.lower():
        return True
    if any(x in model_id for x in ["sonnet-4-5", "opus-4-5", "haiku-4-5"]):
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
    
    def generate(
        self,
        prompt: Union[str, List[Dict]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        web_search: bool = False,
    ) -> AIResponse:
        """
        Generate a response using Claude.
        """
        # If web search requested, delegate to native web search method
        if web_search and _supports_native_web_search(self.model):
            return self._generate_with_web_search(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens or 8192,
            )

        try:
            parts = self._normalize_input(prompt)
            claude_content = self._map_parts(parts)
            
            # Claude requires max_tokens to be specified.
            # Use a reasonable default if caller didn't provide one.
            # Callers should check ModelInfo.max_output_tokens from the catalog
            # to pass the right value for their model.
            if max_tokens is None:
                max_tokens = 8192
            
            # Build request kwargs
            kwargs = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": claude_content}],
            }
            
            if system_prompt:
                kwargs["system"] = system_prompt
            
            # Generate response
            response = self._client.messages.create(**kwargs)
            
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
                usage = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                }
            
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
    
    def _generate_with_web_search(
        self,
        prompt: Union[str, List[Dict]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> AIResponse:
        """
        Generate a response using Claude with native web search.
        """
        url = "https://api.anthropic.com/v1/messages"
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": WEB_SEARCH_BETA_HEADER,
            "content-type": "application/json"
        }
        
        parts = self._normalize_input(prompt)
        claude_content = self._map_parts(parts)

        data = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": claude_content}],
            "tools": [WEB_SEARCH_TOOL],
        }
        
        if system_prompt:
            data["system"] = system_prompt
        
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode('utf-8'),
                headers=headers,
                method='POST'
            )
            
            with urllib.request.urlopen(req) as response:
                result = json.load(response)
            
            content = ""
            output_parts = []
            content_blocks = result.get('content', [])
            
            for block in content_blocks:
                block_type = block.get('type', '')
                if block_type == 'text':
                    text = block.get('text', '')
                    content += text
                    output_parts.append({"type": "text", "text": text})
            
            usage_data = result.get('usage', {})
            usage = {
                "input_tokens": usage_data.get('input_tokens', 0),
                "output_tokens": usage_data.get('output_tokens', 0),
            }
            
            # Detect output truncation in raw HTTP response
            stop_reason = result.get('stop_reason')
            is_truncated = (stop_reason == "max_tokens")
            
            ai_response = AIResponse(
                content=content,
                model=self.model,
                provider=self.PROVIDER_NAME,
                usage=usage,
                parts=output_parts,
                raw_response=result,
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
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            error_body_lower = error_body.lower()
            # Detect context length exceeded in raw HTTP 400 responses
            if e.code == 400 and ("too many" in error_body_lower and "token" in error_body_lower) or \
               ("context" in error_body_lower and "length" in error_body_lower) or \
               ("prompt is too long" in error_body_lower):
                raise AIContextLengthError(
                    f"Input context too long for model '{self.model}': {error_body}",
                    provider=self.PROVIDER_NAME,
                )
            elif e.code == 400 and "beta" in error_body_lower:
                raise AIProviderError(
                    f"Web search not available for model '{self.model}'.",
                    provider=self.PROVIDER_NAME
                )
            elif e.code == 401:
                raise AIAuthenticationError("Invalid API key", provider=self.PROVIDER_NAME)
            elif e.code == 429:
                raise AIRateLimitError("Rate limit exceeded", provider=self.PROVIDER_NAME)
            else:
                raise AIProviderError(f"HTTP {e.code}: {error_body}", provider=self.PROVIDER_NAME)
        except Exception as e:
            raise AIProviderError(f"Web search generation failed: {e}", provider=self.PROVIDER_NAME, original_error=e)
    
    def generate_json(
        self,
        prompt: Union[str, List[Dict]],
        schema: Union[Dict, Type],
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        web_search: bool = False,
        force: bool = False,
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
            web_search: If True, enable native Claude web search (4.5+ models).

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

        # Claude requires max_tokens. Use the caller's value if provided,
        # otherwise default to 8192. Callers should check
        # ModelInfo.max_output_tokens from the catalog for their model.
        if max_tokens is None:
            max_tokens = 8192
        
        if web_search:
            if not _supports_native_web_search(self.model):
                raise AIProviderError(
                    f"Web search not supported for model '{self.model}'.",
                    provider=self.PROVIDER_NAME
                )
            # Web search path uses raw HTTP; include schema guidance in
            # the system prompt since the beta endpoint may not support
            # output_config yet.
            json_system = "You must respond with valid JSON only. No additional text or explanation."
            if system_prompt:
                json_system = f"{system_prompt}\n\n{json_system}"
            return self._generate_with_web_search(
                prompt=prompt,
                system_prompt=json_system,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        
        try:
            parts = self._normalize_input(prompt)
            claude_content = self._map_parts(parts)
            
            kwargs = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": claude_content}],
                # Anthropic Constraint Decoding via output_config.format
                "output_config": {
                    "format": {
                        "type": "json_schema",
                        "schema": json_schema,
                    }
                },
            }
            
            if system_prompt:
                kwargs["system"] = system_prompt
            
            response = self._client.messages.create(**kwargs)
            
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
                usage = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                }
            
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
        try:
            self._client.messages.create(
                model=self.model, max_tokens=1024,
                messages=[{"role": "user", "content": "Say hi."}],
                thinking={"type": "adaptive", "budget_tokens": 1024},
            )
            return True
        except Exception as e:
            err = str(e).lower()
            if "rate" in err or "timeout" in err:
                return None
            return False

    def probe_structured_json(self) -> Optional[bool]:
        """Probe whether this Claude model supports output_config JSON schema mode."""
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

    def discover_modalities(self, model_id: str) -> Dict[str, List[str]]:
        """Discover modalities for Claude models."""
        input_modalities = ["text"]
        output_modalities = ["text"]
        
        low_id = model_id.lower()
        if any(x in low_id for x in ["claude-3", "claude-4"]):
            input_modalities.append("vision")
            
        return {"input": input_modalities, "output": output_modalities}
