"""
Anthropic Claude AI Provider

Wraps the Anthropic SDK for Claude models.
Supports native web search for Claude 4.5+ models.
"""

import json
import urllib.request
import urllib.error
from typing import Optional

from .base_provider import (
    BaseAIProvider,
    AIResponse,
    AIProviderError,
    AIRateLimitError,
    AIAuthenticationError,
    AIModelNotFoundError,
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
    
    Only Claude 4.5+ models support the web-search-2025-03-05 beta.
    """
    # Check for 4.5+ model indicators
    if "4-5" in model_id or "4.5" in model_id:
        return True
    if "latest" in model_id.lower():
        # Latest models typically support newest features
        return True
    # Explicit model checks for future-proofing
    if any(x in model_id for x in ["sonnet-4-5", "opus-4-5", "haiku-4-5"]):
        return True
    return False


class ClaudeProvider(BaseAIProvider):
    """
    Anthropic Claude AI provider implementation.
    
    Uses the anthropic SDK directly.
    Supports native web search for Claude 4.5+ models via the beta API.
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
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AIResponse:
        """
        Generate a response using Claude.
        
        Args:
            prompt: The user prompt/message
            system_prompt: Optional system instruction
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate (default 4096)
            
        Returns:
            AIResponse with the generated content
        """
        try:
            # Build the messages
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # Claude requires max_tokens to be specified
            if max_tokens is None:
                max_tokens = 4096
            
            # Build request kwargs
            kwargs = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages,
            }
            
            # Add system prompt if provided
            if system_prompt:
                kwargs["system"] = system_prompt
            
            # Generate response
            response = self._client.messages.create(**kwargs)
            
            # Extract content
            content = ""
            if response.content:
                # Claude returns a list of content blocks
                for block in response.content:
                    if hasattr(block, 'text'):
                        content += block.text
            
            # Extract usage info
            usage = {}
            if response.usage:
                usage = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                }
            
            return AIResponse(
                content=content,
                model=self.model,
                provider=self.PROVIDER_NAME,
                usage=usage,
                raw_response=response
            )
            
        except Exception as e:
            error_message = str(e).lower()
            error_type = type(e).__name__
            
            # Check for specific error types
            if "authentication" in error_message or "api_key" in error_message or "AuthenticationError" in error_type:
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
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> AIResponse:
        """
        Generate a response using Claude with native web search.
        
        Uses the web-search-2025-03-05 beta API directly via HTTP
        since the SDK may not fully support it yet.
        """
        url = "https://api.anthropic.com/v1/messages"
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": WEB_SEARCH_BETA_HEADER,
            "content-type": "application/json"
        }
        
        data = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
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
            
            # Extract content from response
            content = ""
            content_blocks = result.get('content', [])
            
            for block in content_blocks:
                block_type = block.get('type', '')
                if block_type == 'text':
                    content += block.get('text', '')
                elif block_type == 'tool_use':
                    # Tool was called - content may be in the result
                    pass
            
            # Extract usage
            usage_data = result.get('usage', {})
            usage = {
                "input_tokens": usage_data.get('input_tokens', 0),
                "output_tokens": usage_data.get('output_tokens', 0),
            }
            
            return AIResponse(
                content=content,
                model=self.model,
                provider=self.PROVIDER_NAME,
                usage=usage,
                raw_response=result
            )
            
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            if e.code == 400 and "beta" in error_body.lower():
                raise AIProviderError(
                    f"Web search not available for model '{self.model}'. "
                    "Only Claude 4.5+ models support native web search.",
                    provider=self.PROVIDER_NAME
                )
            elif e.code == 401:
                raise AIAuthenticationError(
                    "Invalid API key",
                    provider=self.PROVIDER_NAME
                )
            elif e.code == 429:
                raise AIRateLimitError(
                    "Rate limit exceeded",
                    provider=self.PROVIDER_NAME
                )
            else:
                raise AIProviderError(
                    f"HTTP {e.code}: {error_body}",
                    provider=self.PROVIDER_NAME
                )
        except Exception as e:
            raise AIProviderError(
                f"Web search generation failed: {e}",
                provider=self.PROVIDER_NAME,
                original_error=e
            )
    
    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        web_search: bool = False,
    ) -> AIResponse:
        """
        Generate a JSON response using Claude.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system instruction
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            web_search: If True, enable web search for current information.
                       Only supported on Claude 4.5+ models.
        
        Raises:
            AIProviderError: If web_search=True but model doesn't support it.
        """
        # Cap max_tokens to avoid Anthropic's streaming requirement for long operations
        if max_tokens is not None and max_tokens > 8192:
            max_tokens = 8192
        elif max_tokens is None:
            max_tokens = 4096
        
        # Enhance system prompt for JSON output
        json_system = "You must respond with valid JSON only. No additional text or explanation."
        if system_prompt:
            json_system = f"{system_prompt}\n\n{json_system}"
        
        # Handle web search
        if web_search:
            if not _supports_native_web_search(self.model):
                raise AIProviderError(
                    f"Web search not supported for model '{self.model}'. "
                    "Only Claude 4.5+ models (e.g., claude-sonnet-4-5-20250929) support web search.",
                    provider=self.PROVIDER_NAME
                )
            return self._generate_with_web_search(
                prompt=prompt,
                system_prompt=json_system,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        
        return self.generate(
            prompt=prompt,
            system_prompt=json_system,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    def is_available(self) -> bool:
        """Check if Claude is available and configured."""
        if not self.api_key:
            return False
        
        try:
            # Try a minimal API call to verify connectivity
            # Using count_tokens as a lightweight check
            self._client.count_tokens(
                model=self.model,
                messages=[{"role": "user", "content": "test"}]
            )
            return True
        except Exception:
            # Fall back to just checking if client initialized
            return self._client is not None

    def list_models(self) -> list[dict]:
        """List available models from Claude."""
        if not self.api_key:
            return []
            
        try:
            # New Anthropic SDKs support models.list()
            # It returns a SyncCursorPage[Model]
            models = self._client.models.list(limit=100)
            
            models_list = []
            for model in models:
                # model attributes: id, display_name, created_at, type
                model_id = model.id
                name = getattr(model, "display_name", model_id)
                
                # Heuristic for context window and cost
                context = 200000 # Default for Claude 3/3.5
                cost = "standard"
                
                if "opus" in model_id:
                    cost = "premium"
                elif "haiku" in model_id:
                    cost = "economical"
                
                # Check web search capability
                capabilities = ["text", "vision"]
                if _supports_native_web_search(model_id):
                    capabilities.append("web_search")
                
                models_list.append({
                    "id": model_id,
                    "name": name,
                    "context_window": context,
                    "capabilities": capabilities,
                    "cost_tier": cost
                })
            
            return models_list
        except Exception as e:
            print(f"Error listing Claude models: {e}")
            return []
