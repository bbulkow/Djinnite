"""
OpenAI Provider

Wraps the OpenAI SDK.
Supports web search via Gemini's native Google Search grounding.
"""

from typing import Optional, Union, List, Dict

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
    OpenAI provider implementation.
    
    Uses the openai SDK.
    Supports web search by delegating to Gemini's native Google Search.
    """
    
    PROVIDER_NAME = "chatgpt"
    
    def __init__(self, api_key: str, model: str, gemini_api_key: Optional[str] = None):
        """
        Initialize the OpenAI provider.
        
        Args:
            api_key: OpenAI API key
            model: Model ID to use
            gemini_api_key: Optional Gemini API key for web search capability
        """
        self._gemini_api_key = gemini_api_key
        super().__init__(api_key, model)
    
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

    def _map_parts(self, parts: List[Dict]) -> List:
        """Map internal parts to OpenAI SDK message content."""
        openai_content = []
        for part in parts:
            if part["type"] == "text":
                openai_content.append({"type": "text", "text": part["text"]})
            elif part["type"] == "image":
                if "image_data" in part:
                    # OpenAI supports base64 in data URLs
                    import base64
                    data = part["image_data"]
                    if isinstance(data, bytes):
                        data = base64.b64encode(data).decode("utf-8")
                    mime = part.get("mime_type", "image/jpeg")
                    openai_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{data}"}
                    })
                elif "file_uri" in part:
                    openai_content.append({
                        "type": "image_url",
                        "image_url": {"url": part["file_uri"]}
                    })
            # OpenAI's chat completion API (at least currently) primarily supports vision for multimodal
            # Audio/Video are handled via other endpoints or specifically formatted if supported
        return openai_content
    
    def generate(
        self,
        prompt: Union[str, List[Dict]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AIResponse:
        """
        Generate a response using OpenAI.
        """
        try:
            parts = self._normalize_input(prompt)
            openai_content = self._map_parts(parts)

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": openai_content})
            
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
                
            response = self._client.chat.completions.create(**kwargs)
            
            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            
            usage = {}
            if response.usage:
                usage = {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                }
            
            # Detect output truncation: OpenAI returns finish_reason="length"
            # when the output was cut short due to max_tokens.
            # This is an HTTP 200 response — the SDK does NOT raise an exception.
            is_truncated = (finish_reason == "length")
            
            ai_response = AIResponse(
                content=content,
                model=self.model,
                provider=self.PROVIDER_NAME,
                usage=usage,
                raw_response=response,
                truncated=is_truncated,
                finish_reason=finish_reason,
            )
            
            if is_truncated:
                raise AIOutputTruncatedError(
                    f"Output truncated: model hit max output token limit "
                    f"(finish_reason='length', output_tokens={usage.get('output_tokens', '?')})",
                    provider=self.PROVIDER_NAME,
                    partial_response=ai_response,
                )
            
            return ai_response
            
        except (AIOutputTruncatedError, AIContextLengthError):
            raise  # Never swallow our own semantic errors
        except Exception as e:
            error_message = str(e).lower()
            
            # Detect context length exceeded: OpenAI SDK raises
            # openai.BadRequestError (HTTP 400) with code="context_length_exceeded"
            error_code = getattr(e, 'code', None) or ''
            if str(error_code) == 'context_length_exceeded' or 'context_length_exceeded' in error_message:
                raise AIContextLengthError(
                    f"Input context too long for model '{self.model}': {e}",
                    provider=self.PROVIDER_NAME,
                    original_error=e
                )
            elif "api_key" in error_message or "auth" in error_message:
                raise AIAuthenticationError(
                    "Invalid API key or authentication failed",
                    provider=self.PROVIDER_NAME,
                    original_error=e
                )
            elif "rate" in error_message or "quota" in error_message:
                raise AIRateLimitError(
                    "Rate limit exceeded",
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
    
    def _search_with_gemini(self, query: str) -> str:
        """
        Perform a web search using Gemini's native Google Search grounding.
        """
        if not self._gemini_api_key:
            raise AIProviderError(
                "Web search requires Gemini API key. "
                "Configure gemini_api_key in OpenAIProvider or use Gemini directly.",
                provider=self.PROVIDER_NAME
            )
        
        try:
            from google import genai
            from google.genai import types
            
            client = genai.Client(api_key=self._gemini_api_key)
            
            config = {
                "temperature": 0.3,
                "tools": [types.Tool(google_search=types.GoogleSearch())],
            }
            
            response = client.models.generate_content(
                model="gemini-2.5-flash",  # Use fast model for search
                contents=f"Search the web and provide current information about: {query}",
                config=config
            )
            
            return response.text
            
        except Exception as e:
            raise AIProviderError(
                f"Gemini search failed: {e}",
                provider=self.PROVIDER_NAME,
                original_error=e
            )
    
    def generate_json(
        self,
        prompt: Union[str, List[Dict]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        web_search: bool = False,
    ) -> AIResponse:
        """
        Generate a JSON response using OpenAI.
        """
        # Pass through the caller's max_tokens value without capping.
        # Callers should check ModelInfo.max_output_tokens from the catalog
        # to pass the right value for their model.
        
        try:
            # If web search requested, get current info from Gemini first
            if web_search:
                if not self._gemini_api_key:
                    raise AIProviderError(
                        "Web search requires Gemini API key. "
                        "Pass gemini_api_key to OpenAIProvider constructor.",
                        provider=self.PROVIDER_NAME
                    )
                
                # Extract search context from Gemini
                search_results = self._search_with_gemini(str(prompt))
                
                # Augment the prompt
                if isinstance(prompt, str):
                    prompt = f"Based on web search results: {search_results}\n\n{prompt}"
                else:
                    prompt.insert(0, {"type": "text", "text": f"Web search results: {search_results}"})
            
            parts = self._normalize_input(prompt)
            openai_content = self._map_parts(parts)

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": openai_content})
            
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "response_format": {"type": "json_object"},  # Enable JSON mode
            }
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
                
            response = self._client.chat.completions.create(**kwargs)
            
            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            
            usage = {}
            if response.usage:
                usage = {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                }
            
            # Detect output truncation — same check as generate()
            is_truncated = (finish_reason == "length")
            
            ai_response = AIResponse(
                content=content,
                model=self.model,
                provider=self.PROVIDER_NAME,
                usage=usage,
                raw_response=response,
                truncated=is_truncated,
                finish_reason=finish_reason,
            )
            
            if is_truncated:
                raise AIOutputTruncatedError(
                    f"JSON output truncated: model hit max output token limit "
                    f"(finish_reason='length', output_tokens={usage.get('output_tokens', '?')})",
                    provider=self.PROVIDER_NAME,
                    partial_response=ai_response,
                )
            
            return ai_response
            
        except AIProviderError:
            raise  # Re-raise all our own errors (including truncation/context)
        except Exception as e:
            # Only fall back to plain generate() for JSON-mode-specific failures
            # (e.g. model doesn't support response_format). Truncation and
            # context errors are already re-raised above.
            return self.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
    
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
                    "cost_tier": "standard"
                })
            
            return models_list
        except Exception as e:
            print(f"Error listing OpenAI models: {e}")
            return []

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
