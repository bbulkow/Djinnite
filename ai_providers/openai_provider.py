"""
OpenAI Provider

Wraps the OpenAI SDK.
Supports web search via Gemini's native Google Search grounding.
"""

from typing import Optional

from .base_provider import (
    BaseAIProvider,
    AIResponse,
    AIProviderError,
    AIRateLimitError,
    AIAuthenticationError,
    AIModelNotFoundError,
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
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AIResponse:
        """
        Generate a response using OpenAI.
        """
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
                
            response = self._client.chat.completions.create(**kwargs)
            
            content = response.choices[0].message.content
            
            usage = {}
            if response.usage:
                usage = {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
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
            
            if "api_key" in error_message or "auth" in error_message:
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
        
        Args:
            query: The search query
            
        Returns:
            Search results as text
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
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        web_search: bool = False,
    ) -> AIResponse:
        """
        Generate a JSON response using OpenAI.
        
        Uses OpenAI's JSON mode for more reliable JSON output.
        When web_search=True, uses Gemini to fetch current info first,
        then passes that context to OpenAI for structured output.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system instruction
            temperature: Sampling temperature
            max_tokens: Maximum output tokens (capped at 16384 for gpt-4o)
            web_search: If True, use Gemini to search web first
        """
        # Cap max_tokens for OpenAI (gpt-4o max is 16384 completion tokens)
        if max_tokens is not None and max_tokens > 16384:
            max_tokens = 16384
        
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
                search_results = self._search_with_gemini(prompt)
                
                # Augment the prompt with search results
                augmented_prompt = f"""Based on the following current information from web search:

---SEARCH RESULTS---
{search_results}
---END SEARCH RESULTS---

Now answer the original question:
{prompt}"""
                prompt = augmented_prompt
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
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
            
            usage = {}
            if response.usage:
                usage = {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                }
            
            return AIResponse(
                content=content,
                model=self.model,
                provider=self.PROVIDER_NAME,
                usage=usage,
                raw_response=response
            )
            
        except AIProviderError:
            raise  # Re-raise our own errors
        except Exception as e:
            # Fall back to regular generation if JSON mode fails
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
            # Try to list models (lightweight check)
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
                
                # Heuristic filter for chat models
                if "gpt" not in model_id:
                    continue
                    
                context = 128000 # Default assumption for modern GPT-4
                if "gpt-3.5" in model_id:
                    context = 16000
                elif "mini" in model_id:
                    context = 256000 # gpt-4o-mini has larger context
                
                models_list.append({
                    "id": model_id,
                    "name": model_id, # OpenAI doesn't give display names usually
                    "context_window": context,
                    "capabilities": ["text", "vision"] if "vision" in model_id or "gpt-4" in model_id else ["text"],
                    "cost_tier": "standard"
                })
            
            return models_list
        except Exception as e:
            print(f"Error listing OpenAI models: {e}")
            return []
