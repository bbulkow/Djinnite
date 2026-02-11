"""
Google Gemini AI Provider

Wraps the Google Gen AI SDK (google-genai) for Gemini models.
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


class GeminiProvider(BaseAIProvider):
    """
    Google Gemini AI provider implementation.
    
    Uses the google-genai SDK.
    """
    
    PROVIDER_NAME = "gemini"
    
    def _initialize_client(self) -> None:
        """Initialize the Gemini client."""
        try:
            from google import genai
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
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AIResponse:
        """
        Generate a response using Gemini.
        
        Args:
            prompt: The user prompt/message
            system_prompt: Optional system instruction
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            AIResponse with the generated content
        """
        try:
            # Build configuration
            config = {
                "temperature": temperature,
            }
            if max_tokens:
                config["max_output_tokens"] = max_tokens
            if system_prompt:
                config["system_instruction"] = system_prompt
            
            # Generate response
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config
            )
            
            # Extract usage info if available
            usage = {}
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                metadata = response.usage_metadata
                usage = {
                    "input_tokens": getattr(metadata, 'prompt_token_count', 0),
                    "output_tokens": getattr(metadata, 'candidates_token_count', 0),
                }
            
            return AIResponse(
                content=response.text,
                model=self.model,
                provider=self.PROVIDER_NAME,
                usage=usage,
                raw_response=response
            )
            
        except Exception as e:
            error_message = str(e).lower()
            
            # Check for specific error types
            if "api_key" in error_message or "authentication" in error_message:
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
    
    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        web_search: bool = False,
    ) -> AIResponse:
        """
        Generate a JSON response using Gemini.
        
        Uses Gemini's JSON mode for more reliable JSON output.
        NOTE: Gemini does NOT support JSON mode + web_search together.
        When web_search=True, we skip JSON mode and rely on the prompt for JSON output.
        
        Args:
            web_search: If True, enable Google Search grounding for current info
        """
        try:
            from google.genai import types
            
            # Build configuration
            config = {
                "temperature": temperature,
            }
            
            # IMPORTANT: Gemini does NOT support response_mime_type with tools!
            # Error: "Tool use with a response mime type: 'application/json' is unsupported"
            # When web_search is enabled, skip JSON mode - prompt instructs JSON output
            if not web_search:
                config["response_mime_type"] = "application/json"
            
            if max_tokens:
                config["max_output_tokens"] = max_tokens
            if system_prompt:
                config["system_instruction"] = system_prompt
            
            # Enable Google Search grounding for current information
            if web_search:
                config["tools"] = [types.Tool(google_search=types.GoogleSearch())]
            
            # Generate response
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config
            )
            
            # Extract text from candidates directly instead of using response.text
            # This ensures we get the complete response
            text_content = ""
            if response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    text_content = "".join(
                        part.text for part in candidate.content.parts 
                        if hasattr(part, 'text')
                    )
            
            # Fall back to response.text if candidates extraction failed
            if not text_content and hasattr(response, 'text'):
                text_content = response.text
            
            # Extract usage info if available
            usage = {}
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                metadata = response.usage_metadata
                usage = {
                    "input_tokens": getattr(metadata, 'prompt_token_count', 0),
                    "output_tokens": getattr(metadata, 'candidates_token_count', 0),
                }
            
            return AIResponse(
                content=text_content,
                model=self.model,
                provider=self.PROVIDER_NAME,
                usage=usage,
                raw_response=response
            )
            
        except Exception as e:
            # If web_search was requested and failed, RAISE the error - don't silently degrade
            if web_search:
                raise AIProviderError(
                    f"Web search generation failed: {e}",
                    provider=self.PROVIDER_NAME,
                    original_error=e
                )
            # Only fall back if web_search was NOT requested
            return self.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
    
    def is_available(self) -> bool:
        """Check if Gemini is available and configured."""
        if not self.api_key:
            return False
        
        try:
            # Try to list models as a connectivity check
            # We iterate properly since it might return a generator
            models = self._client.models.list(config={"page_size": 1})
            # Just trying to access it is enough
            return True
        except Exception:
            return False

    def list_models(self) -> list[dict]:
        """List available models from Gemini."""
        if not self.api_key:
            return []
            
        try:
            models_list = []
            pager = self._client.models.list()
            
            for model in pager:
                name = getattr(model, "name", "")
                
                # Filter for Gemini models, exclude tuning/embedding models if desired
                # Usually we just want 'generateContent' supported models
                # But checking name for 'gemini' is a good heuristic
                if "gemini" not in name.lower():
                    continue
                
                # Extract ID from resource name (e.g. models/gemini-1.5-pro -> gemini-1.5-pro)
                model_id = name.split("/")[-1] if "/" in name else name
                
                # Determine capabilities
                capabilities = ["text"]
                if "vision" in model_id.lower() or "gemini" in model_id.lower():
                    capabilities.append("vision")
                
                models_list.append({
                    "id": model_id,
                    "name": getattr(model, "display_name", model_id),
                    "context_window": getattr(model, "input_token_limit", 0),
                    "capabilities": capabilities,
                    "cost_tier": "standard"
                })
            
            return models_list
        except Exception as e:
            print(f"Error listing Gemini models: {e}")
            return []
