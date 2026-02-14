"""
Google Gemini AI Provider

Wraps the Google Gen AI SDK (google-genai) for Gemini models.
"""

from typing import Optional, Union, List, Dict

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
    
    Supports both Google AI Studio (backend="gemini") and Vertex AI (backend="vertexai").
    Uses the google-genai SDK.
    """
    
    PROVIDER_NAME = "gemini"
    
    def __init__(self, api_key: str, model: str, backend: str = "gemini", project_id: Optional[str] = None):
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
        super().__init__(api_key, model)

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

    def generate(
        self,
        prompt: Union[str, List[Dict]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AIResponse:
        """
        Generate a response using Gemini.
        """
        try:
            parts = self._normalize_input(prompt)
            gemini_parts = self._map_parts(parts)
            
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
                contents=gemini_parts,
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
            
            # Extract parts and text
            output_parts = []
            text_content = ""
            if response.candidates:
                candidate = response.candidates[0]
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

            return AIResponse(
                content=text_content,
                model=self.model,
                provider=self.PROVIDER_NAME,
                usage=usage,
                parts=output_parts,
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
        prompt: Union[str, List[Dict]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        web_search: bool = False,
    ) -> AIResponse:
        """
        Generate a JSON response using Gemini.
        """
        try:
            from google.genai import types
            
            parts = self._normalize_input(prompt)
            gemini_parts = self._map_parts(parts)

            # Build configuration
            config = {
                "temperature": temperature,
            }
            
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
                contents=gemini_parts,
                config=config
            )
            
            # Extract text from candidates
            text_content = ""
            output_parts = []
            if response.candidates:
                candidate = response.candidates[0]
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
                usage = {
                    "input_tokens": getattr(metadata, 'prompt_token_count', 0),
                    "output_tokens": getattr(metadata, 'candidates_token_count', 0),
                }
            
            return AIResponse(
                content=text_content,
                model=self.model,
                provider=self.PROVIDER_NAME,
                usage=usage,
                parts=output_parts,
                raw_response=response
            )
            
        except Exception as e:
            if web_search:
                raise AIProviderError(
                    f"Web search generation failed: {e}",
                    provider=self.PROVIDER_NAME,
                    original_error=e
                )
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
            self._client.models.list(config={"page_size": 1})
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
                if "gemini" not in name.lower():
                    continue
                
                model_id = name.split("/")[-1] if "/" in name else name
                
                # Determine capabilities
                modalities = ["text"]
                if any(x in model_id.lower() for x in ["vision", "flash", "pro"]):
                    modalities.extend(["vision", "audio", "video"])
                
                models_list.append({
                    "id": model_id,
                    "name": getattr(model, "display_name", model_id),
                    "context_window": getattr(model, "input_token_limit", 0),
                    "modalities": modalities,
                    "cost_tier": "standard"
                })
            
            return models_list
        except Exception as e:
            print(f"Error listing Gemini models: {e}")
            return []

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
