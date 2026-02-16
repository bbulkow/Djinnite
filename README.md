# Djinnite

**Lightweight AI abstraction layer for multiple providers (Gemini, Claude, OpenAI)**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Djinnite is a purpose-built AI abstraction layer that wraps provider SDKs directly—no heavy frameworks, no vendor lock-in, maximum control. Built for developers who need reliable, switchable AI capabilities across multiple projects.

**🔍 Breakthrough Feature: Unified Multimodality**

Stop juggling provider-specific formats like Gemini `parts` vs. OpenAI `content` blocks. Djinnite standardizes multimodal interaction into a single, normative schema.

- **Unified Input**: Pass images, audio, video, or text using a simple list-of-dicts.
- **Interleaved Output**: Receive text plus non-text parts (like cropped images or audio) in a consistent format.
- **Safety First**: Global `modality_policy` allows you to disable high-cost modalities (like video) across your entire organization.
- **Automatic Conversion**: Plain strings are auto-converted to multimodal parts—your existing code just works.

```python
# Multimodal: Mix text and images with any supporting provider
prompt = [
    {"type": "text", "text": "What is in this receipt?"},
    {"type": "image", "image_data": open("receipt.jpg", "rb").read(), "mime_type": "image/jpeg"}
]
response = provider.generate(prompt)
```

**🔍 Breakthrough Feature: Universal Knowledge Grounding**

Escape the knowledge cutoff trap! Every AI provider has opaque training data cutoffs. Djinnite solves this with **universal grounding/web search** across ALL providers:
- **Gemini**: Uses native search grounding 
- **Claude**: Uses native search capabilities (when available)
- **OpenAI**: Uses intelligent Gemini-powered web search (since OpenAI doesn't support this yet)
- **Single API**: Same `web_search=True` parameter works everywhere
- **Automatic fallback**: Graceful degradation when search isn't available

**Enterprise-Ready Google Integration**
Djinnite supports both paths for Google Gemini:
- **Google AI Studio**: Fast, free-tier friendly setup for developers.
- **Vertex AI (Google Cloud)**: Secure, production-ready infrastructure for enterprises.

Your agents get **current information** regardless of which provider you use, without being "mired in the past."

Future goals may include:
- **Automatic Training Horizon Detection**: Systematic discovery of training data cutoffs by querying each model directly.
- **Enhanced Web Search Capability Detection**: Better heuristics for identifying models with native search capabilities (e.g., newer OpenAI search-specific models).
- **Stateful Streaming**: High-performance streaming for long-form content.

Pull requests accepted.

NOTE: this project coded with Gemini (Pro 3), and Anthropic Claude (mostly Opus 4.5), using the CLINE vscode plugin, which is awesome.

---

## 🎯 Why Djinnite?

### **True Multi-Provider Orchestration**
Use **Google Gemini**, **Anthropic Claude**, and **OpenAI ChatGPT** side-by-side with identical code. No provider-specific syntax to learn or maintain, allowing you to mix and match the best models for each specific task in your application.

```python
# Same code works with any provider
from djinnite.ai_providers import get_provider

provider = get_provider("gemini", api_key="...", model="gemini-2.5-flash")
response = provider.generate("Explain quantum computing")

provider = get_provider("claude", api_key="...", model="claude-3-5-sonnet-20241022") 
response = provider.generate("Explain quantum computing")  # Identical API
```

### **Perfect for Agentic Development**
Designed from day one for **AI agents and automated systems**:

- **Unified Multimodality** - standardized vision, audio, and video support
- **Standardized responses** with consistent token counting across providers
- **Robust error handling** with retry-friendly exception hierarchy
- **JSON generation** optimized for structured agent outputs
- **Provider fallback chains** - switch providers when one hits limits
- **Request/response logging** for debugging agent conversations

```python
# Ideal for agents: structured JSON with Guaranteed Schema Enforcement
schema = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"}
                },
                "required": ["name", "type"],
                "additionalProperties": False
            }
        }
    },
    "required": ["entities"],
    "additionalProperties": False
}
response = provider.generate_json(
    "Extract entities from this text and return as JSON",
    schema=schema,
    temperature=0.1  # Low temperature for consistent agent behavior
)
```

### **Always Up-to-Date Models**
Never fall behind on the latest AI capabilities:

- **Automatic model discovery** - refresh available models from provider APIs
- **AI-powered cost estimation** - intelligent cost scoring across providers  
- **Model deprecation tracking** - get warnings before models disappear
- **Beta model detection** - early access to experimental capabilities

```bash
# Stay current with one command
uv run python -m djinnite.scripts.update_models
# Updates: gemini-2.5-flash, claude-3-5-sonnet-20241022, gpt-4o, etc.
```

### **Optimized Multi-Model Orchestration**
Don't just switch providers—**use them in parallel**. Djinnite lets you orchestrate multiple models across different providers simultaneously, choosing the best tool for every specific task:

- **Claude 3.5 Sonnet** for complex reasoning and coding
- **Gemini 2.5 Flash** for high-speed, high-volume extraction
- **GPT-4o** for specialized creative tasks
- **Unified Interface**: Use them all together in the same application without juggling multiple SDKs or different response formats.

```json
// Configure multiple providers for simultaneous use
{
  "default_provider": "gemini",
  "providers": {
    "gemini": { 
      "default_model": "gemini-2.5-flash",
      "use_cases": { "extraction": "gemini-2.5-flash" } 
    },
    "claude": { 
      "default_model": "claude-3-5-sonnet-20241022",
      "use_cases": { "reasoning": "claude-3-5-sonnet-20241022" }
    }
  }
}
```

### **Universal Grounding & Web Search**
Get current information across all providers:

- **Web search abstraction** - works with OpenAI and Gemini (more coming)
- **Consistent grounding interface** - same API regardless of how each provider implements real-time data
- **Fallback strategies** - graceful degradation when web search isn't available

```python
# Web search works across providers that support it
headlines_schema = {
    "type": "object",
    "properties": {
        "headlines": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["headlines"],
    "additionalProperties": False
}
response = provider.generate_json(
    "What are today's top tech news headlines?",
    schema=headlines_schema,
    web_search=True  # Automatic provider-specific implementation
)
```

### **Enterprise-Grade Credential Management**
Secure, scalable API key management:

- **Host project configuration** - keeps secrets in your main project
- **Multiple API key support** - different keys for different use cases
- **Safe submodule sharing** - no secrets embedded in shared code
- **Provider-specific settings** - custom rate limits, model preferences per provider

### **Submodule-First Architecture**
Built to be **shared across multiple projects** without conflicts:

- **Git submodule optimized** - stable API guarantees across versions
- **Zero dependency conflicts** - only wraps official provider SDKs
- **Breaking change protection** - explicit compatibility guarantees
- **Project isolation** - each project maintains its own config and model preferences

---

## 🚀 Quick Start

### Installation

To use Djinnite, it is recommended to install it in editable mode so that the `djinnite` package is available in your Python environment:

```bash
# As a git submodule (recommended for sharing across projects)
git submodule add https://github.com/bbulkow/Djinnite.git djinnite
pip install -e djinnite/

# Or as a standalone package
git clone https://github.com/bbulkow/Djinnite.git
cd Djinnite
pip install -e .
```

### Basic Usage

```python
from djinnite.ai_providers import get_provider
from djinnite.config_loader import load_ai_config

# Load configuration
config = load_ai_config()
provider_name, model = config.get_model_for_use_case("general")

# Create provider
provider = get_provider(provider_name, config.providers[provider_name].api_key, model)

# Generate response
response = provider.generate(
    prompt="Write a Python function to calculate fibonacci numbers",
    temperature=0.7
)

print(f"Model: {response.model}")
print(f"Content: {response.content}")
print(f"Tokens: {response.total_tokens}")
```

### Multimodal Usage

```python
# Pass a list of parts for multimodal interaction
prompt = [
    {"type": "text", "text": "What color is the object in this image?"},
    {"type": "image", "image_data": open("image.png", "rb").read(), "mime_type": "image/png"}
]

response = provider.generate(prompt)
print(response.content)

# Access interleaved output parts if returned by the model
for part in response.parts:
    if part["type"] == "text":
        print(f"Text: {part['text']}")
    elif part["type"] == "inline_data":
        print(f"Received data: {part['mime_type']}")
```

### Configuration & Maintenance

Djinnite requires a `config/ai_config.json` file in your project root to manage API keys and model preferences.

For detailed instructions on setup, maintenance scripts, and integration, see **[USE.md](USE.md)**.

---

## 📖 Core Concepts

### Providers

Each AI provider (Gemini, Claude, OpenAI) is wrapped in a standardized interface:

```python
from djinnite.ai_providers import get_provider, list_available_providers

# See what's available
print(list_available_providers())  # ['gemini', 'claude', 'chatgpt']

# Create any provider with identical interface
provider = get_provider("gemini", "your-api-key", "gemini-2.5-flash")
```

### Responses

All providers return the same `AIResponse` structure:

```python
response = provider.generate("Hello, world!")

# Standardized across all providers
response.content        # Generated text
response.model         # Actual model used
response.provider      # Provider name
response.usage         # Token usage dict
response.parts         # Multimodal output parts (interleaved)
response.input_tokens  # Tokens in prompt
response.output_tokens # Tokens in response
response.total_tokens  # Combined total
response.raw_response  # Original provider response
response.truncated     # True if output was cut short by token limit
response.finish_reason # Provider-native stop reason (e.g. "stop", "length", "max_tokens")
```

### Multimodal Schema

Djinnite uses a standardized "Part" schema for all multimodal inputs:

```python
# Standard Input Parts
[
    {"type": "text", "text": "Describe this audio and image."},
    {"type": "image", "image_data": b"...", "mime_type": "image/jpeg"},
    {"type": "audio", "file_uri": "gs://...", "mime_type": "audio/mp3"},
    {"type": "video", "file_uri": "https://...", "mime_type": "video/mp4"}
]
```

### Error Handling

Comprehensive exception hierarchy for robust applications. Djinnite **never silently returns partial data** — if the model output is truncated or the context is too long, you get a specific exception.

```python
from djinnite import (
    AIProviderError,          # Base class for all provider errors
    AIOutputTruncatedError,   # Output hit max token limit (HTTP 200 with partial content!)
    AIContextLengthError,     # Input too long for model (HTTP 400)
    AIRateLimitError,         # Rate limit exceeded (HTTP 429)
    AIAuthenticationError,    # Bad API key (HTTP 401)
    AIModelNotFoundError,     # Model doesn't exist (HTTP 404)
    DjinniteModalityError,    # Unsupported modality (client-side check)
)

try:
    response = provider.generate(prompt, max_tokens=500)
except AIOutputTruncatedError as e:
    # CRITICAL: The model's output was cut short by the token limit.
    # The API returned HTTP 200 but the response is incomplete!
    # The partial content is available for inspection:
    print(f"Truncated! Got {e.partial_response.output_tokens} tokens")
    print(f"Partial content: {e.partial_response.content[:100]}...")
    # Retry with higher max_tokens, or raise to the caller
except AIContextLengthError as e:
    # The input prompt was too long for the model's context window.
    # The API returned HTTP 400. Shorten the prompt or use a bigger model.
    print(f"Prompt too long: {e}")
except DjinniteModalityError as e:
    # Model doesn't support one of the requested modalities (e.g. video)
    print(f"Unsupported: {e.requested_modalities}")
except AIRateLimitError:
    # Switch to different provider or implement backoff
    pass
except AIProviderError as e:
    # General provider error (catches all the above too)
    print(f"Provider {e.provider} failed: {e}")
```

---


## 🛠 Advanced Usage

### JSON Generation (Schema-Enforced)

**Strict Constraint Decoding** — guaranteed structure, not "best-effort" JSON:

```python
# Define the schema (dict or Pydantic BaseModel)
resume_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "role": {"type": "string"},
        "skills": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["name", "role", "skills"],
    "additionalProperties": False
}

# generate_json() uses provider-native Constraint Decoding
response = provider.generate_json(
    prompt="Extract the name, role, and skills from this resume: ...",
    schema=resume_schema,
    system_prompt="You are a resume parsing expert.",
    temperature=0.1,  # Lower temperature for deterministic output
    max_tokens=1000
)

# Parse the guaranteed-structure response
import json
data = json.loads(response.content)
# data is guaranteed to have "name", "role", "skills" keys
```

Or use a **Pydantic BaseModel** for type-safe schema definition:

```python
from pydantic import BaseModel

class ResumeData(BaseModel):
    name: str
    role: str
    skills: list[str]

response = provider.generate_json(
    prompt="Extract the name, role, and skills from this resume: ...",
    schema=ResumeData,
)
data = ResumeData.model_validate_json(response.content)
```

### Web Search & Grounding

Access real-time information across providers:

```python
# OpenAI with web search via Gemini
provider = get_provider("chatgpt", openai_key, "gpt-4o", gemini_api_key=gemini_key)

news_schema = {
    "type": "object",
    "properties": {
        "headlines": {
            "type": "array",
            "items": {"type": "object", "properties": {"title": {"type": "string"}, "summary": {"type": "string"}}, "required": ["title", "summary"], "additionalProperties": False}
        }
    },
    "required": ["headlines"],
    "additionalProperties": False
}
response = provider.generate_json(
    "What are the latest developments in quantum computing this week?",
    schema=news_schema,
    web_search=True  # Enables real-time information
)
```

### Use Case-Specific Models

Configure different models for different purposes:

```json
{
  "providers": {
    "gemini": {
      "api_key": "...",
      "default_model": "gemini-2.5-flash", 
      "use_cases": {
        "coding": "gemini-2.5-flash",      // Fast for code
        "analysis": "gemini-2.5-pro",      // Deep for analysis  
        "creative": "gemini-2.5-pro",      // Creative for writing
        "cheap": "gemini-2.5-flash"        // Economical for bulk
      }
    }
  }
}
```

```python
# Automatic model selection by use case
config = load_ai_config()
provider_name, model = config.get_model_for_use_case("coding")
provider = get_provider(provider_name, config.providers[provider_name].api_key, model)
```

### Provider Fallback Chains

Implement robust fallback for production systems:

```python
def generate_with_fallback(prompt, providers=["gemini", "claude", "chatgpt"]):
    """Try multiple providers until one succeeds."""
    config = load_ai_config()
    
    for provider_name in providers:
        try:
            provider_config = config.get_provider(provider_name)
            if not provider_config:
                continue
                
            provider = get_provider(
                provider_name, 
                provider_config.api_key, 
                provider_config.default_model
            )
            return provider.generate(prompt)
            
        except AIRateLimitError:
            continue  # Try next provider
        except AIProviderError:
            continue  # Try next provider
    
    raise Exception("All providers failed")
```

### Request/Response Logging

Debug and monitor AI interactions:

```python
from djinnite.llm_logger import LLMLogger

logger = LLMLogger()

# Log request
request_id = logger.log_request(
    prompt="Hello world",
    system_prompt=None,
    model="gemini-2.5-flash", 
    provider="gemini"
)

# ... make request ...

# Log response
logger.log_response(
    request_id=request_id,
    response_content="Hello! How can I help?",
    success=True,
    usage={"input_tokens": 2, "output_tokens": 6}
)
```

---

## 🏗 Architecture

### Design Philosophy

**Direct SDK Wrapping**: No heavy frameworks like LangChain or LiteLLM. Each provider implementation directly wraps the official SDK for maximum performance and feature access.

**Submodule-First**: Built to be shared as a git submodule across multiple projects. Strict API compatibility guarantees prevent breaking changes from affecting downstream projects.

**Configuration Convention**: Host projects maintain their own `config/ai_config.json` and `config/model_catalog.json`. Djinnite automatically discovers the host project's config directory.

### Project Structure

```
djinnite/
├── ai_providers/           # Provider implementations
│   ├── base_provider.py    # Abstract interface + AIResponse
│   ├── gemini_provider.py  # Google Gemini wrapper  
│   ├── claude_provider.py  # Anthropic Claude wrapper
│   └── openai_provider.py  # OpenAI ChatGPT wrapper
├── config_loader.py        # Configuration management
├── llm_logger.py          # Request/response logging
├── scripts/               # Utility commands
│   ├── validate_ai.py     # Test connectivity
│   ├── validate_models.py # Comprehensive modality test
│   ├── update_models.py   # Refresh model catalog
│   └── update_model_costs.py  # Cost estimation
├── prompts/               # Shared prompt templates
├── config/                # Example configurations
└── tests/                 # Test suite
```

### 🗓️ Future Roadmap (Agentic Tasks)

- [ ] **Agentic Codex / Artifact Interface**: Support OpenAI Codex and similar agentic models that use artifact-based output systems rather than pure streaming. These models (gpt-5-codex, gpt-5.1-codex, etc.) require a different interaction paradigm than chat completions.
- [ ] **Thinking Abstraction**: Unified `thinking` parameter across providers (Claude `budget`, OpenAI `reasoning_effort`, Gemini `thinking_mode`). The model catalog already records `capabilities.thinking` and `capabilities.thinking_style` — the provider abstraction layer needs to map the unified parameter to provider-native APIs.
- [ ] **Temperature-Aware Generation**: Use `capabilities.temperature` from the catalog to automatically omit temperature for reasoning models that reject it, instead of forcing callers to handle the error.
- [ ] **Modality-Aware Web Search Discovery**: Refine `discover_modalities` to identify models with native "tools" for web search.
- [ ] **Training Horizon Probe**: Implement a script to automatically verify training data cutoffs for all models in the catalog.

### Provider Interface

All providers implement the same abstract interface:

```python
class BaseAIProvider(ABC):
    def generate(self, prompt: Union[str, List[Dict]], system_prompt: Optional[str] = None, 
                 temperature: float = 0.7, max_tokens: Optional[int] = None) -> AIResponse
    
    def generate_json(self, prompt: Union[str, List[Dict]], schema: Union[Dict, Type],
                      system_prompt: Optional[str] = None, temperature: float = 0.3,
                      max_tokens: Optional[int] = None,
                      web_search: bool = False) -> AIResponse
    
    def is_available(self) -> bool
    def list_models(self) -> list[dict]
```

---

## 🔐 Security & Configuration

### Credential Management

Djinnite follows secure configuration patterns:

- **API keys** stored in host project's `config/ai_config.json` (not in git)
- **Example config** provided in `djinnite/config/ai_config.example.json`
- **Environment variable** fallback support
- **Per-provider** key isolation

### Configuration Discovery

Djinnite automatically finds your project's config:

```python
# Project structure:
my-project/
├── config/
│   └── ai_config.json     # Djinnite finds this automatically
└── djinnite/              # Git submodule
    └── ...
```

The `config_loader.py` uses `PROJECT_ROOT = Path(__file__).parent.parent` to locate the host project's config directory.

### Multi-Project Safety

When used as a git submodule:

- **No shared secrets** - each project has its own API keys
- **API compatibility** - strict versioning prevents breaking changes
- **Configuration isolation** - projects can't interfere with each other

---


## 🤝 Contributing

### Adding a New Provider

1. **Create provider implementation**:
   ```python
   # djinnite/ai_providers/new_provider.py
   from .base_provider import BaseAIProvider
   
   class NewProvider(BaseAIProvider):
       PROVIDER_NAME = "new_provider"
       # Implement abstract methods...
   ```

2. **Register in factory**:
   ```python
   # djinnite/ai_providers/__init__.py
   PROVIDERS = {
       "gemini": GeminiProvider,
       "claude": ClaudeProvider,
       "chatgpt": OpenAIProvider,
       "new_provider": NewProvider,  # Add here
   }
   ```

3. **Add dependencies**:
   ```toml
   # pyproject.toml
   dependencies = [
       "google-genai>=1.0.0",
       "anthropic>=0.8.0", 
       "openai>=1.6.0",
       "new-provider-sdk>=1.0.0",  # Add here
   ]
   ```

### API Compatibility

⚠️ **Djinnite is used as a git submodule across multiple projects.** Breaking changes affect all consumers.

**Safe changes:**
- ✅ Adding new functions, classes, or optional parameters
- ✅ Adding new providers
- ✅ Bug fixes that don't change behavior

**Requires coordination:**
- ⚠️ Changing function signatures or return types
- ⚠️ Modifying existing behavior

**Never do without approval:**
- 🚫 Renaming or removing existing functions
- 🚫 Changing required parameters
- 🚫 Moving modules (breaks imports)

See [DEVELOPMENT.md](DEVELOPMENT.md) for complete guidelines.

---

## 📋 Comparison with Alternatives

| Feature | Djinnite | LangChain | LiteLLM | Direct SDKs |
|---------|----------|-----------|---------|-------------|
| **Setup Complexity** | ⚡ Simple | 📚 Complex | 🔧 Moderate | ⚡ Simple |
| **Provider Switching** | 🔄 Instant | 🔄 Instant | 🔄 Instant | 💼 Rewrite Code |
| **Dependencies** | 📦 Minimal | 📦 Heavy | 📦 Moderate | 📦 Provider-specific |
| **Performance** | 🚀 Native SDK | 🐌 Abstraction Overhead | 🚀 Native SDK | 🚀 Native SDK |
| **Multimodality** | 🖼️ Unified | 📚 Complex | ❌ Basic | 🔧 Provider-specific |
| **Model Discovery** | 🤖 AI-Powered | ❌ Manual | ❌ Manual | ❌ Manual |
| **Cost Tracking** | 📊 AI-Estimated | ❌ Manual | ❌ Manual | ❌ Manual |
| **Web Search** | 🌐 Unified API | 🌐 Various Tools | ❌ No | 🔧 Provider-specific |
| **Error Handling** | 🛡️ Standardized | 🛡️ Standardized | 🛡️ Standardized | 🔧 Provider-specific |
| **Agentic Features** | 🤖 Built-in | 🤖 Extensive | ❌ Basic | 🔧 Manual |
| **Submodule Safe** | ✅ Designed for it | ❌ Version Hell | ❌ Dependency Conflicts | ✅ If Managed |

---

## 📊 Use Cases

### **AI Agents & Automation**
- **Structured JSON responses** for agent communication
- **Robust error handling** for production reliability  
- **Token usage tracking** for cost monitoring
- **Provider fallback chains** for high availability

### **Cost-Sensitive Applications**
- **AI-powered cost estimation** for intelligent provider selection
- **Real-time cost tracking** with token usage monitoring
- **Automatic model switching** based on cost/performance requirements

### **Multi-Project Organizations**
- **Git submodule sharing** across projects
- **Centralized provider management** with project-specific configs
- **API compatibility guarantees** prevent breaking downstream projects

### **Rapid Prototyping**
- **Instant provider switching** to test different AI capabilities  
- **Minimal setup** - just API keys in JSON config
- **Latest models** automatically available through model discovery

### **Production Systems**
- **Provider redundancy** with automatic fallback
- **Request/response logging** for debugging and monitoring
- **Model validation** prevents deployment with deprecated models

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built for the [EventFinder](https://github.com/bbulkow/EventFinder) project
- Inspired by the need for reliable, switchable AI across multiple applications
- And to use AI to choose the AI models that both perform best and are lowest cost
- Thanks to Google, Anthropic, and OpenAI for excellent AI APIs
- But more importantly to Gemini and Claude coding agents without which I wouldn't have bothered with this layer

---

## 🔗 Related Projects

- **[EventFinder](https://github.com/bbulkow/EventFinder)** - Cultural event discovery using AI (primary consumer)
- **More coming soon** - Djinnite is designed to power multiple AI-driven applications

---

*Djinnite: Because your AI shouldn't be tied to a single provider.* ⚡
