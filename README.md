# Djinnite

**Lightweight AI abstraction layer for multiple providers (Gemini, Claude, OpenAI)**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Djinnite is a purpose-built AI abstraction layer that wraps provider SDKs directly—no heavy frameworks, no vendor lock-in, maximum control. Built for developers who need reliable, switchable AI capabilities across multiple projects.

**🔍 Breakthrough Feature: Universal Knowledge Grounding**

Escape the knowledge cutoff trap! Every AI provider has opaque training data cutoffs—Gemini knows about events until some unknown date, Claude until another, OpenAI until yet another. When you ask "What happened this week?" you get outdated information without even knowing it.

**Djinnite solves this with universal grounding/web search** across ALL providers:
- **Gemini**: Uses native search grounding 
- **Claude**: Uses native search capabilities (when available)
- **OpenAI**: Uses intelligent Gemini-powered web search (since OpenAI doesn't support this yet)
- **Single API**: Same `web_search=True` parameter works everywhere
- **Automatic fallback**: Graceful degradation when search isn't available

Your agents get **current information** regardless of which provider you use, without being "mired in the past."

Future goals may include stateful streaming, more providers.

Pull requests accepted.

NOTE: this project coded with Gemini (Pro 3), and Anthropic Claude (mostly Opus 4.5), using the CLINE vscode plugin, which is awesome.

---

## 🎯 Why Djinnite?

### **True Multi-Provider Abstraction**
Switch between **Google Gemini**, **Anthropic Claude**, and **OpenAI ChatGPT** with identical code. No provider-specific syntax to learn or maintain.

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

- **Standardized responses** with consistent token counting across providers
- **Robust error handling** with retry-friendly exception hierarchy
- **JSON generation** optimized for structured agent outputs
- **Provider fallback chains** - switch providers when one hits limits
- **Request/response logging** for debugging agent conversations

```python
# Ideal for agents: structured JSON responses
response = provider.generate_json(
    "Extract entities from this text and return as JSON",
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
python -m djinnite.scripts.update_models
# Updates: gemini-2.5-flash, claude-3-5-sonnet-20241022, gpt-4o, etc.
```

### **Effortless Provider Switching**
Change AI providers in **seconds, not hours**:

```json
// Switch from expensive to economical with one config change
{
  "default_provider": "gemini",  // Was "claude"
  "providers": {
    "gemini": { "default_model": "gemini-2.5-flash" }  // 40x cheaper than Claude
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
response = provider.generate_json(
    "What are today's top tech news headlines?",
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

### Configuration

Create `config/ai_config.json` in your project:

```json
{
  "providers": {
    "gemini": {
      "api_key": "your-gemini-api-key",
      "enabled": true,
      "default_model": "gemini-2.5-flash",
      "use_cases": {
        "coding": "gemini-2.5-flash",
        "analysis": "gemini-2.5-pro"
      }
    },
    "claude": {
      "api_key": "your-claude-api-key", 
      "enabled": true,
      "default_model": "claude-3-5-sonnet-20241022"
    }
  },
  "default_provider": "gemini"
}
```

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
response.input_tokens  # Tokens in prompt
response.output_tokens # Tokens in response
response.total_tokens  # Combined total
response.raw_response  # Original provider response
```

### Error Handling

Comprehensive exception hierarchy for robust applications:

```python
from djinnite.ai_providers import AIProviderError, AIRateLimitError, AIAuthenticationError

try:
    response = provider.generate("Hello")
except AIRateLimitError:
    # Switch to different provider or implement backoff
    pass
except AIAuthenticationError:
    # Invalid API key
    pass  
except AIProviderError as e:
    # General provider error
    print(f"Provider {e.provider} failed: {e}")
```

---

## 🔧 Available Scripts

Djinnite includes powerful utility scripts for managing AI providers:

### Validate Connectivity

Test your API keys and provider setup:

```bash
python -m djinnite.scripts.validate_ai
```

**Output:**
```
Testing Gemini provider...
✅ Gemini: Successfully connected with gemini-2.5-flash
✅ Generated 25 tokens in 1.2s

Testing Claude provider...
✅ Claude: Successfully connected with claude-3-5-sonnet-20241022  
✅ Generated 31 tokens in 0.8s
```

### Update Model Catalog

Refresh available models from provider APIs:

```bash
python -m djinnite.scripts.update_models
```

**What it does:**
- Fetches latest model lists from Gemini, Claude, OpenAI APIs
- Updates `config/model_catalog.json` with new models
- Preserves existing cost scores and capabilities
- Warns about deprecated or removed models

### Update Model Costs

AI-powered cost estimation for intelligent provider selection:

```bash
python -m djinnite.scripts.update_model_costs --dry-run
```

**Features:**
- Uses AI to estimate relative costs between providers
- Anchored to Gemini 2.5 Flash (cost_score = 1.0) 
- Calculates costs from official pricing when available
- Enables cost-aware model selection in your applications

---

## 🛠 Advanced Usage

### JSON Generation

Optimized for structured outputs and agent communication:

```python
# JSON mode with lower temperature for consistency
response = provider.generate_json(
    prompt="Extract the name, role, and skills from this resume: ...",
    system_prompt="You are a resume parsing expert. Return valid JSON.",
    temperature=0.1,  # Lower temperature for deterministic output
    max_tokens=1000
)

# Parse the structured response
import json
data = json.loads(response.content)
```

### Web Search & Grounding

Access real-time information across providers:

```python
# OpenAI with web search via Gemini
provider = get_provider("chatgpt", openai_key, "gpt-4o", gemini_api_key=gemini_key)

response = provider.generate_json(
    "What are the latest developments in quantum computing this week?",
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
│   ├── update_models.py   # Refresh model catalog
│   └── update_model_costs.py  # Cost estimation
├── prompts/               # Shared prompt templates
├── config/                # Example configurations
└── tests/                 # Test suite
```

### Provider Interface

All providers implement the same abstract interface:

```python
class BaseAIProvider(ABC):
    def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                 temperature: float = 0.7, max_tokens: Optional[int] = None) -> AIResponse
    
    def generate_json(self, prompt: str, system_prompt: Optional[str] = None,
                      temperature: float = 0.3, max_tokens: Optional[int] = None,
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

## 🔄 Model Management

### Automatic Model Discovery

Stay current with the latest AI capabilities:

```bash
# Refresh available models from provider APIs
python -m djinnite.scripts.update_models
```

Updates `config/model_catalog.json` with:
- **New models** from provider APIs (e.g., gemini-2.5-flash, claude-3-5-sonnet-20241022)
- **Context windows** and capabilities
- **Deprecation status** for model lifecycle management

### Intelligent Cost Estimation

AI-powered cost analysis for optimal provider selection:

```bash
# Estimate costs using AI
python -m djinnite.scripts.update_model_costs
```

**How it works:**
- Anchors to **Gemini 2.5 Flash** (cost_score = 1.0)
- Uses **AI analysis** of provider pricing pages
- **Calculates** from API pricing data when available
- **Estimates** relative costs between providers for cost-aware switching

Example cost scores:
```json
{
  "gemini-2.5-flash": 1.0,      // Anchor (very economical)
  "claude-3-5-sonnet": 40.0,    // 40x more expensive
  "gpt-4o": 15.0                 // 15x more expensive
}
```

### Model Validation

Prevent broken deployments with model validation:

```python
# Automatic disabled model checking
provider = get_provider("gemini", api_key, "deprecated-model")
# Raises AIProviderError: Model 'deprecated-model' is disabled: model deprecated as of 2024-01-15
```

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
| **Model Discovery** | 🤖 AI-Powered | ❌ Manual | ❌ Manual | ❌ Manual |
| **Cost Tracking** | 📊 AI-Estimated | ❌ Manual | ❌ Manual | ❌ Manual |
| **Web Search** | 🌐 Unified API | 🌐 Various Tools | ❌ No | 🔧 Provider-specific |
| **Error Handling** | 🛡️ Standardized | 🛡️ Standardized | 🛡️ Standardized | 🔧 Provider-specific |
| **Agentic Features** | 🤖 Built-in | 🤖 Extensive | ❌ Basic | 🔧 Manual |
| **Submodule Safe** | ✅ Designed for it | ❌ Version Hell | ❌ Dependency Conflicts | ✅ If Managed |

### Why Not LangChain?

- **Complexity**: LangChain is a full framework with hundreds of dependencies
- **Performance**: Multiple abstraction layers slow down simple AI calls  
- **Overkill**: Most projects just need basic text generation with provider switching

### Why Not LiteLLM?

- **Static Model Lists**: Requires manual updates for new models
- **No Intelligence**: No cost estimation or automatic model discovery
- **Limited Features**: Basic abstraction without advanced capabilities

### Why Not Direct SDKs?

- **Vendor Lock-in**: Code tied to specific provider APIs
- **Maintenance**: Need to update code when switching providers
- **Inconsistency**: Different error handling and response formats per provider

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
- Thanks to Google, Anthropic, and OpenAI for excellent AI APIs

---

## 🔗 Related Projects

- **[EventFinder](https://github.com/bbulkow/EventFinder)** - Cultural event discovery using AI (primary consumer)
- **More coming soon** - Djinnite is designed to power multiple AI-driven applications

---

*Djinnite: Because your AI shouldn't be tied to a single provider.* ⚡