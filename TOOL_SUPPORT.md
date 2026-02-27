# Unified Tool Calling — Feature Proposal

> **Status:** Proposal  
> **Author:** Djinnite maintainers  
> **Date:** February 2026

## Summary

Add a **unified tool calling abstraction** to Djinnite: a cross-provider interface
that lets callers define tools once and have the model invoke them transparently,
regardless of which AI provider is being used. Djinnite handles the entire
tool-use loop — model requests a tool call, Djinnite executes it, feeds the
result back via the provider-specific API, and repeats until the model produces
a final answer.

Tools can be **Python callables** (in-process functions) or **MCP servers**
(Model Context Protocol, over transport). Both use the same definition schema;
they are parallel execution backends.

---

## Motivation

Djinnite already unifies multimodality, structured JSON, thinking, and web search
across providers. But the most important primitive for agentic AI — **tool
calling** — still requires provider-specific code.

Every major provider supports tool/function calling, but with different formats:

| Concern | OpenAI | Claude | Gemini |
|---|---|---|---|
| Tool definition | `tools=[{"type":"function", ...}]` | `tools=[{"name":..., "input_schema":...}]` | `Tool(function_declarations=[...])` |
| Tool call in response | `tool_calls` output items | `tool_use` content blocks | `function_call` Parts |
| Feeding results back | Tool result input items | `tool_result` content blocks | `function_response` Parts |
| Stop reason | `"incomplete"` with tool call | `"tool_use"` | `"STOP"` with function call |

A caller building an agent today must handle all of these. Djinnite should
abstract this away — same pattern as everything else in the library.

---

## Design Principles

1. **Transparent loop.** The caller provides tools and a prompt. Djinnite
   returns the **final answer**. All intermediate tool-call rounds are handled
   internally, like `web_search=True` works today.

2. **Definition ≠ Execution.** Tool *definitions* (name, description, schema)
   use a single format (MCP-compatible). Tool *execution* is pluggable:
   Python callable or MCP transport, as parallel backends.

3. **Djinnite stays a library.** Djinnite does not manage MCP server lifecycle,
   does not impose an agent framework, and does not own business logic.
   The caller provides tools; Djinnite orchestrates the model interaction.

4. **Provider-native tools remain separate.** Provider-specific capabilities
   like `web_search`, `code_interpreter`, etc. are not user-defined tools.
   They coexist alongside user tools but are toggled by dedicated flags.

---

## Caller Interface

### Defining Tools

Tools are dictionaries using the MCP-compatible schema. Each tool has a
**definition** (sent to the model) and an **executor** (how to run it).

Python callables and MCP servers can be mixed freely in the same tool list:

```python
def get_weather(city: str) -> str:
    """Fetch current weather for a city."""
    resp = requests.get(f"https://api.weather.com/v1/{city}")
    return resp.json()["summary"]

tools = [
    # Python callable — executed in-process, zero overhead
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"]
        },
        "callable": get_weather,
    },
    # MCP server — executed over transport, leverages MCP ecosystem
    {
        "name": "lookup_user",
        "description": "Look up a user record by their numeric ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "integer", "description": "The user's ID"}
            },
            "required": ["user_id"]
        },
        "mcp_server": "my-user-db",    # MCP server reference
    },
]
```

The model sees a flat list of tool definitions — it has no idea which are
Python functions and which are MCP servers. Djinnite routes each call to
the correct executor transparently.

### Using Tools in generate()

```python
response = provider.generate(
    "What's the weather where user 42 lives?",
    tools=tools,
)

# The caller gets the FINAL answer — all tool calls happened internally.
print(response.content)
# "It's 72°F and sunny in San Francisco, where user 42 (Alice) lives."
```

The same works with `generate_json()`:

```python
response = provider.generate_json(
    "What's the weather where user 42 lives?",
    schema=WeatherReport,
    tools=tools,
)
```

### Inspecting Tool Call History

The intermediate tool calls are available as metadata for debugging, logging,
and cost attribution:

```python
for call in response.tool_calls_made:
    print(f"  Called: {call['name']}({call['arguments']})")
    print(f"  Result: {call['result']}")
    print(f"  Duration: {call['duration_ms']}ms")

# Example output:
#   Called: lookup_user({"user_id": 42})
#   Result: {"name": "Alice", "city": "San Francisco"}
#   Duration: 12ms
#   Called: get_weather({"city": "San Francisco"})
#   Result: "72°F and sunny"
#   Duration: 340ms
```

### Combining Tools with Existing Features

Tools compose with all existing Djinnite features:

```python
response = provider.generate(
    "Based on today's headlines, what should user 42 wear?",
    tools=tools,
    web_search=True,       # Provider-native web search (runs alongside user tools)
    thinking=True,          # Extended reasoning
    system_prompt="You are a helpful personal assistant.",
)
```

---

## Tool Execution Backends

Tool definitions and tool execution are decoupled. The definition schema is
always the same; the executor varies.

### Python Callable (In-Process)

The simplest and lowest-latency option. The caller provides a Python function
via the `"callable"` key. Djinnite calls it directly with the model's arguments.

```python
{
    "name": "calculate",
    "description": "Evaluate a math expression.",
    "inputSchema": {
        "type": "object",
        "properties": {"expr": {"type": "string"}},
        "required": ["expr"]
    },
    "callable": lambda expr: str(eval(expr)),   # In-process Python
}
```

**Characteristics:**
- Zero transport overhead — direct function call
- Caller controls the implementation entirely
- Exceptions in the callable are caught and fed back to the model as error
  results (the model can recover gracefully)

### MCP Server (Over Transport)

For tools provided by external MCP servers, the caller specifies an
`"mcp_server"` reference instead of a `"callable"`. The host project manages
server lifecycle; Djinnite dispatches tool calls over the MCP transport.

```python
{
    "name": "query_database",
    "description": "Execute a read-only SQL query.",
    "inputSchema": {
        "type": "object",
        "properties": {"sql": {"type": "string"}},
        "required": ["sql"]
    },
    "mcp_server": "my-postgres-server",   # MCP server reference
}
```

**Characteristics:**
- Leverages the MCP ecosystem (file systems, databases, APIs, etc.)
- Server runs as a separate process (stdio/SSE transport)
- Host project is responsible for server lifecycle and configuration
- Djinnite provides a thin MCP client adapter for dispatch

### Mixed: Both in One Request

Python callables and MCP tools can coexist in the same `tools` list. The model
sees a flat list of tool definitions; Djinnite routes each call to the correct
executor.

```python
tools = [
    {"name": "get_weather", ..., "callable": get_weather},      # Python
    {"name": "query_database", ..., "mcp_server": "my-db"},     # MCP
    {"name": "send_email", ..., "callable": send_email},         # Python
]
```

---

## The Internal Loop

When the caller invokes `generate()` with tools, Djinnite runs an internal
loop that is invisible to the caller:

```
Caller: generate(prompt, tools=[...])
  │
  ▼
┌──────────────────────────────────────────────────────────────┐
│  Djinnite Tool Loop                                          │
│                                                              │
│  1. Translate tool definitions → provider-native format      │
│     • OpenAI:  tools=[{"type":"function", "function":{...}}] │
│     • Claude:  tools=[{"name":"...", "input_schema":{...}}]  │
│     • Gemini:  Tool(function_declarations=[...])             │
│                                                              │
│  2. Send prompt + tool definitions to model                  │
│     ─────────────────────────────────► Provider API           │
│                                                              │
│  3. Model responds with tool call(s)                         │
│     ◄─────────────────────────────────                       │
│     (finish_reason = tool_use / tool_calls / function_call)  │
│                                                              │
│  4. For each tool call:                                      │
│     a. Look up executor (callable or MCP)                    │
│     b. Deserialize arguments from model output               │
│     c. Execute: callable(**args) or mcp_dispatch(name, args) │
│     d. Capture result (or error if execution fails)          │
│     e. Record in tool_calls_made metadata                    │
│                                                              │
│  5. Feed ALL results back using provider-specific format     │
│     • OpenAI:  tool result input items                       │
│     • Claude:  tool_result content blocks                    │
│     • Gemini:  function_response Parts                       │
│     ─────────────────────────────────► Provider API           │
│                                                              │
│  6. Model responds — is it another tool call or final text?  │
│     ◄─────────────────────────────────                       │
│     • Tool call → go to step 4 (loop)                        │
│     • Final text → exit loop                                 │
│                                                              │
│  7. Build AIResponse with:                                   │
│     • content = final text answer                            │
│     • usage = aggregated tokens across ALL rounds            │
│     • tool_calls_made = history of all calls + results       │
│                                                              │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
Caller receives: AIResponse(content="...", tool_calls_made=[...])
```

### Sequential Tool Calling

A critical capability: the model calls tools **sequentially across rounds**,
where the result of one tool informs the next tool call. This is the most
common pattern in real agentic use.

**Example: "What's the weather where user 42 lives?"**

```
Round 1: Model → call lookup_user(user_id=42)
         Djinnite executes → {"name": "Alice", "city": "San Francisco"}
         Djinnite feeds result back to model

Round 2: Model → call get_weather(city="San Francisco")
         Djinnite executes → "72°F and sunny"
         Djinnite feeds result back to model

Round 3: Model → "It's 72°F and sunny in San Francisco, where Alice lives."
         (final text — loop exits)
```

The model could not have called `get_weather` in round 1 because it didn't
yet know which city to query — it needed the result of `lookup_user` first.
Each provider handles this multi-turn conversation differently:

| Provider | How sequential rounds work |
|---|---|
| **OpenAI** | Each round is a new `responses.create()` call using the previous response ID, or by appending tool result items to the input |
| **Claude** | Multi-turn message list: assistant message with `tool_use` block → user message with `tool_result` block → next assistant turn |
| **Gemini** | Contents list extended with the model's `function_call` Part followed by a `function_response` Part |

Djinnite manages this conversation state internally. The caller never sees
the intermediate turns.

### Parallel Tool Calls Within a Round

Some providers support multiple tool calls in a **single response** — the
model asks for several tools at once when the calls are independent:

```
Round 1: Model → call get_weather(city="SF") AND call get_weather(city="NYC")
         Djinnite executes both concurrently
         Djinnite feeds both results back in one round

Round 2: Model → "SF is 72°F, NYC is 45°F."
         (final text — loop exits)
```

OpenAI and Gemini support parallel tool calls natively. Claude currently
issues one tool call per turn. Djinnite handles both patterns transparently.

### Loop Safety

- **Max iterations**: Configurable limit (default: 10) to prevent runaway loops.
  Raises `AIProviderError` if exceeded.
- **Tool execution timeout**: Per-tool timeout for callables (default: 30s).
- **Error recovery**: If a callable raises an exception, the error message is
  sent back to the model as a tool result so it can retry or adapt. The loop
  does not abort on tool errors.

---

## Cross-Provider Translation

Djinnite translates tool definitions and tool results to/from each provider's
native format. The caller never sees provider-specific structures.

### Tool Definition Translation

**Djinnite canonical format** (what the caller provides):
```json
{
    "name": "get_weather",
    "description": "Get current weather for a city.",
    "inputSchema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
    }
}
```

**OpenAI Responses API** (what Djinnite sends):
```json
{
    "type": "function",
    "name": "get_weather",
    "description": "Get current weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": false
    },
    "strict": true
}
```

**Anthropic Claude** (what Djinnite sends):
```json
{
    "name": "get_weather",
    "description": "Get current weather for a city.",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": false
    }
}
```

**Google Gemini** (what Djinnite sends):
```python
types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="get_weather",
        description="Get current weather for a city.",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
            # No additionalProperties — Gemini rejects it
        }
    )
])
```

Note: `additionalProperties` handling follows the same per-provider rules as
`generate_json()` — added for OpenAI/Claude, stripped for Gemini. The caller
never specifies it (same contract as structured JSON schemas).

### Tool Call Response Parsing

When the model decides to call a tool, each provider returns it differently.
Djinnite normalizes all of them into:

```python
# Normalized tool call (internal, used by the loop)
{
    "id": "call_abc123",            # Provider-assigned call ID
    "name": "get_weather",          # Tool name
    "arguments": {"city": "SF"},    # Parsed arguments (dict)
}
```

### Tool Result Submission

After executing the tool, Djinnite feeds the result back in the provider's
native format:

| Provider | Format |
|---|---|
| **OpenAI** | Input item: `{"type": "function_call_output", "call_id": "...", "output": "..."}` |
| **Claude** | Message content block: `{"type": "tool_result", "tool_use_id": "...", "content": "..."}` |
| **Gemini** | Part: `types.Part.from_function_response(name="...", response={...})` |

---

## AIResponse Changes

### New Fields

```python
@dataclass
class AIResponse:
    # ... existing fields ...

    tool_calls_made: list[dict] = field(default_factory=list)
    """
    History of all tool calls executed during the generation loop.

    Each entry:
        {
            "name": str,           # Tool name
            "arguments": dict,     # Arguments the model provided
            "result": Any,         # Return value from the callable / MCP
            "error": Optional[str],# Error message if execution failed
            "duration_ms": int,    # Execution time in milliseconds
            "round": int,          # Which round of the loop (1-indexed)
        }

    Empty list if no tools were used.
    """
```

### Token Aggregation

When the tool loop runs multiple rounds, `usage` reflects the **total** across
all rounds:

```python
response.input_tokens   # Sum of input tokens across all rounds
response.output_tokens  # Sum of output tokens across all rounds
response.total_tokens   # Grand total
```

---

## Relationship to Existing web_search

The current `web_search=True` parameter uses **provider-native tools** (OpenAI
`web_search_preview`, Gemini `google_search`, Claude `web_search_20250305`).
These are provider-side capabilities — the provider handles execution
internally.

User-defined tools are different: **Djinnite handles execution**.

Both can coexist in a single request:

```python
response = provider.generate(
    "What's the weather in the city with today's biggest news story?",
    tools=[weather_tool],    # User-defined tool — Djinnite executes
    web_search=True,          # Provider-native tool — provider executes
)
```

The provider-native web search runs on the provider's side; the user-defined
tools run on Djinnite's side. The model can interleave calls to both.

---

## Catalog Integration

### Do We Need a New Capability?

**Yes.** Not all models in the catalog support tool/function calling. The
catalog currently tracks 80+ models across three providers. While the
mainstream text-generation models all support tool calling, many specialty
models do not:

**Models that will NOT support tool calling:**
- **TTS models**: `gemini-2.5-flash-preview-tts`, `gemini-2.5-pro-preview-tts`,
  `gpt-4o-mini-tts`, `gpt-4o-mini-tts-*` — audio output only
- **Embedding models**: `gemini-embedding-001` — vector output only
- **Image generation models**: `gpt-image-1`, `gpt-image-1-mini`,
  `gpt-image-1.5`, `chatgpt-image-latest` — image output only
- **Realtime/streaming models**: `gpt-4o-realtime-preview-*`,
  `gpt-realtime-*`, `gpt-realtime-mini-*` — different interaction paradigm
- **Audio-specific models**: `gpt-audio-*`, `gpt-4o-audio-preview-*` —
  specialized audio pipelines
- **Transcription models**: `gpt-4o-transcribe-*`, `gpt-4o-mini-transcribe-*`
- **Robotics models**: `gemini-robotics-er-1.5-preview`
- **Image generation (Gemini)**: `gemini-2.5-flash-image`,
  `gemini-3-pro-image-preview`, `gemini-3.1-flash-image-preview`

These models currently have `capabilities: {structured_json: null, ...}` —
all nulls — meaning they haven't been probed. Tool calling will follow the
same pattern: `null` = unknown, `true` = confirmed, `false` = confirmed not
supported.

**Models that WILL support tool calling** (core text-generation models):
- All `gemini-2.5-*`, `gemini-3-*`, `gemini-3.1-*` text models
- All `claude-*` models (Haiku, Sonnet, Opus)
- All `gpt-4o*`, `gpt-4.1*`, `gpt-5*` text models

Without a catalog capability, Djinnite would either (a) try tool calling on a
TTS model and get a confusing API error, or (b) need hardcoded model-name
heuristics. The probe-and-record approach is consistent with how `structured_json`,
`thinking`, and `temperature` are already handled.

### New Capability Field

The `ModelCapabilities` dataclass and catalog JSON gain `tool_calling`:

```python
@dataclass
class ModelCapabilities:
    structured_json: Optional[bool] = None
    temperature: Optional[bool] = None
    thinking: Optional[bool] = None
    web_search: Optional[bool] = None
    thinking_style: Optional[str] = None
    tool_calling: Optional[bool] = None    # NEW
```

```json
{
    "id": "gemini-2.5-flash",
    "capabilities": {
        "structured_json": true,
        "thinking": true,
        "web_search": true,
        "tool_calling": true
    }
}
```

### Pre-Flight Check

Like `generate_json()` checks for `structured_json` support, `generate()` with
tools checks for `tool_calling` support:

```python
if tools and not force:
    self._check_capability("tool_calling")
```

Models with `tool_calling: null` (unknown) are allowed through — the call
may succeed or fail at the API level. Models with `tool_calling: false` are
rejected before making the API call, with a clear error message.

### Probe

A new `probe_tool_calling()` method on each provider, following the same
pattern as `probe_structured_json()` and `probe_thinking()`:

```python
def probe_tool_calling(self) -> Optional[bool]:
    """
    Probe whether this model supports function/tool calling.

    Sends a minimal request with a simple tool definition.
    If the model accepts tool definitions → True.
    If the provider returns a 400/unsupported error → False.
    If the result is ambiguous (rate limit, auth error) → None.
    """
```

This probe is integrated into `update_models.py` to populate the capability
during catalog refresh, alongside existing probes.

---

## Multimodal Integration

Tool calling and multimodality interact at three points: the initial prompt,
tool call arguments, and tool results. Each has different constraints.

### Multimodal Prompts with Tools

The initial prompt can be fully multimodal — images, audio, video alongside
text — even when tools are present. This is already supported by Djinnite's
`List[Dict]` prompt format and works naturally:

```python
tools = [
    {
        "name": "identify_building",
        "description": "Identify a building by name and return its address.",
        "inputSchema": {
            "type": "object",
            "properties": {"building_name": {"type": "string"}},
            "required": ["building_name"]
        },
        "callable": lookup_building_address,
    },
    {
        "name": "get_weather",
        "description": "Get weather for a city.",
        "inputSchema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        },
        "callable": get_weather,
    },
]

# Multimodal prompt + tools work together
prompt = [
    {"type": "text", "text": "What building is in this photo? Look up its address and tell me the weather there."},
    {"type": "image", "image_data": open("photo.jpg", "rb").read(), "mime_type": "image/jpeg"},
]

response = provider.generate(prompt, tools=tools)
# Model sees the image → identifies "Empire State Building"
# Round 1: calls identify_building(building_name="Empire State Building") → "350 Fifth Ave, New York, NY"
# Round 2: calls get_weather(city="New York") → "45°F, cloudy"
# Round 3: "The building in the photo is the Empire State Building at 350 Fifth Ave, New York. It's currently 45°F and cloudy there."
```

### Tool Call Arguments: Always JSON

Tool call arguments are **always JSON** — structured text. No provider
currently supports passing binary data (images, audio) as tool arguments.
The model describes what it wants in text, and the tool interprets it:

```python
# The model can't pass an image as a tool argument.
# Instead, it describes what it needs in text:
#   call analyze_chart(chart_type="bar", data_description="quarterly revenue")
# NOT:
#   call analyze_chart(image=<binary data>)
```

This is a fundamental constraint of all three provider APIs (OpenAI, Claude,
Gemini) — tool `inputSchema` is JSON Schema, and arguments are JSON objects.
Djinnite does not need to work around this; it's the correct design.

### Tool Results: Native Multimodal I/O (NMIO)

As of February 2026, the industry has transitioned from text-only tool results
to **Native Multimodal I/O** — the model can ingest raw binary payloads
(images, audio, video) directly from tool results into its next reasoning
cycle, without an intermediate "describe-the-image-in-text" step.

Each provider supports this differently:

| Feature | Gemini 3.1 | GPT-5.2 | Claude 4.6 |
|---|---|---|---|
| **Tool returns raw image** | Native (`inlineData`) | Via File ID reference | Requires new turn / user message |
| **Video/audio tool input** | Supported (native) | Supported (Realtime API) | Text/image only |
| **Parallel image tooling** | High (multi-part) | Medium | Low |
| **State management** | Interactions API | Responses API | Stateless / client-managed |

**Gemini** leads: a tool's `function_result` can contain `inlineData` media
parts (base64 images, audio) rather than just strings. The model receives
the image in its tool-role history and reasons about it directly — no
vision-to-text round trip. Gemini also supports `media_resolution` control
per tool call for managing token backpressure and latency.

**OpenAI** supports multimodal tool results via the File ID pattern — the
tool uploads binary data to the `/files` endpoint and returns the file ID.
The model can then reference it. This adds a round-trip hop compared to
Gemini's inline approach, but it works.

**Claude** remains text-primary for tool results. Claude 4.6 has excellent
visual reasoning for *inputs* (interpreting complex schematics to trigger
tools), but tool *results* must be text/JSON. Returning an image from a
tool requires injecting it as a new user message rather than a native tool
result. Claude's focus has been on structured JSON reliability for tool
call schemas rather than multimodal tool payloads.

### Djinnite's Multimodal Tool Result Abstraction

Djinnite provides a unified callable return convention that handles all
three provider patterns. This is a **Phase 1** feature — multimodal tool
results are too fundamental to defer.

**Simple case — text result** (works everywhere):

```python
def get_weather(city: str) -> str:
    """Returns a text description."""
    return "72°F and sunny in San Francisco"
```

Djinnite serializes via `str()` or `json.dumps()` and passes to all providers.

**Multimodal case — return parts** (provider-adaptive):

A callable returns a dict with a `"parts"` key to signal multimodal content,
using the same part schema as Djinnite's prompt format:

```python
def render_cad_model(part_id: str) -> dict:
    """Render a CAD model and return the image for the model to reason about."""
    image_bytes = cad_engine.render(part_id)
    return {
        "parts": [
            {"type": "text", "text": f"3D render of part {part_id}:"},
            {"type": "image", "image_data": image_bytes, "mime_type": "image/png"},
        ]
    }
```

Djinnite translates this into the provider-native format:

| Provider | How Djinnite handles multimodal tool results |
|---|---|
| **Gemini** | Passes `inlineData` media parts directly in the `function_response` — zero-copy, native multimodal reasoning |
| **OpenAI** | Uploads binary to `/files` endpoint, returns file ID reference in the tool result — adds one round-trip |
| **Claude** | Falls back to text description for the tool result. If the result contains images, injects them as content in the next user message turn so the model can still "see" them |

The caller doesn't need to know which provider is in use — the same
`"parts"` return convention works everywhere. Djinnite smooths over the
provider differences automatically, just like it does for multimodal
*inputs*.

**MCP tools with multimodal results** follow the same pattern — the MCP
server returns content with MIME-typed parts, and Djinnite routes them
through the same provider-adaptive pipeline.

---

## Public API Changes

### generate() Signature

```python
def generate(
    self,
    prompt: Union[str, List[Dict]],
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    web_search: bool = False,
    thinking: Union[bool, int, str, None] = None,
    tools: Optional[List[Dict]] = None,            # NEW
    max_tool_rounds: int = 10,                      # NEW
) -> AIResponse:
```

### generate_json() Signature

```python
def generate_json(
    self,
    prompt: Union[str, List[Dict]],
    schema: Union[Dict, Type],
    system_prompt: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: Optional[int] = None,
    web_search: bool = False,
    force: bool = False,
    thinking: Union[bool, int, str, None] = None,
    tools: Optional[List[Dict]] = None,            # NEW
    max_tool_rounds: int = 10,                      # NEW
) -> AIResponse:
```

### New Exports

```python
from djinnite import AIToolExecutionError   # Tool callable raised an exception
                                             # (after all retries exhausted)
```

---

## Implementation Phases

### Phase 1: Python Callables + Core Loop + Multimodal Results

- Add `tools` parameter to `generate()` and `generate_json()`
- Implement tool definition translation for all three providers
- Implement the internal execution loop (call → execute → feed back)
- Implement sequential tool calling (multi-round conversation state per provider)
- Implement tool call response parsing for all three providers
- Implement tool result submission for all three providers
- **Multimodal tool results**: support `"parts"` return convention from callables
  - Gemini: native `inlineData` in `function_response`
  - OpenAI: upload to `/files`, return file ID reference
  - Claude: inject images as user message content (graceful fallback)
- Add `tool_calls_made` to `AIResponse`
- Add `tool_calling` to model catalog capabilities
- Add `probe_tool_calling()` to each provider
- Support parallel tool calls (multiple calls per round)
- Error handling: callable exceptions → error results to model

### Phase 2: MCP Executor

- Add MCP client adapter (thin wrapper for dispatching tool calls over MCP)
- Support `"mcp_server"` key on tool definitions
- Host project passes MCP server references; Djinnite dispatches
- MCP and Python callables coexist in the same tools list

### Phase 3: Advanced

- Streaming tool calls (receive tool calls as they arrive, execute eagerly)
- Tool call cancellation (caller can abort mid-loop)
- Tool result caching (skip re-execution for identical calls within a session)
- Async callable support (`async def` tools)

---

## Compatibility

This proposal follows Djinnite's change policy:

- ✅ **Adds** new optional parameters (`tools`, `max_tool_rounds`) with defaults
- ✅ **Adds** new fields to `AIResponse` (`tool_calls_made`, default empty list)
- ✅ **Adds** new exception class (`AIToolExecutionError`)
- ✅ **Adds** new capability to model catalog (`tool_calling`)
- ✅ No changes to existing behavior when `tools` is not provided
- ✅ No changes to existing function signatures or return types

All existing code continues to work unchanged.
