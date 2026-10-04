"""
Offline stubs shared by the platform / history / thinking tests.

No network, no keys. Providers are built with ``bare()`` (skips SDK client
construction, like ``test_empty_response._bare``) and given a stub client
that records every request it receives.
"""

from types import SimpleNamespace as NS

import anthropic
import httpx2

from djinnite.config_loader import ModelInfo, ModelCosting, ModelCapabilities


INFO = ModelInfo(
    id="test-model", name="Test", context_window=100_000, max_output_tokens=1000,
    costing=ModelCosting(input_per_1m=1.0, output_per_1m=2.0, source="published"),
)


def info(**caps) -> ModelInfo:
    """A priced ModelInfo with the given capabilities."""
    return ModelInfo(
        id="test-model", name="Test", context_window=100_000, max_output_tokens=1000,
        costing=ModelCosting(input_per_1m=1.0, output_per_1m=2.0, source="published"),
        capabilities=ModelCapabilities(**caps),
    )


def bare(cls, client, model_info=INFO, model="test-model", **attrs):
    """Provider instance with a stub client, skipping SDK construction."""
    p = cls.__new__(cls)
    p.api_key = "x"
    p.model = model
    p._model_info = model_info
    p._require_pricing = True
    p._client = client
    p._anthropic = anthropic
    for k, v in attrs.items():
        setattr(p, k, v)
    return p


# ---------------------------------------------------------------- Claude

def claude_response(text='{"v": 1}', stop="end_turn", input_tokens=100,
                    output_tokens=40, thinking_tokens=None, report_details=True):
    """A Message-like object. ``report_details=False`` omits output_tokens_details."""
    details = NS(thinking_tokens=thinking_tokens) if (report_details and thinking_tokens is not None) else None
    usage = NS(input_tokens=input_tokens, output_tokens=output_tokens,
               output_tokens_details=details, server_tool_use_input_tokens=0)
    return NS(content=[NS(type="text", text=text)], stop_reason=stop, usage=usage)


class _Stream:
    def __init__(self, response):
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._response


class StubAnthropic:
    """Records ``messages.stream`` / ``create`` / ``count_tokens`` calls.

    ``responses`` are returned in order by ``stream``; an Exception instance
    in the list is raised instead. ``create_error`` / ``count_error`` make
    those calls raise.
    """

    def __init__(self, responses=(), create_error=None, count_error=None):
        self.calls = []
        self.create_calls = []
        self.count_calls = []
        self._responses = list(responses)
        self._create_error = create_error
        self._count_error = count_error
        self.messages = NS(stream=self._stream, create=self._create, count_tokens=self._count)

    def _stream(self, **kw):
        self.calls.append(kw)
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return _Stream(item)

    def _create(self, **kw):
        self.create_calls.append(kw)
        if self._create_error is not None:
            raise self._create_error
        return claude_response()

    def _count(self, **kw):
        self.count_calls.append(kw)
        if self._count_error is not None:
            raise self._count_error
        return NS(input_tokens=1)


def anthropic_error(cls, status, message="boom", google_status=None):
    """A real anthropic APIStatusError subclass, shaped like a Vertex error body."""
    request = httpx2.Request("POST", "https://aiplatform.googleapis.com/v1/x")
    response = httpx2.Response(status, request=request)
    err = {"code": status, "message": message}
    if google_status:
        err["status"] = google_status
    return cls(f"Error code: {status}", response=response, body=[{"error": err}])


# ---------------------------------------------------------------- Gemini

def gemini_response(text='{"v": 1}', thoughts=None, candidates=7, prompt=11):
    usage = NS(prompt_token_count=prompt, candidates_token_count=candidates,
               total_token_count=prompt + candidates + (thoughts or 0),
               thoughts_token_count=thoughts)
    cand = NS(finish_reason="STOP", finish_message=None, safety_ratings=None,
              content=NS(parts=[NS(text=text, inline_data=None)]), grounding_metadata=None)
    return NS(candidates=[cand], text=text, usage_metadata=usage,
              prompt_feedback=NS(block_reason=None, block_reason_message=None, safety_ratings=None))


class StubGenai:
    """Records ``models.generate_content`` calls; raises ``error`` if given."""

    def __init__(self, response=None, error=None):
        self.calls = []
        self._response = response or gemini_response()
        self._error = error
        self.models = NS(generate_content=self._gen, count_tokens=self._count,
                         list=lambda **kw: [])

    def _gen(self, **kw):
        self.calls.append(kw)
        if self._error is not None:
            raise self._error
        return self._response

    def _count(self, **kw):
        self.calls.append(kw)
        if self._error is not None:
            raise self._error
        return NS(total_tokens=1)


# ------------------------------------------------- OpenAI / Grok (Responses)

def responses_response(text='{"v": 1}'):
    msg = NS(type="message", content=[NS(type="output_text", text=text)])
    return NS(output=[msg], output_text=text, status="completed", incomplete_details=None,
              usage=NS(input_tokens=10, output_tokens=5, total_tokens=15,
                       output_tokens_details=NS(reasoning_tokens=0)))


class StubResponses:
    def __init__(self, response=None):
        self.calls = []
        self._response = response or responses_response()
        self.responses = NS(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        return self._response
