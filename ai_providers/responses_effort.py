"""
Reasoning-effort probes shared by the Responses-API providers (OpenAI, Grok).

Both vendors take ``reasoning={"effort": <level>}``. OpenAI answers a probe
in one of three machine-readable ways (verified 2026-11-03):

* accepted -- the model takes that level;
* HTTP 400, ``param == "reasoning.effort"``, ``code == "unsupported_value"``
  -- the model reasons but lacks that level; the message lists the levels it
  accepts ("Supported values are: 'medium', 'high', and 'xhigh'.");
* HTTP 400, ``param == "reasoning.effort"``, ``code ==
  "unsupported_parameter"`` -- the model does not reason at all.

Anything else (404 for a shut-down model, 5xx, rate limit, an error shape
this code does not recognise) is inconclusive. That matters because
``update_models`` stores a confirmed-empty style list permanently: only the
vendor saying "no reasoning" may produce ``[]``.

xAI's error shape is unverified (no credentials); on an unrecognised shape
these probes return ``None`` and the catalog keeps what it had.
"""

import re
from typing import Optional

try:
    from ..config_loader import EFFORT_LEVEL_VALUES  # type: ignore
except ImportError:  # pragma: no cover - flat-layout import
    from config_loader import EFFORT_LEVEL_VALUES  # type: ignore


_SUPPORTED_VALUES = re.compile(r"Supported values are:(.*)$", re.S)


def effort_error_kind(exc: Exception) -> str:
    """``"level"``, ``"no_reasoning"`` or ``"other"`` for a probe exception."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict) and body.get("param") == "reasoning.effort":
        if body.get("code") == "unsupported_value":
            return "level"
        if body.get("code") == "unsupported_parameter":
            return "no_reasoning"
    return "other"


def listed_effort_levels(exc: Exception) -> list[str]:
    """The levels an ``unsupported_value`` rejection lists, in ladder order.

    ``none`` is dropped: it is the vendor's off switch, which Djinnite
    spells ``thinking=False``. Names outside ``EFFORT_LEVEL_VALUES`` are
    dropped too.
    """
    body = getattr(exc, "body", None)
    message = body.get("message", "") if isinstance(body, dict) else ""
    match = _SUPPORTED_VALUES.search(message or "")
    listed = set(re.findall(r"'([^']+)'", match.group(1))) if match else set()
    return [level for level in EFFORT_LEVEL_VALUES if level in listed]


def combination_level(levels: list[str]) -> str:
    """``medium``, or the accepted level nearest it (a tie goes lower).

    The combination probe asks whether thinking combines with another
    capability. An end of the ladder answers a narrower question: OpenAI
    rejects web search at ``minimal``, and probing there recorded
    ``{thinking: on, web_search: on}`` as incompatible on the whole gpt-5
    family (2026-11-03).
    """
    ladder = list(EFFORT_LEVEL_VALUES)
    target = ladder.index("medium")
    return min(levels, key=lambda lvl: (abs(ladder.index(lvl) - target), ladder.index(lvl)))


class ResponsesEffortProbes:
    """``probe_thinking_style`` / ``probe_thinking_disable`` for the Responses API.

    ``probe_thinking_style`` also records, on the instance:

    * ``probed_effort_level`` -- the level the combination probe sends:
      ``combination_level`` of the accepted levels. A fixed ``"low"`` is
      rejected by models such as ``gpt-5-pro`` (``high`` only), and every
      thinking-on combination would be recorded as incompatible;
    * ``probed_effort_levels`` -- the model's accepted levels in ladder
      order, when a rejection listed them; ``update_models`` writes them to
      the catalog's ``effort_levels``.
    """

    probed_effort_level: Optional[str] = None
    probed_effort_levels: Optional[list[str]] = None

    def _send_effort(self, level: str) -> None:
        self._client.responses.create(
            model=self.model,
            input="Say hi.",
            reasoning={"effort": level},
        )

    def probe_thinking_style(self) -> Optional[list[str]]:
        """``["effort"]``, ``[]`` (no reasoning) or ``None`` (inconclusive).

        Sends ``minimal`` and then ``max``, the two ends of the ladder, so a
        model missing either one returns its accepted list. If neither end
        was accepted, the combination level is sent to confirm the model
        takes it. A model that accepts both ends is recorded as supporting
        effort with its levels unknown, and ``medium`` is confirmed for the
        combination probe.
        """
        self.probed_effort_level = None
        self.probed_effort_levels = None
        accepted = None
        rejection = None
        for level in ("minimal", "max"):
            try:
                self._send_effort(level)
            except Exception as e:
                kind = effort_error_kind(e)
                if kind == "no_reasoning":
                    return []
                if kind != "level":
                    return None
                rejection = e
                break
            accepted = accepted or level

        if rejection is not None:
            levels = listed_effort_levels(rejection)
            if not levels:
                return None
            self.probed_effort_levels = levels
            self.probed_effort_level = combination_level(levels)
            if accepted is None:
                try:
                    self._send_effort(self.probed_effort_level)
                except Exception:
                    return None
            return ["effort"]

        try:
            self._send_effort("medium")
            self.probed_effort_level = "medium"
        except Exception:
            self.probed_effort_level = accepted
        return ["effort"]

    def probe_thinking_disable(self) -> Optional[bool]:
        """Whether ``effort: "none"`` (what ``thinking=False`` sends) is accepted.

        ``False`` when the vendor rejects it as a level (always-on models
        such as the ``-pro`` family) or rejects reasoning altogether;
        ``None`` for anything else.
        """
        try:
            self._send_effort("none")
            return True
        except Exception as e:
            if effort_error_kind(e) in ("level", "no_reasoning"):
                return False
            return None
