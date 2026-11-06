"""
Harness for the live platform tier (``--live-platform``): Djinnite through
Google Vertex AI, against the real service.

Everything the tier needs comes from ``config/ai_config.json``, the same
file every other part of Djinnite is configured from: the ``platforms``
block names the project, quota project and locations, and its optional
``testing`` sub-block holds the test-only knobs (which models to use, a
decoy project, the cost cap). Credentials are Application Default
Credentials, resolved by google-auth exactly as they are for a consumer.
Nothing is read from the environment.

    {
      "platforms": {
        "vertexai": {
          "project_id": "<the operator's Vertex project>",
          "quota_project": "<the project to bill, or null>",
          "locations": ["global", "us"],
          "testing": {
            "gemini_model": "gemini-3.5-flash",
            "claude_model": "claude-haiku-4-5-20251001",
            "claude_canary": "claude-sonnet-5-5",
            "claude_5x_model": null,
            "decoy_project": null,
            "max_cost": 0.50
          }
        }
      }
    }

Every ``testing`` key is optional. A missing piece disables the tests that
need it, by name, in the summary; it never fails the run and it never
silently passes. See PLATFORM_LIVE_TEST_DESIGN.md for the test matrix.

All output is ASCII (Windows consoles are cp1252).
"""

import os
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from typing import Callable, Optional

from djinnite.ai_providers.base_provider import AIRateLimitError
from djinnite.config_loader import AIConfig

PLATFORM = "vertexai"

DEFAULTS = {
    "gemini_model": "gemini-3.5-flash",
    "claude_model": "claude-haiku-4-5-20251001",
    "claude_canary": "claude-sonnet-5-5",
    "max_cost": 0.50,
}


@dataclass(frozen=True)
class PlatformLiveConfig:
    project: str
    quota_project: Optional[str]
    decoy_project: Optional[str]
    locations: tuple
    gemini_model: str
    claude_model: str
    claude_canary: str
    claude_5x_model: Optional[str]
    max_cost: float

    @classmethod
    def from_ai_config(cls, cfg: AIConfig) -> "PlatformLiveConfig":
        """Read ``platforms.vertexai`` (and its ``testing`` block). Call ``missing_config`` first."""
        block = cfg.platforms[PLATFORM]
        t = block.testing or {}
        return cls(
            project=block.project_id or "",
            quota_project=block.quota_project or None,
            decoy_project=t.get("decoy_project") or None,
            locations=tuple(loc.strip().lower() for loc in block.locations if loc.strip()),
            gemini_model=t.get("gemini_model") or DEFAULTS["gemini_model"],
            claude_model=t.get("claude_model") or DEFAULTS["claude_model"],
            claude_canary=t.get("claude_canary") or DEFAULTS["claude_canary"],
            claude_5x_model=t.get("claude_5x_model") or None,
            max_cost=float(t.get("max_cost") or DEFAULTS["max_cost"]),
        )


def missing_config(cfg: AIConfig) -> Optional[str]:
    """What an opted-in run lacks in ai_config.json, or None. Checked at pytest_configure."""
    block = cfg.platforms.get(PLATFORM)
    if block is None:
        return ("ai_config.json has no platforms.vertexai block. Add one with project_id "
                "and locations (see PLATFORM_LIVE_TEST_DESIGN.md, 'Configuration').")
    if not block.enabled:
        return "ai_config.json platforms.vertexai is disabled (enabled: false)."
    if not block.project_id:
        return "ai_config.json platforms.vertexai.project_id is not set."
    locations = [loc.strip().lower() for loc in block.locations if loc.strip()]
    if "global" not in locations:
        return "ai_config.json platforms.vertexai.locations must include 'global'."
    return None


def say(line: str = "") -> None:
    """Print one ASCII line (shown with -s; captured otherwise)."""
    print(line.encode("ascii", "replace").decode("ascii"), flush=True)


def describe_credentials() -> str:
    """Which principal ADC resolves to -- printed by the preflight."""
    import google.auth
    creds, adc_project = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"])
    kind = type(creds).__name__
    who = (getattr(creds, "service_account_email", None)
           or getattr(creds, "_target_principal", None)
           or "user credentials")
    quota = getattr(creds, "quota_project_id", None) or "-"
    return f"{kind} principal={who} adc_project={adc_project or '-'} adc_quota_project={quota}"


# ----------------------------------------------------------- cost ledger

class CostLedger:
    """Every billed call in the session, with a hard total cap."""

    def __init__(self, cap: float):
        self.cap = cap
        self.rows: list = []

    def record(self, label: str, response) -> None:
        if response is None:
            return
        u = response.usage or {}
        self.rows.append((
            label, u.get("input_tokens", 0), u.get("output_tokens", 0),
            u.get("thinking_tokens"), u.get("total_cost") or 0.0,
            u.get("price_multiplier"),
        ))

    @property
    def total(self) -> float:
        return sum(r[4] for r in self.rows)

    def report(self) -> str:
        lines = ["", "[INFO] platform cost ledger",
                 f"  {'call':<48} {'in':>7} {'out':>7} {'think':>7} {'cost':>11} mult"]
        for label, i, o, t, c, m in self.rows:
            lines.append(f"  {label[:48]:<48} {i:>7} {o:>7} {str(t):>7} ${c:>10.6f} "
                         f"{'' if m is None else f'x{m:.2f}'}")
        lines.append(f"  {'TOTAL':<48} {'':>7} {'':>7} {'':>7} ${self.total:>10.6f} "
                     f"(cap ${self.cap:.2f})")
        return "\n".join(lines)


# ---------------------------------------------------------------- retry

def gemini_call(fn: Callable, *, wait: float = 10.0):
    """Gemini's quota is shared and dynamic: one retry after a 429.

    Never used for Claude, where a 429 is the result under test.
    """
    try:
        return fn()
    except AIRateLimitError:
        say(f"[WARN] Gemini 429; retrying once in {wait:.0f}s")
        time.sleep(wait)
        return fn()


# ------------------------------------------------------ missing ADC probe

def run_without_adc(provider: str, model: str, project: str, location: str) -> str:
    """Call a platform provider in a subprocess whose ADC points nowhere.

    Returns the exception class name the call raised (or "OK"). No request
    reaches Google: credential resolution fails locally.
    """
    code = textwrap.dedent(f"""
        from djinnite.ai_providers import get_provider
        try:
            p = get_provider({provider!r}, model={model!r}, platform="vertexai",
                             project_id={project!r}, location={location!r},
                             require_pricing=False)
            p.generate("hi", max_output_tokens=16)
            print("RESULT OK")
        except Exception as e:
            print("RESULT " + type(e).__name__)
    """)
    env = dict(os.environ)
    env["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(
        os.path.dirname(__file__), "_no_such_adc_file.json")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                         text=True, timeout=120)
    for line in out.stdout.splitlines():
        if line.startswith("RESULT "):
            return line.split(" ", 1)[1].strip()
    raise AssertionError(f"subprocess gave no result:\n{out.stdout}\n{out.stderr}")
