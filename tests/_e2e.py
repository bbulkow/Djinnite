"""
Harness for the platform-mode end-to-end tier (``--e2e-platform``).

Real calls to Google Vertex AI in a dedicated test project, made with
platform credentials (an impersonated service account -- never a key file).
See PLATFORM_E2E_TEST_DESIGN.md for the GCP setup and the test matrix.

Configuration is environment variables (DJINNITE_E2E_*), so a run is one
command and nothing project-specific is committed:

    $env:DJINNITE_E2E_PROJECT       = "<e2e project id>"
    $env:DJINNITE_E2E_DECOY_PROJECT = "<decoy project id>"
    $env:DJINNITE_E2E_CREDENTIALS   = "<runner gcloud dir>\\application_default_credentials.json"
    uv run pytest tests/ --e2e-platform -rA -s

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


@dataclass(frozen=True)
class E2EConfig:
    project: str
    credentials: Optional[str]
    decoy_project: Optional[str]
    locations: tuple
    gemini_model: str
    claude_model: str
    claude_canary: str
    claude_5x_model: Optional[str]
    max_cost: float

    @classmethod
    def from_env(cls) -> "E2EConfig":
        env = os.environ.get
        return cls(
            project=env("DJINNITE_E2E_PROJECT", ""),
            credentials=env("DJINNITE_E2E_CREDENTIALS") or None,
            decoy_project=env("DJINNITE_E2E_DECOY_PROJECT") or None,
            locations=tuple(
                loc.strip().lower()
                for loc in env("DJINNITE_E2E_LOCATIONS", "global,us").split(",")
                if loc.strip()
            ),
            gemini_model=env("DJINNITE_E2E_GEMINI_MODEL", "gemini-3.5-flash"),
            claude_model=env("DJINNITE_E2E_CLAUDE_MODEL", "claude-haiku-4-5-20251001"),
            claude_canary=env("DJINNITE_E2E_CLAUDE_CANARY", "claude-sonnet-5-5"),
            claude_5x_model=env("DJINNITE_E2E_CLAUDE_5X_MODEL") or None,
            max_cost=float(env("DJINNITE_E2E_MAX_COST", "0.50")),
        )


def missing_config() -> Optional[str]:
    """What an opted-in run lacks, or None. Checked at pytest_configure."""
    cfg = E2EConfig.from_env()
    if not cfg.project:
        return ("DJINNITE_E2E_PROJECT is not set. Set it to the e2e test project "
                "(see PLATFORM_E2E_TEST_DESIGN.md, 'GCP setup').")
    if cfg.credentials and not os.path.exists(cfg.credentials):
        return f"DJINNITE_E2E_CREDENTIALS points to a missing file: {cfg.credentials}"
    if "global" not in cfg.locations:
        return "DJINNITE_E2E_LOCATIONS must include 'global'."
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
        lines = ["", "[INFO] e2e cost ledger",
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
