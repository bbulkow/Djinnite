"""
Validate AI Configuration

Tests connectivity and authentication for every configured ai_config.json
entry by attempting a minimal generation request. Entries are access paths
(ACCESS_PATHS_DESIGN.md): several may share a provider type, e.g. "claude"
direct and "claude-vertex" on Vertex AI, and each is checked.

Also reports ``deny`` model keys that name no catalog model of the entry's
provider type (a typo, or a model the catalog has since dropped).

Usage:
    python -m djinnite.scripts.validate_ai
    python -m djinnite.scripts.validate_ai --config path/to/ai_config.json

The helpers ``entry_label``, ``select_entries``, ``entry_skip_reason`` and
``unknown_deny_models`` are pure (no network) and shared by validate_json
and validate_models.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

# Support direct execution (adds the package's parent to path)
_project_root = str(Path(__file__).parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from djinnite.config_loader import (
        DENY_ALL_MODELS, PROVIDER_TYPES, load_ai_config, load_model_catalog,
    )
    from djinnite.ai_providers.base_provider import AIOutputTruncatedError
    from djinnite.ai_providers.platforms import PLATFORMS
except ImportError:
    _pkg_root = str(Path(__file__).resolve().parent.parent)
    if _pkg_root not in sys.path:
        sys.path.insert(0, _pkg_root)
    from config_loader import (  # type: ignore
        DENY_ALL_MODELS, PROVIDER_TYPES, load_ai_config, load_model_catalog,
    )
    from ai_providers.base_provider import AIOutputTruncatedError  # type: ignore
    from ai_providers.platforms import PLATFORMS  # type: ignore


# ======================================================================
# Pure helpers (no network) -- unit-tested in tests/test_access_paths_validate.py
# ======================================================================

def entry_label(config, name: str) -> str:
    """How an entry is shown: ``claude (claude, direct)`` or
    ``claude-vertex (claude, platform vertexai@global)``.

    The platform location is the entry's own, else the platform registry's
    default for the provider type, else ``default``.
    """
    pc = config.providers[name]
    ptype = config.provider_type(name)
    if pc.mode != "platform":
        return f"{name} ({ptype}, direct)"
    location = pc.location
    if not location:
        spec = PLATFORMS.get(pc.platform)
        location = (spec.default_location.get(ptype) if spec else None) or "default"
    return f"{name} ({ptype}, platform {pc.platform}@{location})"


def select_entries(config, selector: Optional[str] = None) -> list:
    """Entry names to check, in config order.

    With no ``selector``, every entry (disabled and unusable ones included, so
    each gets its own skip line). Otherwise the entries whose name is
    ``selector`` or whose provider type is ``selector``: an entry name picks
    that entry, a type picks every entry of that type.

    Raises:
        ValueError: ``selector`` is neither an entry name nor the type of any entry.
    """
    names = list(config.providers)
    if not selector:
        return names
    picked = [n for n in names if n == selector or config.provider_type(n) == selector]
    if not picked:
        types = sorted({config.provider_type(n) for n in names})
        raise ValueError(
            f"'{selector}' is neither an ai_config entry nor the type of one. "
            f"Entries: {', '.join(names) or '(none)'}; "
            f"types configured: {', '.join(types) or '(none)'}"
        )
    return picked


def entry_skip_reason(config, name: str) -> Optional[str]:
    """Why entry ``name`` cannot be checked, or ``None`` if it can."""
    pc = config.providers[name]
    if not pc.enabled:
        return "disabled in config"
    ptype = config.provider_type(name)
    if ptype not in PROVIDER_TYPES:
        return (f"'{ptype}' is not a provider type {PROVIDER_TYPES}; "
                f"add \"provider\": \"<type>\" to the entry")
    if not config.is_usable(name):
        return "API key is missing or a placeholder (direct mode needs one)"
    return None


def unknown_deny_models(config, catalog) -> list:
    """``(entry, model_id)`` for each ``deny`` model key the catalog does not have.

    A model key is checked against the catalog section of the entry's
    provider type. ``"*"`` (every model) is never unknown.
    """
    out = []
    for name, pc in config.providers.items():
        ptype = config.provider_type(name)
        for model_id in pc.deny:
            if model_id == DENY_ALL_MODELS:
                continue
            if catalog.get_model(ptype, model_id) is None:
                out.append((name, model_id))
    return out


# ======================================================================
# Main
# ======================================================================

def _report_unknown_deny_models(config) -> None:
    try:
        catalog = load_model_catalog()
    except Exception as e:
        print(f"[WARN] Could not load the model catalog to check deny model keys: {e}")
        return
    for name, model_id in unknown_deny_models(config, catalog):
        print(f"[WARN] {name}: deny names model '{model_id}' which is not in the "
              f"catalog for {config.provider_type(name)}")


def validate_ai():
    parser = argparse.ArgumentParser(description="Validate AI provider configuration")
    parser.add_argument("--config", type=str, help="Path to ai_config.json")
    args = parser.parse_args()

    print("Loading configuration...")
    config_path = Path(args.config) if args.config else None
    config = load_ai_config(config_path)

    _report_unknown_deny_models(config)

    print("\nValidating AI provider entries...")
    print("=" * 60)

    success_count = 0
    fail_count = 0
    skip_count = 0

    if not config.providers:
        print("[WARN] No entries under 'providers' in ai_config.json")

    for name in select_entries(config):
        label = entry_label(config, name)
        reason = entry_skip_reason(config, name)
        if reason and config.providers[name].enabled:
            # An enabled entry that cannot be built (placeholder key, unknown
            # type) is a failure, not a skip: catching that is this script's job.
            print(f"[FAIL] {label}: {reason}")
            fail_count += 1
            continue
        if reason:
            print(f"[SKIP] {label}: {reason}")
            skip_count += 1
            continue

        model = config.providers[name].default_model
        model_info = f" model {model}" if model else ""
        print(f"Testing {label}{model_info}...", end=" ", flush=True)

        try:
            # build_provider() loads model_info from the catalog (enabling
            # catalog-aware features like temperature stripping for
            # reasoning models) and applies the entry's deny restrictions.
            provider = config.build_provider(name)

            # Connectivity check (is_available usually does a lightweight check)
            if not provider.is_available():
                print(f"\r[FAIL] {label}: Connection failed (API unreachable or credentials invalid)")
                fail_count += 1
                continue

            # Functional Generation Check (The "Unit Test" part)
            try:
                # Try a small generation to prove auth works.
                # Use max_output_tokens=50 instead of 1 -- reasoning models
                # (e.g. GPT-5) reject very low values.  Truncation is still
                # OK; it proves the API accepted the request (HTTP 200).
                provider.generate("Test", max_output_tokens=50)
                print(f"\r[OK] {label}: Success! (Authenticated & Generating)")
                success_count += 1
            except AIOutputTruncatedError:
                # Truncation is expected -- the API worked!
                print(f"\r[OK] {label}: Success! (Authenticated & Generating)")
                success_count += 1
            except Exception as e:
                # Clean up error message
                error_details = str(e).replace("\n", " ").strip()
                if hasattr(e, 'message'):
                    error_details += f" | Details: {e.message}"

                print(f"\r[FAIL] {label}: Generation failed: {error_details}")

                # DEBUG: Try to list available models to help user find a valid one
                try:
                    if hasattr(provider, 'list_models'):
                        print(f"   [INFO]  Debugging: Checking available models for {name}...")
                        models = provider.list_models()
                        if models:
                            ids = [m['id'] for m in models]
                            print(f"   [OK] Available models: {', '.join(ids[:5])}...")
                            if model and model not in ids:
                                print(f"   [WARN]  Configured model '{model}' is NOT in this list.")
                        else:
                            print("   [WARN]  No models returned by API.")
                except Exception as list_err:
                    print(f"   [WARN]  Could not list models: {list_err}")

                fail_count += 1

        except Exception as e:
            error_details = str(e).replace("\n", " ").strip()
            print(f"\r[FAIL] {label}: Initialization failed: {error_details}")
            fail_count += 1

    print("=" * 60)
    print(f"Summary: {success_count} passed, {fail_count} failed, {skip_count} skipped.")

    # If no success at all, or if there were failures, provide help
    if fail_count > 0 or success_count == 0:
        print("\n[TIP] TROUBLESHOOTING TIP:")
        print("   1. Ensure you have copied the config template:")
        print("      (Windows) copy config\\ai_config.example.json config\\ai_config.json")
        print("      (Mac/Linux) cp config/ai_config.example.json config/ai_config.json")
        print("   2. Edit config/ai_config.json and add your real API keys")
        print("   3. Ensure 'enabled': true is set for your provider entries")

    if fail_count > 0 or success_count == 0:
        sys.exit(1)

if __name__ == "__main__":
    validate_ai()
