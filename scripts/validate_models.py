"""
Validate Models Script

Comprehensive validation for AI models, testing both basic text-to-text
and advanced multimodal capabilities (vision, audio, video).

Iterates ai_config.json entries (access paths, ACCESS_PATHS_DESIGN.md);
each entry's catalog models are listed by its provider type. Platform-mode
entries are skipped: the catalog's top-level facts are direct-mode ones, so
use validate_ai or probe_platform for a platform.

Usage:
    python scripts/validate_models.py [--multimodal] [--provider ENTRY_OR_TYPE]
"""

import sys
import base64
import argparse
from pathlib import Path
from datetime import datetime

try:
    from djinnite.config_loader import load_ai_config, load_model_catalog
    from djinnite.ai_providers import PROVIDERS
    from djinnite.scripts.validate_ai import entry_label, entry_skip_reason, select_entries
except ImportError:
    # Direct execution: add the package root to sys.path
    _project_root = Path(__file__).resolve().parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    from config_loader import load_ai_config, load_model_catalog  # type: ignore
    from ai_providers import PROVIDERS  # type: ignore
    from scripts.validate_ai import entry_label, entry_skip_reason, select_entries  # type: ignore

# 1x1 Transparent PNG Pixel
TINY_IMAGE_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

def get_provider_class(provider_type):
    """The provider class for a provider *type* (not an entry name), or None."""
    return PROVIDERS.get(provider_type)

def test_model(provider_instance, model_id, model_modalities, test_multimodal=False):
    """Test a single model's capabilities."""
    results = []
    
    # 1. Basic Text Test (Always)
    # Check if text is a supported input modality
    if "text" in model_modalities.input:
        print(f"    - Testing TEXT...", end=" ", flush=True)
        try:
            resp = provider_instance.generate("Say 'Djinnite OK'")
            if "OK" in resp.content.upper():
                print("[OK]")
                results.append(("text", True))
            else:
                print(f"[WARN]  Unexpected response: {resp.content[:20]}...")
                results.append(("text", True))
        except Exception as e:
            print(f"[FAIL] {e}")
            results.append(("text", False))
    else:
        print(f"    - Skipping TEXT (not supported input)")

    # 2. Multimodal Tests
    if test_multimodal:
        if "vision" in model_modalities.input:
            print(f"    - Testing VISION...", end=" ", flush=True)
            try:
                img_data = base64.b64decode(TINY_IMAGE_BASE64)
                prompt = [
                    {"type": "text", "text": "What color is this pixel?"},
                    {"type": "image", "image_data": img_data, "mime_type": "image/png"}
                ]
                resp = provider_instance.generate(prompt)
                print(f"[OK] ({resp.content[:20]}...)")
                results.append(("vision", True))
            except Exception as e:
                print(f"[FAIL] {e}")
                results.append(("vision", False))
        
        # Audio/Video tests can be added here
        if "audio" in model_modalities.input:
             print(f"    - Testing AUDIO input (skip - needs asset)...")
            
    # 3. Output Modalities Report
    print(f"    - Output Modalities: {model_modalities.output}")
            
    return results

def validate_models():
    parser = argparse.ArgumentParser(description="Validate Djinnite models and modalities")
    parser.add_argument("--multimodal", action="store_true", help="Perform advanced multimodal tests")
    parser.add_argument("--provider", type=str,
                        help="Limit to one ai_config entry, or every entry of a provider type")
    parser.add_argument("--config", type=str, help="Path to ai_config.json")
    args = parser.parse_args()

    config = load_ai_config(Path(args.config) if args.config else None)
    catalog = load_model_catalog()

    try:
        entry_names = select_entries(config, args.provider)
    except ValueError as e:
        print(f"[FAIL] {e}")
        sys.exit(1)

    print(f"\nDjinnite Model Validator - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    for name in entry_names:
        p_config = config.providers[name]
        label = entry_label(config, name)
        reason = entry_skip_reason(config, name)
        if reason:
            print(f"\n[SKIP] {label}: {reason}")
            continue
        if p_config.mode == "platform":
            print(f"\n[SKIP] {label}: platform mode -- use validate_ai / probe_platform")
            continue

        ptype = config.provider_type(name)
        print(f"\nEntry: {label}")
        print("-" * 30)

        provider_cls = get_provider_class(ptype)
        if not provider_cls:
            print(f"  [FAIL] Unknown provider class for type {ptype}")
            continue

        models = catalog.list_models(ptype)
        if not models:
            print(f"  [WARN]  No models found in catalog for {ptype}")
            continue

        for model in models:
            print(f"  Model: {model.id}")
            if model.disabled:
                # get_provider refuses disabled models at runtime.
                print(f"    [SKIP] disabled in catalog: {model.disabled_reason or 'no reason given'}")
                continue
            print(f"    Inputs: {model.modalities.input}")
            print(f"    Outputs: {model.modalities.output}")

            try:
                # build_provider constructs the class for the entry's type
                # (provider_cls), loads the model's catalog entry and applies
                # the entry's deny restrictions.
                instance = config.build_provider(name, model.id)
                test_model(instance, model.id, model.modalities, args.multimodal)
            except Exception as e:
                print(f"    [FAIL] Initialization failed: {e}")

    print("\n" + "=" * 70)
    print("Validation Complete.")

if __name__ == "__main__":
    validate_models()
