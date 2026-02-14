"""
Validate Models Script

Comprehensive validation for AI models, testing both basic text-to-text
and advanced multimodal capabilities (vision, audio, video).

Usage:
    python scripts/validate_models.py [--multimodal] [--provider PROVIDER]
"""

import os
import sys
import base64
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import config_loader
from ai_providers.gemini_provider import GeminiProvider
from ai_providers.claude_provider import ClaudeProvider
from ai_providers.openai_provider import OpenAIProvider

# 1x1 Transparent PNG Pixel
TINY_IMAGE_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

def get_provider_class(provider_name):
    providers = {
        "gemini": GeminiProvider,
        "claude": ClaudeProvider,
        "chatgpt": OpenAIProvider
    }
    return providers.get(provider_name)

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
                print("✅")
                results.append(("text", True))
            else:
                print(f"⚠️  Unexpected response: {resp.content[:20]}...")
                results.append(("text", True))
        except Exception as e:
            print(f"❌ {e}")
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
                print(f"✅ ({resp.content[:20]}...)")
                results.append(("vision", True))
            except Exception as e:
                print(f"❌ {e}")
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
    parser.add_argument("--provider", type=str, help="Limit to specific provider")
    parser.add_argument("--config", type=str, help="Path to ai_config.json")
    args = parser.parse_args()

    config = config_loader.load_ai_config(Path(args.config) if args.config else None)
    catalog = config_loader.load_model_catalog()

    print(f"\nDjinnite Model Validator - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    for p_name, p_config in config.providers.items():
        if not p_config.enabled:
            continue
        if args.provider and p_name != args.provider:
            continue
            
        print(f"\nProvider: {p_name.upper()}")
        print("-" * 30)
        
        provider_cls = get_provider_class(p_name)
        if not provider_cls:
            print(f"  ❌ Unknown provider class for {p_name}")
            continue

        models = catalog.list_models(p_name)
        if not models:
            print(f"  ⚠️  No models found in catalog for {p_name}")
            continue

        for model in models:
            print(f"  Model: {model.id}")
            print(f"    Inputs: {model.modalities.input}")
            print(f"    Outputs: {model.modalities.output}")
            
            try:
                # Initialize provider for this model
                kwargs = {}
                if p_name == "gemini":
                    kwargs["backend"] = p_config.backend
                    kwargs["project_id"] = p_config.project_id
                
                instance = provider_cls(api_key=p_config.api_key, model=model.id, **kwargs)
                test_model(instance, model.id, model.modalities, args.multimodal)
            except Exception as e:
                print(f"    ❌ Initialization failed: {e}")

    print("\n" + "=" * 70)
    print("Validation Complete.")

if __name__ == "__main__":
    validate_models()
