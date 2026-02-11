"""
One-time script to clean up disabled_reason values that reference patterns.
Changes pattern-based reasons to more descriptive explicit reasons.

Usage:
    python -m djinnite.scripts.clean_disabled_reasons
"""
import json
import sys
from pathlib import Path

# Support direct execution (adds project root to path)
_project_root = str(Path(__file__).parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from djinnite.config_loader import CONFIG_DIR

CATALOG_PATH = CONFIG_DIR / "model_catalog.json"

# Map pattern types to meaningful disabled reasons
PATTERN_TO_REASON = {
    "image": "image generation model - not for text extraction",
    "tts": "text-to-speech model - not for text extraction",  
    "audio": "audio model - not for text extraction",
    "realtime": "realtime/streaming model - not for batch text extraction",
    "transcribe": "transcription model - not for text extraction",
    "embedding": "embedding model - not for text generation",
    "robotics": "robotics model - not for text extraction",
    "instruct": "legacy instruct model - deprecated",
    "gpt-3.5": "legacy model - deprecated",
    "gpt-4-0613": "legacy model - deprecated",
    "gpt-4-1106": "legacy preview model - use gpt-4-turbo instead",
    "gpt-4-0125": "legacy preview model - use gpt-4-turbo instead",
    "gpt-4-turbo-preview": "preview alias - use gpt-4-turbo instead",
}

def clean_disabled_reasons():
    with open(CATALOG_PATH, 'r', encoding='utf-8') as f:
        catalog = json.load(f)
    
    updated = 0
    
    for provider_name, provider_data in catalog.items():
        for model in provider_data.get("models", []):
            reason = model.get("disabled_reason", "")
            
            # Check if it's a pattern-based reason
            if "matches exclusion pattern" in reason:
                # Extract the pattern name
                # Format: "matches exclusion pattern 'xxx'"
                for pattern, new_reason in PATTERN_TO_REASON.items():
                    if f"'{pattern}'" in reason or reason.endswith(f"'{pattern}'"):
                        model["disabled_reason"] = new_reason
                        updated += 1
                        print(f"  {model['id']}: {new_reason}")
                        break
                else:
                    # No specific mapping found
                    model["disabled_reason"] = "specialty model - not for general text extraction"
                    updated += 1
                    print(f"  {model['id']}: specialty model (unmapped)")
    
    # Save
    with open(CATALOG_PATH, 'w', encoding='utf-8') as f:
        json.dump(catalog, f, indent=2)
    
    print(f"\n✅ Updated {updated} disabled_reason values")

if __name__ == "__main__":
    clean_disabled_reasons()
