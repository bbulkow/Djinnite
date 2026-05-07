"""
One-shot migration: rewrite every model entry's capability fields from
boolean / single-string shape to the list-of-states shape.

Rules (mirroring config_loader._coerce_states):
  on/off vocabularies (structured_json, thinking, web_search,
  json_with_search):
      True  -> ["on", "off"]
      False -> ["off"]
      str / list / null -> passthrough
  temperature vocabulary:
      True  -> ["any", "default"]
      False -> ["default"]
  thinking_style vocabulary:
      "adaptive"|"budget"|"effort" -> [<value>]
      list -> kept (filtered to known tokens)
      null -> null

Re-runnable. Idempotent on already-migrated catalogs.
"""

import json
from pathlib import Path
from typing import Any

CATALOG_PATH = Path(__file__).parent.parent / "config" / "model_catalog.json"

ON_OFF = ("on", "off")
TEMP = ("any", "default")
STYLES = ("adaptive", "budget", "effort")

ON_OFF_FIELDS = ("structured_json", "thinking", "web_search", "json_with_search")


def coerce_on_off(raw: Any):
    if raw is None:
        return None
    if raw is True:
        return ["on", "off"]
    if raw is False:
        return ["off"]
    if isinstance(raw, str):
        return [raw] if raw in ON_OFF else None
    if isinstance(raw, list):
        kept = [v for v in raw if v in ON_OFF]
        return kept or None
    return None


def coerce_temperature(raw: Any):
    if raw is None:
        return None
    if raw is True:
        return ["any", "default"]
    if raw is False:
        return ["default"]
    if isinstance(raw, str):
        return [raw] if raw in TEMP else None
    if isinstance(raw, list):
        kept = [v for v in raw if v in TEMP]
        return kept or None
    return None


def coerce_thinking_style(raw: Any):
    if raw is None:
        return None
    if isinstance(raw, str):
        return [raw] if raw in STYLES else None
    if isinstance(raw, list):
        kept = [v for v in raw if v in STYLES]
        return kept or None
    return None


def migrate_capabilities(caps: dict) -> dict:
    out = dict(caps)
    for k in ON_OFF_FIELDS:
        if k in out:
            out[k] = coerce_on_off(out[k])
    if "temperature" in out:
        out["temperature"] = coerce_temperature(out["temperature"])
    if "thinking_style" in out:
        out["thinking_style"] = coerce_thinking_style(out["thinking_style"])
    return out


def main() -> None:
    data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))

    n = 0
    # Catalog is keyed at the top level by provider name.
    for provider_name, provider_block in data.items():
        if not isinstance(provider_block, dict):
            continue
        for m in provider_block.get("models", []):
            caps = m.get("capabilities")
            if isinstance(caps, dict):
                m["capabilities"] = migrate_capabilities(caps)
                n += 1
            elif "supports_structured_json" in m:
                # Legacy flat field migration
                ssj = m.pop("supports_structured_json")
                m["capabilities"] = {
                    "structured_json": coerce_on_off(ssj),
                    "temperature": None,
                    "thinking": None,
                    "web_search": None,
                    "json_with_search": None,
                    "thinking_style": None,
                }
                n += 1

    CATALOG_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Migrated {n} model entries in {CATALOG_PATH}")


if __name__ == "__main__":
    main()
