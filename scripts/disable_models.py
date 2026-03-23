"""
Disable Models Script

Applies config/disabled_models.json to the model catalog.  Models listed
in the disable file are marked ``disabled: true`` in the catalog; models
NOT listed are re-enabled (the JSON file is the single source of truth).

Usage:
    python -m djinnite.scripts.disable_models              # Apply to catalog
    python -m djinnite.scripts.disable_models --dry-run    # Preview only
    python -m djinnite.scripts.disable_models --list       # Show disabled models
"""

import argparse
import json
import sys
from pathlib import Path

try:
    from djinnite.config_loader import CONFIG_DIR, _resolve_config_file
except ImportError:
    _project_root = str(Path(__file__).resolve().parent.parent)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from config_loader import CONFIG_DIR, _resolve_config_file


def load_disable_list() -> dict[str, str]:
    """Load the disable list from config/disabled_models.json.

    Returns:
        dict mapping model_id -> reason.
    """
    path = _resolve_config_file("disabled_models.json")
    if not path.exists():
        print(f"[WARN] {path} not found -- no models to disable")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("models", {})


def load_catalog(catalog_path=None) -> dict:
    path = catalog_path or CONFIG_DIR / "model_catalog.json"
    if not path.exists():
        print(f"[FAIL] Catalog not found at {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_catalog(catalog: dict, catalog_path=None) -> None:
    path = catalog_path or CONFIG_DIR / "model_catalog.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)


def list_disabled(catalog: dict) -> None:
    """Print all currently disabled models in the catalog."""
    count = 0
    for provider_name, provider_data in catalog.items():
        for model in provider_data.get("models", []):
            if model.get("disabled"):
                reason = model.get("disabled_reason", "")
                print(f"  {provider_name}/{model['id']}: {reason}")
                count += 1
    if count == 0:
        print("  (no disabled models)")
    else:
        print(f"\n  Total: {count} disabled models")


def apply_disable_list(
    catalog: dict,
    disable_list: dict[str, str],
    dry_run: bool = False,
) -> dict[str, int]:
    """Apply the disable list to the catalog.

    Returns stats dict with counts.
    """
    stats = {"disabled": 0, "re_enabled": 0, "unchanged": 0}

    for provider_name, provider_data in catalog.items():
        for model in provider_data.get("models", []):
            model_id = model["id"]
            currently_disabled = model.get("disabled", False)

            if model_id in disable_list:
                reason = disable_list[model_id]
                if not currently_disabled:
                    print(f"  [DISABLE] {provider_name}/{model_id}: {reason}")
                    if not dry_run:
                        model["disabled"] = True
                        model["disabled_reason"] = reason
                    stats["disabled"] += 1
                else:
                    # Already disabled -- update reason if changed
                    old_reason = model.get("disabled_reason", "")
                    if old_reason != reason:
                        print(f"  [UPDATE]  {provider_name}/{model_id}: {reason}")
                        if not dry_run:
                            model["disabled_reason"] = reason
                    stats["unchanged"] += 1
            else:
                if currently_disabled:
                    print(f"  [ENABLE]  {provider_name}/{model_id} (removed from disable list)")
                    if not dry_run:
                        model["disabled"] = False
                        model["disabled_reason"] = ""
                    stats["re_enabled"] += 1
                else:
                    stats["unchanged"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Apply disabled_models.json to the model catalog."
    )
    parser.add_argument(
        "--dry-run", "-n", action="store_true",
        help="Preview changes without saving",
    )
    parser.add_argument(
        "--list", "-l", action="store_true", dest="list_only",
        help="Show currently disabled models and exit",
    )
    parser.add_argument(
        "--catalog", type=str, help="Path to model_catalog.json",
    )
    args = parser.parse_args()

    catalog_path = Path(args.catalog) if args.catalog else None
    catalog = load_catalog(catalog_path)

    if args.list_only:
        print("[TOOL] Disabled Models")
        print("-" * 40)
        list_disabled(catalog)
        return

    disable_list = load_disable_list()
    if not disable_list:
        return

    print("[TOOL] Disable Models")
    print("-" * 40)
    print(f"Disable list: {len(disable_list)} models")
    if args.dry_run:
        print("Mode: DRY RUN")
    print()

    stats = apply_disable_list(catalog, disable_list, dry_run=args.dry_run)

    print()
    print("-" * 40)
    if args.dry_run:
        print("[CHECK] DRY RUN -- no changes saved")
    else:
        save_catalog(catalog, catalog_path)
        print(f"[SAVE] Saved to {catalog_path or CONFIG_DIR / 'model_catalog.json'}")
    print(f"  Disabled:    {stats['disabled']}")
    print(f"  Re-enabled:  {stats['re_enabled']}")
    print(f"  Unchanged:   {stats['unchanged']}")


if __name__ == "__main__":
    main()
