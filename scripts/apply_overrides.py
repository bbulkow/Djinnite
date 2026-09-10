"""
Apply Overrides Script

Applies config/model_overrides.json to the model catalog.

This is the command to run after hand-editing model_overrides.json. It is also
run automatically as the last step of update_models and update_model_costs, so
a refresh can never land without human decisions applied.

Usage:
    python -m djinnite.scripts.apply_overrides              # Apply to catalog
    python -m djinnite.scripts.apply_overrides --dry-run    # Preview only
    python -m djinnite.scripts.apply_overrides --list       # Show overridden models
"""

import argparse
import json
import sys
from pathlib import Path

try:
    from djinnite.config_loader import CONFIG_DIR
    from djinnite.scripts.model_overrides import (
        PROVENANCE_KEY, apply_overrides, load_overrides, save_catalog,
    )
except ImportError:
    _project_root = str(Path(__file__).resolve().parent.parent)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from config_loader import CONFIG_DIR
    from scripts.model_overrides import (
        PROVENANCE_KEY, apply_overrides, load_overrides, save_catalog,
    )


def list_overridden(catalog: dict) -> None:
    """Print every field in the catalog that a human set, and what it replaced."""
    count = 0
    for provider_name, provider_data in catalog.items():
        if not isinstance(provider_data, dict):
            continue
        for model in provider_data.get("models", []):
            prov = model.get(PROVENANCE_KEY)
            if not prov:
                continue
            count += 1
            print(f"\n  {provider_name}/{model['id']}")
            for path, record in sorted(prov.items()):
                was = record.get("was")
                shown = "(absent)" if was == "__absent__" else repr(was)
                now = model
                for part in path.split("."):
                    now = now.get(part) if isinstance(now, dict) else None
                print(f"    {path}: {now!r}   (discovery said {shown})")
    if not count:
        print("  (no overridden models)")
    else:
        print(f"\n  Total: {count} models carry human overrides")


def main():
    parser = argparse.ArgumentParser(
        description="Apply model_overrides.json to the model catalog."
    )
    parser.add_argument("--catalog", type=str, help="Path to model_catalog.json")
    parser.add_argument("--overrides", type=str, help="Path to model_overrides.json")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would change without writing the catalog",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Show currently overridden models and exit",
    )
    args = parser.parse_args()

    catalog_path = Path(args.catalog) if args.catalog else CONFIG_DIR / "model_catalog.json"
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    print("[TOOL] Apply Overrides")
    print("-" * 40)

    if args.list:
        list_overridden(catalog)
        return

    overrides = load_overrides(Path(args.overrides) if args.overrides else None)
    print(f"Override entries: {len(overrides)}")
    if args.dry_run:
        print("Mode: DRY RUN\n")
        stats = apply_overrides(catalog, overrides)
        print("\n" + "-" * 40)
        print("[CHECK] DRY RUN -- no changes saved")
    else:
        print()
        stats = save_catalog(catalog, catalog_path, overrides)
        print("-" * 40)

    print(f"  Models touched:    {stats['models']}")
    print(f"  Fields set:        {stats['fields']}")
    print(f"  Already matching:  {stats['noop']}")
    print(f"  Reverted:          {stats['reverted']}")
    if stats["stale"]:
        print(f"  Stale entries:     {stats['stale']} (match no catalog model)")


if __name__ == "__main__":
    main()
