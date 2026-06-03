"""
Audit Pricing Script (read-only)

Classifies every model in the catalog as fixed/floating, checks staleness, and
reports what update_model_costs would re-price on its next run -- WITHOUT making
any API calls or writing the catalog.  Use this for human review before
committing a price refresh.

Usage:
    python -m djinnite.scripts.audit_pricing
    python -m djinnite.scripts.audit_pricing --provider gemini
    python -m djinnite.scripts.audit_pricing --staleness-days 90
    python -m djinnite.scripts.audit_pricing --fail-on-unknown   # nonzero exit if any unknown
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timezone

# Add project root to sys.path (mirrors the other scripts)
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import config_loader
from pricing_class import classify_model

DEFAULT_STALENESS_DAYS = 180


def _effective_class(provider, model, sibling_ids):
    """Return (effective_class, auto_class, drift_msg_or_None)."""
    costing = model.costing
    auto = classify_model(provider, model.id, sibling_ids)
    if costing.pricing_class_source == "manual" and costing.pricing_class:
        drift = None
        if costing.pricing_class != auto:
            drift = f"{model.id}: manual={costing.pricing_class} but classifier={auto}"
        return costing.pricing_class, auto, drift
    return auto, auto, None


def audit_pricing():
    parser = argparse.ArgumentParser(description="Audit model pricing (read-only)")
    parser.add_argument("--provider", type=str, help="Limit to a specific provider")
    parser.add_argument("--catalog", type=str, help="Path to model_catalog.json")
    parser.add_argument("--staleness-days", type=int, default=DEFAULT_STALENESS_DAYS,
                        help=f"Treat fixed prices older than this as stale (default: {DEFAULT_STALENESS_DAYS})")
    parser.add_argument("--fail-on-unknown", action="store_true",
                        help="Exit nonzero if any model has source=unknown")
    parser.add_argument("--fail-on-stale", action="store_true",
                        help="Exit nonzero if any fixed model is stale")
    args = parser.parse_args()

    catalog = config_loader.load_model_catalog(
        Path(args.catalog) if args.catalog else None)
    today = datetime.now(timezone.utc).date()

    print(f"\nDjinnite Pricing Audit - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Staleness threshold: {args.staleness_days} days")
    print("=" * 70)

    totals = {"floating": 0, "fixed_fresh": 0, "fixed_stale": 0, "unknown": 0,
              "failed": 0, "manual": 0, "disabled": 0, "missing": 0}
    would_reprice = []
    unknowns = []
    drifts = []

    for p_name in catalog.providers:
        if args.provider and p_name != args.provider:
            continue
        models = catalog.list_models(p_name)
        if not models:
            continue

        sibling_ids = [m.id for m in models]
        print(f"\nProvider: {p_name.upper()} ({len(models)} models)")
        print("-" * 50)

        for model in models:
            costing = model.costing
            eff_class, _auto, drift = _effective_class(p_name, model, sibling_ids)
            if drift:
                drifts.append(drift)

            if model.disabled:
                totals["disabled"] += 1
                continue

            source = costing.source
            has_price = costing.input_per_1m is not None and costing.updated != ""
            stale = costing.is_stale(args.staleness_days, today=today)
            days = costing.days_since_update(today=today)
            age = f"{days}d" if days is not None else "never"

            if source == "manual":
                totals["manual"] += 1
                tag = "MANUAL"
                reprice = False
            elif source == "unknown":
                totals["unknown"] += 1
                unknowns.append(f"{p_name}/{model.id}")
                tag = "UNKNOWN"
                reprice = False  # stable; needs --refresh-unknown to retry
            elif source == "failed":
                totals["failed"] += 1
                tag = "FAILED"
                reprice = True
            elif not has_price:
                totals["missing"] += 1
                tag = "MISSING"
                reprice = True
            elif eff_class == "floating":
                totals["floating"] += 1
                tag = "FLOATING"
                reprice = True
            elif stale:
                totals["fixed_stale"] += 1
                tag = "FIXED-STALE"
                reprice = True
            else:
                totals["fixed_fresh"] += 1
                tag = "FIXED-FRESH"
                reprice = False

            price = f"${costing.input_per_1m}/{costing.output_per_1m}" if has_price else "(no price)"
            flag = " -> RE-PRICE" if reprice else ""
            print(f"  [{tag:12}] {model.id:48} {price:16} {age:7} {source}{flag}")
            if reprice:
                would_reprice.append(f"{p_name}/{model.id} [{tag}] {price}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("-" * 70)
    for key in ("floating", "fixed_stale", "fixed_fresh", "missing", "unknown",
                "failed", "manual", "disabled"):
        print(f"  {key:14}: {totals[key]}")

    if drifts:
        print(f"\nCLASS DRIFT (manual pricing_class disagrees with classifier) ({len(drifts)}):")
        for d in drifts:
            print(f"  - {d}")

    if unknowns:
        print(f"\nUNKNOWN (no public price -- needs manual search) ({len(unknowns)}):")
        for u in unknowns:
            print(f"  - {u}")

    print(f"\nWOULD RE-PRICE ON NEXT RUN ({len(would_reprice)}):")
    for w in would_reprice:
        print(f"  - {w}")

    exit_code = 0
    if args.fail_on_unknown and totals["unknown"] > 0:
        exit_code = 1
    if args.fail_on_stale and totals["fixed_stale"] > 0:
        exit_code = 1
    sys.exit(exit_code)


if __name__ == "__main__":
    audit_pricing()
