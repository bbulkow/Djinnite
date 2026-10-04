#!/usr/bin/env python3
"""
Probe what a cloud platform serves: per-model availability at each location,
and optionally per-model capabilities on the platform.

This is the platform-mode counterpart of ``update_models``. ``update_models``
refreshes the catalog's top-level fields -- direct-mode facts -- through each
provider's own API. A platform (Google Vertex AI today) rarely offers a
model-discovery endpoint for the providers it hosts, and the set of models it
serves grows and shrinks over time. So this script takes the locations to
check from ``ai_config.json``'s ``platforms.<name>`` block, probes the
catalog's models there, and records the results in each model's generated
``platforms.<name>`` block.

Run it observably (AGENTS.md: unbuffered, teed to a log):

    uv run python -u -m djinnite.scripts.probe_platform --platform vertexai 2>&1 | tee <log>

Cost:
    default         One token-count call per model x location. Token counting
                    is not billed.
    --capabilities  The full capability probe suite per model (the same suite
                    update_models runs), through the platform. These are real
                    generation calls, billed to the platform project.

Nothing is written to the catalog without ``--write``; without it the run is
a verification that prints what it found.

Location statuses: available | not_found (404) | no_access (401/403) |
no_quota (429 / RESOURCE_EXHAUSTED) | unknown.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

try:
    from djinnite.config_loader import load_ai_config, CONFIG_DIR
    from djinnite.scripts.model_overrides import save_catalog
    from djinnite.scripts.update_models import _ensure_effort_style, _probe_all_capabilities_for_models
    from djinnite.ai_providers import PROVIDERS
    from djinnite.ai_providers.platforms import get_platform
except ImportError:
    _project_root = str(Path(__file__).resolve().parent.parent)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from config_loader import load_ai_config, CONFIG_DIR  # type: ignore
    from scripts.model_overrides import save_catalog  # type: ignore
    from scripts.update_models import _ensure_effort_style, _probe_all_capabilities_for_models  # type: ignore
    from ai_providers import PROVIDERS  # type: ignore
    from ai_providers.platforms import get_platform  # type: ignore


# Capability fields compared and recorded for the platform.
_CAP_FIELDS = (
    "structured_json", "temperature", "thinking", "web_search",
    "json_with_search", "thinking_style", "incompatible",
)

_STATUS_TAG = {
    "available": "[OK]",
    "not_found": "[WARN]",
    "no_access": "[WARN]",
    "no_quota": "[WARN]",
    "unknown": "[WARN]",
}


def _say(line: str = "") -> None:
    print(line, flush=True)


def _candidates(catalog: dict, provider: str, model_ids: Optional[list]) -> list[dict]:
    """Catalog models of ``provider`` worth probing (or exactly ``model_ids``)."""
    models = catalog.get(provider, {}).get("models", [])
    if model_ids:
        wanted = set(model_ids)
        return [m for m in models if m.get("id") in wanted]
    out = []
    for m in models:
        if m.get("disabled"):
            continue
        mods = m.get("modalities") or {}
        if isinstance(mods, dict):
            if "text" not in mods.get("input", []) or "text" not in mods.get("output", ["text"]):
                continue
        out.append(m)
    return out


def _default_factory(platform: str, project: str, quota: Optional[str]) -> Callable:
    """Build ``make(provider_name, model_id, location)`` for platform mode."""
    def make(provider_name: str, model_id: str, location: str):
        cls = PROVIDERS[provider_name]
        return cls(
            api_key=None,
            model=model_id,
            require_pricing=False,
            platform=platform,
            project_id=project,
            location=location,
            quota_project=quota,
        )
    return make


def _fmt(value) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)


def run(argv: Optional[list] = None, make_provider: Optional[Callable] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Probe a cloud platform's model availability and capabilities")
    parser.add_argument("--platform", required=True, help="Platform name, e.g. vertexai")
    parser.add_argument("--provider", action="append", default=None,
                        help="Provider to probe (repeatable). Default: every provider the platform hosts.")
    parser.add_argument("--model", action="append", default=None,
                        help="Model ID to probe (repeatable). Default: the catalog's text models.")
    parser.add_argument("--location", action="append", default=None,
                        help="Location to probe (repeatable). Default: platforms.<name>.locations in ai_config.json.")
    parser.add_argument("--project", default=None, help="Override platforms.<name>.project_id")
    parser.add_argument("--quota-project", default=None, help="Override platforms.<name>.quota_project")
    parser.add_argument("--capabilities", action="store_true",
                        help="Also run the capability probe suite on the platform (billed generation calls).")
    parser.add_argument("--write", action="store_true",
                        help="Persist results to the catalog (via model_overrides.save_catalog).")
    parser.add_argument("--config", default=None, help="Path to ai_config.json")
    parser.add_argument("--catalog", default=None, help="Path to model_catalog.json")
    args = parser.parse_args(argv)

    spec = get_platform(args.platform)
    ai_config = load_ai_config(Path(args.config) if args.config else None)
    pconf = ai_config.platforms.get(args.platform)
    project = args.project or (pconf.project_id if pconf else None)
    quota = args.quota_project or (pconf.quota_project if pconf else None)
    locations = args.location or (list(pconf.locations) if pconf else [])
    providers = args.provider or sorted(spec.providers)

    if not project:
        _say(f"[FAIL] No project for platform '{args.platform}': set "
             f"platforms.{args.platform}.project_id in ai_config.json or pass --project")
        return 2
    if not locations:
        _say(f"[FAIL] No locations for platform '{args.platform}': set "
             f"platforms.{args.platform}.locations in ai_config.json or pass --location")
        return 2
    for p in providers:
        if p not in spec.providers:
            _say(f"[FAIL] Platform '{args.platform}' does not host provider '{p}' "
                 f"(hosts: {sorted(spec.providers)})")
            return 2

    catalog_path = Path(args.catalog) if args.catalog else CONFIG_DIR / "model_catalog.json"
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    make = make_provider or _default_factory(args.platform, project, quota)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    _say(f"[INFO] platform={args.platform} project={project} quota_project={quota or '-'}")
    _say(f"[INFO] locations={','.join(locations)} providers={','.join(providers)} "
         f"capabilities={'yes' if args.capabilities else 'no'} write={'yes' if args.write else 'no'}")

    counts = {s: 0 for s in _STATUS_TAG}
    probed_models = 0
    for provider in providers:
        models = _candidates(catalog, provider, args.model)
        _say(f"\n{provider}: {len(models)} models x {len(locations)} locations")
        for model in models:
            model_id = model["id"]
            probed_models += 1
            statuses: dict = {}
            for loc in locations:
                try:
                    status, detail = make(provider, model_id, loc).probe_availability()
                except Exception as e:  # construction failed (e.g. no ADC)
                    status, detail = "unknown", f"{type(e).__name__}: {e}"
                statuses[loc] = status
                counts[status] = counts.get(status, 0) + 1
                tail = f" ({' '.join(detail.split())[:140]})" if detail else ""
                _say(f"  {_STATUS_TAG.get(status, '[WARN]')} {provider} {model_id} @{loc} {status}{tail}")

            prior = (model.get("platforms") or {}).get(args.platform) or {}
            block = {
                "locations": {**(prior.get("locations") or {}), **statuses},
                "probed": today,
            }
            if prior.get("capabilities") is not None:
                block["capabilities"] = prior["capabilities"]

            if args.capabilities:
                at = next((loc for loc in locations if statuses.get(loc) == "available"), None)
                if at is None:
                    _say(f"  [SKIP] {provider} {model_id}: capabilities not probed "
                         f"(not available at any probed location)")
                else:
                    _say(f"  [CHECK] {provider} {model_id}: probing capabilities @{at}")
                    results = _probe_all_capabilities_for_models(
                        [model], PROVIDERS[provider], provider, None,
                        make_instance=lambda mid, _p=provider, _l=at: make(_p, mid, _l),
                    )
                    caps = results.get(model_id)
                    if caps is not None:
                        recorded = {k: caps.get(k) for k in _CAP_FIELDS}
                        direct = model.get("capabilities") or {}
                        # Probes report only the thinking shapes they send.
                        # Effort comes from the provider's effort_levels,
                        # exactly as in update_models -- otherwise the
                        # platform block would hide "effort" on the platform.
                        effort_view = {"thinking_style": recorded.get("thinking_style"),
                                       "effort_levels": direct.get("effort_levels")}
                        _ensure_effort_style(effort_view)
                        recorded["thinking_style"] = effort_view["thinking_style"]
                        block["capabilities"] = recorded
                        diffs = 0
                        for k in _CAP_FIELDS:
                            pv, dv = recorded.get(k), direct.get(k)
                            if pv is not None and pv != dv:
                                diffs += 1
                                _say(f"    [DIFF] {k}: direct={_fmt(dv)} platform={_fmt(pv)}")
                        if not diffs:
                            _say("    [SAME] platform capabilities match direct mode")

            model.setdefault("platforms", {})[args.platform] = block

    summary = " ".join(f"{s}={n}" for s, n in counts.items() if n)
    _say(f"\n[INFO] Probed {probed_models} models: {summary or 'nothing'}")
    if args.write:
        save_catalog(catalog, catalog_path)
    else:
        _say("[INFO] Dry run: catalog not written (pass --write to persist)")
    return 0


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
