"""
scripts/probe_platform.py against the real platform (opt-in: ``--e2e-platform``).

Runs the script as a subprocess, exactly as a maintainer would, against a
TEMPORARY copy of the catalog -- the real catalog is never written.

    uv run pytest tests/test_e2e_probe_platform.py --e2e-platform -rA -s
"""

import hashlib
import json
import shutil
import subprocess
import sys

import pytest

from djinnite.config_loader import CONFIG_DIR, LOCATION_STATUS_VALUES
from djinnite.tests._e2e import say

pytestmark = pytest.mark.e2e_platform

REAL_CATALOG = CONFIG_DIR / "model_catalog.json"


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _probe(cfg, tmp_path, *extra):
    catalog = tmp_path / "model_catalog.json"
    shutil.copyfile(REAL_CATALOG, catalog)
    config = tmp_path / "ai_config.json"
    config.write_text(json.dumps({
        "platforms": {"vertexai": {"project_id": cfg.project,
                                   "locations": list(cfg.locations)}},
        "providers": {},
    }), encoding="utf-8")
    cmd = [sys.executable, "-u", "-m", "djinnite.scripts.probe_platform",
           "--platform", "vertexai", "--config", str(config), "--catalog", str(catalog),
           "--model", cfg.gemini_model, "--model", cfg.claude_model, "--write", *extra]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    for line in out.stdout.splitlines():
        say(f"    | {line}")
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.isascii()
    return json.loads(catalog.read_text(encoding="utf-8"))


def _block(catalog, provider, model_id):
    for m in catalog[provider]["models"]:
        if m["id"] == model_id:
            return m.get("platforms", {}).get("vertexai")
    raise AssertionError(f"{model_id} missing from the catalog copy")


def test_s1_availability_written_to_a_catalog_copy(e2e_session, e2e_availability, tmp_path):
    cfg = e2e_session
    before = _digest(REAL_CATALOG)
    written = _probe(cfg, tmp_path)
    for provider, model in (("gemini", cfg.gemini_model), ("claude", cfg.claude_model)):
        block = _block(written, provider, model)
        assert block and block["probed"], (provider, model)
        for loc in cfg.locations:
            status = block["locations"][loc]
            assert status in LOCATION_STATUS_VALUES
            # Same calls as the session's discovery: same answer.
            assert status == e2e_availability[(provider, model, loc)][0], (provider, model, loc)
    assert _digest(REAL_CATALOG) == before, "probe_platform touched the real catalog"


@pytest.mark.e2e_extended
def test_x1_capabilities_probe(e2e_session, tmp_path):
    """The full capability suite through Vertex (billed: ~10 small calls per model)."""
    cfg = e2e_session
    written = _probe(cfg, tmp_path, "--provider", "gemini", "--capabilities")
    caps = _block(written, "gemini", cfg.gemini_model).get("capabilities")
    assert caps, "no platform capabilities recorded"
    for field in ("structured_json", "thinking", "thinking_style"):
        assert caps.get(field), f"{field} not probed on the platform: {caps}"
