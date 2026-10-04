"""
``update_models --reprobe <model-id>`` touches only the models it names (offline).

Regression for the 2026-10-04 bug: reprobing two Claude models also re-priced
every floating model on every provider (paid AI calls), added "effort" to an
untargeted Claude model, and refreshed every listed Claude model's fields.

    uv run pytest tests/test_update_models_scope.py -v
"""

import copy
import json

from djinnite.config_loader import AIConfig
from djinnite.scripts import update_model_costs as costs_mod
from djinnite.scripts.update_models import _refresh_provider, _reprobe_scope

CATALOG = {
    "claude": {"models": [{"id": "claude-a"}, {"id": "claude-b"}]},
    "gemini": {"models": [{"id": "gemini-x"}]},
}


# ------------------------------------------------------------------ scope

def test_scope_whole_catalog():
    assert _reprobe_scope(None, CATALOG) is None
    assert _reprobe_scope({"all"}, CATALOG) is None
    assert _reprobe_scope({"all", "claude-a"}, CATALOG) is None


def test_scope_provider_wildcard_and_model_ids():
    assert _reprobe_scope({"claude:all"}, CATALOG) == {"claude": None}
    assert _reprobe_scope({"claude-a", "gemini-x"}, CATALOG) == {
        "claude": {"claude-a"}, "gemini": {"gemini-x"}}
    # A wildcard swallows bare ids of the same provider.
    assert _reprobe_scope({"claude:all", "claude-b"}, CATALOG) == {"claude": None}


def test_scope_unknown_id_is_reported_and_ignored(capsys):
    assert _reprobe_scope({"no-such-model"}, CATALOG) == {}
    assert "no catalog model has this id" in capsys.readouterr().out


# -------------------------------------------------------- provider refresh

class _Fake:
    """A provider whose list_models() returns a target, a bystander and a new model."""
    PROVIDER_NAME = "claude"
    probed: list = []

    def __init__(self, api_key=None, model=None, **kw):
        self.model = model

    def list_models(self):
        common = {"context_window": 1_000_000, "max_output_tokens": 128_000,
                  "modalities": ["text"], "cost_tier": "standard"}
        return [
            {"id": "target", "name": "Target", "effort_levels": ["low", "high"], **common},
            {"id": "bystander", "name": "Bystander", "effort_levels": ["low", "high"], **common},
            {"id": "brand-new", "name": "New", **common},
        ]

    def discover_modalities(self, model_id):
        return {"input": ["text"], "output": ["text"]}

    def _record(self):
        _Fake.probed.append(self.model)

    def probe_structured_json(self):
        self._record()
        return True

    def probe_temperature(self): return False
    def probe_json_with_search(self): return True
    def probe_web_search(self): return True
    def probe_thinking_style(self): return ["adaptive", "between_tools"]
    def probe_thinking_disable(self): return False
    def probe_incompatible_combinations(self, states): return []


def _existing():
    def model(mid):
        return {
            "id": mid, "name": mid, "context_window": 200_000, "max_output_tokens": 64_000,
            "modalities": {"input": ["text"], "output": ["text"]},
            "costing": {"input_per_1m": 2.0, "output_per_1m": 10.0, "source": "published",
                        "updated": "2026-01-01"},
            "capabilities": {"thinking": ["on", "off"], "thinking_style": ["adaptive"],
                             "effort_levels": ["low", "high"],
                             "incompatible": [{"thinking": "on", "structured_json": "on"}]},
        }
    return {"models": [model("target"), model("bystander")], "last_updated": "2026-01-01"}


def _p_config():
    return type("PC", (), {"api_key": "k", "default_model": "target"})()


def test_scoped_refresh_touches_only_the_target():
    _Fake.probed = []
    existing = _existing()
    before = copy.deepcopy(existing)
    block = _refresh_provider(_Fake, _p_config(), existing, AIConfig(),
                              reprobe={"target"}, targets={"target"})

    # Membership, order and last_updated unchanged; the new model is not added.
    assert [m["id"] for m in block["models"]] == ["target", "bystander"]
    assert block["last_updated"] == "2026-01-01"
    # The bystander is exactly what it was -- no effort pass, no field refresh.
    assert block["models"][1] == before["models"][1]
    # Only the target was probed, and it was re-probed.
    assert _Fake.probed == ["target"]
    caps = block["models"][0]["capabilities"]
    assert caps["thinking"] == ["on"]
    assert caps["thinking_style"] == ["adaptive", "between_tools", "effort"]
    assert caps["incompatible"] == []
    assert block["models"][0]["context_window"] == 1_000_000  # refreshed from the API


def test_unscoped_refresh_is_unchanged_behavior():
    _Fake.probed = []
    block = _refresh_provider(_Fake, _p_config(), _existing(), AIConfig(),
                              reprobe=None, targets=None)
    assert [m["id"] for m in block["models"]] == ["target", "bystander", "brand-new"]
    assert block["last_updated"] != "2026-01-01"


# --------------------------------------------------------------- cost pass

def _cost_files(tmp_path):
    catalog = {
        "gemini": {"models": [{  # floating: re-priced on every unscoped run
            "id": "gemini-flash-latest", "name": "Flash latest",
            "costing": {"input_per_1m": 0.75, "output_per_1m": 3.75, "source": "published",
                        "updated": "2026-10-01"}}]},
        "claude": {"models": [{  # fixed and priced: never re-estimated
            "id": "claude-a", "name": "A",
            "costing": {"input_per_1m": 2.0, "output_per_1m": 10.0, "source": "published",
                        "updated": "2026-10-01"}}]},
    }
    cat = tmp_path / "model_catalog.json"
    cat.write_text(json.dumps(catalog), encoding="utf-8")
    cfg = tmp_path / "ai_config.json"
    cfg.write_text(json.dumps({"providers": {
        "claude": {"api_key": "k"}, "gemini": {"api_key": "k"}}}), encoding="utf-8")
    return cat, cfg


def _run_costs(tmp_path, monkeypatch, scope):
    calls = []
    monkeypatch.setattr(costs_mod, "estimate_costs_with_ai",
                        lambda models, provider, *a, **k: calls.append(
                            (provider, [m["id"] for m in models])) or {})
    cat, cfg = _cost_files(tmp_path)
    before = json.loads(cat.read_text(encoding="utf-8"))
    costs_mod.update_model_costs(catalog_path=cat, config_path=cfg, scope=scope)
    return calls, before, json.loads(cat.read_text(encoding="utf-8"))


def test_cost_pass_respects_a_model_scope(tmp_path, monkeypatch):
    calls, before, after = _run_costs(tmp_path, monkeypatch, {"claude": {"claude-a"}})
    assert calls == []                              # nothing re-priced
    assert after["gemini"] == before["gemini"]      # out-of-scope provider untouched


def test_cost_pass_unscoped_still_reprices_floating(tmp_path, monkeypatch):
    calls, _, _ = _run_costs(tmp_path, monkeypatch, None)
    assert ("gemini", ["gemini-flash-latest"]) in calls
