"""
Maintenance scripts and access paths (offline; no network, no paid calls).

ACCESS_PATHS_DESIGN.md "Maintenance scripts": the estimator resolver shared
by ``update_models`` and ``update_model_costs``, the per-type direct-entry
choice of ``update_models``, and ``update_model_costs`` stopping without a
write when no direct-mode estimator is configured.

Every AIConfig here comes from ``load_ai_config`` on JSON written to
``tmp_path``, so the real loader is exercised. No test argument is named
``provider``: conftest treats that name as a live test.

    uv run pytest tests/test_access_paths_scripts.py -v -rs
"""

import json
import sys

import pytest

from djinnite.config_loader import load_ai_config
from djinnite.scripts import update_model_costs as costs_mod
from djinnite.scripts import update_models as um
from djinnite.scripts.estimator import Estimator, EstimatorUnavailable, resolve_estimator

VERTEX = {"vertexai": {"project_id": "my-project", "locations": ["global"]}}
CLAUDE_VERTEX = {"provider": "claude", "mode": "platform", "platform": "vertexai",
                 "location": "global", "default_model": "claude-v"}


def _config(tmp_path, providers, default="gemini", platforms=VERTEX):
    path = tmp_path / "ai_config.json"
    path.write_text(json.dumps({"providers": providers, "default_provider": default,
                                "platforms": platforms}), encoding="utf-8")
    return load_ai_config(path)


# ------------------------------------------------------- estimator precedence

def test_cli_model_takes_its_type_from_the_catalog(tmp_path):
    cfg = _config(tmp_path, {
        "gemini": {"api_key": "kg", "default_model": "gemini-x"},
        "chatgpt": {"api_key": "ko", "default_model": "gpt-x"},
    })
    catalog = {"chatgpt": {"models": [{"id": "gpt-big"}]}, "gemini": {"models": []}}
    est = resolve_estimator(cfg, cli_model="gpt-big", catalog=catalog,
                            known_defaults={"provider": "gemini", "model": "gemini-x"})
    assert est == Estimator("chatgpt", "gpt-big", "chatgpt", "ko")


def test_cli_model_not_in_catalog_uses_the_default_entry_type(tmp_path):
    cfg = _config(tmp_path, {
        "work": {"provider": "claude", "api_key": "kw", "default_model": "claude-w"},
        "gemini": {"api_key": "kg"},
    }, default="work")
    est = resolve_estimator(cfg, cli_model="claude-new", catalog={},
                            known_defaults={"provider": "gemini", "model": "gemini-x"})
    assert est == Estimator("claude", "claude-new", "work", "kw")


def test_known_defaults_provider_is_a_type(tmp_path):
    cfg = _config(tmp_path, {
        "gemini": {"api_key": "kg", "default_model": "gemini-x"},
        "claude": {"api_key": "kc", "default_model": "claude-c"},
    }, default="gemini")
    est = resolve_estimator(cfg, known_defaults={"provider": "claude", "model": "claude-est"})
    assert est == Estimator("claude", "claude-est", "claude", "kc")


def test_known_defaults_without_provider_keeps_the_gemini_default(tmp_path):
    cfg = _config(tmp_path, {"gemini": {"api_key": "kg"}})
    est = resolve_estimator(cfg, known_defaults={"model": "gemini-est"})
    assert (est.provider_type, est.entry) == ("gemini", "gemini")


def test_estimator_runs_through_direct_entry_not_the_default_entry(tmp_path):
    """One selection rule: direct_entry(type), the entry update_models refreshes with."""
    cfg = _config(tmp_path, {
        "claude": {"api_key": "kc"},
        "claude-work": {"provider": "claude", "api_key": "kw", "default_model": "claude-w"},
    }, default="claude-work")
    est = resolve_estimator(cfg, known_defaults={"provider": "claude", "model": "claude-est"})
    assert est.entry == "claude" == cfg.direct_entry("claude")
    assert est.api_key == "kc"


def test_falls_back_to_the_default_entry_type_and_model(tmp_path):
    cfg = _config(tmp_path, {
        "claude": {"api_key": "kc", "default_model": "claude-c"},
        "gemini": {"api_key": "kg", "default_model": "gemini-g"},
    }, default="claude")
    est = resolve_estimator(cfg, known_defaults={})
    assert est == Estimator("claude", "claude-c", "claude", "kc")


def test_platform_default_entry_falls_back_to_the_direct_entry_of_its_type(tmp_path):
    cfg = _config(tmp_path, {
        "claude-vertex": CLAUDE_VERTEX,
        "claude-direct": {"provider": "claude", "api_key": "kd", "default_model": "claude-d"},
    }, default="claude-vertex")
    est = resolve_estimator(cfg, known_defaults={})
    # Type and model from the default entry; it runs through the direct one.
    assert est == Estimator("claude", "claude-v", "claude-direct", "kd")


def test_platform_only_type_is_unavailable(tmp_path):
    cfg = _config(tmp_path, {"claude-vertex": CLAUDE_VERTEX,
                             "gemini": {"api_key": "kg"}}, default="gemini")
    with pytest.raises(EstimatorUnavailable) as exc:
        resolve_estimator(cfg, known_defaults={"provider": "claude", "model": "claude-est"})
    msg = str(exc.value)
    assert "no usable direct-mode entry for estimator type 'claude'" in msg
    assert "claude-vertex [platform]" in msg
    assert "needs a provider API key" in msg


def test_placeholder_key_is_unavailable_and_says_why(tmp_path):
    cfg = _config(tmp_path, {"claude": {"api_key": "your-anthropic-api-key-here"}})
    with pytest.raises(EstimatorUnavailable, match=r"claude \[direct, no usable api_key\]"):
        resolve_estimator(cfg, known_defaults={"provider": "claude", "model": "claude-est"})


def test_ambiguous_direct_entries_are_unavailable(tmp_path):
    cfg = _config(tmp_path, {
        "claude-a": {"provider": "claude", "api_key": "ka"},
        "claude-b": {"provider": "claude", "api_key": "kb"},
        "gemini": {"api_key": "kg"},
    }, default="gemini")
    with pytest.raises(EstimatorUnavailable, match="several direct-mode entries"):
        resolve_estimator(cfg, known_defaults={"provider": "claude", "model": "claude-est"})


def test_ambiguity_is_not_resolved_by_the_default_entry(tmp_path):
    cfg = _config(tmp_path, {
        "claude-a": {"provider": "claude", "api_key": "ka"},
        "claude-b": {"provider": "claude", "api_key": "kb"},
    }, default="claude-b")
    with pytest.raises(EstimatorUnavailable, match="several direct-mode entries"):
        resolve_estimator(cfg, known_defaults={"provider": "claude", "model": "claude-est"})


@pytest.mark.parametrize("deny, needs, blocked", [
    (["web_search"], ("web_search",), True),
    ({"claude-est": ["structured_json"]}, ("structured_json",), True),
    ({"other-model": ["web_search"]}, ("web_search",), False),
    (["web_search"], ("structured_json",), False),
    (["web_search"], (), False),
])
def test_estimator_unavailable_when_its_entry_denies_what_estimation_needs(tmp_path, deny, needs, blocked):
    cfg = _config(tmp_path, {"claude": {"api_key": "kc", "deny": deny,
                                        "deny_reason": "test policy"}})
    kd = {"provider": "claude", "model": "claude-est"}
    if blocked:
        with pytest.raises(EstimatorUnavailable, match="denies .* test policy"):
            resolve_estimator(cfg, known_defaults=kd, needs=needs)
    else:
        assert resolve_estimator(cfg, known_defaults=kd, needs=needs).entry == "claude"


def test_unknown_estimator_type_and_missing_model_are_unavailable(tmp_path):
    cfg = _config(tmp_path, {"gemini": {"api_key": "kg"}})
    with pytest.raises(EstimatorUnavailable, match="not a provider type"):
        resolve_estimator(cfg, known_defaults={"provider": "openai", "model": "gpt-x"})
    with pytest.raises(EstimatorUnavailable, match="no estimator model"):
        resolve_estimator(cfg, known_defaults={})  # gemini entry has no default_model
    with pytest.raises(EstimatorUnavailable, match="not a configured, enabled entry"):
        resolve_estimator(_config(tmp_path, {"gemini": {"api_key": "kg"}}, default="nope"),
                          known_defaults={})


# ------------------------------------------------- update_models entry choice

@pytest.mark.parametrize("providers, ptype, expected", [
    # legacy: one entry named after its type
    ({"claude": {"api_key": "kc"}}, "claude",
     ("claude", "Updating claude models (entry 'claude')...")),
    # mixed direct + platform of one type: the direct one
    ({"claude-vertex": CLAUDE_VERTEX,
      "claude-direct": {"provider": "claude", "api_key": "kd"}}, "claude",
     ("claude-direct", "Updating claude models (entry 'claude-direct')...")),
    # platform only
    ({"claude-vertex": CLAUDE_VERTEX}, "claude",
     (None, "[SKIP] claude: platform mode only (claude-vertex) -- use probe_platform")),
    # nothing of that type configured
    ({"gemini": {"api_key": "kg"}}, "claude",
     (None, "[WARN] Provider claude not configured, skipping.")),
    # a direct entry without a usable key is "not configured", not platform-only
    ({"claude": {"api_key": "your-key-here"}, "claude-vertex": CLAUDE_VERTEX}, "claude",
     (None, "[WARN] Provider claude not configured, skipping.")),
])
def test_entry_for_refresh(tmp_path, providers, ptype, expected):
    assert um._entry_for_refresh(_config(tmp_path, providers), ptype) == expected


def test_entry_for_refresh_ambiguous_fails(tmp_path):
    cfg = _config(tmp_path, {"claude-a": {"provider": "claude", "api_key": "ka"},
                             "claude-b": {"provider": "claude", "api_key": "kb"}})
    entry, message = um._entry_for_refresh(cfg, "claude")
    assert entry is None
    assert message.startswith("[FAIL] claude: ")
    assert "claude-a, claude-b" in message


def test_update_models_refreshes_through_the_direct_entry(tmp_path, monkeypatch, capsys):
    """The main loop hands each type's direct entry to _refresh_provider."""
    _config(tmp_path, {
        "claude-vertex": CLAUDE_VERTEX,
        "claude-direct": {"provider": "claude", "api_key": "kd", "default_model": "claude-d"},
        "gemini-vertex": {"provider": "gemini", "mode": "platform", "platform": "vertexai"},
    }, default="claude-direct")
    cat = tmp_path / "model_catalog.json"
    cat.write_text("{}", encoding="utf-8")

    refreshed = []
    monkeypatch.setattr(um, "_refresh_provider",
                        lambda cls, pc, *a, **k: refreshed.append((cls.__name__, pc.name)))
    monkeypatch.setattr(um, "save_catalog", lambda *a, **k: None)
    monkeypatch.setattr(costs_mod, "update_model_costs", lambda **k: True)
    monkeypatch.setattr(sys, "argv", ["update_models", "--config",
                                      str(tmp_path / "ai_config.json"), "--catalog", str(cat)])
    um.update_models()

    assert refreshed == [("ClaudeProvider", "claude-direct")]
    out = capsys.readouterr().out
    assert "Updating claude models (entry 'claude-direct')..." in out
    assert "[SKIP] gemini: platform mode only (gemini-vertex) -- use probe_platform" in out
    assert "[WARN] Provider chatgpt not configured, skipping." in out


# ------------------------------------------------ update_models estimation

def test_limit_estimation_skips_without_a_direct_estimator(tmp_path, monkeypatch, capsys):
    cfg = _config(tmp_path, {"claude-vertex": CLAUDE_VERTEX}, default="claude-vertex")
    monkeypatch.setattr(um, "_estimator_config", {"provider": "claude", "model": "claude-est"})
    monkeypatch.setattr(type(cfg), "build_provider",
                        lambda *a, **k: pytest.fail("must not build an estimator"))
    assert um.estimate_output_limits_with_ai([{"id": "m"}], "claude", cfg) == {}
    assert um.estimate_modalities_with_ai([{"id": "m"}], "claude", cfg) == {}
    assert "no usable direct-mode entry" in capsys.readouterr().out


def test_estimation_builds_from_the_estimator_entry(tmp_path, monkeypatch):
    cfg = _config(tmp_path, {
        "claude-vertex": CLAUDE_VERTEX,
        "claude-direct": {"provider": "claude", "api_key": "kd"},
    }, default="claude-vertex")
    monkeypatch.setattr(um, "_estimator_config", {"provider": "claude", "model": "claude-est"})
    built = []

    class _Resp:
        content = json.dumps({"models": [{"id": "m", "input": ["text"], "output": ["text"]}]})

    class _Est:
        def generate_json(self, **kw):
            return _Resp()

    def fake_build(self, name, model=None, **overrides):
        built.append((name, model, overrides))
        return _Est()

    monkeypatch.setattr(type(cfg), "build_provider", fake_build)
    got = um.estimate_modalities_with_ai([{"id": "m"}], "claude", cfg)
    assert got == {"m": {"input": ["text"], "output": ["text"]}}
    assert built == [("claude-direct", "claude-est", {"require_pricing": False})]


# ------------------------------------------------------------- cost pass

def _cost_catalog(tmp_path):
    catalog = {"claude": {"models": [
        {"id": "claude-priced", "name": "Priced",   # fixed price: kept
         "costing": {"input_per_1m": 2.0, "output_per_1m": 10.0, "source": "published",
                     "updated": "2026-10-01"}},
        {"id": "claude-x-latest", "name": "Latest",  # floating: always re-priced
         "costing": {"input_per_1m": 3.0, "output_per_1m": 15.0, "source": "published",
                     "updated": "2026-10-01"}},
        {"id": "claude-new", "name": "New"},         # no price yet: needs one
    ]}}
    path = tmp_path / "model_catalog.json"
    path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    return path


@pytest.mark.parametrize("estimator_defaults", [
    {"provider": "claude", "model": "claude-est"},  # known_model_defaults names the type
    {},                                             # the default entry's type
])
def test_cost_pass_with_only_a_platform_claude_entry_writes_nothing(
        tmp_path, monkeypatch, capsys, estimator_defaults):
    _config(tmp_path, {"claude-vertex": CLAUDE_VERTEX}, default="claude-vertex")
    cat = _cost_catalog(tmp_path)
    before = cat.read_bytes()
    monkeypatch.setattr(costs_mod, "_estimator_config", estimator_defaults)
    monkeypatch.setattr(costs_mod, "estimate_costs_with_ai",
                        lambda *a, **k: pytest.fail("must not estimate"))
    monkeypatch.setattr(costs_mod, "save_catalog",
                        lambda *a, **k: pytest.fail("must not write"))

    ok = costs_mod.update_model_costs(catalog_path=cat,
                                      config_path=tmp_path / "ai_config.json")

    assert ok is False
    assert cat.read_bytes() == before   # existing prices untouched, nothing nulled
    out = capsys.readouterr().out
    assert "[FAIL] Estimator: no usable direct-mode entry for estimator type 'claude'" in out
    assert "claude-vertex [platform]" in out
    assert out.rstrip().endswith("-- no prices changed")


def test_cost_cli_exits_1_without_an_estimator(tmp_path, monkeypatch):
    _config(tmp_path, {"claude-vertex": CLAUDE_VERTEX}, default="claude-vertex")
    cat = _cost_catalog(tmp_path)
    before = cat.read_bytes()
    monkeypatch.setattr(costs_mod, "_estimator_config", {})
    monkeypatch.setattr(sys, "argv", ["update_model_costs", "--config",
                                      str(tmp_path / "ai_config.json"), "--catalog", str(cat)])
    with pytest.raises(SystemExit) as exc:
        costs_mod.main()
    assert exc.value.code == 1
    assert cat.read_bytes() == before


def test_cost_pass_estimates_through_the_direct_entry(tmp_path, monkeypatch, capsys):
    cfg_path = tmp_path / "ai_config.json"
    _config(tmp_path, {
        "claude-vertex": CLAUDE_VERTEX,
        "claude-direct": {"provider": "claude", "api_key": "kd"},
    }, default="claude-vertex")
    cat = _cost_catalog(tmp_path)
    monkeypatch.setattr(costs_mod, "_estimator_config",
                        {"provider": "claude", "model": "claude-est"})
    built, calls = [], []

    def fake_build(self, name, model=None, **overrides):
        built.append((name, model, overrides))
        return object()

    def fake_estimate(models, section, est_type, est_model, api_key, logger=None, **k):
        calls.append((section, [m["id"] for m in models], est_type, est_model, api_key))
        k["make_provider"]()  # the factory builds from the estimator's entry
        return {}

    from djinnite.config_loader import AIConfig
    monkeypatch.setattr(AIConfig, "build_provider", fake_build)
    monkeypatch.setattr(costs_mod, "estimate_costs_with_ai", fake_estimate)

    assert costs_mod.update_model_costs(catalog_path=cat, config_path=cfg_path) is True
    assert "Estimator: claude/claude-est (entry 'claude-direct')" in capsys.readouterr().out
    assert calls == [("claude", ["claude-x-latest", "claude-new"], "claude", "claude-est", "kd")]
    assert built == [("claude-direct", "claude-est", {"require_pricing": False})]


def test_cost_pass_survives_two_catalog_sections_and_saves(tmp_path, monkeypatch):
    """Regression: the per-model result once shadowed the estimator, crashing section 2."""
    cfg_path = tmp_path / "ai_config.json"
    _config(tmp_path, {"claude": {"api_key": "kc"}, "gemini": {"api_key": "kg"}},
            default="claude")
    catalog = {
        "claude": {"models": [{"id": "claude-x-latest", "name": "C"}]},
        "gemini": {"models": [{"id": "gemini-y-latest", "name": "G"}]},
    }
    cat = tmp_path / "model_catalog.json"
    cat.write_text(json.dumps(catalog), encoding="utf-8")
    monkeypatch.setattr(costs_mod, "_estimator_config",
                        {"provider": "claude", "model": "claude-est"})
    sections, saved = [], []

    def fake_estimate(models, section, est_type, est_model, api_key, logger=None, **k):
        sections.append((section, est_type, est_model, api_key))
        return {m["id"]: {"input_per_1m": 1.0, "output_per_1m": 2.0,
                          "search_cost_per_unit": None,
                          "source_url": "https://example.com/pricing",
                          "published_figure": "$1 / $2 per 1M (Standard)"}
                for m in models}

    monkeypatch.setattr(costs_mod, "estimate_costs_with_ai", fake_estimate)
    monkeypatch.setattr(costs_mod, "save_catalog", lambda data, *a, **k: saved.append(data))

    assert costs_mod.update_model_costs(catalog_path=cat, config_path=cfg_path) is True
    assert [s[0] for s in sections] == ["claude", "gemini"]
    assert all(s[1:] == ("claude", "claude-est", "kc") for s in sections)
    assert saved, "the cost pass must save after estimating every section"
    for ptype, mid in (("claude", "claude-x-latest"), ("gemini", "gemini-y-latest")):
        costing = saved[-1][ptype]["models"][0]["costing"]
        assert (costing["input_per_1m"], costing["output_per_1m"]) == (1.0, 2.0)


def test_cost_pass_stops_when_the_estimator_entry_denies_web_search(tmp_path, monkeypatch, capsys):
    _config(tmp_path, {"claude": {"api_key": "kc", "deny": ["web_search"]}}, default="claude")
    cat = _cost_catalog(tmp_path)
    before = cat.read_bytes()
    monkeypatch.setattr(costs_mod, "_estimator_config",
                        {"provider": "claude", "model": "claude-est"})
    monkeypatch.setattr(costs_mod, "estimate_costs_with_ai",
                        lambda *a, **k: pytest.fail("must not estimate"))
    monkeypatch.setattr(costs_mod, "save_catalog",
                        lambda *a, **k: pytest.fail("must not write"))
    ok = costs_mod.update_model_costs(catalog_path=cat, config_path=tmp_path / "ai_config.json")
    assert ok is False
    assert cat.read_bytes() == before
    out = capsys.readouterr().out
    assert "[FAIL] Estimator: estimator entry 'claude' denies web_search" in out


def test_failed_estimate_keeps_an_existing_price_and_marks_only_unpriced(tmp_path, monkeypatch):
    """R8: an estimator that resolves but whose requests all fail changes no price."""
    _config(tmp_path, {"claude": {"api_key": "kc"}}, default="claude")
    cat = _cost_catalog(tmp_path)  # claude-x-latest priced (floating), claude-new unpriced
    monkeypatch.setattr(costs_mod, "_estimator_config",
                        {"provider": "claude", "model": "claude-est"})
    monkeypatch.setattr(costs_mod, "estimate_costs_with_ai", lambda *a, **k: {})
    saved = []
    monkeypatch.setattr(costs_mod, "save_catalog", lambda data, *a, **k: saved.append(data))

    costs_mod.update_model_costs(catalog_path=cat, config_path=tmp_path / "ai_config.json")

    models = {m["id"]: m for m in saved[-1]["claude"]["models"]}
    kept = models["claude-x-latest"]["costing"]
    assert (kept["input_per_1m"], kept["output_per_1m"], kept["source"]) == (3.0, 15.0, "published")
    assert models["claude-new"]["costing"]["source"] == "failed"  # had no price to lose


def test_update_models_reports_a_cost_pass_that_did_not_run(tmp_path, monkeypatch, capsys):
    _config(tmp_path, {"claude": {"api_key": "kc"}}, default="claude")
    cat = tmp_path / "model_catalog.json"
    cat.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(um, "_refresh_provider", lambda *a, **k: None)
    monkeypatch.setattr(um, "save_catalog", lambda *a, **k: None)
    monkeypatch.setattr(costs_mod, "update_model_costs", lambda **k: False)
    monkeypatch.setattr(sys, "argv", ["update_models", "--config",
                                      str(tmp_path / "ai_config.json"), "--catalog", str(cat)])
    um.update_models()
    assert "[WARN] Cost estimation did not run" in capsys.readouterr().out
