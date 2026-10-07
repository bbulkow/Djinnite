"""
Offline tests for the validation scripts' access-path handling.

validate_ai, validate_json and validate_models iterate ai_config entries
(ACCESS_PATHS_DESIGN.md) rather than a hard-coded list of provider types.
Their pure helpers live in scripts/validate_ai.py and are tested here; the
scripts' main functions are run only against configs whose every entry is
skipped, so no provider is built and nothing touches the network.

No test parameter is named ``provider``: conftest treats that as a live test.
"""

import importlib
import json
import sys
from pathlib import Path

import pytest

_project_root = str(Path(__file__).parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from djinnite.config_loader import ModelCatalog, ModelInfo, load_ai_config
from djinnite.scripts import validate_ai as va


def _config(tmp_path, providers, platforms=None, name="ai_config.json"):
    data = {"providers": providers}
    if platforms is not None:
        data["platforms"] = platforms
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return load_ai_config(path), path


MIXED = {
    "claude": {"api_key": "sk-ant-real-looking", "default_model": "claude-sonnet-5-5"},
    "claude-vertex": {
        "provider": "claude", "mode": "platform", "platform": "vertexai",
        "location": "us", "default_model": "claude-sonnet-5-5",
    },
    "claude-vertex-default": {
        "provider": "claude", "platform": "vertexai", "default_model": "claude-sonnet-5-5",
    },
    "gemini": {"api_key": "AIza-real-looking", "default_model": "gemini-3.5-flash"},
    "grok": {"api_key": "xai-real-looking", "default_model": "grok-5"},
}
PLATFORMS = {"vertexai": {"project_id": "my-project", "locations": ["global", "us"]}}


@pytest.fixture
def mixed(tmp_path):
    return _config(tmp_path, MIXED, PLATFORMS)[0]


# ----------------------------------------------------------------------
# Scripts import cleanly
# ----------------------------------------------------------------------

@pytest.mark.parametrize("script, main", [
    ("validate_ai", "validate_ai"),
    ("validate_json", "validate_json"),
    ("validate_models", "validate_models"),
])
def test_script_imports_cleanly(script, main):
    module = importlib.import_module(f"djinnite.scripts.{script}")
    assert callable(getattr(module, main))


def test_validate_json_and_models_share_validate_ai_helpers():
    vj = importlib.import_module("djinnite.scripts.validate_json")
    vm = importlib.import_module("djinnite.scripts.validate_models")
    for module in (vj, vm):
        assert module.select_entries is va.select_entries
        assert module.entry_label is va.entry_label
        assert module.entry_skip_reason is va.entry_skip_reason


def test_validate_models_class_is_chosen_by_type():
    vm = importlib.import_module("djinnite.scripts.validate_models")
    from djinnite.ai_providers import PROVIDERS
    for ptype, cls in PROVIDERS.items():
        assert vm.get_provider_class(ptype) is cls
    assert vm.get_provider_class("claude-vertex") is None


# ----------------------------------------------------------------------
# entry_label
# ----------------------------------------------------------------------

def test_label_direct(mixed):
    assert va.entry_label(mixed, "claude") == "claude (claude, direct)"
    assert va.entry_label(mixed, "grok") == "grok (grok, direct)"


def test_label_platform_with_location(mixed):
    assert va.entry_label(mixed, "claude-vertex") == "claude-vertex (claude, platform vertexai@us)"


def test_label_platform_default_location_comes_from_registry(mixed):
    from djinnite.ai_providers.platforms import PLATFORMS as REGISTRY
    expected = REGISTRY["vertexai"].default_location["claude"]
    assert va.entry_label(mixed, "claude-vertex-default") == (
        f"claude-vertex-default (claude, platform vertexai@{expected})"
    )


def test_label_legacy_gemini_vertex_backend(tmp_path):
    cfg, _ = _config(tmp_path, {"gemini": {"backend": "vertexai", "project_id": "my-project"}})
    assert va.entry_label(cfg, "gemini").startswith("gemini (gemini, platform vertexai@")


# ----------------------------------------------------------------------
# select_entries (--provider)
# ----------------------------------------------------------------------

def test_select_all_entries_in_config_order(mixed):
    assert va.select_entries(mixed) == list(MIXED)
    assert va.select_entries(mixed, None) == list(MIXED)
    assert va.select_entries(mixed, "") == list(MIXED)


def test_select_by_type_picks_every_entry_of_that_type(mixed):
    assert va.select_entries(mixed, "claude") == [
        "claude", "claude-vertex", "claude-vertex-default",
    ]


def test_select_by_entry_name_picks_only_that_entry(mixed):
    assert va.select_entries(mixed, "claude-vertex") == ["claude-vertex"]
    assert va.select_entries(mixed, "grok") == ["grok"]


def test_select_type_with_no_entry_named_after_it(tmp_path):
    cfg, _ = _config(tmp_path, {
        "work": {"provider": "chatgpt", "api_key": "sk-real-looking"},
        "home": {"provider": "chatgpt", "api_key": "sk-real-looking-2"},
    })
    assert va.select_entries(cfg, "chatgpt") == ["work", "home"]
    assert va.select_entries(cfg, "home") == ["home"]


def test_select_includes_disabled_entries_so_they_are_reported(tmp_path):
    cfg, _ = _config(tmp_path, {
        "claude": {"api_key": "sk-ant-real-looking", "enabled": False},
    })
    assert va.select_entries(cfg, "claude") == ["claude"]


def test_select_unknown_selector_raises_naming_entries(mixed):
    with pytest.raises(ValueError) as exc:
        va.select_entries(mixed, "chatgpt")
    msg = str(exc.value)
    assert "chatgpt" in msg
    assert "claude-vertex" in msg


# ----------------------------------------------------------------------
# entry_skip_reason
# ----------------------------------------------------------------------

def test_skip_reason(tmp_path):
    cfg, _ = _config(tmp_path, {
        "claude": {"api_key": "sk-ant-real-looking"},
        "chatgpt": {"api_key": "your-openai-key-here"},
        "gemini": {"api_key": "AIza-real-looking", "enabled": False},
        "openai": {"api_key": "sk-real-looking"},
        "claude-vertex": {"provider": "claude", "platform": "vertexai"},
    }, PLATFORMS)
    assert va.entry_skip_reason(cfg, "claude") is None
    assert va.entry_skip_reason(cfg, "claude-vertex") is None
    assert "disabled" in va.entry_skip_reason(cfg, "gemini")
    assert "API key" in va.entry_skip_reason(cfg, "chatgpt")
    reason = va.entry_skip_reason(cfg, "openai")
    assert "not a provider type" in reason and '"provider"' in reason


# ----------------------------------------------------------------------
# unknown_deny_models
# ----------------------------------------------------------------------

def _catalog():
    return ModelCatalog(providers={
        "claude": [
            ModelInfo(id="claude-sonnet-5-5", name="Sonnet", context_window=1000000),
            ModelInfo(id="claude-haiku-4-5-20251001", name="Haiku", context_window=200000),
        ],
        "gemini": [ModelInfo(id="gemini-3.5-flash", name="Flash", context_window=1048576)],
    })


def test_unknown_deny_models(tmp_path):
    cfg, _ = _config(tmp_path, {
        "claude-vertex": {
            "provider": "claude", "platform": "vertexai",
            "deny": {
                "*": ["web_search"],
                "claude-haiku-4-5-20251001": ["structured_json"],
                "claude-haiku-4-5": ["structured_json"],   # Google's short ID: a typo here
            },
        },
        # A model of another type is unknown for this entry's type.
        "gemini-vertex": {
            "provider": "gemini", "platform": "vertexai",
            "deny": {"claude-sonnet-5-5": ["web_search"], "gemini-3.5-flash": ["web_search"]},
        },
    }, PLATFORMS)
    assert va.unknown_deny_models(cfg, _catalog()) == [
        ("claude-vertex", "claude-haiku-4-5"),
        ("gemini-vertex", "claude-sonnet-5-5"),
    ]


def test_unknown_deny_models_star_and_list_forms_are_never_unknown(tmp_path):
    cfg, _ = _config(tmp_path, {
        "claude": {"api_key": "sk-ant-real-looking", "deny": ["web_search"]},
        "claude-vertex": {"provider": "claude", "platform": "vertexai",
                          "deny": {"*": ["json_with_search"]}},
        "gemini": {"api_key": "AIza-real-looking"},
    }, PLATFORMS)
    assert va.unknown_deny_models(cfg, ModelCatalog()) == []


# ----------------------------------------------------------------------
# Main functions, offline: every entry is skipped before a provider is built
# ----------------------------------------------------------------------

SKIPPED_ONLY = {
    "claude": {"api_key": "your-anthropic-key-here", "deny": {"no-such-model-x": ["web_search"]}},
    "claude-off": {"provider": "claude", "api_key": "sk-ant-real-looking", "enabled": False},
    "openai": {"api_key": "sk-real-looking"},
}


def _run_main(monkeypatch, module_name, main_name, argv):
    module = importlib.import_module(f"djinnite.scripts.{module_name}")
    monkeypatch.setattr(sys, "argv", [module_name, *argv])
    with pytest.raises(SystemExit) as exc:
        getattr(module, main_name)()
    return exc.value.code


def test_validate_ai_main_skips_and_warns_offline(tmp_path, monkeypatch, capsys):
    _, path = _config(tmp_path, SKIPPED_ONLY)
    code = _run_main(monkeypatch, "validate_ai", "validate_ai", ["--config", str(path)])
    out = capsys.readouterr().out
    assert code == 1  # a placeholder key is a failure
    assert ("[WARN] claude: deny names model 'no-such-model-x' which is not "
            "in the catalog for claude") in out
    # An enabled entry of a known type that cannot authenticate is what this
    # script exists to catch: a failure, as before named entries.
    assert "[FAIL] claude (claude, direct): API key" in out
    assert "Summary: 0 passed, 2 failed, 1 skipped." in out
    assert "[SKIP] claude-off (claude, direct): disabled" in out
    assert "[FAIL] openai (openai, direct): 'openai' is not a provider type" in out
    out.encode("ascii")  # cp1252-safe output


def test_validate_json_main_unknown_selector_fails(tmp_path, monkeypatch, capsys):
    _, path = _config(tmp_path, SKIPPED_ONLY)
    code = _run_main(monkeypatch, "validate_json", "validate_json",
                     ["--config", str(path), "--provider", "grok"])
    assert code == 1
    assert "[FAIL] 'grok' is neither an ai_config entry" in capsys.readouterr().out


def test_validate_json_main_skips_denied_structured_json(tmp_path, monkeypatch, capsys):
    _, path = _config(tmp_path, {
        **SKIPPED_ONLY,
        "claude-vertex": {
            "provider": "claude", "platform": "vertexai", "default_model": "claude-sonnet-5-5",
            "deny": ["structured_json"], "deny_reason": "org policy in my-project",
        },
    }, PLATFORMS)
    code = _run_main(monkeypatch, "validate_json", "validate_json",
                     ["--config", str(path), "--provider", "claude"])
    out = capsys.readouterr().out
    assert code == 1
    assert ("[SKIP] claude-vertex (claude, platform vertexai@global): entry denies "
            "structured_json for model claude-sonnet-5-5 (org policy in my-project)") in out
    assert "[SKIP] claude-off (claude, direct): disabled" in out
    assert "openai" not in out  # not of type claude
    out.encode("ascii")


def test_validate_models_main_skips_platform_entries(tmp_path, monkeypatch, capsys):
    _, path = _config(tmp_path, {
        **SKIPPED_ONLY,
        "claude-vertex": {"provider": "claude", "platform": "vertexai"},
    }, PLATFORMS)
    module = importlib.import_module("djinnite.scripts.validate_models")
    monkeypatch.setattr(module, "load_model_catalog", lambda: ModelCatalog())
    monkeypatch.setattr(sys, "argv", ["validate_models", "--config", str(path)])
    module.validate_models()
    out = capsys.readouterr().out
    assert "[SKIP] claude-vertex (claude, platform vertexai@global): platform mode -- use validate_ai / probe_platform" in out
    assert "[SKIP] claude (claude, direct): API key" in out
    assert "Validation Complete." in out
    out.encode("ascii")
