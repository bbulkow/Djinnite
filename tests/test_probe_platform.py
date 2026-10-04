"""
scripts/probe_platform.py (offline; providers are faked).

The script records, per model, which locations a platform serves it at and
(optionally) the capabilities probed there. It writes only with --write,
and only through model_overrides.save_catalog.

    uv run pytest tests/test_probe_platform.py -v
"""

import json

import pytest

from djinnite.scripts import probe_platform


CATALOG = {
    "claude": {"models": [
        {"id": "claude-sonnet-5-5", "name": "Sonnet 5.5",
         "modalities": {"input": ["text", "vision"], "output": ["text"]},
         "capabilities": {"thinking": ["on", "off"], "thinking_style": ["adaptive"],
                          "structured_json": ["on", "off"]},
         "platforms": {"vertexai": {"locations": {"eu": "not_found"},
                                    "capabilities": {"thinking": ["on"]},
                                    "probed": "2026-09-01"}}},
        {"id": "claude-old", "name": "Old", "disabled": True,
         "modalities": {"input": ["text"], "output": ["text"]}},
    ]},
    "gemini": {"models": [
        {"id": "gemini-3.5-flash", "name": "Flash",
         "modalities": {"input": ["text"], "output": ["text"]}, "capabilities": {}},
        {"id": "gemini-tts", "name": "TTS",
         "modalities": {"input": ["text"], "output": ["audio"]}},
    ]},
}

STATUS = {
    ("claude-sonnet-5-5", "global"): ("available", ""),
    ("claude-sonnet-5-5", "us"): ("no_quota", "AIRateLimitError: 429 RESOURCE_EXHAUSTED"),
    ("gemini-3.5-flash", "global"): ("available", ""),
    ("gemini-3.5-flash", "us"): ("not_found", "ClientError: 404 NOT_FOUND"),
}


class _Fake:
    """Answers availability from STATUS and capability probes like Sonnet 5.5."""
    made = []

    def __init__(self, provider, model, location):
        self.provider, self.model, self.location = provider, model, location
        _Fake.made.append((provider, model, location))

    def probe_availability(self):
        return STATUS[(self.model, self.location)]

    def probe_structured_json(self): return True
    def probe_temperature(self): return False
    def probe_json_with_search(self): return True
    def probe_web_search(self): return True
    def probe_thinking_style(self): return ["adaptive", "between_tools"]
    def probe_thinking_disable(self): return False
    def probe_incompatible_combinations(self, states): return []


@pytest.fixture
def files(tmp_path):
    cat = tmp_path / "model_catalog.json"
    cat.write_text(json.dumps(CATALOG), encoding="utf-8")
    cfg = tmp_path / "ai_config.json"
    cfg.write_text(json.dumps({
        "platforms": {"vertexai": {"project_id": "proj", "quota_project": "proj",
                                   "locations": ["global", "us"]}},
        "providers": {},
    }), encoding="utf-8")
    _Fake.made = []
    return cfg, cat


def _run(files, *extra):
    cfg, cat = files
    return probe_platform.run(
        ["--platform", "vertexai", "--config", str(cfg), "--catalog", str(cat), *extra],
        make_provider=_Fake,
    )


def test_dry_run_reports_and_writes_nothing(files, capsys):
    before = files[1].read_text(encoding="utf-8")
    assert _run(files) == 0
    out = capsys.readouterr().out
    assert "[OK] claude claude-sonnet-5-5 @global available" in out
    assert "[WARN] claude claude-sonnet-5-5 @us no_quota (AIRateLimitError: 429 RESOURCE_EXHAUSTED)" in out
    assert "[WARN] gemini gemini-3.5-flash @us not_found" in out
    assert "Dry run" in out
    assert out.isascii()
    assert files[1].read_text(encoding="utf-8") == before
    # Disabled and non-text models are not probed.
    assert {m for _, m, _ in _Fake.made} == {"claude-sonnet-5-5", "gemini-3.5-flash"}


def test_write_records_availability_and_keeps_prior_facts(files):
    assert _run(files, "--write") == 0
    data = json.loads(files[1].read_text(encoding="utf-8"))
    block = data["claude"]["models"][0]["platforms"]["vertexai"]
    # New statuses merged over the earlier probe's; capabilities kept.
    assert block["locations"] == {"eu": "not_found", "global": "available", "us": "no_quota"}
    assert block["capabilities"] == {"thinking": ["on"]}
    assert block["probed"] != "2026-09-01"
    flash = data["gemini"]["models"][0]["platforms"]["vertexai"]
    assert flash["locations"] == {"global": "available", "us": "not_found"}
    assert "capabilities" not in flash


def test_capabilities_probe_records_and_diffs(files, capsys):
    assert _run(files, "--provider", "claude", "--capabilities", "--write") == 0
    out = capsys.readouterr().out
    assert "[CHECK] claude claude-sonnet-5-5: probing capabilities @global" in out
    assert '[DIFF] thinking: direct=["on","off"] platform=["on"]' in out
    caps = json.loads(files[1].read_text(encoding="utf-8"))["claude"]["models"][0][
        "platforms"]["vertexai"]["capabilities"]
    assert caps["thinking"] == ["on"]
    assert caps["thinking_style"] == ["adaptive", "between_tools"]
    assert caps["incompatible"] == []


def test_cli_overrides_and_scoping(files):
    assert _run(files, "--provider", "gemini", "--model", "gemini-3.5-flash",
                "--location", "global") == 0
    assert _Fake.made == [("gemini", "gemini-3.5-flash", "global")]


def test_missing_project_fails_without_probing(files, tmp_path, capsys):
    cfg = tmp_path / "empty.json"
    cfg.write_text(json.dumps({"providers": {}}), encoding="utf-8")
    rc = probe_platform.run(["--platform", "vertexai", "--config", str(cfg),
                             "--catalog", str(files[1])], make_provider=_Fake)
    assert rc == 2
    assert "[FAIL] No project" in capsys.readouterr().out
    assert _Fake.made == []


def test_unhosted_provider_fails(files, capsys):
    assert _run(files, "--provider", "chatgpt") == 2
    assert "does not host provider 'chatgpt'" in capsys.readouterr().out
