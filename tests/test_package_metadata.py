"""
Package metadata: one version, floors without ceilings, requirements in sync.

AGENTS.md "Dependencies have floors, never ceilings": the provider SDKs are
declared with ``>=`` only, because a library's upper bound becomes every
consumer's upper bound. This test fails on any ceiling, on drift between
``pyproject.toml`` and ``requirements.txt``, and on the two version strings
disagreeing (they did, 0.5.0 vs 0.6.0, until 0.7.0).

    uv run pytest tests/test_package_metadata.py -v
"""

import re
import tomllib
from pathlib import Path

import djinnite

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
PROJECT = PYPROJECT["project"]

# The SDKs this rule is about. Other dependencies (pytest, ...) are dev-only.
SDKS = ("google-genai", "anthropic", "openai")

_CEILING = re.compile(r"<|==|~=|!=")


def _name(spec: str) -> str:
    """``anthropic[vertex]>=1.0`` -> ``anthropic``."""
    return re.split(r"[\[><=~!;\s]", spec, maxsplit=1)[0]


def _sdk_specs(specs):
    return {_name(s): s.strip() for s in specs if _name(s) in SDKS}


def test_version_string_matches_pyproject():
    assert djinnite.__version__ == PROJECT["version"]


def test_requires_python_is_3_11():
    """google-genai 3.x needs 3.11; 3.10 reached end of life 2026-10-01."""
    assert PROJECT["requires-python"] == ">=3.11"
    assert "Programming Language :: Python :: 3.10" not in PROJECT["classifiers"]


def test_sdk_dependencies_have_floors_and_no_ceilings():
    specs = _sdk_specs(PROJECT["dependencies"])
    assert set(specs) == set(SDKS), specs
    for name, spec in specs.items():
        assert ">=" in spec, f"{spec!r}: an SDK floor is required"
        assert not _CEILING.search(spec.split(">=", 1)[1]), (
            f"{spec!r} caps the SDK version. Floors only, never ceilings: "
            "fix the incompatibility in Djinnite instead (AGENTS.md)."
        )


def test_requirements_txt_matches_pyproject():
    lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    reqs = _sdk_specs(l for l in lines if l.strip() and not l.lstrip().startswith("#"))
    assert reqs == _sdk_specs(PROJECT["dependencies"])
