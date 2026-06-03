"""
Pricing Classification Tests (offline)

Verifies the fixed/floating classifier per vendor scheme and the staleness
helpers on ModelCosting.  Runs entirely offline -- no API keys required.

Usage:
    uv run python -m djinnite.tests.test_pricing_class
"""

import sys
from datetime import date
from pathlib import Path

# Support direct execution
_project_root = str(Path(__file__).parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from djinnite.pricing_class import (
    classify_model, strip_date_pin, has_dated_sibling,
    has_date_pin, has_latest_suffix, has_numeric_version,
)
from djinnite.config_loader import ModelCosting


# Representative sibling sets per provider (bare aliases + their dated twins).
GEMINI = [
    "gemini-2.5-flash", "gemini-flash-latest", "gemini-flash-lite-latest",
    "gemini-pro-latest", "gemini-2.5-computer-use-preview-10-2025",
    "gemini-2.5-flash-native-audio-latest", "gemini-embedding-001",
]
OPENAI = [
    "gpt-5.2", "gpt-5.2-2025-12-11", "gpt-4o", "gpt-4o-2024-08-06",
    "gpt-5.2-chat-latest", "chatgpt-image-latest", "gpt-image-1",
]
CLAUDE = ["claude-opus-4-8", "claude-opus-4-5-20251101", "claude-sonnet-4-6"]


# (provider, model_id, siblings, expected_class)
CLASSIFY_CASES = [
    # Gemini: -latest and version-less aliases float; numbered/dated are fixed.
    ("gemini", "gemini-flash-latest", GEMINI, "floating"),
    ("gemini", "gemini-flash-lite-latest", GEMINI, "floating"),
    ("gemini", "gemini-pro-latest", GEMINI, "floating"),
    ("gemini", "gemini-2.5-flash-native-audio-latest", GEMINI, "floating"),
    ("gemini", "gemini-2.5-flash", GEMINI, "fixed"),
    ("gemini", "gemini-2.5-computer-use-preview-10-2025", GEMINI, "fixed"),
    ("gemini", "gemini-embedding-001", GEMINI, "fixed"),
    # OpenAI: -latest and bare-with-dated-sibling float; dated snapshots fixed.
    ("chatgpt", "gpt-5.2", OPENAI, "floating"),
    ("chatgpt", "gpt-5.2-2025-12-11", OPENAI, "fixed"),
    ("chatgpt", "gpt-4o", OPENAI, "floating"),
    ("chatgpt", "gpt-4o-2024-08-06", OPENAI, "fixed"),
    ("chatgpt", "gpt-5.2-chat-latest", OPENAI, "floating"),
    ("chatgpt", "chatgpt-image-latest", OPENAI, "floating"),
    ("chatgpt", "gpt-image-1", OPENAI, "fixed"),  # no dated sibling
    # Claude: every id pins its price; only an explicit -latest would float.
    ("claude", "claude-opus-4-8", CLAUDE, "fixed"),
    ("claude", "claude-opus-4-5-20251101", CLAUDE, "fixed"),
    ("claude", "claude-sonnet-4-6", CLAUDE, "fixed"),
]


def test_classify() -> int:
    fails = 0
    for provider, model_id, siblings, expected in CLASSIFY_CASES:
        got = classify_model(provider, model_id, siblings)
        ok = got == expected
        print(f"  [{'OK' if ok else 'FAIL'}] {provider}/{model_id}: {got} (expected {expected})")
        if not ok:
            fails += 1
    return fails


def test_strip_and_sibling() -> int:
    fails = 0
    cases = [
        ("gpt-5.2-2025-12-11", "gpt-5.2"),          # ISO
        ("claude-opus-4-5-20251101", "claude-opus-4-5"),  # compact 8-digit
        ("gemini-2.5-computer-use-preview-10-2025", "gemini-2.5-computer-use-preview"),  # -MM-YYYY
        ("claude-opus-4-8", "claude-opus-4-8"),     # no date -> unchanged
    ]
    for model_id, expected in cases:
        got = strip_date_pin(model_id)
        ok = got == expected
        print(f"  [{'OK' if ok else 'FAIL'}] strip_date_pin({model_id}) = {got} (expected {expected})")
        if not ok:
            fails += 1

    # has_dated_sibling: bare alias detects its dated twin; dated id does not.
    sib_cases = [
        ("gpt-5.2", OPENAI, True),
        ("gpt-4o", OPENAI, True),
        ("gpt-5.2-2025-12-11", OPENAI, False),  # a dated id is never a bare alias
        ("gpt-image-1", OPENAI, False),         # no dated twin
    ]
    for model_id, sibs, expected in sib_cases:
        got = has_dated_sibling(model_id, sibs)
        ok = got == expected
        print(f"  [{'OK' if ok else 'FAIL'}] has_dated_sibling({model_id}) = {got} (expected {expected})")
        if not ok:
            fails += 1
    return fails


def test_staleness() -> int:
    fails = 0
    today = date(2026, 6, 2)
    cases = [
        # (updated, expected_is_stale_at_180)
        ("", True),                 # missing -> stale
        ("not-a-date", True),       # unparseable -> stale
        ("2026-05-01", False),      # ~32 days -> fresh
        ("2025-12-01", True),       # ~183 days -> stale
    ]
    for updated, expected in cases:
        c = ModelCosting(input_per_1m=1.0, output_per_1m=2.0, updated=updated)
        got = c.is_stale(180, today=today)
        ok = got == expected
        print(f"  [{'OK' if ok else 'FAIL'}] is_stale(updated={updated!r}) = {got} (expected {expected})")
        if not ok:
            fails += 1
    return fails


def test_helpers() -> int:
    fails = 0
    checks = [
        (has_date_pin("gpt-5.2-2025-12-11"), True),
        (has_date_pin("gpt-5.2"), False),
        (has_latest_suffix("gpt-5.2-chat-latest"), True),
        (has_latest_suffix("gpt-5.2"), False),
        (has_numeric_version("gemini-2.5-flash"), True),
        (has_numeric_version("gemini-flash"), False),
    ]
    for got, expected in checks:
        ok = got == expected
        print(f"  [{'OK' if ok else 'FAIL'}] helper -> {got} (expected {expected})")
        if not ok:
            fails += 1
    return fails


def run():
    print("\nDjinnite Pricing Classification Tests")
    print("=" * 60)
    fails = 0
    print("\nclassify_model:")
    fails += test_classify()
    print("\nstrip_date_pin / has_dated_sibling:")
    fails += test_strip_and_sibling()
    print("\nstaleness:")
    fails += test_staleness()
    print("\nhelpers:")
    fails += test_helpers()
    print("\n" + "=" * 60)
    print(f"Result: {'ALL PASS' if fails == 0 else str(fails) + ' FAILED'}")
    if fails:
        sys.exit(1)


if __name__ == "__main__":
    run()
