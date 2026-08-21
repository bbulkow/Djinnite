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


def _check_classify() -> list:
    fails = []
    for provider, model_id, siblings, expected in CLASSIFY_CASES:
        got = classify_model(provider, model_id, siblings)
        ok = got == expected
        msg = f"{provider}/{model_id}: {got} (expected {expected})"
        print(f"  [{'OK' if ok else 'FAIL'}] {msg}")
        if not ok:
            fails.append(msg)
    return fails


def _check_strip_and_sibling() -> list:
    fails = []
    cases = [
        ("gpt-5.2-2025-12-11", "gpt-5.2"),          # ISO
        ("claude-opus-4-5-20251101", "claude-opus-4-5"),  # compact 8-digit
        ("gemini-2.5-computer-use-preview-10-2025", "gemini-2.5-computer-use-preview"),  # -MM-YYYY
        ("claude-opus-4-8", "claude-opus-4-8"),     # no date -> unchanged
    ]
    for model_id, expected in cases:
        got = strip_date_pin(model_id)
        ok = got == expected
        msg = f"strip_date_pin({model_id}) = {got} (expected {expected})"
        print(f"  [{'OK' if ok else 'FAIL'}] {msg}")
        if not ok:
            fails.append(msg)

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
        msg = f"has_dated_sibling({model_id}) = {got} (expected {expected})"
        print(f"  [{'OK' if ok else 'FAIL'}] {msg}")
        if not ok:
            fails.append(msg)
    return fails


def _check_staleness() -> list:
    fails = []
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
        msg = f"is_stale(updated={updated!r}) = {got} (expected {expected})"
        print(f"  [{'OK' if ok else 'FAIL'}] {msg}")
        if not ok:
            fails.append(msg)
    return fails


def _check_helpers() -> list:
    fails = []
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
        msg = f"helper -> {got} (expected {expected})"
        print(f"  [{'OK' if ok else 'FAIL'}] {msg}")
        if not ok:
            fails.append(msg)
    return fails


# ------------------------------------------------------------------
# Price provenance
#
# `is_stale` answers "how old is this number", which cannot distinguish a
# price read off the vendor's pricing page from one a model guessed. An
# estimate re-dated yesterday looks fresher than a verified price from last
# month. That gap let gemini-flash-latest sit at an estimated $0.50/$3.00
# when the real price was $1.50/$7.50 -- a 3x error that never tripped the
# staleness check because it was never old enough.
# ------------------------------------------------------------------

def test_unverified_is_independent_of_age():
    """A brand-new estimate is unverified; a verified price never is."""
    fresh_guess = ModelCosting(input_per_1m=1.0, output_per_1m=2.0,
                               source="estimated", updated=date.today().isoformat())
    assert not fresh_guess.is_stale(180), "guard: this fixture must look fresh by age"
    assert fresh_guess.is_unverified(), "a never-checked price is not fresh"
    assert fresh_guess.needs_repricing(180)

    verified = ModelCosting(input_per_1m=1.0, output_per_1m=2.0,
                            source="published", updated=date.today().isoformat(),
                            source_url="https://example.invalid/pricing")
    assert not verified.is_unverified()
    assert not verified.needs_repricing(180)


def test_manual_pricing_counts_as_verified():
    """A human-entered price is deliberate, not a guess."""
    manual = ModelCosting(input_per_1m=1.0, output_per_1m=2.0,
                          source="manual", updated=date.today().isoformat())
    assert not manual.is_unverified()


def test_unpriced_entry_is_not_flagged_unverified():
    """With no price there is nothing to verify -- that is `missing`, not stale."""
    assert not ModelCosting(source="unknown").is_unverified()


def test_needs_repricing_still_catches_plain_age():
    """Provenance is an additional trigger, not a replacement for age."""
    old_but_verified = ModelCosting(input_per_1m=1.0, output_per_1m=2.0,
                                    source="published", updated="2020-01-01",
                                    source_url="https://example.invalid/pricing")
    assert old_but_verified.is_stale(180)
    assert old_but_verified.needs_repricing(180)


def test_runtime_staleness_stays_age_only():
    """is_stale gates the runtime hard-fail, so provenance must not leak in.

    Making every estimate instantly stale would turn ~73% of the catalog
    into AIPricingError at request time.
    """
    guess = ModelCosting(input_per_1m=1.0, output_per_1m=2.0,
                         source="estimated", updated=date.today().isoformat())
    assert guess.is_unverified()
    assert not guess.is_stale(180)


# ------------------------------------------------------------------
# pytest entry points -- these assert, so a regression actually fails.
# ------------------------------------------------------------------

def test_classify():
    fails = _check_classify()
    assert not fails, "classify_model mismatches:\n  " + "\n  ".join(fails)


def test_strip_and_sibling():
    fails = _check_strip_and_sibling()
    assert not fails, "strip/sibling mismatches:\n  " + "\n  ".join(fails)


def test_staleness():
    fails = _check_staleness()
    assert not fails, "staleness mismatches:\n  " + "\n  ".join(fails)


def test_helpers():
    fails = _check_helpers()
    assert not fails, "helper mismatches:\n  " + "\n  ".join(fails)


def run():
    print("\nDjinnite Pricing Classification Tests")
    print("=" * 60)
    fails = []
    print("\nclassify_model:")
    fails += _check_classify()
    print("\nstrip_date_pin / has_dated_sibling:")
    fails += _check_strip_and_sibling()
    print("\nstaleness:")
    fails += _check_staleness()
    print("\nhelpers:")
    fails += _check_helpers()
    print("\n" + "=" * 60)
    print(f"Result: {'ALL PASS' if not fails else str(len(fails)) + ' FAILED'}")
    if fails:
        sys.exit(1)


if __name__ == "__main__":
    run()
