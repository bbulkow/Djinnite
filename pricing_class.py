"""
Pricing Classification

Classifies an AI model id as ``fixed`` or ``floating`` for the purposes of cost
maintenance:

    fixed     The id pins a price that does not change under us.  Re-price only
              when missing or past the staleness vector (see update_model_costs).
    floating  The id is an alias that the vendor can silently re-point to a
              different underlying model (and therefore a different price).
              Re-price on every run.

The axis that matters is whether the *price* can change under a stable model id,
NOT whether the underlying model rolls.  Pricing history shows this is
PER-VENDOR, so a uniform naming heuristic is wrong:

  * Anthropic -- a price change ships as a NEW version id, never a silent
    re-price.  Opus 4.1 -> 4.6 -> 4.7; the 67% cut ($15/$75 -> $5/$25) landed on
    the new id ``claude-opus-4-6``, and 4.7 kept it.  Sonnet 4.6 stayed $3/$15.
    So every Claude id pins a price, including bare ``claude-opus-4-8``.
  * OpenAI -- bare aliases float (``gpt-4o`` went $5/$15 -> $2.50/$10 when it
    re-pointed from the 2024-05-13 snapshot to 2024-08-06; each dated snapshot
    keeps its own price).  A bare id that shadows a dated sibling is therefore a
    floating alias; a dated snapshot is fixed.  (Dated ids are not *guaranteed*
    immutable -- OpenAI has cut prices on existing ids like o3 and gpt-3.5 --
    which is why "fixed" still gets a 180-day staleness re-check upstream.)
  * Gemini -- version-less aliases float across MAJOR versions
    (``gemini-flash-latest`` rolled 2.5 -> 3.5).  Numbered/dated entries
    (``gemini-2.5-flash``, ``...-10-2025``) are stable.

The per-vendor table below is a human judgement of *current* vendor behaviour.
If a vendor changes how it re-prices ids, revise ``_VENDOR_RULES``.
"""

import re
from typing import Iterable, Literal

PricingClass = Literal["fixed", "floating"]

FIXED: PricingClass = "fixed"
FLOATING: PricingClass = "floating"

# A date pin anywhere in the id marks an immutable snapshot:
#   2025-12-11    ISO            (OpenAI)
#   20251101      compact 8-digit (Claude)
#   -09-2025      -MM-YYYY        (Gemini previews: -09-2025, -10-2025)
_DATE_PIN = re.compile(
    r"\d{4}-\d{2}-\d{2}"     # ISO        2025-12-11
    r"|\b\d{8}\b"            # compact    20251101
    r"|-\d{2}-20\d{2}\b"     # -MM-YYYY   -10-2025
)

# Same shapes, but TAIL-anchored so strip_date_pin only removes a trailing
# stamp and never mangles an internal one (gpt-4o-2024-08-06 -> gpt-4o).
_DATE_PIN_TAIL = re.compile(r"(?:-\d{4}-\d{2}-\d{2}|-\d{8}|-\d{2}-20\d{2})$")

_LATEST_SUFFIX = re.compile(r"-latest$")

# A numeric version token: 2.5 / 3 / 3.1 / 4-8 (Claude's dashed minor).
_NUMERIC_VERSION = re.compile(r"(?:^|[-.])\d+(?:[.-]\d+)*(?:$|[-.])")


def has_date_pin(model_id: str) -> bool:
    """True if the id contains a date stamp (ISO, compact 8-digit, or -MM-YYYY)."""
    return _DATE_PIN.search(model_id) is not None


def has_latest_suffix(model_id: str) -> bool:
    """True if the id ends in ``-latest`` (an explicit floating alias)."""
    return _LATEST_SUFFIX.search(model_id) is not None


def has_numeric_version(model_id: str) -> bool:
    """True if the id carries a numeric version token (e.g. 2.5, 3.1, 4-8)."""
    return _NUMERIC_VERSION.search(model_id) is not None


def strip_date_pin(model_id: str) -> str:
    """Remove a trailing date stamp, if any (gpt-5.2-2025-12-11 -> gpt-5.2)."""
    return _DATE_PIN_TAIL.sub("", model_id)


def has_dated_sibling(model_id: str, sibling_ids: Iterable[str]) -> bool:
    """
    True if another id in the same provider is this id plus a trailing date
    stamp (i.e. ``model_id`` is a bare alias shadowing a dated snapshot).

    ``gpt-5.2`` has the dated sibling ``gpt-5.2-2025-12-11`` -> True.
    The model itself is ignored, and dated ids never have a "bare alias of"
    relationship to themselves.
    """
    if has_date_pin(model_id):
        return False
    for sib in sibling_ids:
        if sib == model_id:
            continue
        if has_date_pin(sib) and strip_date_pin(sib) == model_id:
            return True
    return False


def _classify_claude(model_id: str, sibling_ids: Iterable[str]) -> PricingClass:
    # Every Claude id pins its price; new prices ship as new ids.
    return FIXED


def _classify_chatgpt(model_id: str, sibling_ids: Iterable[str]) -> PricingClass:
    # A bare name that shadows a dated snapshot is a re-pointing alias.
    return FLOATING if has_dated_sibling(model_id, sibling_ids) else FIXED


def _classify_gemini(model_id: str, sibling_ids: Iterable[str]) -> PricingClass:
    # Version-less aliases float across major versions; numbered ids are stable.
    return FIXED if has_numeric_version(model_id) else FLOATING


_VENDOR_RULES = {
    "claude": _classify_claude,
    "chatgpt": _classify_chatgpt,
    "gemini": _classify_gemini,
}


def classify_model(
    provider: str,
    model_id: str,
    sibling_ids: Iterable[str],
) -> PricingClass:
    """
    Classify ``model_id`` as ``fixed`` or ``floating``.

    Args:
        provider: Provider key (``claude`` / ``chatgpt`` / ``gemini``).
        model_id: The model id to classify.
        sibling_ids: All model ids for the SAME provider (used to detect a bare
            name shadowing a dated snapshot).

    Decision order:
        1. ``-latest`` suffix          -> floating (explicit alias).
        2. otherwise a date pin        -> fixed (a dated snapshot pins a price).
        3. otherwise the per-vendor rule for the remaining bare names.
    """
    # Universal rules first.
    if has_latest_suffix(model_id):
        return FLOATING
    if has_date_pin(model_id):
        return FIXED

    # Per-vendor rule for bare names.  Materialize siblings once so the rule may
    # iterate them more than once.
    siblings = list(sibling_ids)
    rule = _VENDOR_RULES.get(provider)
    if rule is None:
        # Unknown provider: conservative default; staleness still re-checks.
        return FIXED
    return rule(model_id, siblings)
