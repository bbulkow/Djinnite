"""
Platform registry: providers' models served through a cloud platform.

Djinnite reaches a model in one of two **access modes**:

* ``direct``   -- the provider's own API (Anthropic API, Google AI Studio,
                  OpenAI, xAI), authenticated with that provider's API key.
* ``platform`` -- a cloud platform that hosts the provider's models (Google
                  Vertex AI today; others such as Amazon Bedrock or Microsoft
                  Foundry later), authenticated with the platform's own
                  credentials.

A platform serves some subset of each hosted provider's models, and that
subset grows and shrinks over time. Djinnite therefore implements the
platform's *mechanics* here and in each provider's client code (auth,
locations, endpoint rules, error shapes, pricing modifiers) and **probes**
which models and capabilities the platform actually offers
(``scripts/probe_platform.py``), rather than encoding model lists.

This module holds per-platform rules only. Per-model facts belong in the
catalog (``platforms.<name>`` block per model), never in Python -- see
DEVELOPMENT.md "Static Data Policy".
"""

from dataclasses import dataclass, field
from typing import Optional

from .base_provider import AIProviderError


ACCESS_MODES = ("direct", "platform")


@dataclass(frozen=True)
class PlatformSpec:
    """Rules for one cloud platform.

    Attributes:
        name: Platform identifier used in config, kwargs, and the catalog
            (e.g. ``"vertexai"``).
        providers: Djinnite provider names whose models the platform hosts.
        default_location: Per-provider location used when the caller names
            none.
        premium_free_locations: Locations billed at the provider's standard
            (catalog) price.
        location_premium: Per-provider price multiplier applied at every
            location NOT in ``premium_free_locations``. Providers absent from
            this map carry no location premium.
    """
    name: str
    providers: frozenset
    default_location: dict = field(default_factory=dict)
    premium_free_locations: frozenset = frozenset()
    location_premium: dict = field(default_factory=dict)

    def price_multiplier(self, provider: str, location: Optional[str]) -> float:
        """Token-price multiplier for ``provider`` at ``location``."""
        premium = self.location_premium.get(provider)
        if premium is None:
            return 1.0
        loc = (location or self.default_location.get(provider) or "").lower()
        if loc in self.premium_free_locations:
            return 1.0
        return premium


PLATFORMS: dict = {
    "vertexai": PlatformSpec(
        name="vertexai",
        providers=frozenset({"gemini", "claude"}),
        default_location={
            # Gemini's default predates platform mode and is kept so existing
            # callers are unchanged. Claude defaults to "global": Anthropic's
            # recommended endpoint, the only premium-free one, and the 5.x
            # models are not served at single-region endpoints.
            "gemini": "us-central1",
            "claude": "global",
        },
        premium_free_locations=frozenset({"global"}),
        # "Regional and multi-region endpoints include a 10% pricing premium
        # over global endpoints" -- Claude Sonnet 4.5 and later.
        # https://platform.claude.com/docs/en/build-with-claude/claude-on-vertex-ai
        location_premium={"claude": 1.10},
    ),
}


def get_platform(name: str, provider: Optional[str] = None) -> PlatformSpec:
    """Return the spec for platform ``name``.

    Raises:
        AIProviderError: unknown platform, or ``provider`` given and not
            hosted by the platform.
    """
    spec = PLATFORMS.get(name)
    if spec is None:
        raise AIProviderError(
            f"Unknown platform '{name}'. Known platforms: {sorted(PLATFORMS)}",
            provider=provider or "platform",
        )
    if provider is not None and provider not in spec.providers:
        raise AIProviderError(
            f"Platform '{name}' does not host provider '{provider}' "
            f"(hosts: {sorted(spec.providers)})",
            provider=provider,
        )
    return spec


def resolve_platform(
    platform: Optional[str] = None,
    backend: Optional[str] = None,
    provider: Optional[str] = None,
) -> Optional[str]:
    """Resolve the platform name from ``platform`` and the legacy ``backend``.

    ``backend="vertexai"`` predates platform mode (Gemini's original Vertex
    switch) and remains an alias for ``platform="vertexai"``. Any other
    ``backend`` value (``"gemini"``, ``"anthropic"``, ...) means direct mode.

    Returns:
        The platform name, or ``None`` for direct mode.

    Raises:
        AIProviderError: unknown platform, unhosted provider, or
            ``platform`` and ``backend`` naming different platforms.
    """
    from_backend = backend if backend in PLATFORMS else None
    if platform and from_backend and platform != from_backend:
        raise AIProviderError(
            f"platform='{platform}' conflicts with backend='{backend}'",
            provider=provider or "platform",
        )
    name = platform or from_backend
    if name is None:
        return None
    get_platform(name, provider)
    return name


__all__ = [
    "ACCESS_MODES",
    "PlatformSpec",
    "PLATFORMS",
    "get_platform",
    "resolve_platform",
]
