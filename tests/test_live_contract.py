"""
The call-shape contract in DIRECT mode (opt-in: ``--live``).

The same checks the platform e2e tier runs through Vertex AI
(``tests/_contract.py``), run here through each configured provider's own
API with its own key -- so verification does not rest on one platform.
Runs once per provider in PROVIDER_NAMES that ai_config.json configures.

    uv run pytest tests/test_live_contract.py --live -rA
"""

from djinnite.tests import _contract as contract


def test_generate_json(provider):
    contract.assert_usage(contract.check_generate_json(provider))


def test_history(provider):
    contract.assert_usage(contract.check_history(provider))


def test_truncation(provider):
    contract.assert_usage(contract.check_truncation(provider))
