"""
Disable Models Script -- SUPERSEDED

Disable state moved out of ``disabled_models.json`` and into
``config/model_overrides.json``, the single human-editable file for per-model
decisions. Keeping a dedicated file per parameter did not scale: "disabled
models" has no natural sibling for a pinned context window or a hand-verified
price without spawning "enabled models", "verified pricing", and so on.

This shim stays so existing habits and any scripted callers fail loudly with a
pointer rather than silently doing nothing. It does not modify the catalog.

To disable a model now:

    1. Add it to config/model_overrides.json:
         "chatgpt/gpt-4o": {
           "disabled": true,
           "disabled_reason": "deprecated by OpenAI; use gpt-5 or gpt-4.1"
         }
    2. Run: python -m djinnite.scripts.apply_overrides
"""

import sys

_MESSAGE = """
[FAIL] disable_models has been superseded.

Disable state now lives in config/model_overrides.json, alongside every other
per-model human decision, and is applied by:

    uv run python -m djinnite.scripts.apply_overrides

To disable a model, add an entry to config/model_overrides.json:

    "chatgpt/gpt-4o": {
      "disabled": true,
      "disabled_reason": "deprecated by OpenAI; use gpt-5 or gpt-4.1"
    }

To see what is currently overridden (including everything disabled):

    uv run python -m djinnite.scripts.apply_overrides --list
"""


def main():
    print(_MESSAGE.strip())
    sys.exit(1)


if __name__ == "__main__":
    main()
