"""Lightweight, training-script-only MoE monitoring utilities.

Enable by setting `MOE_PATCH_DIR` and making sure this repo is on
`PYTHONPATH` so `sitecustomize.py` can auto-apply monkey patches.
"""

from __future__ import annotations

from .moe import enable_moe_monitor

__all__ = ['enable_moe_monitor']

