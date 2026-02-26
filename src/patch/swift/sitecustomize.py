from __future__ import annotations

import os
import sys
import traceback


def _maybe_enable_swift_moe_monitor() -> None:
    if not os.environ.get('MOE_PATCH_DIR'):
        return
    try:
        from moe import enable_moe_monitor

        enable_moe_monitor()
    except Exception:
        if os.environ.get('SWIFT_MOE_MONITOR_STRICT', '').strip() == '1':
            raise
        sys.stderr.write('[swift-moe-monitor] enable failed; continuing without monitor.\n')
        if os.environ.get('SWIFT_MOE_MONITOR_DEBUG', '').strip() == '1':
            sys.stderr.write(traceback.format_exc() + '\n')


_maybe_enable_swift_moe_monitor()
