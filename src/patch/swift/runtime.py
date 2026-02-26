from __future__ import annotations

import atexit
import json
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


def _get_dist_info() -> tuple[int, int]:
    """Best-effort (rank, world_size) without assuming torch is installed."""
    rank = int(os.environ.get('RANK', '0') or '0')
    world_size = int(os.environ.get('WORLD_SIZE', '1') or '1')
    try:
        import torch.distributed as dist  # type: ignore

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
    except Exception:
        pass
    return rank, world_size


def resolve_jsonl_path(path_env: str) -> Path:
    """Resolve `MOE_PATCH_DIR` into a rank-specific JSONL file path.

    Layout:
    - Directory: `logs/moe_monitor` -> `logs/moe_monitor/swift_moe_lb_step_all_rank_{rank}.jsonl`
    """
    rank, _world_size = _get_dist_info()

    raw = (path_env or '').strip()
    if not raw:
        raise ValueError('Empty MOE_PATCH_DIR')

    p = Path(raw)
    # Treat as directory root; if a file path is passed, fall back to its parent dir.
    if p.suffix:
        p = p.parent
    return p / f'swift_moe_lb_step_all_rank_{rank}.jsonl'


@dataclass(frozen=True)
class AsyncJSONLWriterConfig:
    path: Path
    flush_interval_s: float = 2.0


class AsyncJSONLWriter:
    def __init__(self, cfg: AsyncJSONLWriterConfig) -> None:
        self._cfg = cfg
        self._queue: queue.Queue[str] = queue.Queue(maxsize=10_000)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name='moe-jsonl-writer', daemon=True)
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._cfg.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread.start()
        atexit.register(self.close)

    def submit(self, record: dict[str, Any]) -> None:
        if not self._started:
            self.start()
        line = json.dumps(record, ensure_ascii=False) + '\n'
        try:
            self._queue.put_nowait(line)
        except queue.Full:
            # Monitoring must never block training.
            pass

    def close(self) -> None:
        if not self._started:
            return
        self._stop.set()
        try:
            self._thread.join(timeout=2.0)
        except Exception:
            pass

    def _run(self) -> None:
        last_flush = time.time()
        f = self._cfg.path.open('a', encoding='utf-8')
        try:
            while not self._stop.is_set() or not self._queue.empty():
                try:
                    line = self._queue.get(timeout=0.2)
                except queue.Empty:
                    line = None
                if line is not None:
                    f.write(line)
                now = time.time()
                if now - last_flush >= self._cfg.flush_interval_s:
                    f.flush()
                    last_flush = now
            f.flush()
        finally:
            try:
                f.close()
            except Exception:
                pass


_WRITER: Optional[AsyncJSONLWriter] = None


def get_writer_from_env() -> Optional[AsyncJSONLWriter]:
    global _WRITER
    if _WRITER is not None:
        return _WRITER

    path_env = os.environ.get('MOE_PATCH_DIR', '').strip()
    if not path_env:
        return None

    path = resolve_jsonl_path(path_env)
    _WRITER = AsyncJSONLWriter(AsyncJSONLWriterConfig(path=path))
    _WRITER.start()
    return _WRITER
