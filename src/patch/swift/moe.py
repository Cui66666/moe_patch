from __future__ import annotations

import atexit
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

try:
    # When used as a package: `PYTHONPATH=/path/to/ms-swift`
    from .runtime import get_writer_from_env  # type: ignore
except Exception:
    # When used as a standalone patch dir: `PYTHONPATH=/path/to/monitor_patch`
    from runtime import get_writer_from_env  # type: ignore


def _get_megatron_step() -> tuple[Optional[int], Optional[int]]:
    """Best-effort (iteration, consumed_train_samples)."""
    try:
        from megatron.training import get_args  # type: ignore

        args = get_args()
        iteration = getattr(args, 'curr_iteration', None)
        consumed = getattr(args, 'consumed_train_samples', None)
        return iteration, consumed
    except Exception:
        return None, None


def _get_rank() -> int:
    try:
        import torch.distributed as dist  # type: ignore

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return int(os.environ.get('RANK', '0') or '0')


@dataclass
class _LayerAccum:
    num_experts: int
    top_k: int
    assignments: Any  # torch.Tensor[int64] on device
    tokens: int = 0


@dataclass
class MoELoadMonitor:
    interval: int = 1
    count_mode: str = 'no_grad'  # no_grad | grad | all
    _writer: Any = field(default=None, init=False)
    _current_iteration: Optional[int] = field(default=None, init=False)
    _tracking: bool = field(default=False, init=False)
    _layers: Dict[str, _LayerAccum] = field(default_factory=dict, init=False)
    _router_id_to_layer: Dict[int, str] = field(default_factory=dict, init=False)
    _unknown_layer_counter: int = field(default=0, init=False)

    def enabled(self) -> bool:
        return self._writer is not None

    def _should_count(self) -> bool:
        if self.count_mode == 'all':
            return True
        try:
            import torch  # type: ignore

            is_grad = torch.is_grad_enabled()
        except Exception:
            return False
        if self.count_mode == 'grad':
            return bool(is_grad)
        # default: count only in `no_grad` (avoids activation recomputation double-counting).
        return not bool(is_grad)

    def _layer_name(self, router: Any) -> str:
        layer_number = getattr(router, 'layer_number', None)
        if isinstance(layer_number, int):
            return f'layer_{layer_number}'
        rid = id(router)
        name = self._router_id_to_layer.get(rid)
        if name is None:
            self._unknown_layer_counter += 1
            name = f'layer_unknown_{self._unknown_layer_counter}'
            self._router_id_to_layer[rid] = name
        return name

    def observe(self, router: Any, routing_map: Any) -> None:
        if not self.enabled():
            return
        if not getattr(router, 'training', False):
            return
        if not self._should_count():
            return

        iteration, _consumed = _get_megatron_step()
        if iteration is None:
            return

        if self._current_iteration != iteration:
            if self._tracking:
                self.flush()
            self._layers.clear()
            self._current_iteration = int(iteration)
            self._tracking = (int(iteration) % self.interval == 0)

        if not self._tracking:
            return

        try:
            import torch  # type: ignore

            with torch.no_grad():
                per_expert = routing_map.sum(dim=0).to(torch.int64)
        except Exception:
            return

        layer = self._layer_name(router)
        num_experts = int(getattr(router, 'num_experts', per_expert.numel()))
        top_k = int(getattr(router, 'topk', 1))

        acc = self._layers.get(layer)
        if acc is None:
            device = per_expert.device
            acc = _LayerAccum(
                num_experts=num_experts,
                top_k=top_k,
                assignments=per_expert.new_zeros((num_experts,), dtype=torch.int64, device=device),
            )
            self._layers[layer] = acc

        # Guard in case model changes its expert count mid-run (shouldn't happen).
        if int(acc.assignments.numel()) != int(per_expert.numel()):
            return

        try:
            acc.assignments.add_(per_expert)
            acc.tokens += int(routing_map.shape[0])
        except Exception:
            return

    def flush(self) -> None:
        if not self.enabled():
            return
        if not self._tracking or self._current_iteration is None or not self._layers:
            return

        rank = _get_rank()
        iteration = int(self._current_iteration)

        for layer, acc in self._layers.items():
            try:
                actual = acc.assignments.detach().cpu().tolist()
            except Exception:
                continue

            tokens = int(acc.tokens)
            record: Dict[str, Any] = {
                'iteration': iteration,
                'rank': rank,
                'layer': layer,
                'num_experts': int(acc.num_experts),
                'top_k': int(acc.top_k),
                'tokens': tokens,
                'actual_assignments': actual,
            }
            self._writer.submit(record)

        self._layers.clear()

    def close(self) -> None:
        self.flush()
        try:
            self._writer.close()
        except Exception:
            pass


_MONITOR: Optional[MoELoadMonitor] = None


def get_monitor_from_env() -> Optional[MoELoadMonitor]:
    global _MONITOR
    if _MONITOR is not None:
        return _MONITOR

    writer = get_writer_from_env()
    if writer is None:
        return None

    interval = int(os.environ.get('SWIFT_MOE_MONITOR_INTERVAL', '1') or '1')
    count_mode = (os.environ.get('SWIFT_MOE_MONITOR_COUNT_MODE', 'no_grad') or 'no_grad').strip().lower()
    if count_mode not in {'no_grad', 'grad', 'all'}:
        count_mode = 'no_grad'

    monitor = MoELoadMonitor(interval=max(1, interval), count_mode=count_mode)
    monitor._writer = writer
    _MONITOR = monitor
    atexit.register(monitor.close)
    return monitor


def patch_topk_router() -> None:
    """Monkey-patch Megatron-Core `TopKRouter.forward` to record `routing_map` stats."""
    try:
        from megatron.core.transformer.moe.router import TopKRouter  # type: ignore
    except Exception:
        return

    if getattr(TopKRouter, '_swift_moe_monitor_patched', False):
        return

    import functools

    orig_forward = TopKRouter.forward

    @functools.wraps(orig_forward)
    def wrapped_forward(self: Any, *args: Any, **kwargs: Any):
        probs, routing_map = orig_forward(self, *args, **kwargs)
        monitor = get_monitor_from_env()
        if monitor is not None:
            monitor.observe(self, routing_map)
        return probs, routing_map

    TopKRouter.forward = wrapped_forward  # type: ignore[method-assign]
    TopKRouter._swift_moe_monitor_patched = True
    TopKRouter._swift_moe_monitor_orig_forward = orig_forward


def enable_moe_monitor() -> None:
    """Entry point: enable only if `MOE_PATCH_DIR` is set."""
    if not os.environ.get('MOE_PATCH_DIR'):
        return
    if get_monitor_from_env() is None:
        return
    patch_topk_router()
