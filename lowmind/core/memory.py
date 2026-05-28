"""
LowMind Memory Manager
Optimized for low-end devices like Raspberry Pi
"""
import gc
import os
import time

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


class MemoryManager:
    """
    Advanced memory manager optimized for resource-constrained devices.
    Uses LRU eviction strategy and aggressive garbage collection.
    """

    def __init__(self, max_memory_mb=256):
        self.max_memory = max_memory_mb * 1024 * 1024
        self.allocated_memory = 0
        self.tensors = {}          # name -> (tensor, size, last_used_ts)
        self.memory_history = []
        self.peak_memory = 0
        self.low_memory_mode = max_memory_mb <= 128
        self._counter = 0          # unique ID counter for auto-named tensors

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def allocate(self, tensor, name=None):
        size = tensor.data.nbytes
        if getattr(tensor, 'grad', None) is not None:
            size += tensor.grad.nbytes

        for retry in range(3):
            if self.allocated_memory + size <= self.max_memory:
                break
            if retry == 0:
                self.free_unused()
            elif retry == 1:
                self.free_all_non_essential()
            else:
                self.clear_cache()
            gc.collect()
        else:
            raise MemoryError(
                f"Memory limit exceeded: "
                f"used={self.allocated_memory / 1e6:.1f}MB  "
                f"requested={size / 1e6:.2f}MB  "
                f"max={self.max_memory / 1e6:.0f}MB"
            )

        self.allocated_memory += size
        self.peak_memory = max(self.peak_memory, self.allocated_memory)

        if name:
            self.tensors[name] = (tensor, size, time.time())

        self.memory_history.append((time.time(), self.allocated_memory))
        if len(self.memory_history) > 100:
            self.memory_history.pop(0)

        return tensor

    def auto_name(self):
        """Generate a unique tensor name."""
        self._counter += 1
        return f"_tensor_{self._counter}"

    # ------------------------------------------------------------------
    # Deallocation helpers
    # ------------------------------------------------------------------

    def free(self, name):
        if name in self.tensors:
            _, size, _ = self.tensors.pop(name)
            self.allocated_memory = max(0, self.allocated_memory - size)

    def free_unused(self):
        """Evict tensors not recently used and not marked as parameters."""
        now = time.time()
        to_remove = [
            n for n, (t, _, ts) in self.tensors.items()
            if (not getattr(t, '_is_parameter', False) or (now - ts > 60))
            and not getattr(t, 'requires_grad', False)
        ]
        for n in to_remove:
            self.free(n)

    def free_all_non_essential(self):
        """Keep only parameter tensors."""
        to_remove = [
            n for n, (t, _, _) in self.tensors.items()
            if not getattr(t, '_is_parameter', False)
        ]
        for n in to_remove:
            self.free(n)

    def clear_cache(self):
        for name in list(self.tensors.keys()):
            self.free(name)

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    def get_memory_info(self):
        info = {
            'allocated_mb': self.allocated_memory / 1e6,
            'max_mb': self.max_memory / 1e6,
            'usage_percent': (self.allocated_memory / self.max_memory) * 100,
            'tensors_count': len(self.tensors),
            'peak_memory_mb': self.peak_memory / 1e6,
        }
        if _PSUTIL_AVAILABLE:
            try:
                proc = psutil.Process(os.getpid())
                vm = psutil.virtual_memory()
                info['process_memory_mb'] = proc.memory_info().rss / 1e6
                info['system_memory_percent'] = vm.percent
            except Exception:
                pass
        return info

    def optimize_for_inference(self):
        """Drop all gradient buffers (inference mode)."""
        for name, (tensor, _, _) in self.tensors.items():
            if getattr(tensor, 'grad', None) is not None:
                tensor.grad = None
        gc.collect()

    def __repr__(self):
        info = self.get_memory_info()
        return (
            f"MemoryManager(allocated={info['allocated_mb']:.1f}MB / "
            f"{info['max_mb']:.0f}MB, tensors={info['tensors_count']})"
        )


# Module-level singleton — can be reconfigured by the user
memory_manager = MemoryManager(max_memory_mb=256)


def configure_memory(max_mb=256, low_memory_mode=None):
    """
    Reconfigure the global memory manager.

    Args:
        max_mb: Maximum memory in megabytes (default 256).
        low_memory_mode: If True, enables aggressive cleanup. Auto-detected if None.
    """
    global memory_manager
    memory_manager = MemoryManager(max_memory_mb=max_mb)
    if low_memory_mode is not None:
        memory_manager.low_memory_mode = low_memory_mode
