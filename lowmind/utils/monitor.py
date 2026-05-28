"""
LowMind System Monitor — works on Raspberry Pi and regular machines
"""
import os
import time
from ..core.memory import memory_manager

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False


class SystemMonitor:
    """
    Real-time system health monitor optimized for Raspberry Pi.

    Args:
        max_samples: History buffer size (default 100).

    Example::

        monitor = lm.SystemMonitor()
        monitor.print_status()
        score = monitor.health_score()
    """

    def __init__(self, max_samples=100):
        self.max_samples = max_samples
        self.history = []
        self._start = time.time()

    def get_stats(self):
        """Return current system statistics as a dict."""
        stats = {
            'uptime_s': time.time() - self._start,
            'cpu_temp_c': self._get_cpu_temp(),
        }
        if _PSUTIL:
            try:
                vm = psutil.virtual_memory()
                cpu = psutil.cpu_percent(interval=0.05)
                disk = psutil.disk_usage('/')
                proc = psutil.Process(os.getpid())
                stats.update({
                    'cpu_percent': cpu,
                    'ram_percent': vm.percent,
                    'ram_available_mb': vm.available / 1e6,
                    'disk_percent': disk.percent,
                    'process_rss_mb': proc.memory_info().rss / 1e6,
                })
            except Exception:
                pass
        stats.update(memory_manager.get_memory_info())
        return stats

    @staticmethod
    def _get_cpu_temp():
        # Raspberry Pi
        try:
            out = os.popen('vcgencmd measure_temp').read()
            if 'temp=' in out:
                return float(out.split('=')[1].replace("'C\n", ""))
        except Exception:
            pass
        # Linux thermal zone
        try:
            with open('/sys/class/thermal/thermal_zone0/temp') as f:
                return int(f.read().strip()) / 1000.0
        except Exception:
            pass
        return None

    def update(self):
        stats = self.get_stats()
        self.history.append(stats)
        if len(self.history) > self.max_samples:
            self.history.pop(0)
        return stats

    def health_score(self):
        """
        Compute a 0-100 health score based on memory, CPU temp, and RAM.

        Returns:
            float — higher is healthier.
        """
        stats = self.get_stats()
        scores = []

        mem_info = memory_manager.get_memory_info()
        scores.append(max(0.0, 100.0 - mem_info['usage_percent']))

        temp = stats.get('cpu_temp_c')
        if temp is not None:
            scores.append(max(0.0, 100.0 - max(0.0, temp - 40) * 2))

        ram_pct = stats.get('ram_percent')
        if ram_pct is not None:
            scores.append(max(0.0, 100.0 - ram_pct))

        return sum(scores) / max(len(scores), 1)

    def print_status(self):
        """Print a formatted status report."""
        stats = self.get_stats()
        print("=" * 56)
        print("  LowMind System Monitor")
        print("=" * 56)
        mem = stats.get('allocated_mb', 0)
        max_mem = stats.get('max_mb', 0)
        print(f"  Framework RAM   : {mem:.1f} MB / {max_mem:.0f} MB")
        print(f"  Tensor Count    : {stats.get('tensors_count', '?')}")
        if 'ram_percent' in stats:
            print(f"  System RAM      : {stats['ram_percent']:.1f}% used  "
                  f"({stats.get('ram_available_mb', 0):.0f} MB free)")
        if 'cpu_percent' in stats:
            print(f"  CPU Usage       : {stats['cpu_percent']:.1f}%")
        if 'process_rss_mb' in stats:
            print(f"  Process Memory  : {stats['process_rss_mb']:.1f} MB")
        temp = stats.get('cpu_temp_c')
        if temp is not None:
            flag = " [HOT]" if temp > 75 else (" [WARM]" if temp > 60 else "")
            print(f"  CPU Temp        : {temp:.1f} C{flag}")
        print(f"  Uptime          : {stats['uptime_s']:.1f}s")
        score = self.health_score()
        bar = "#" * int(score // 5) + "-" * (20 - int(score // 5))
        print(f"  Health Score    : {score:.0f}/100  [{bar}]")
        print("=" * 56)


class memory_trace:
    """
    Context manager to measure memory and time usage of a code block.

    Example::

        with lm.memory_trace("Forward Pass"):
            out = model(x)
    """

    def __init__(self, name="block"):
        self.name = name
        self._start_mem = 0
        self._start_time = 0

    def __enter__(self):
        self._start_mem = memory_manager.allocated_memory
        self._start_time = time.time()
        return self

    def __exit__(self, *args):
        end_mem = memory_manager.allocated_memory
        elapsed = time.time() - self._start_time
        delta = (end_mem - self._start_mem) / 1e6
        print(f"[{self.name}]  time={elapsed*1000:.1f}ms  "
              f"mem_delta={delta:+.2f}MB  total={end_mem/1e6:.2f}MB")
