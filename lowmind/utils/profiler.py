"""
LowMind Model Profiler
Measure FLOPs, parameters, memory, and throughput.
"""
import time
import numpy as np
from ..core.tensor import Tensor
from ..core.module import Module


class ModelProfiler:
    """
    Profile a LowMind model: parameters, estimated FLOPs, memory, and throughput.

    Example::

        profiler = lm.ModelProfiler(model)
        report = profiler.profile(input_shape=(1, 3, 32, 32))
        profiler.print_report(report)
    """

    def __init__(self, model: Module):
        self.model = model

    def count_parameters(self):
        """Count total and trainable parameters."""
        total = 0
        trainable = 0
        for p in self.model.parameters():
            n = p.data.size
            total += n
            if p.requires_grad:
                trainable += n
        return {'total': total, 'trainable': trainable, 'non_trainable': total - trainable}

    def estimate_flops(self, input_shape):
        """
        Estimate FLOPs for a forward pass.

        Args:
            input_shape: Input tensor shape (without batch dim), e.g. (3, 32, 32).

        Returns:
            dict with 'total_flops', 'total_mflops', per-layer breakdown.
        """
        flops_by_layer = {}
        total_flops = 0

        for name, mod in self.model.named_modules():
            mod_type = type(mod).__name__

            if mod_type == 'Linear':
                flops = 2 * mod.in_features * mod.out_features
                if hasattr(mod, 'bias'):
                    flops += mod.out_features
            elif mod_type == 'Conv2d':
                # FLOPs = 2 * C_in * kH * kW * C_out * H_out * W_out
                kH, kW = mod.kernel_size
                # Rough estimate without knowing output spatial size
                flops = 2 * mod.in_channels * kH * kW * mod.out_channels
            elif mod_type in ('BatchNorm1d', 'BatchNorm2d'):
                flops = 4 * (mod.num_features if hasattr(mod, 'num_features') else 0)
            elif mod_type == 'LSTMCell':
                h = mod.hidden_size
                i = mod.input_size
                flops = 8 * (i + h) * h  # 4 gates × 2 ops each
            elif mod_type == 'GRUCell':
                h = mod.hidden_size
                i = mod.input_size
                flops = 6 * (i + h) * h  # 3 gates × 2 ops each
            else:
                continue

            if name == 'root':
                continue

            flops_by_layer[name] = {'type': mod_type, 'flops': flops}
            total_flops += flops

        return {
            'total_flops': total_flops,
            'total_mflops': total_flops / 1e6,
            'total_gflops': total_flops / 1e9,
            'layers': flops_by_layer,
        }

    def estimate_memory(self):
        """Estimate memory for parameters + gradients in bytes."""
        param_bytes = 0
        grad_bytes = 0
        for p in self.model.parameters():
            nb = p.data.nbytes
            param_bytes += nb
            if p.requires_grad:
                grad_bytes += nb

        return {
            'param_mb': param_bytes / 1e6,
            'grad_mb': grad_bytes / 1e6,
            'total_mb': (param_bytes + grad_bytes) / 1e6,
        }

    def measure_throughput(self, input_shape, batch_size=1, n_runs=20, warmup=3):
        """
        Measure actual forward-pass throughput.

        Args:
            input_shape: Shape without batch dim, e.g. (784,) or (3,32,32).
            batch_size:  Number of samples per run.
            n_runs:      Number of timing runs.
            warmup:      Warm-up runs (not counted).

        Returns:
            dict with latency_ms (mean/std) and samples_per_sec.
        """
        self.model.eval()
        full_shape = (batch_size,) + tuple(input_shape)
        x = Tensor(np.random.randn(*full_shape).astype(np.float32))

        # Warm-up
        for _ in range(warmup):
            _ = self.model(x)

        # Timed runs
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = self.model(x)
            times.append(time.perf_counter() - t0)

        times = np.array(times) * 1000  # ms
        return {
            'latency_ms_mean': float(times.mean()),
            'latency_ms_std': float(times.std()),
            'latency_ms_min': float(times.min()),
            'latency_ms_max': float(times.max()),
            'samples_per_sec': float(batch_size * 1000 / times.mean()),
        }

    def profile(self, input_shape, batch_size=1, n_runs=20):
        """
        Run a full profiling pass.

        Args:
            input_shape: Without batch dim, e.g. (784,) or (3, 32, 32).
            batch_size:  Batch size for throughput test.
            n_runs:      Number of timing iterations.

        Returns:
            Full profiling report dict.
        """
        params = self.count_parameters()
        flops = self.estimate_flops(input_shape)
        memory = self.estimate_memory()
        throughput = self.measure_throughput(input_shape, batch_size, n_runs)
        return {
            'model': type(self.model).__name__,
            'parameters': params,
            'flops': flops,
            'memory': memory,
            'throughput': throughput,
        }

    @staticmethod
    def print_report(report):
        """Pretty-print a profiling report."""
        print("=" * 62)
        print(f"  Model Profiler — {report['model']}")
        print("=" * 62)

        p = report['parameters']
        print(f"\n  Parameters")
        print(f"    Total       : {p['total']:>12,}")
        print(f"    Trainable   : {p['trainable']:>12,}")
        print(f"    Frozen      : {p['non_trainable']:>12,}")

        f = report['flops']
        print(f"\n  Estimated FLOPs (1 forward pass, batch=1)")
        print(f"    Total MFLOPs: {f['total_mflops']:>12.2f}")
        if f['layers']:
            print(f"    Per Layer:")
            for name, info in list(f['layers'].items())[:8]:
                print(f"      {name:<30} {info['type']:<14} {info['flops']:>10,}")

        m = report['memory']
        print(f"\n  Memory Estimate")
        print(f"    Parameters  : {m['param_mb']:>10.2f} MB")
        print(f"    Gradients   : {m['grad_mb']:>10.2f} MB")
        print(f"    Total       : {m['total_mb']:>10.2f} MB")

        t = report['throughput']
        print(f"\n  Throughput")
        print(f"    Latency     : {t['latency_ms_mean']:.2f} ± {t['latency_ms_std']:.2f} ms")
        print(f"    Min/Max     : {t['latency_ms_min']:.2f} / {t['latency_ms_max']:.2f} ms")
        print(f"    Throughput  : {t['samples_per_sec']:.1f} samples/sec")

        print("=" * 62)
