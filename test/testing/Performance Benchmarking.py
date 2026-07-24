import lowmind as lm
import numpy as np
import time

def benchmark_lowmind():
    """Comprehensive performance testing"""
    print("🧪 LowMind Performance Benchmark")
    print("=" * 50)
    
    # Test 1: Tensor creation speed
    print("1. Tensor Creation Speed:")
    start_time = time.time()
    tensors = [lm.Tensor(np.random.randn(100, 100)) for _ in range(10)]
    creation_time = time.time() - start_time
    print(f"   ✅ 10 tensors created in {creation_time:.4f}s")
    
    # Test 2: Operation speed
    print("\n2. Operation Speed:")
    a = lm.Tensor(np.random.randn(500, 500))
    b = lm.Tensor(np.random.randn(500, 500))
    
    start_time = time.time()
    c = a + b
    add_time = time.time() - start_time
    print(f"   ✅ Addition: {add_time:.4f}s")
    
    start_time = time.time()
    d = a * b
    mul_time = time.time() - start_time
    print(f"   ✅ Multiplication: {mul_time:.4f}s")
    
    # Test 3: Backward pass speed
    print("\n3. Backward Pass Speed:")
    a = lm.Tensor(np.random.randn(100, 100), requires_grad=True)
    b = lm.Tensor(np.random.randn(100, 100), requires_grad=True)
    c = (a * b).sum()
    
    start_time = time.time()
    c.backward()
    backward_time = time.time() - start_time
    print(f"   ✅ Backward pass: {backward_time:.4f}s")
    
    # Test 4: Memory efficiency
    print("\n4. Memory Efficiency:")
    lm.memory_manager.clear_cache()
    
    initial_memory = lm.memory_manager.get_memory_info()['allocated_mb']
    
    # Create large tensors
    large_tensors = []
    for i in range(5):
        t = lm.Tensor(np.random.randn(500, 500), name=f"large_{i}")
        large_tensors.append(t)
    
    final_memory = lm.memory_manager.get_memory_info()['allocated_mb']
    memory_used = final_memory - initial_memory
    
    print(f"   ✅ Memory used: {memory_used:.2f}MB")
    print(f"   ✅ Memory per tensor: {memory_used/5:.2f}MB")
    
    # Cleanup
    del large_tensors
    lm.memory_manager.clear_cache()

# Run benchmark
benchmark_lowmind()