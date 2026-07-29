
import lowmind as lm
import numpy as np
import time

def performance_benchmark_visible():
    print("⚡ Performance Benchmark")
    print("=" * 40)
    
    # Test different model sizes
    model_sizes = [
        ("Tiny", 10, 5),
        ("Small", 50, 10), 
        ("Medium", 100, 20),
        ("Large", 200, 50)
    ]
    
    print("Model Size | Inference Time | Memory Used")
    print("-" * 45)
    
    for model_name, input_size, hidden_size in model_sizes:
        # Create model
        model = lm.Linear(input_size, hidden_size)
        
        # Create test data
        test_input = lm.Tensor(np.random.randn(1, input_size))
        
        # Time inference
        start_time = time.time()
        for _ in range(10):  # Multiple runs for average
            output = model(test_input)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10 * 1000  # ms per inference
        
        # Memory usage
        mem_info = lm.memory_manager.get_memory_info()
        memory_used = mem_info['allocated_mb']
        
        print(f"{model_name:10} | {avg_time:8.2f} ms | {memory_used:8.2f} MB")
        
        # Cleanup
        lm.memory_manager.clear_cache()
    
    print("-" * 45)
    print("✅ Benchmark Completed!")

# Run benchmark
if __name__ == "__main__":
    performance_benchmark_visible()