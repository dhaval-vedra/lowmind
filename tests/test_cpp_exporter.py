import os
import subprocess
import numpy as np
import pytest
import lowmind as lm

def test_cpp_exporter_end_to_end():
    np.random.seed(42)

    # 1. Build a diverse Sequential model including Conv2d, BatchNorm, Activations, MaxPool, Flatten, and Linear
    model = lm.Sequential(
        lm.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1),
        lm.BatchNorm2d(num_features=2),
        lm.ReLU(),
        lm.MaxPool2d(kernel_size=2, stride=2),
        lm.Flatten(),
        lm.Linear(in_features=32, out_features=4),
        lm.Sigmoid()
    )
    model.eval()

    # 2. Define input shape and generate sample input
    input_shape = (1, 8, 8) # (C, H, W)
    x = lm.randn(1, *input_shape) # Batch size 1 for microcontroller scenario
    python_out = model(x).data[0] # Extract the single flat array output

    # 3. Export model to C++ header
    header_path = "temp_model.h"
    lm.export_to_cpp(model, input_shape, header_path, namespace="my_embedded_model")

    # 4. Generate C++ Verification Test Runner Code
    runner_code = f"""#include <iostream>
#include <iomanip>
#include "{header_path}"

int main() {{
    // Input values matching the Python test input
    const float input[{np.prod(input_shape)}] = {{
        {", ".join(f"{float(val):.9g}f" for val in x.data.flatten())}
    }};

    float output[my_embedded_model::OUTPUT_SIZE] = {{0.0f}};

    // Call the exported static model forward predict function
    my_embedded_model::predict(input, output);

    // Print the output array values in a parsed format
    std::cout << std::scientific << std::setprecision(9);
    for (int i = 0; i < my_embedded_model::OUTPUT_SIZE; ++i) {{
        std::cout << output[i] << (i == my_embedded_model::OUTPUT_SIZE - 1 ? "" : " ");
    }}
    std::cout << std::endl;
    return 0;
}}
"""
    runner_path = "temp_runner.cpp"
    with open(runner_path, "w") as f:
        f.write(runner_code)

    # 5. Compile the C++ program using g++
    executable_path = "./temp_runner"
    compile_cmd = ["g++", "-O3", runner_path, "-o", executable_path]
    compile_res = subprocess.run(compile_cmd, capture_output=True, text=True)

    # Assert successful compilation
    assert compile_res.returncode == 0, f"Compilation failed: {compile_res.stderr}"

    # 6. Run the compiled binary and capture the outputs
    run_res = subprocess.run([executable_path], capture_output=True, text=True)
    assert run_res.returncode == 0, f"Execution failed: {run_res.stderr}"

    cpp_out_strs = run_res.stdout.strip().split()
    cpp_out = np.array([float(val) for val in cpp_out_strs], dtype=np.float32)

    # Clean up temporary compile files
    for path in (header_path, runner_path, executable_path):
        if os.path.exists(path):
            os.remove(path)

    # 7. Assert numeric parity to a high degree of precision (tolerance < 1e-4)
    print("Python Output:", python_out)
    print("C++ Output:   ", cpp_out)
    assert np.allclose(python_out, cpp_out, atol=1e-4)
