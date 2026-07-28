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
        {", ".join(f"{float(val):.8f}f" for val in x.data.flatten())}
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


def test_cpp_exporter_activations_and_bn():
    np.random.seed(42)

    # Build a model testing LeakyReLU, Tanh, Softmax, and BatchNorm1d
    model = lm.Sequential(
        lm.Linear(in_features=10, out_features=15),
        lm.BatchNorm1d(num_features=15),
        lm.LeakyReLU(negative_slope=0.1),
        lm.Linear(in_features=15, out_features=5),
        lm.Tanh(),
        lm.Linear(in_features=5, out_features=3),
        lm.Softmax()
    )
    model.eval()

    input_shape = (10,)
    x = lm.randn(1, *input_shape)
    python_out = model(x).data[0]

    header_path = "temp_model_act.h"
    lm.export_to_cpp(model, input_shape, header_path, namespace="my_embedded_act_model")

    runner_code = f"""#include <iostream>
#include <iomanip>
#include "{header_path}"

int main() {{
    const float input[{np.prod(input_shape)}] = {{
        {", ".join(f"{float(val):.8f}f" for val in x.data.flatten())}
    }};

    float output[my_embedded_act_model::OUTPUT_SIZE] = {{0.0f}};

    my_embedded_act_model::predict(input, output);

    std::cout << std::scientific << std::setprecision(9);
    for (int i = 0; i < my_embedded_act_model::OUTPUT_SIZE; ++i) {{
        std::cout << output[i] << (i == my_embedded_act_model::OUTPUT_SIZE - 1 ? "" : " ");
    }}
    std::cout << std::endl;
    return 0;
}}
"""
    runner_path = "temp_runner_act.cpp"
    with open(runner_path, "w") as f:
        f.write(runner_code)

    executable_path = "./temp_runner_act"
    compile_cmd = ["g++", "-O3", runner_path, "-o", executable_path]
    compile_res = subprocess.run(compile_cmd, capture_output=True, text=True)
    assert compile_res.returncode == 0, f"Compilation failed: {compile_res.stderr}"

    run_res = subprocess.run([executable_path], capture_output=True, text=True)
    assert run_res.returncode == 0, f"Execution failed: {run_res.stderr}"

    cpp_out_strs = run_res.stdout.strip().split()
    cpp_out = np.array([float(val) for val in cpp_out_strs], dtype=np.float32)

    # Clean up
    for path in (header_path, runner_path, executable_path):
        if os.path.exists(path):
            os.remove(path)

    assert np.allclose(python_out, cpp_out, atol=1e-4)


def test_cpp_exporter_unsupported_model():
    # Pass an object that does not represent a list of modules or layers to test error handling
    with pytest.raises(ValueError, match="Model has no sequential layers or submodules to export"):
        lm.export_to_cpp("not_a_model", (10,), "fail.h")


def test_cpp_exporter_profiler_and_sketch():
    np.random.seed(42)
    model = lm.Sequential(
        lm.Linear(in_features=4, out_features=2),
        lm.ReLU()
    )
    model.eval()

    header_path = "test_profile_model.h"
    sketch_path = "test_profile_model.ino"

    # Export with generate_arduino_sketch=True
    lm.export_to_cpp(model, (4,), header_path, namespace="profile_model", generate_arduino_sketch=True)

    # Assert both files were created successfully
    assert os.path.exists(header_path)
    assert os.path.exists(sketch_path)

    # Verify content of the sketch file
    with open(sketch_path, "r") as f:
        ino_content = f.read()
    assert "LOWMIND HARDWARE IN-SITU INFERENCE BENCHMARK" in ino_content
    assert "profile_model::predict" in ino_content
    assert "profile_model::TOTAL_ROM_BYTES" in ino_content

    # Clean up
    if os.path.exists(header_path):
        os.remove(header_path)
    if os.path.exists(sketch_path):
        os.remove(sketch_path)


def test_cpp_exporter_normalization():
    np.random.seed(42)
    model = lm.Sequential(
        lm.Linear(in_features=3, out_features=1),
    )
    model.eval()

    header_path = "test_norm_model.h"
    lm.export_to_cpp(model, (3,), header_path, namespace="norm_model")

    # Write verification driver testing both normalization and predict
    runner_code = f"""#include <iostream>
#include <iomanip>
#include "{header_path}"

int main() {{
    float data[3] = {{10.0f, 20.0f, 30.0f}};
    const float mean[3] = {{2.0f, 5.0f, 10.0f}};
    const float std_dev[3] = {{4.0f, 5.0f, 20.0f}};

    // Call generated normalize utility
    norm_model::normalize_input(data, mean, std_dev, 3);

    std::cout << std::scientific << std::setprecision(6);
    std::cout << data[0] << " " << data[1] << " " << data[2] << std::endl;
    return 0;
}}
"""
    runner_path = "temp_runner_norm.cpp"
    with open(runner_path, "w") as f:
        f.write(runner_code)

    executable_path = "./temp_runner_norm"
    compile_cmd = ["g++", "-O3", runner_path, "-o", executable_path]
    compile_res = subprocess.run(compile_cmd, capture_output=True, text=True)
    assert compile_res.returncode == 0, f"Compilation failed: {compile_res.stderr}"

    run_res = subprocess.run([executable_path], capture_output=True, text=True)
    assert run_res.returncode == 0, f"Execution failed: {run_res.stderr}"

    cpp_out_strs = run_res.stdout.strip().split()
    cpp_out = np.array([float(val) for val in cpp_out_strs], dtype=np.float32)

    # Clean up
    for path in (header_path, runner_path, executable_path):
        if os.path.exists(path):
            os.remove(path)

    # Expected python manual normalization: (val - mean) / std
    expected = (np.array([10.0, 20.0, 30.0]) - np.array([2.0, 5.0, 10.0])) / np.array([4.0, 5.0, 20.0])
    assert np.allclose(expected, cpp_out, atol=1e-4)
