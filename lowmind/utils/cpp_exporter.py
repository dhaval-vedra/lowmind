"""
LowMind Embedded C++ Inference Engine Exporter — pure C++ model export for microcontrollers
"""
import os
import numpy as np

def export_to_cpp(model, input_shape, output_path, namespace="lowmind_model", generate_arduino_sketch=False):
    """
    Exports a LowMind Sequential (or compatible) model into a single self-contained C++ header file.
    Uses a highly efficient, deterministic Ping-Pong Static Buffer design to avoid dynamic allocation.

    Auxiliary production features added:
    1. ROM/RAM Footprint & FLOP Profiler layer-by-layer.
    2. Embedded C++ Input Normalization Helper generator.
    3. Arduino/ESP32 Prototyping Sketch Generator.

    Args:
        model: LowMind Model or Sequential instance.
        input_shape: tuple of ints (e.g. (3, 28, 28) or (784,)) representing a single sample shape.
        output_path: file path to write the generated C++ header file.
        namespace: C++ namespace for the model.
        generate_arduino_sketch: If True, writes a ready-to-flash Arduino .ino file next to the header.
    """
    # Collect all modules
    if hasattr(model, "layers"):
        layers = model.layers
    elif hasattr(model, "_modules"):
        layers = list(model._modules.values())
    else:
        raise ValueError("Model has no sequential layers or submodules to export.")

    # 1. Compute intermediate shapes and required maximum static buffer size
    current_shape = list(input_shape)
    input_size = int(np.prod(current_shape))
    max_buffer_size = input_size
    layer_shapes = [list(current_shape)]

    for layer in layers:
        name = layer.__class__.__name__
        if name == "Linear":
            current_shape = [layer.out_features]
        elif name == "Conv2d":
            if len(current_shape) != 3:
                raise ValueError("Conv2d expects a 3D input shape (C, H, W)")
            C, H, W = current_shape
            kH, kW = layer.kernel_size
            sH, sW = layer.stride
            pH, pW = layer.padding
            out_H = (H + 2 * pH - kH) // sH + 1
            out_W = (W + 2 * pW - kW) // sW + 1
            out_C = layer.out_channels
            current_shape = [out_C, out_H, out_W]
        elif name in ("MaxPool2d", "AvgPool2d"):
            if len(current_shape) != 3:
                raise ValueError(f"{name} expects a 3D input shape (C, H, W)")
            C, H, W = current_shape
            kH, kW = layer.kernel_size
            sH, sW = layer.stride
            pH, pW = layer.padding
            out_H = (H + 2 * pH - kH) // sH + 1
            out_W = (W + 2 * pW - kW) // sW + 1
            current_shape = [C, out_H, out_W]
        elif name == "Flatten":
            current_shape = [int(np.prod(current_shape))]
        elif name in ("BatchNorm2d", "BatchNorm1d", "Dropout", "ReLU", "LeakyReLU", "ELU", "GELU", "Sigmoid", "Tanh", "Softmax", "LogSoftmax"):
            pass
        else:
            # Custom or ignored layer, shape unchanged
            pass

        size = int(np.prod(current_shape))
        if size > max_buffer_size:
            max_buffer_size = size
        layer_shapes.append(list(current_shape))

    output_size = int(np.prod(current_shape))

    # Helper function to generate C++ float array from NumPy array
    def array_to_cpp_initializer(arr):
        flat = arr.ravel()
        return ", ".join(f"{float(val):.8f}f" for val in flat)

    # 2. FLOPs and Footprint profiling
    print("\n" + "="*55)
    print("🚀  LOWMIND EMBEDDED FOOTPRINT & PROFILER REPORT")
    print("="*55)
    print(f"{'Layer Type':<20} | {'Output Shape':<12} | {'ROM (Bytes)':<10} | {'FLOPs':<8}")
    print("-"*55)

    total_flops = 0
    total_rom_bytes = 0

    for idx, layer in enumerate(layers):
        name = layer.__class__.__name__
        in_sh = layer_shapes[idx]
        out_sh = layer_shapes[idx + 1]
        in_size = int(np.prod(in_sh))
        out_size = int(np.prod(out_sh))

        flops = 0
        rom_bytes = 0

        if name == "Linear":
            w_size = layer.weight.data.size
            b_size = layer.bias.data.size if (hasattr(layer, "bias") and layer.bias is not None) else 0
            rom_bytes = (w_size + b_size) * 4
            # 2 FLOPs (multiply and add) per weight connection
            flops = out_size * (in_size * 2)
        elif name == "Conv2d":
            w_size = layer.weight.data.size
            b_size = layer.bias.data.size if (hasattr(layer, "bias") and layer.bias is not None) else 0
            rom_bytes = (w_size + b_size) * 4
            flops = out_sh[0] * out_sh[1] * out_sh[2] * (in_sh[0] * layer.kernel_size[0] * layer.kernel_size[1] * 2)
        elif name in ("BatchNorm1d", "BatchNorm2d"):
            m_size = layer.running_mean.size
            v_size = layer.running_var.size
            g_size = layer.gamma.data.size if getattr(layer, "affine", True) else 0
            b_size = layer.beta.data.size if getattr(layer, "affine", True) else 0
            rom_bytes = (m_size + v_size + g_size + b_size) * 4
            # 4 FLOPs: sub mean, div std, mul gamma, add beta per element
            flops = in_size * 4
        elif name in ("MaxPool2d", "AvgPool2d"):
            # kH * kW comparisons or additions per output element
            kH, kW = layer.kernel_size
            flops = out_size * (kH * kW)
        elif name in ("ReLU", "LeakyReLU", "Sigmoid", "Tanh", "Softmax"):
            flops = in_size

        total_flops += flops
        total_rom_bytes += rom_bytes

        print(f"{name:<20} | {str(out_sh):<12} | {rom_bytes:<10,} | {flops:<,}")

    ram_scratchpad_bytes = 2 * max_buffer_size * 4
    print("-"*55)
    print(f"📊  Total ROM Usage:    {total_rom_bytes:,} Bytes")
    print(f"🐏  Static RAM Footprint: {ram_scratchpad_bytes:,} Bytes (using Ping-Pong Buffers)")
    print(f"⚡  Total Model FLOPs:   {total_flops:,} Floating-Point Operations")
    print("="*55 + "\n")

    # Begin writing C++ content
    cpp = []
    cpp.append(f"// Generated by LowMind Embedded C++ Exporter")
    cpp.append(f"#ifndef {namespace.upper()}_H")
    cpp.append(f"#define {namespace.upper()}_H\n")
    cpp.append(f"#include <cmath>")
    cpp.append(f"#include <algorithm>\n")
    cpp.append(f"namespace {namespace} {{\n")

    # Define metadata constants
    cpp.append(f"// Model Meta Data")
    cpp.append(f"constexpr int INPUT_SIZE = {input_size};")
    cpp.append(f"constexpr int OUTPUT_SIZE = {output_size};")
    cpp.append(f"constexpr int MAX_BUFFER_SIZE = {max_buffer_size};")
    cpp.append(f"constexpr int TOTAL_ROM_BYTES = {total_rom_bytes};")
    cpp.append(f"constexpr int STATIC_RAM_BYTES = {ram_scratchpad_bytes};")
    cpp.append(f"constexpr int TOTAL_FLOPS = {total_flops};\n")

    # Write C++ Layer Math Primitives
    cpp.append("""// --- Highly Optimized Self-Contained Layer Implementations ---

inline void linear_forward(const float* x, const float* w, const float* b, float* out, int in_features, int out_features) {
    for (int i = 0; i < out_features; ++i) {
        float val = (b != nullptr) ? b[i] : 0.0f;
        for (int j = 0; j < in_features; ++j) {
            val += x[j] * w[i * in_features + j];
        }
        out[i] = val;
    }
}

inline void conv2d_forward(const float* x, const float* w, const float* b, float* out,
                           int in_C, int in_H, int in_W,
                           int out_C, int kH, int kW,
                           int stride_H, int stride_W,
                           int padding_H, int padding_W,
                           int out_H, int out_W) {
    for (int o = 0; o < out_C; ++o) {
        for (int oh = 0; oh < out_H; ++oh) {
            for (int ow = 0; ow < out_W; ++ow) {
                float val = (b != nullptr) ? b[o] : 0.0f;
                int h_start = oh * stride_H - padding_H;
                int w_start = ow * stride_W - padding_W;

                for (int c = 0; c < in_C; ++c) {
                    for (int kh = 0; kh < kH; ++kh) {
                        int h_pos = h_start + kh;
                        if (h_pos < 0 || h_pos >= in_H) continue;

                        for (int kw = 0; kw < kW; ++kw) {
                            int w_pos = w_start + kw;
                            if (w_pos < 0 || w_pos >= in_W) continue;

                            int x_idx = c * (in_H * in_W) + h_pos * in_W + w_pos;
                            int w_idx = o * (in_C * kH * kW) + c * (kH * kW) + kh * kW + kw;

                            val += x[x_idx] * w[w_idx];
                        }
                    }
                }
                int out_idx = o * (out_H * out_W) + oh * out_W + ow;
                out[out_idx] = val;
            }
        }
    }
}

inline void maxpool2d_forward(const float* x, float* out,
                              int C, int H, int W,
                              int kH, int kW,
                              int stride_H, int stride_W,
                              int padding_H, int padding_W,
                              int out_H, int out_W) {
    for (int c = 0; c < C; ++c) {
        for (int oh = 0; oh < out_H; ++oh) {
            for (int ow = 0; ow < out_W; ++ow) {
                float max_val = -1e30f;
                int h_start = oh * stride_H - padding_H;
                int w_start = ow * stride_W - padding_W;

                for (int kh = 0; kh < kH; ++kh) {
                    int h_pos = h_start + kh;
                    if (h_pos < 0 || h_pos >= H) continue;

                    for (int kw = 0; kw < kW; ++kw) {
                        int w_pos = w_start + kw;
                        if (w_pos < 0 || w_pos >= W) continue;

                        int x_idx = c * (H * W) + h_pos * W + w_pos;
                        if (x[x_idx] > max_val) {
                            max_val = x[x_idx];
                        }
                    }
                }
                int out_idx = c * (out_H * out_W) + oh * out_W + ow;
                out[out_idx] = max_val;
            }
        }
    }
}

inline void avgpool2d_forward(const float* x, float* out,
                              int C, int H, int W,
                              int kH, int kW,
                              int stride_H, int stride_W,
                              int padding_H, int padding_W,
                              int out_H, int out_W) {
    for (int c = 0; c < C; ++c) {
        for (int oh = 0; oh < out_H; ++oh) {
            for (int ow = 0; ow < out_W; ++ow) {
                float sum_val = 0.0f;
                int count = 0;
                int h_start = oh * stride_H - padding_H;
                int w_start = ow * stride_W - padding_W;

                for (int kh = 0; kh < kH; ++kh) {
                    int h_pos = h_start + kh;
                    if (h_pos < 0 || h_pos >= H) continue;

                    for (int kw = 0; kw < kW; ++kw) {
                        int w_pos = w_start + kw;
                        if (w_pos < 0 || w_pos >= W) continue;

                        int x_idx = c * (H * W) + h_pos * W + w_pos;
                        sum_val += x[x_idx];
                        count++;
                    }
                }
                int out_idx = c * (out_H * out_W) + oh * out_W + ow;
                out[out_idx] = (count > 0) ? (sum_val / count) : 0.0f;
            }
        }
    }
}

inline void relu_forward(const float* x, float* out, int size) {
    for (int i = 0; i < size; ++i) {
        out[i] = (x[i] > 0.0f) ? x[i] : 0.0f;
    }
}

inline void leaky_relu_forward(const float* x, float* out, int size, float alpha) {
    for (int i = 0; i < size; ++i) {
        out[i] = (x[i] > 0.0f) ? x[i] : (alpha * x[i]);
    }
}

inline void sigmoid_forward(const float* x, float* out, int size) {
    for (int i = 0; i < size; ++i) {
        out[i] = 1.0f / (1.0f + expf(-x[i]));
    }
}

inline void tanh_forward(const float* x, float* out, int size) {
    for (int i = 0; i < size; ++i) {
        out[i] = tanhf(x[i]);
    }
}

inline void softmax_forward(const float* x, float* out, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; ++i) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        out[i] = expf(x[i] - max_val);
        sum += out[i];
    }
    for (int i = 0; i < size; ++i) {
        out[i] /= sum;
    }
}

inline void batchnorm2d_forward(const float* x, const float* gamma, const float* beta,
                                 const float* mean, const float* var, float* out,
                                 int C, int H, int W, float eps) {
    for (int c = 0; c < C; ++c) {
        float g = (gamma != nullptr) ? gamma[c] : 1.0f;
        float b = (beta != nullptr) ? beta[c] : 0.0f;
        float m = mean[c];
        float v = var[c];
        float inv_std = 1.0f / sqrtf(v + eps);

        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                int idx = c * (H * W) + h * W + w;
                out[idx] = (x[idx] - m) * inv_std * g + b;
            }
        }
    }
}

inline void batchnorm1d_forward(const float* x, const float* gamma, const float* beta,
                                 const float* mean, const float* var, float* out,
                                 int size, float eps) {
    for (int i = 0; i < size; ++i) {
        float g = (gamma != nullptr) ? gamma[i] : 1.0f;
        float b = (beta != nullptr) ? beta[i] : 0.0f;
        float m = mean[i];
        float v = var[i];
        float inv_std = 1.0f / sqrtf(v + eps);
        out[i] = (x[i] - m) * inv_std * g + b;
    }
}

// --- Microcontroller Input Normalization Utility ---
inline void normalize_input(float* data, const float* mean, const float* std, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = (data[i] - mean[i]) / std[i];
    }
}
""")

    # 3. Export Weights and Biases of each layer as static float arrays
    cpp.append("// --- Static Model Parameters (Weights & Biases) ---")
    for idx, layer in enumerate(layers):
        name = layer.__class__.__name__
        if name == "Linear":
            w_init = array_to_cpp_initializer(layer.weight.data)
            cpp.append(f"const float weight_{idx}[] = {{ {w_init} }};")
            if hasattr(layer, "bias") and layer.bias is not None:
                b_init = array_to_cpp_initializer(layer.bias.data)
                cpp.append(f"const float bias_{idx}[] = {{ {b_init} }};")
        elif name == "Conv2d":
            w_init = array_to_cpp_initializer(layer.weight.data)
            cpp.append(f"const float weight_{idx}[] = {{ {w_init} }};")
            if hasattr(layer, "bias") and layer.bias is not None:
                b_init = array_to_cpp_initializer(layer.bias.data)
                cpp.append(f"const float bias_{idx}[] = {{ {b_init} }};")
        elif name in ("BatchNorm1d", "BatchNorm2d"):
            m_init = array_to_cpp_initializer(layer.running_mean)
            v_init = array_to_cpp_initializer(layer.running_var)
            cpp.append(f"const float running_mean_{idx}[] = {{ {m_init} }};")
            cpp.append(f"const float running_var_{idx}[] = {{ {v_init} }};")
            if getattr(layer, "affine", True):
                g_init = array_to_cpp_initializer(layer.gamma.data)
                b_init = array_to_cpp_initializer(layer.beta.data)
                cpp.append(f"const float gamma_{idx}[] = {{ {g_init} }};")
                cpp.append(f"const float beta_{idx}[] = {{ {b_init} }};")

    cpp.append("")

    # 4. Generate Static Ping-Pong Inference Predict Call
    cpp.append("// --- Sequential Inference Function ---")
    cpp.append("inline void predict(const float* input, float* output) {")
    cpp.append("    // Ping-Pong static scratchpad buffers")
    cpp.append("    static float buf1[MAX_BUFFER_SIZE];")
    cpp.append("    static float buf2[MAX_BUFFER_SIZE];\n")
    cpp.append("    // Copy input into buf1 to kick off pipeline")
    cpp.append("    for (int i = 0; i < INPUT_SIZE; ++i) buf1[i] = input[i];\n")

    current_src = "buf1"
    current_dst = "buf2"

    for idx, layer in enumerate(layers):
        name = layer.__class__.__name__
        in_sh = layer_shapes[idx]
        out_sh = layer_shapes[idx + 1]
        in_size = int(np.prod(in_sh))
        out_size = int(np.prod(out_sh))

        cpp.append(f"    // Layer {idx}: {name}")

        if name == "Linear":
            bias_arg = f"bias_{idx}" if (hasattr(layer, "bias") and layer.bias is not None) else "nullptr"
            cpp.append(f"    linear_forward({current_src}, weight_{idx}, {bias_arg}, {current_dst}, {in_size}, {out_size});")
        elif name == "Conv2d":
            bias_arg = f"bias_{idx}" if (hasattr(layer, "bias") and layer.bias is not None) else "nullptr"
            kH, kW = layer.kernel_size
            sH, sW = layer.stride
            pH, pW = layer.padding
            cpp.append(f"    conv2d_forward({current_src}, weight_{idx}, {bias_arg}, {current_dst}, "
                       f"{in_sh[0]}, {in_sh[1]}, {in_sh[2]}, {out_sh[0]}, "
                       f"{kH}, {kW}, {sH}, {sW}, {pH}, {pW}, {out_sh[1]}, {out_sh[2]});")
        elif name == "MaxPool2d":
            kH, kW = layer.kernel_size
            sH, sW = layer.stride
            pH, pW = layer.padding
            cpp.append(f"    maxpool2d_forward({current_src}, {current_dst}, "
                       f"{in_sh[0]}, {in_sh[1]}, {in_sh[2]}, "
                       f"{kH}, {kW}, {sH}, {sW}, {pH}, {pW}, {out_sh[1]}, {out_sh[2]});")
        elif name == "AvgPool2d":
            kH, kW = layer.kernel_size
            sH, sW = layer.stride
            pH, pW = layer.padding
            cpp.append(f"    avgpool2d_forward({current_src}, {current_dst}, "
                       f"{in_sh[0]}, {in_sh[1]}, {in_sh[2]}, "
                       f"{kH}, {kW}, {sH}, {sW}, {pH}, {pW}, {out_sh[1]}, {out_sh[2]});")
        elif name == "Flatten":
            cpp.append(f"    // Flatten is a conceptual pointer copy")
            cpp.append(f"    for (int i = 0; i < {in_size}; ++i) {current_dst}[i] = {current_src}[i];")
        elif name == "ReLU":
            cpp.append(f"    relu_forward({current_src}, {current_dst}, {in_size});")
        elif name == "LeakyReLU":
            cpp.append(f"    leaky_relu_forward({current_src}, {current_dst}, {in_size}, {layer.negative_slope:.9g}f);")
        elif name == "Sigmoid":
            cpp.append(f"    sigmoid_forward({current_src}, {current_dst}, {in_size});")
        elif name == "Tanh":
            cpp.append(f"    tanh_forward({current_src}, {current_dst}, {in_size});")
        elif name == "Softmax":
            cpp.append(f"    softmax_forward({current_src}, {current_dst}, {in_size});")
        elif name == "BatchNorm2d":
            g_arg = f"gamma_{idx}" if getattr(layer, "affine", True) else "nullptr"
            b_arg = f"beta_{idx}" if getattr(layer, "affine", True) else "nullptr"
            cpp.append(f"    batchnorm2d_forward({current_src}, {g_arg}, {b_arg}, "
                       f"running_mean_{idx}, running_var_{idx}, {current_dst}, "
                       f"{in_sh[0]}, {in_sh[1]}, {in_sh[2]}, {layer.eps:.9g}f);")
        elif name == "BatchNorm1d":
            g_arg = f"gamma_{idx}" if getattr(layer, "affine", True) else "nullptr"
            b_arg = f"beta_{idx}" if getattr(layer, "affine", True) else "nullptr"
            cpp.append(f"    batchnorm1d_forward({current_src}, {g_arg}, {b_arg}, "
                       f"running_mean_{idx}, running_var_{idx}, {current_dst}, "
                       f"{in_size}, {layer.eps:.9g}f);")
        elif name == "Dropout":
            cpp.append(f"    // Dropout is a no-op during inference")
            cpp.append(f"    for (int i = 0; i < {in_size}; ++i) {current_dst}[i] = {current_src}[i];")
        else:
            cpp.append(f"    // Unsupported layer '{name}' bypassed")
            cpp.append(f"    for (int i = 0; i < {in_size}; ++i) {current_dst}[i] = {current_src}[i];")

        # Swap ping pong buffers
        current_src, current_dst = current_dst, current_src
        cpp.append("")

    # Final copy to output
    cpp.append(f"    // Copy final output to target pointer")
    cpp.append(f"    for (int i = 0; i < OUTPUT_SIZE; ++i) output[i] = {current_src}[i];")
    cpp.append("}")

    cpp.append(f"\n}} // namespace {namespace}")
    cpp.append(f"#endif // {namespace.upper()}_H")

    # Write output header to file
    with open(output_path, "w") as f:
        f.write("\n".join(cpp))
    print(f"Successfully exported LowMind model to C++ Embedded Inference Engine → {output_path}")

    # 5. Optionally generate ready-to-flash Arduino Sketch (.ino)
    if generate_arduino_sketch:
        header_filename = os.path.basename(output_path)
        sketch_path = os.path.splitext(output_path)[0] + ".ino"

        sketch = f"""// Ready-to-flash Arduino/ESP32 Benchmark Sketch for LowMind
#include "{header_filename}"

// Mock inputs initialized to 0.5
float input_data[{namespace}::INPUT_SIZE];
float output_data[{namespace}::OUTPUT_SIZE];

void setup() {{
    Serial.begin(115200);
    while (!Serial) delay(10);

    Serial.println("================================================");
    Serial.println("🍓 LOWMIND HARDWARE IN-SITU INFERENCE BENCHMARK");
    Serial.println("================================================");

    // Fill mock inputs
    for (int i = 0; i < {namespace}::INPUT_SIZE; ++i) {{
        input_data[i] = 0.5f;
    }}

    Serial.print("Model ROM Footprint: ");
    Serial.print({namespace}::TOTAL_ROM_BYTES);
    Serial.println(" Bytes");

    Serial.print("Model RAM Footprint: ");
    Serial.print({namespace}::STATIC_RAM_BYTES);
    Serial.println(" Bytes");

    Serial.print("Total Layer FLOPs:   ");
    Serial.println({namespace}::TOTAL_FLOPS);

    // Warm-up inference
    {namespace}::predict(input_data, output_data);
}}

void loop() {{
    unsigned long start_time = micros();

    // Run hardware inference
    {namespace}::predict(input_data, output_data);

    unsigned long duration = micros() - start_time;

    Serial.print("⚡ Hardware Inference completed in: ");
    Serial.print(duration);
    Serial.println(" microseconds");

    Serial.print("Prediction Output (first element): ");
    Serial.println(output_data[0], 6);

    delay(2000); // Wait 2 seconds before next benchmark
}}
"""
        with open(sketch_path, "w") as f:
            f.write(sketch)
        print(f"Successfully generated ready-to-flash Arduino Prototyping Sketch → {sketch_path}")
