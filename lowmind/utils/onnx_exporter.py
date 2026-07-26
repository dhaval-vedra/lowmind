"""
LowMind ONNX Exporter — production-grade ONNX model export for edge deployment
"""
import numpy as np
from ..core.tensor import Tensor
from ..core.module import Module


def export_to_onnx(model, dummy_input, filepath="model.onnx"):
    try:
        import onnx
        from onnx import helper, TensorProto
    except ImportError:
        raise ImportError("The 'onnx' package is required to use the ONNX model exporter. "
                          "Please install it using: pip install onnx")
    """
    Exports a LowMind model to standard ONNX format.

    This enables you to run models trained in LowMind directly on PyTorch,
    TensorFlow, ONNX Runtime, or any ONNX-supported edge hardware accelerator!

    Args:
        model:        A LowMind `Module` or `Sequential` instance.
        dummy_input:  A dummy input Tensor or numpy array (matches expected input shape).
        filepath:     Destination filepath for the exported ONNX model.
    """
    if isinstance(dummy_input, Tensor):
        input_data = dummy_input.numpy()
    else:
        input_data = np.array(dummy_input, dtype=np.float32)

    input_shape = list(input_data.shape)

    # 1. Define ONNX Graph Inputs & Outputs
    # Input info
    onnx_inputs = [
        helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
    ]

    onnx_nodes = []
    onnx_initializers = []

    current_input_name = "input"
    node_counter = 0

    # Extract modules/layers in order
    modules_list = []
    if hasattr(model, '_modules') and len(model._modules) > 0:
        modules_list = list(model._modules.values())
    else:
        modules_list = [model]

    # Helper to generate unique names
    def get_unique_name(prefix):
        nonlocal node_counter
        node_counter += 1
        return f"{prefix}_{node_counter}"

    for idx, layer in enumerate(modules_list):
        layer_type = type(layer).__name__
        output_name = get_unique_name("out")

        if layer_type == "Linear":
            # Map Linear to Gemm (General Matrix Multiplication)
            # Y = alpha * A * B' + beta * C
            w_name = get_unique_name("w")
            b_name = get_unique_name("b")

            # Weight initializer (ONNX expects weights of shape [out_features, in_features] for Gemm with transB=1)
            weight_proto = helper.make_tensor(
                name=w_name,
                data_type=TensorProto.FLOAT,
                dims=list(layer.weight.shape),
                vals=layer.weight.numpy().flatten().tolist()
            )
            onnx_initializers.append(weight_proto)

            # Bias initializer
            if hasattr(layer, 'bias') and layer.bias is not None:
                bias_proto = helper.make_tensor(
                    name=b_name,
                    data_type=TensorProto.FLOAT,
                    dims=list(layer.bias.shape),
                    vals=layer.bias.numpy().flatten().tolist()
                )
                onnx_initializers.append(bias_proto)
                inputs_list = [current_input_name, w_name, b_name]
            else:
                inputs_list = [current_input_name, w_name]

            # Create Gemm node
            node = helper.make_node(
                op_type="Gemm",
                inputs=inputs_list,
                outputs=[output_name],
                name=get_unique_name("Gemm"),
                alpha=1.0,
                beta=1.0,
                transB=1  # Transpose weights to match Linear weight shape [out_features, in_features]
            )
            onnx_nodes.append(node)
            current_input_name = output_name

        elif layer_type == "Conv2d":
            # Map Conv2d to Conv
            w_name = get_unique_name("w")
            b_name = get_unique_name("b")

            # Weight initializer [out_channels, in_channels, kH, kW]
            weight_proto = helper.make_tensor(
                name=w_name,
                data_type=TensorProto.FLOAT,
                dims=list(layer.weight.shape),
                vals=layer.weight.numpy().flatten().tolist()
            )
            onnx_initializers.append(weight_proto)

            # Bias initializer
            if hasattr(layer, 'bias') and layer.bias is not None:
                bias_proto = helper.make_tensor(
                    name=b_name,
                    data_type=TensorProto.FLOAT,
                    dims=list(layer.bias.shape),
                    vals=layer.bias.numpy().flatten().tolist()
                )
                onnx_initializers.append(bias_proto)
                inputs_list = [current_input_name, w_name, b_name]
            else:
                inputs_list = [current_input_name, w_name]

            # Create Conv node
            # Pad is [pH_start, pW_start, pH_end, pW_end]
            pH, pW = layer.padding
            pads = [pH, pW, pH, pW]

            node = helper.make_node(
                op_type="Conv",
                inputs=inputs_list,
                outputs=[output_name],
                name=get_unique_name("Conv"),
                kernel_shape=list(layer.kernel_size),
                strides=list(layer.stride),
                pads=pads,
                dilations=[1, 1]
            )
            onnx_nodes.append(node)
            current_input_name = output_name

        elif layer_type == "ReLU":
            node = helper.make_node(
                op_type="Relu",
                inputs=[current_input_name],
                outputs=[output_name],
                name=get_unique_name("Relu")
            )
            onnx_nodes.append(node)
            current_input_name = output_name

        elif layer_type == "Sigmoid":
            node = helper.make_node(
                op_type="Sigmoid",
                inputs=[current_input_name],
                outputs=[output_name],
                name=get_unique_name("Sigmoid")
            )
            onnx_nodes.append(node)
            current_input_name = output_name

        elif layer_type == "Tanh":
            node = helper.make_node(
                op_type="Tanh",
                inputs=[current_input_name],
                outputs=[output_name],
                name=get_unique_name("Tanh")
            )
            onnx_nodes.append(node)
            current_input_name = output_name

        elif layer_type == "Softmax":
            node = helper.make_node(
                op_type="Softmax",
                inputs=[current_input_name],
                outputs=[output_name],
                name=get_unique_name("Softmax"),
                axis=-1
            )
            onnx_nodes.append(node)
            current_input_name = output_name

        elif layer_type == "Flatten":
            node = helper.make_node(
                op_type="Flatten",
                inputs=[current_input_name],
                outputs=[output_name],
                name=get_unique_name("Flatten"),
                axis=layer.start_dim
            )
            onnx_nodes.append(node)
            current_input_name = output_name

        elif layer_type in ("MaxPool2d", "AvgPool2d"):
            op_type = "MaxPool" if layer_type == "MaxPool2d" else "AveragePool"
            pH, pW = layer.padding
            pads = [pH, pW, pH, pW]

            node = helper.make_node(
                op_type=op_type,
                inputs=[current_input_name],
                outputs=[output_name],
                name=get_unique_name(op_type),
                kernel_shape=list(layer.kernel_size),
                strides=list(layer.stride),
                pads=pads
            )
            onnx_nodes.append(node)
            current_input_name = output_name

        elif layer_type in ("BatchNorm1d", "BatchNorm2d"):
            # Map BatchNorm to BatchNormalization
            scale_name = get_unique_name("scale")
            bias_name = get_unique_name("bias")
            mean_name = get_unique_name("mean")
            var_name = get_unique_name("var")

            C = layer.num_features

            scale_proto = helper.make_tensor(
                name=scale_name,
                data_type=TensorProto.FLOAT,
                dims=[C],
                vals=layer.gamma.numpy().flatten().tolist() if layer.affine else [1.0] * C
            )
            onnx_initializers.append(scale_proto)

            bias_proto = helper.make_tensor(
                name=bias_name,
                data_type=TensorProto.FLOAT,
                dims=[C],
                vals=layer.beta.numpy().flatten().tolist() if layer.affine else [0.0] * C
            )
            onnx_initializers.append(bias_proto)

            mean_proto = helper.make_tensor(
                name=mean_name,
                data_type=TensorProto.FLOAT,
                dims=[C],
                vals=layer.running_mean.flatten().tolist()
            )
            onnx_initializers.append(mean_proto)

            var_proto = helper.make_tensor(
                name=var_name,
                data_type=TensorProto.FLOAT,
                dims=[C],
                vals=layer.running_var.flatten().tolist()
            )
            onnx_initializers.append(var_proto)

            node = helper.make_node(
                op_type="BatchNormalization",
                inputs=[current_input_name, scale_name, bias_name, mean_name, var_name],
                outputs=[output_name],
                name=get_unique_name("BatchNormalization"),
                epsilon=layer.eps,
                momentum=1.0 - layer.momentum
            )
            onnx_nodes.append(node)
            current_input_name = output_name

        else:
            # Skip or pass-through for unknown/custom layers
            pass

    # Trace output shape using a test forward pass of the model
    with onnx_no_grad_ctx():
        out_lm = model(Tensor(input_data))
    output_shape = list(out_lm.shape)

    # 2. Define ONNX Graph Outputs
    onnx_outputs = [
        helper.make_tensor_value_info(current_input_name, TensorProto.FLOAT, output_shape)
    ]

    # 3. Assemble ONNX Graph & Model
    graph_proto = helper.make_graph(
        nodes=onnx_nodes,
        name="lowmind_exported_graph",
        inputs=onnx_inputs,
        outputs=onnx_outputs,
        initializer=onnx_initializers
    )

    # Create the model protocol buffer
    onnx_model = helper.make_model(graph_proto, producer_name="lowmind")

    # Verify the model
    onnx.checker.check_model(onnx_model)

    # Write to file
    with open(filepath, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"ONNX Model successfully exported and verified → {filepath}")
    return onnx_model


class onnx_no_grad_ctx:
    """Helper context manager to disable grad locally if needed."""
    def __enter__(self):
        try:
            from ..core.no_grad import no_grad
            self.ctx = no_grad()
            self.ctx.__enter__()
        except ImportError:
            self.ctx = None

    def __exit__(self, *args):
        if self.ctx is not None:
            self.ctx.__exit__(*args)
