typetorch_module_files = {
    "src/typetorch/types.mpp",
    "src/typetorch/torch_mappings.mpp",
    "src/typetorch/shape_meta.mpp",
    "src/typetorch/common.mpp",
    "src/typetorch/runtime_checks.mpp",
    "src/typetorch/tensor/core.mpp",
    "src/typetorch/tensor/factory.mpp",
    "src/typetorch/tensor/arithmetic.mpp",
    "src/typetorch/tensor/view.mpp",
    "src/typetorch/tensor/nn.mpp",
    "src/typetorch/tensor/tensor.mpp",
    "src/typetorch/nnModules/core.mpp",
    "src/typetorch/nnModules/linear.mpp",
    "src/typetorch/nnModules/conv2d.mpp",
    "src/typetorch/nnModules/embedding.mpp",
    "src/typetorch/nnModules/layer_norm.mpp",
    "src/typetorch/nnModules/rms_norm.mpp",
    "src/typetorch/nnModules/sequential.mpp",
    "src/typetorch/nnModules/mlp.mpp",
    "src/typetorch/nnModules/activation.mpp",
    "src/typetorch/nnModules/transformer.mpp",
    "src/typetorch/nnModules/flatten.mpp",
    "src/typetorch/nnModules/pooling.mpp",
    "src/typetorch/nnModules/nnModules.mpp",
    "src/typetorch/typetorch.mpp"
}

libtorch_module_files = {"src/libtorch.mpp"}

fast_io_module_files = {"src/fast_io.mpp", "src/fast_io.cpp"}