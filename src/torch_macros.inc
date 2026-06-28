#ifndef TYPETORCH_TORCH_MACROS
#define TYPETORCH_TORCH_MACROS
#pragma once

#define TYPETORCH_UNPAREN(...) __VA_ARGS__

#define TYPETORCH_SHAPE(dims) \
	::typetorch::Shape<TYPETORCH_UNPAREN dims>

#define TYPETORCH_TENSOR(dims) \
	TYPETORCH_TENSOR_LAYOUT(dims, ::typetorch::DType::F32, ::typetorch::Device::CPU, ::typetorch::Layout::Contiguous)

#define TYPETORCH_TENSOR_OPT(dims, dtype, device) \
	TYPETORCH_TENSOR_LAYOUT(dims, dtype, device, ::typetorch::Layout::Contiguous)

#define TYPETORCH_TENSOR_LAYOUT(dims, dtype, device, layout) \
	::typetorch::Tensor<TYPETORCH_SHAPE(dims), dtype, device, layout>

#define TYPETORCH_RETAIN(raw, dims) \
	TYPETORCH_TENSOR(dims)::unsafe_retain(raw)

#define TYPETORCH_RETAIN_OPT(raw, dims, dtype, device) \
	TYPETORCH_TENSOR_OPT(dims, dtype, device)::unsafe_retain(raw)

#define TYPETORCH_RETAIN_LAYOUT(raw, dims, dtype, device, layout) \
	TYPETORCH_TENSOR_LAYOUT(dims, dtype, device, layout)::unsafe_retain(raw)

#define TYPETORCH_FROM_TORCH_OP(torch_op, dims) \
	TYPETORCH_TENSOR(dims)::unsafe_retain(torch_op({TYPETORCH_UNPAREN dims}))


#define TYPETORCH_FROM_TORCH_OP_OPT(torch_op, dims, dtype, device)    \
	TYPETORCH_TENSOR_OPT(dims, dtype, device)::unsafe_retain(torch_op({TYPETORCH_UNPAREN dims}, ::typetorch::detail::container_options<dtype, device>()))

#define TYPETORCH_RANDN(dims) \
	TYPETORCH_FROM_TORCH_OP(::torch::randn, dims)

#define TYPETORCH_RANDN_OPT(dims, dtype, device) \
	TYPETORCH_FROM_TORCH_OP_OPT(::torch::randn, dims, dtype, device)

#define TYPETORCH_ONES(dims) \
	TYPETORCH_FROM_TORCH_OP(::torch::ones, dims)

#define TYPETORCH_ONES_OPT(dims, dtype, device) \
	TYPETORCH_FROM_TORCH_OP_OPT(::torch::ones, dims, dtype, device)

#define TYPETORCH_RAND(dims) \
	TYPETORCH_FROM_TORCH_OP(::torch::rand, dims)

#define TYPETORCH_RAND_OPT(dims, dtype, device) \
	TYPETORCH_FROM_TORCH_OP_OPT(::torch::rand, dims, dtype, device)

#endif
