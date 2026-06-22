#ifndef TYPETORCH_TYPE_MACROS

#define TYPETORCH_TYPE_MACROS

#pragma once


#define BROADCAST_COMPATIBLE(LShape, RShape) \
	(::typetorch::detail::broadcast_compatible(LShape::dims, RShape::dims))

#define BROADCAST_SHAPE_T(LShape, RShape)                 \
	typename[: ::typetorch::detail::broadcast_shape_meta( \
				   LShape::dims, RShape::dims):]

#define MATMUL_SHAPE_T(LShape, RShape)                 \
	typename[: ::typetorch::detail::matmul_shape_meta( \
				   LShape::dims, RShape::dims):]

#define TRANSPOSE_SHAPE_T(ShapeKind, Dim0, Dim1)          \
	typename[: ::typetorch::detail::transpose_shape_meta( \
				   ShapeKind::dims, Dim0, Dim1):]

#define TRANSPOSE_LAYOUT(ShapeKind, LayoutKind, Dim0, Dim1) \
	(::typetorch::detail::transpose_layout(LayoutKind, ShapeKind::dims, Dim0, Dim1))

#define PERMUTE_SHAPE_T(ShapeKind, PermuteDims)         \
	typename[: ::typetorch::detail::permute_shape_meta( \
				   ShapeKind::dims, PermuteDims):]

#define PERMUTE_LAYOUT(ShapeKind, LayoutKind, PermuteDims) \
	(::typetorch::detail::permute_layout(LayoutKind, ShapeKind::dims, PermuteDims))

#define VIEW_SHAPE_T(ShapeKind, ResultDims)          \
	typename[: ::typetorch::detail::view_shape_meta( \
				   ShapeKind::dims, ResultDims):]

#define FLATTEN_SHAPE_T(ShapeKind, StartDim, EndDim)    \
	typename[: ::typetorch::detail::flatten_shape_meta( \
				   ShapeKind::dims, StartDim, EndDim):]

#define SQUEEZE_SHAPE_T(ShapeKind)                      \
	typename[: ::typetorch::detail::squeeze_shape_meta( \
				   ShapeKind::dims):]

#define SQUEEZE_DIM_SHAPE_T(ShapeKind, Dim)             \
	typename[: ::typetorch::detail::squeeze_shape_meta( \
				   ShapeKind::dims, Dim):]

#define UNSQUEEZE_SHAPE_T(ShapeKind, Dim)                 \
	typename[: ::typetorch::detail::unsqueeze_shape_meta( \
				   ShapeKind::dims, Dim):]

#define NORMALIZED_SHAPE_COMPATIBLE(InputShape, NormalizedShape) \
	(::typetorch::detail::normalized_shape_compatible(           \
		InputShape::dims, NormalizedShape::dims))

#endif