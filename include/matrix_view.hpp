#pragma once

#include <cstddef>
#include <type_traits>

#if defined(__CUDACC__)
#define GPU_FACT_MATRIX_VIEW_HOST_DEVICE __host__ __device__
#else
#define GPU_FACT_MATRIX_VIEW_HOST_DEVICE
#endif

namespace gpu_fact
{

// Non-owning row-major view. row_stride counts elements, including any row padding.
// Storage must cover (rows-1)*row_stride+cols elements for a nonempty view.
template <typename T> struct MatrixView
{
	T* data;
	std::size_t rows;
	std::size_t cols;
	std::size_t row_stride;

	GPU_FACT_MATRIX_VIEW_HOST_DEVICE const T& operator()(std::size_t row, std::size_t col) const
	{
		return data[row * row_stride + col];
	}

	GPU_FACT_MATRIX_VIEW_HOST_DEVICE T& operator()(std::size_t row, std::size_t col)
	{
		return data[row * row_stride + col];
	}

	GPU_FACT_MATRIX_VIEW_HOST_DEVICE MatrixView<const std::remove_const_t<T>> as_const() const
	{
		return MatrixView<const std::remove_const_t<T>>{data, rows, cols, row_stride};
	}
};

} // namespace gpu_fact

#undef GPU_FACT_MATRIX_VIEW_HOST_DEVICE
