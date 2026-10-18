#pragma once

#include <cstddef>

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
};

} // namespace gpu_fact
