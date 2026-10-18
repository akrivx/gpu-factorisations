#pragma once

#include "matrix_view.hpp"
#include <limits>
#include <stdexcept>

namespace gpu_fact
{

// Validates metadata only; allocation sizes and pointer address spaces are the caller's
// responsibility.
inline void validate_lu_shapes(MatrixView<double> a, MatrixView<int> pivots, MatrixView<double> b)
{
	if (!a.data || !pivots.data || !b.data)
		throw std::invalid_argument("LU solve: null data pointer");
	const auto max_int = static_cast<std::size_t>((std::numeric_limits<int>::max)());
	const auto valid_extent = [max_int](std::size_t value)
	{ return value > 0 && value <= max_int; };
	const auto valid_view = [&](const auto& view)
	{ return valid_extent(view.rows) && valid_extent(view.cols) && valid_extent(view.row_stride); };
	if (!valid_view(a) || !valid_view(pivots) || !valid_view(b))
		throw std::invalid_argument("LU solve: dimensions and strides must be in [1, INT_MAX]");
	if (a.cols != a.rows || b.rows != a.rows || pivots.rows != a.rows || pivots.cols != 1)
		throw std::invalid_argument("LU solve: expected square A, n-by-nrhs B, and n-by-1 pivots");
	if (a.row_stride < a.cols || b.row_stride < b.cols || pivots.row_stride != 1)
		throw std::invalid_argument("LU solve: invalid row stride (pivots must be contiguous)");
	const auto max_size = (std::numeric_limits<std::size_t>::max)() / sizeof(double);
	if (a.cols > max_size || b.cols > max_size || a.rows - 1 > (max_size - a.cols) / a.row_stride ||
	    b.rows - 1 > (max_size - b.cols) / b.row_stride)
		throw std::invalid_argument("LU solve: matrix storage size overflow");
}

} // namespace gpu_fact
