#include "lu.hpp"
#include "validate_shape.hpp"

#include <stdexcept>
#include <string>

#include <lapacke.h>

namespace gpu_fact
{

void lu_solve_reference(MatrixView<double> a, MatrixView<int> pivots, MatrixView<double> b)
{
	static_assert(sizeof(lapack_int) == sizeof(int), "The reference solver requires LP64 LAPACKE");
	validate_lu_shapes(a, pivots, b);
	const lapack_int info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, static_cast<lapack_int>(a.rows),
	                                      static_cast<lapack_int>(b.cols), a.data,
	                                      static_cast<lapack_int>(a.row_stride), pivots.data,
	                                      b.data, static_cast<lapack_int>(b.row_stride));
	if (info == LAPACK_WORK_MEMORY_ERROR || info == LAPACK_TRANSPOSE_MEMORY_ERROR)
		throw std::runtime_error("LU solve (reference): LAPACKE_dgesv workspace allocation failed");
	if (info > 0)
		throw std::runtime_error("LU solve (reference): singular matrix");
	if (info < 0)
		throw std::runtime_error("LU solve (reference): invalid argument " + std::to_string(-info));
}

} // namespace gpu_fact
