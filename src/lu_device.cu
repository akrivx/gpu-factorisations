#include "lu.hpp"
#include "validate_shape.hpp"

#include <stdexcept>

namespace gpu_fact
{

void lu_solve_device(MatrixView<double> a, MatrixView<int> pivots, MatrixView<double> b)
{
	validate_lu_shapes(a, pivots, b);
	throw std::runtime_error("LU solve (device): device LU not yet implemented");
}

} // namespace gpu_fact
