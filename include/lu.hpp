#pragma once
#include "matrix_view.hpp"

namespace gpu_fact
{

// Row-major A (n*n), b (n*nrhs), and contiguous pivots (n*1, row_stride=1).
// Views do not own storage; the caller supplies valid, non-overlapping buffers.
// Overwrites A with LU factors and b with x. Pivots are 1-based LAPACK row indices.
void lu_solve_reference(MatrixView<double> a, MatrixView<int> pivots, MatrixView<double> b);
void lu_solve_host(MatrixView<double> a, MatrixView<int> pivots, MatrixView<double> b);
// All view pointers refer to device storage; shape metadata remains on the host.
void lu_solve_device(MatrixView<double> a, MatrixView<int> pivots, MatrixView<double> b);

} // namespace gpu_fact
