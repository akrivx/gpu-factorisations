#pragma once
#include <vector>

namespace factorisations
{
// Double precision, contiguous row-major A (n*n) and B (n*nrhs).
// A[i*n+j], B[i*nrhs+j]; the returned X uses X[i*nrhs+j].
// Inputs are copied; partial pivoting is supplied by LAPACKE dgesv.
std::vector<double> lu_solve(int n, int nrhs, const std::vector<double> &a,
                             const std::vector<double> &b);
} // namespace factorisations
