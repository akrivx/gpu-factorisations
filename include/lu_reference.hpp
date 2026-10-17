#pragma once
#include <vector>

namespace factorisations
{
// Double precision, column-major A (n*n) and B (n*nrhs).
// Inputs are copied; partial pivoting is supplied by LAPACK DGETRF/DGETRS.
std::vector<double> lu_solve(int n, int nrhs, const std::vector<double> &a,
                             const std::vector<double> &b);
} // namespace factorisations
