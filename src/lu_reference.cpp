#include "lu_reference.hpp"
#include <lapacke.h>
#include <stdexcept>
#include <string>

std::vector<double> factorisations::lu_solve(int n, int nrhs, const std::vector<double> &a,
                                             const std::vector<double> &b)
{
  if (n <= 0 || nrhs <= 0 || a.size() != std::size_t(n) * n || b.size() != std::size_t(n) * nrhs)
    throw std::invalid_argument("LU solve: invalid dimensions or storage size");
  auto lu = a;
  auto x = b;
  std::vector<lapack_int> pivots(n);
  const lapack_int info =
      LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs, lu.data(), n, pivots.data(), x.data(), nrhs);
  if (info == LAPACK_WORK_MEMORY_ERROR || info == LAPACK_TRANSPOSE_MEMORY_ERROR)
    throw std::runtime_error("LAPACKE_dgesv: workspace allocation failed");
  if (info > 0)
    throw std::runtime_error("LU solve: singular matrix");
  if (info < 0)
    throw std::runtime_error("LAPACKE_dgesv: invalid argument " + std::to_string(-info));
  return x;
}
