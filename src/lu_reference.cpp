#include "lu_reference.hpp"
#include <stdexcept>
#include <string>

// The pinned reference LAPACK uses the LP64 Fortran ABI (32-bit integers).
extern "C"
{
  void dgetrf_(const int *, const int *, double *, const int *, int *, int *);
  void dgetrs_(const char *, const int *, const int *, const double *, const int *, const int *,
               double *, const int *, int *, std::size_t);
}

std::vector<double> factorisations::lu_solve(int n, int nrhs, const std::vector<double> &a,
                                             const std::vector<double> &b)
{
  if (n <= 0 || nrhs <= 0 || a.size() != std::size_t(n) * n || b.size() != std::size_t(n) * nrhs)
    throw std::invalid_argument("LU solve: invalid dimensions or storage size");
  auto lu = a;
  auto x = b;
  std::vector<int> pivots(n);
  int info = 0;
  dgetrf_(&n, &n, lu.data(), &n, pivots.data(), &info);
  if (info > 0)
    throw std::runtime_error("LU solve: singular matrix");
  if (info < 0)
    throw std::runtime_error("DGETRF: invalid argument " + std::to_string(-info));
  const char trans = 'N';
  dgetrs_(&trans, &n, &nrhs, lu.data(), &n, pivots.data(), x.data(), &n, &info, 1);
  if (info != 0)
    throw std::runtime_error("DGETRS failed: " + std::to_string(info));
  return x;
}
