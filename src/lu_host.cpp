#include "lu.hpp"
#include "validate_shape.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace
{

int find_pivot(gpu_fact::MatrixView<double> a, int k)
{
	int pivot = k;
	for (int i = k + 1; i < static_cast<int>(a.rows); ++i)
		if (std::abs(a(i, k)) > std::abs(a(pivot, k)))
			pivot = i;
	return pivot;
}

void swap_rows(gpu_fact::MatrixView<double> a, int row1, int row2)
{
	for (std::size_t j = 0; j < a.cols; ++j)
		std::swap(a(row1, j), a(row2, j));
}

void compute_multipliers(gpu_fact::MatrixView<double> a, int k)
{
	const double pivot_value = a(k, k);
	if (pivot_value == 0.0)
		throw std::runtime_error("LU solve (host): singular matrix");
	for (int i = k + 1; i < static_cast<int>(a.rows); ++i)
		a(i, k) /= pivot_value;
}

void update_trailing_submatrix(gpu_fact::MatrixView<double> a, int k)
{
	for (int i = k + 1; i < static_cast<int>(a.rows); ++i)
		for (int j = k + 1; j < static_cast<int>(a.cols); ++j)
			a(i, j) -= a(i, k) * a(k, j);
}

void factorise_matrix(gpu_fact::MatrixView<double> a, gpu_fact::MatrixView<int> pivots)
{
	const int n = static_cast<int>(a.rows);
	for (int k = 0; k + 1 < n; ++k)
	{
		const int pivot = find_pivot(a, k);
		pivots(k, 0) = pivot + 1; // Store 1-based index
		if (pivot != k)
		{
			std::swap(pivots(k, 0), pivots(pivot, 0));
			swap_rows(a, k, pivot);
		}

		compute_multipliers(a, k);
		update_trailing_submatrix(a, k);
	}
}

void permute_rhs(const gpu_fact::MatrixView<const int> pivots, gpu_fact::MatrixView<double> b)
{
	for (std::size_t i = 0; i < pivots.rows; ++i)
	{
		const int pivot = pivots(i, 0) - 1; // Convert to 0-based index
		if (pivot != static_cast<int>(i))
			std::swap(b(i, 0), b(pivot, 0));
	}
}

void forward_substitution(const gpu_fact::MatrixView<const double> a,
                          const gpu_fact::MatrixView<const int> pivots,
                          gpu_fact::MatrixView<double> b)
{
	const int n = static_cast<int>(a.rows);
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < i; ++j)
			b(i, 0) -= a(i, j) * b(j, 0);
}

void backward_substitution(const gpu_fact::MatrixView<const double> a,
                           gpu_fact::MatrixView<double> b)
{
    const int n = static_cast<int>(a.rows);
    for (int i = n - 1; i >= 0; --i)
    {
        for (int j = i + 1; j < n; ++j)
            b(i, 0) -= a(i, j) * b(j, 0);
        if (a(i, i) == 0.0)
            throw std::runtime_error("LU solve (host): singular matrix");
        b(i, 0) /= a(i, i);
    }
}

} // namespace

namespace gpu_fact
{

void lu_solve_host(MatrixView<double> a, MatrixView<int> pivots, MatrixView<double> b)
{
	validate_lu_shapes(a, pivots, b);
	factorise_matrix(a, pivots);
	permute_rhs(pivots.as_const(), b);
    forward_substitution(a.as_const(), pivots.as_const(), b);
    backward_substitution(a.as_const(), b);
}

} // namespace gpu_fact
