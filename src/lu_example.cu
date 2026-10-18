#include "blas_handle.hpp"
#include "device_buffer.hpp"
#include "lu.hpp"
#include <cmath>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// A tiny kernel gives new experiments a working CUDA compilation example.
__global__ void subtract(double* x, const double* y, int n)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
		x[i] -= y[i];
}

int main(int argc, char** argv)
{
	try
	{
		const bool cpu_only = argc == 2 && std::string(argv[1]) == "--cpu-only";
		if (argc > 1 && !cpu_only)
			throw std::invalid_argument("Usage: lu_example [--cpu-only]");
		// Requires a pivot in column one. A*[1,2,3]^T = [7,3,5]^T.
		const std::vector<double> a{0, 2, 1, 1, -2, 2, 2, 3, -1};
		const std::vector<double> b{7, 3, 5};
		auto lu = a;
		auto x = b;
		std::vector<int> pivots(3);
		gpu_fact::lu_solve_reference({lu.data(), 3, 3, 3}, {pivots.data(), 3, 1, 1},
		                             {x.data(), 3, 1, 1});
		for (int i = 0; i < 3; ++i)
		{
			if (!std::isfinite(x[i]) || std::abs(x[i] - (i + 1)) > 1e-12)
				throw std::runtime_error("Reference solution mismatch");
			if (pivots[i] < i + 1 || pivots[i] > 3)
				throw std::runtime_error("Invalid pivot index");
		}
		if (pivots[0] == 1 || lu == a)
			throw std::runtime_error("Expected in-place factorisation with row pivoting");
		auto second_lu = a;
		std::vector<double> second_b{2, 3, -4};
		const std::vector<double> expected{-1, 0, 2};
		gpu_fact::lu_solve_reference({second_lu.data(), 3, 3, 3}, {pivots.data(), 3, 1, 1},
		                             {second_b.data(), 3, 1, 1});
		for (int i = 0; i < 3; ++i)
			if (!std::isfinite(second_b[i]) || std::abs(second_b[i] - expected[i]) > 1e-12)
				throw std::runtime_error("Second solution mismatch");
		// Padded rows and two RHS columns exercise the view strides and inferred shapes.
		std::vector<double> padded_a{0, 2, 1, 99, 1, -2, 2, 99, 2, 3, -1, 99};
		std::vector<double> padded_b{7, 2, 99, 3, 3, 99, 5, -4, 99};
		gpu_fact::lu_solve_reference({padded_a.data(), 3, 3, 4}, {pivots.data(), 3, 1, 1},
		                             {padded_b.data(), 3, 2, 3});
		for (int i = 0; i < 3; ++i)
		{
			if (!std::isfinite(padded_b[i * 3]) || !std::isfinite(padded_b[i * 3 + 1]) ||
			    std::abs(padded_b[i * 3] - (i + 1)) > 1e-12 ||
			    std::abs(padded_b[i * 3 + 1] - expected[i]) > 1e-12 || padded_b[i * 3 + 2] != 99 ||
			    padded_a[i * 4 + 3] != 99)
				throw std::runtime_error("Padded multi-RHS solve mismatch");
		}
		// Invalid LP64 extents must be rejected by the solver before accessing storage.
		const auto int_max = static_cast<std::size_t>((std::numeric_limits<int>::max)());
		for (auto invalid_extent : {std::size_t{0}, int_max + 1})
		{
			for (int field = 0; field < 9; ++field)
			{
				gpu_fact::MatrixView<double> av{lu.data(), 3, 3, 3};
				gpu_fact::MatrixView<int> pv{pivots.data(), 3, 1, 1};
				gpu_fact::MatrixView<double> bv{x.data(), 3, 1, 1};
				std::size_t* fields[]{&av.rows, &av.cols, &av.row_stride,
				                      &pv.rows, &pv.cols, &pv.row_stride,
				                      &bv.rows, &bv.cols, &bv.row_stride};
				*fields[field] = invalid_extent;
				bool rejected = false;
				try
				{
					gpu_fact::lu_solve_reference(av, pv, bv);
				}
				catch (const std::invalid_argument&)
				{
					rejected = true;
				}
				if (!rejected)
					throw std::runtime_error("Invalid LP64 extent was accepted");
			}
		}
		auto expect_invalid = [&](gpu_fact::MatrixView<double> av, gpu_fact::MatrixView<int> pv,
		                          gpu_fact::MatrixView<double> bv)
		{
			try
			{
				gpu_fact::lu_solve_reference(av, pv, bv);
			}
			catch (const std::invalid_argument&)
			{
				return;
			}
			throw std::runtime_error("Invalid LU view was accepted");
		};
		expect_invalid({lu.data(), 3, 2, 3}, {pivots.data(), 3, 1, 1}, {x.data(), 3, 1, 1});
		expect_invalid({lu.data(), 3, 3, 2}, {pivots.data(), 3, 1, 1}, {x.data(), 3, 1, 1});
		expect_invalid({lu.data(), 3, 3, 3}, {pivots.data(), 2, 1, 1}, {x.data(), 3, 1, 1});
		expect_invalid({lu.data(), 3, 3, 3}, {pivots.data(), 3, 1, 2}, {x.data(), 3, 1, 1});
		expect_invalid({lu.data(), 3, 3, 3}, {pivots.data(), 3, 1, 1}, {x.data(), 2, 1, 1});
		expect_invalid({lu.data(), 3, 3, 3}, {pivots.data(), 3, 1, 1}, {x.data(), 3, 0, 1});
		expect_invalid({lu.data(), 3, 3, 3}, {pivots.data(), 3, 1, 1}, {x.data(), 3, 1, 0});
		expect_invalid({lu.data(), 3, 3, static_cast<std::size_t>(1) << 31},
		               {pivots.data(), 3, 1, 1}, {x.data(), 3, 1, 1});
		bool invalid_rejected = false;
		try
		{
			gpu_fact::lu_solve_reference({lu.data(), 0, 0, 0}, {pivots.data(), 3, 1, 1},
			                             {x.data(), 3, 1, 1});
		}
		catch (const std::invalid_argument&)
		{
			invalid_rejected = true;
		}
		if (!invalid_rejected)
			throw std::runtime_error("Invalid dimension was accepted");
		bool null_rejected = false;
		try
		{
			gpu_fact::lu_solve_reference({nullptr, 3, 3, 3}, {pivots.data(), 3, 1, 1},
			                             {x.data(), 3, 1, 1});
		}
		catch (const std::invalid_argument&)
		{
			null_rejected = true;
		}
		if (!null_rejected)
			throw std::runtime_error("Null matrix was accepted");
		bool singular_rejected = false;
		try
		{
			std::vector<double> singular_a{1, 2, 2, 4};
			std::vector<double> singular_b{1, 2};
			std::vector<int> singular_pivots(2);
			gpu_fact::lu_solve_reference({singular_a.data(), 2, 2, 2},
			                             {singular_pivots.data(), 2, 1, 1},
			                             {singular_b.data(), 2, 1, 1});
		}
		catch (const std::runtime_error&)
		{
			singular_rejected = true;
		}
		if (!singular_rejected)
			throw std::runtime_error("Singular matrix was accepted");
		std::cout << "LAPACK solution: " << x[0] << ' ' << x[1] << ' ' << x[2] << '\n';
		if (cpu_only)
			return 0;

		gpu_fact::DeviceBuffer<double> da(9), dx(3), db(3), residual(3);
		gpu_fact::cuda_check(
		    cudaMemcpy(da.data(), a.data(), 9 * sizeof(double), cudaMemcpyHostToDevice));
		gpu_fact::cuda_check(
		    cudaMemcpy(dx.data(), x.data(), 3 * sizeof(double), cudaMemcpyHostToDevice));
		gpu_fact::cuda_check(
		    cudaMemcpy(db.data(), b.data(), 3 * sizeof(double), cudaMemcpyHostToDevice));
		gpu_fact::BlasHandle blas;
		const double one = 1, zero = 0;
		// cuBLAS sees row-major A as column-major A^T; transpose it to compute A*x.
		gpu_fact::cublas_check(cublasDgemv(blas.get(), CUBLAS_OP_T, 3, 3, &one, da.data(), 3,
		                                   dx.data(), 1, &zero, residual.data(), 1));
		subtract<<<1, 32>>>(residual.data(), db.data(), 3);
		gpu_fact::cuda_check(cudaGetLastError());
		gpu_fact::cuda_check(cudaDeviceSynchronize());
		double norm = 0;
		gpu_fact::cublas_check(cublasDnrm2(blas.get(), 3, residual.data(), 1, &norm));
		std::cout << "cuBLAS residual ||Ax-b||_2: " << norm << '\n';
		if (!std::isfinite(norm) || norm > 1e-12)
			throw std::runtime_error("Residual too large");
		return 0;
	}
	catch (const std::exception& error)
	{
		std::cerr << error.what() << '\n';
		return 1;
	}
}
