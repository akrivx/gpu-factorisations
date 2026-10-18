#include "lu_reference.hpp"
#include <cmath>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>

void check(cudaError_t status)
{
	if (status != cudaSuccess)
		throw std::runtime_error(cudaGetErrorString(status));
}
void check(cublasStatus_t status)
{
	if (status != CUBLAS_STATUS_SUCCESS)
		throw std::runtime_error("cuBLAS error " + std::to_string(int(status)));
}
struct DeviceBuffer
{
	double* data = nullptr;
	explicit DeviceBuffer(std::size_t n)
	{
		check(cudaMalloc(&data, n * sizeof(double)));
	}
	~DeviceBuffer()
	{
		cudaFree(data);
	}
	DeviceBuffer(const DeviceBuffer&) = delete;
	DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};
struct BlasHandle
{
	cublasHandle_t handle;
	BlasHandle()
	{
		check(cublasCreate(&handle));
	}
	~BlasHandle()
	{
		cublasDestroy(handle);
	}
	BlasHandle(const BlasHandle&) = delete;
	BlasHandle& operator=(const BlasHandle&) = delete;
};

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
		const auto x = factorisations::lu_solve(3, 1, a, b);
		for (int i = 0; i < 3; ++i)
			if (!std::isfinite(x[i]) || std::abs(x[i] - (i + 1)) > 1e-12)
				throw std::runtime_error("Reference solution mismatch");
		// Two distinct RHS columns, stored next to each other in each row.
		// Expected X rows: [1,-1], [2,0], [3,2].
		const std::vector<double> multiple_b{7, 2, 3, 3, 5, -4};
		const std::vector<double> expected_multiple{1, -1, 2, 0, 3, 2};
		const auto multiple = factorisations::lu_solve(3, 2, a, multiple_b);
		for (int j = 0; j < 2; ++j)
			for (int i = 0; i < 3; ++i)
				if (!std::isfinite(multiple[i * 2 + j]) ||
				    std::abs(multiple[i * 2 + j] - expected_multiple[i * 2 + j]) > 1e-12)
					throw std::runtime_error("Multiple right-hand sides mismatch");
		if (a != std::vector<double>{0, 2, 1, 1, -2, 2, 2, 3, -1} ||
		    multiple_b != std::vector<double>{7, 2, 3, 3, 5, -4})
			throw std::runtime_error("Reference solve modified its inputs");
		bool invalid_rejected = false;
		try
		{
			factorisations::lu_solve(3, 1, {1}, b);
		}
		catch (const std::invalid_argument&)
		{
			invalid_rejected = true;
		}
		if (!invalid_rejected)
			throw std::runtime_error("Invalid storage was accepted");
		bool singular_rejected = false;
		try
		{
			factorisations::lu_solve(2, 1, {1, 2, 2, 4}, {1, 2});
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

		DeviceBuffer da(9), dx(3), db(3), residual(3);
		check(cudaMemcpy(da.data, a.data(), 9 * sizeof(double), cudaMemcpyHostToDevice));
		check(cudaMemcpy(dx.data, x.data(), 3 * sizeof(double), cudaMemcpyHostToDevice));
		check(cudaMemcpy(db.data, b.data(), 3 * sizeof(double), cudaMemcpyHostToDevice));
		BlasHandle blas;
		const double one = 1, zero = 0;
		// cuBLAS sees row-major A as column-major A^T; transpose it to compute A*x.
		check(cublasDgemv(blas.handle, CUBLAS_OP_T, 3, 3, &one, da.data, 3, dx.data, 1, &zero,
		                  residual.data, 1));
		subtract<<<1, 32>>>(residual.data, db.data, 3);
		check(cudaGetLastError());
		check(cudaDeviceSynchronize());
		double norm = 0;
		check(cublasDnrm2(blas.handle, 3, residual.data, 1, &norm));
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
