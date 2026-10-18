#include "device_buffer.hpp"
#include "lu.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

void test_case(std::vector<double> a, std::size_t n, bool device, bool expect_singular)
{
	std::vector<double> b(n, 1);
	std::vector<int> pivots(n, -123);
	auto reference_a = a;
	auto reference_b = b;
	std::vector<int> reference_pivots(n);
	if (!expect_singular)
		gpu_fact::lu_solve_reference({reference_a.data(), n, n, n},
		                             {reference_pivots.data(), n, 1, 1},
		                             {reference_b.data(), n, 1, 1});
	bool singular = false;
	try
	{
		if (device)
		{
			gpu_fact::DeviceBuffer<double> da(a.size()), db(n);
			gpu_fact::DeviceBuffer<int> dp(n);
			GPU_FACT_CUDA_CHECK(
			    cudaMemcpy(da.data(), a.data(), a.size() * sizeof(double), cudaMemcpyHostToDevice));
			GPU_FACT_CUDA_CHECK(
			    cudaMemcpy(db.data(), b.data(), n * sizeof(double), cudaMemcpyHostToDevice));
			GPU_FACT_CUDA_CHECK(
			    cudaMemcpy(dp.data(), pivots.data(), n * sizeof(int), cudaMemcpyHostToDevice));
			gpu_fact::lu_solve_device({da.data(), n, n, n}, {dp.data(), n, 1, 1},
			                          {db.data(), n, 1, 1});
			GPU_FACT_CUDA_CHECK(
			    cudaMemcpy(pivots.data(), dp.data(), n * sizeof(int), cudaMemcpyDeviceToHost));
			GPU_FACT_CUDA_CHECK(
			    cudaMemcpy(b.data(), db.data(), n * sizeof(double), cudaMemcpyDeviceToHost));
		}
		else
			gpu_fact::lu_solve_host({a.data(), n, n, n}, {pivots.data(), n, 1, 1},
			                        {b.data(), n, 1, 1});
	}
	catch (const std::runtime_error& error)
	{
		const std::string expected =
		    device ? "LU solve (device): singular matrix" : "LU solve (host): singular matrix";
		if (error.what() != expected)
			throw;
		singular = true;
	}
	if (singular != expect_singular)
		throw std::runtime_error("Unexpected singularity result");
	if (!singular)
	{
		for (std::size_t i = 0; i < n; ++i)
			if (!std::isfinite(b[i]) || std::abs(b[i] - reference_b[i]) > 1e-12)
				throw std::runtime_error("Solution differs from LAPACKE");
		if (pivots.back() != static_cast<int>(n))
			throw std::runtime_error("Final pivot was not initialized");
		for (std::size_t i = 0; i < n; ++i)
			if (pivots[i] < static_cast<int>(i + 1) || pivots[i] > static_cast<int>(n))
				throw std::runtime_error("Invalid pivot history");
	}
}

void test_padded_device_solve()
{
	std::vector<double> a{0, 2, 1, 99, 99, 1, -2, 2, 99, 99, 2, 3, -1, 99, 99};
	std::vector<double> b{7, 2, 99, 3, 3, 99, 5, -4, 99};
	auto expected_a = a;
	auto expected_b = b;
	std::vector<int> pivots(3);
	gpu_fact::lu_solve_reference({expected_a.data(), 3, 3, 5}, {pivots.data(), 3, 1, 1},
	                             {expected_b.data(), 3, 2, 3});
	gpu_fact::DeviceBuffer<double> da(a.size()), db(b.size());
	gpu_fact::DeviceBuffer<int> dp(3);
	GPU_FACT_CUDA_CHECK(
	    cudaMemcpy(da.data(), a.data(), a.size() * sizeof(double), cudaMemcpyHostToDevice));
	GPU_FACT_CUDA_CHECK(
	    cudaMemcpy(db.data(), b.data(), b.size() * sizeof(double), cudaMemcpyHostToDevice));
	gpu_fact::lu_solve_device({da.data(), 3, 3, 5}, {dp.data(), 3, 1, 1}, {db.data(), 3, 2, 3});
	GPU_FACT_CUDA_CHECK(
	    cudaMemcpy(a.data(), da.data(), a.size() * sizeof(double), cudaMemcpyDeviceToHost));
	GPU_FACT_CUDA_CHECK(
	    cudaMemcpy(b.data(), db.data(), b.size() * sizeof(double), cudaMemcpyDeviceToHost));
	for (std::size_t i = 0; i < b.size(); ++i)
		if (!std::isfinite(b[i]) || std::abs(b[i] - expected_b[i]) > 1e-12)
			throw std::runtime_error("Padded multi-RHS device solution mismatch");
	for (std::size_t i = 0; i < 3; ++i)
		if (a[i * 5 + 3] != 99 || a[i * 5 + 4] != 99)
			throw std::runtime_error("Device solve overwrote matrix padding");
}

int main(int argc, char** argv)
{
	try
	{
		const bool cpu_only = argc == 2 && std::string(argv[1]) == "--cpu-only";
		if (argc > 1 && !cpu_only)
			throw std::invalid_argument("Usage: lu_factorisation_test [--cpu-only]");
		for (bool device : {false, true})
		{
			if (device && cpu_only)
				continue;
			test_case({2}, 1, device, false);
			test_case({0}, 1, device, true);
			test_case({0, 1, 2, 3}, 2, device, false);
			test_case({1, 2, 2, 4}, 2, device, true); // Zero final diagonal after elimination.
			test_case({0, 1, 0, 2}, 2, device, true); // Zero earlier pivot, nonzero final diagonal.
		}
		if (!cpu_only)
			test_padded_device_solve();
		return 0;
	}
	catch (const std::exception& error)
	{
		std::cerr << error.what() << '\n';
		return 1;
	}
}
