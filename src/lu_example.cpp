#include "device_buffer.hpp"
#include "lu.hpp"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
struct Options
{
	std::size_t n = 128;
	std::uint32_t seed = 42;
	double tolerance = 1e-10;
	bool cpu_only = false;
	bool help = false;
};

Options parse_options(int argc, char** argv)
{
	Options options;
	for (int i = 1; i < argc; ++i)
	{
		const std::string arg = argv[i];
		if (arg == "--cpu-only")
			options.cpu_only = true;
		else if (arg == "--help" || arg == "-h")
			options.help = true;
		else if (arg == "--size" && i + 1 < argc)
		{
			const std::string value = argv[++i];
			const auto parsed =
			    std::from_chars(value.data(), value.data() + value.size(), options.n);
			if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
			    options.n == 0 ||
			    options.n > static_cast<std::size_t>((std::numeric_limits<int>::max)()))
				throw std::invalid_argument("--size must be an integer in [1, INT_MAX]");
		}
		else if (arg == "--seed" && i + 1 < argc)
		{
			const std::string value = argv[++i];
			const auto parsed =
			    std::from_chars(value.data(), value.data() + value.size(), options.seed);
			if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
				throw std::invalid_argument("--seed must be an integer in [0, 4294967295]");
		}
		else if (arg == "--tolerance" && i + 1 < argc)
		{
			const std::string value = argv[++i];
			const auto parsed =
			    std::from_chars(value.data(), value.data() + value.size(), options.tolerance);
			if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
			    !std::isfinite(options.tolerance) || options.tolerance <= 0)
				throw std::invalid_argument("--tolerance must be a finite positive number");
		}
		else
			throw std::invalid_argument(
			    "Usage: lu_example [--size N] [--seed S] [--tolerance T] [--cpu-only] [--help]");
	}
	return options;
}

// Returns false for numerical failures as well as non-finite outputs.
bool compare(const char* name, std::size_t n, const std::vector<double>& a,
             const std::vector<double>& b, const std::vector<double>& x,
             const std::vector<double>& reference, double tolerance)
{
	double difference = 0, reference_norm = 0, x_norm = 0, b_norm = 0, a_norm = 0, residual = 0;
	for (std::size_t i = 0; i < n; ++i)
	{
		if (!std::isfinite(x[i]))
		{
			std::cout << name << ": FAILED (non-finite solution)\n";
			return false;
		}
		difference = std::max(difference, std::abs(x[i] - reference[i]));
		reference_norm = std::max(reference_norm, std::abs(reference[i]));
		x_norm = std::max(x_norm, std::abs(x[i]));
		b_norm = std::max(b_norm, std::abs(b[i]));
	}
	for (std::size_t i = 0; i < n; ++i)
	{
		double ax = 0, row_norm = 0;
		for (std::size_t j = 0; j < n; ++j)
		{
			ax += a[i * n + j] * x[j];
			row_norm += std::abs(a[i * n + j]);
		}
		residual = std::max(residual, std::abs(ax - b[i]));
		a_norm = std::max(a_norm, row_norm);
	}
	const double relative_error = difference / (reference_norm > 0 ? reference_norm : 1);
	const double scale = a_norm * x_norm + b_norm;
	const double backward_error = residual / (scale > 0 ? scale : 1);
	const bool passed = std::isfinite(relative_error) && std::isfinite(backward_error) &&
	                    relative_error <= tolerance && backward_error <= tolerance;
	std::cout << name << ": solution error (inf)=" << difference
	          << ", relative solution error=" << relative_error << ", ||Ax-b||inf=" << residual
	          << ", scaled residual=" << backward_error << '\n';
	if (!passed)
		std::cerr << name << ": relative solution error or scaled residual exceeds tolerance "
		          << tolerance << " (or is non-finite)\n";
	return passed;
}
} // namespace

int main(int argc, char** argv)
{
	try
	{
		const auto options = parse_options(argc, argv);
		if (options.help)
		{
			std::cout
			    << "Usage: lu_example [--size N] [--seed S] [--tolerance T] [--cpu-only] [--help]\n"
			       "Defaults: size=128, nrhs=1, seed=42, tolerance=1e-10.\n"
			       "Errors use infinity norms; tolerance controls the exit status.\n";
			return 0;
		}
		const auto n = options.n;
		if (n > std::vector<double>{}.max_size() / n)
			throw std::length_error("Matrix size exceeds addressable storage");
		std::vector<double> a(n * n), b(n), known_x(n);
		std::mt19937 generator(options.seed);
		std::uniform_real_distribution<double> random(-1, 1);
		for (auto& value : known_x)
			value = random(generator);
		// Strict diagonal dominance ensures a nonsingular random system. Swapping rows
		// afterward exercises pivoting while preserving nonsingularity.
		for (std::size_t i = 0; i < n; ++i)
		{
			double row_sum = 0;
			for (std::size_t j = 0; j < n; ++j)
			{
				a[i * n + j] = random(generator);
				if (i != j)
					row_sum += std::abs(a[i * n + j]);
			}
			a[i * n + i] = row_sum + 1;
		}
		if (n > 1)
			for (std::size_t j = 0; j < n; ++j)
				std::swap(a[j], a[(n - 1) * n + j]);
		for (std::size_t i = 0; i < n; ++i)
			for (std::size_t j = 0; j < n; ++j)
				b[i] += a[i * n + j] * known_x[j];
		std::cout << std::scientific << std::setprecision(8) << "n=" << n
		          << ", nrhs=1, seed=" << options.seed << ", tolerance=" << options.tolerance
		          << '\n';
		auto reference_a = a;
		auto reference = b;
		std::vector<int> pivots(n);
		gpu_fact::lu_solve_reference({reference_a.data(), n, n, n}, {pivots.data(), n, 1, 1},
		                             {reference.data(), n, 1, 1});
		bool passed = compare("reference (against generated solution)", n, a, b, reference, known_x,
		                      options.tolerance);
		try
		{
			auto host_a = a;
			auto host_x = b;
			gpu_fact::lu_solve_host({host_a.data(), n, n, n}, {pivots.data(), n, 1, 1},
			                        {host_x.data(), n, 1, 1});
			passed = compare("host", n, a, b, host_x, reference, options.tolerance) && passed;
		}
		catch (const std::exception& error)
		{
			if (std::string(error.what()) == "LU solve (host): host LU not yet implemented")
				std::cout << "host: SKIPPED (not yet implemented)\n";
			else
			{
				std::cerr << "host: FAILED: " << error.what() << '\n';
				passed = false;
			}
		}
		if (options.cpu_only)
			std::cout << "device: SKIPPED (--cpu-only)\n";
		else
		{
			try
			{
				gpu_fact::DeviceBuffer<double> device_a(a.size()), device_b(n);
				gpu_fact::DeviceBuffer<int> device_pivots(n);
				GPU_FACT_CUDA_CHECK(cudaMemcpy(device_a.data(), a.data(), a.size() * sizeof(double),
				                               cudaMemcpyHostToDevice));
				GPU_FACT_CUDA_CHECK(cudaMemcpy(device_b.data(), b.data(), b.size() * sizeof(double),
				                               cudaMemcpyHostToDevice));
				gpu_fact::lu_solve_device({device_a.data(), n, n, n},
				                          {device_pivots.data(), n, 1, 1},
				                          {device_b.data(), n, 1, 1});
				GPU_FACT_CUDA_CHECK(cudaDeviceSynchronize());
				std::vector<double> device_x(n);
				GPU_FACT_CUDA_CHECK(cudaMemcpy(device_x.data(), device_b.data(), n * sizeof(double),
				                               cudaMemcpyDeviceToHost));
				passed =
				    compare("device", n, a, b, device_x, reference, options.tolerance) && passed;
			}
			catch (const std::exception& error)
			{
				if (std::string(error.what()) == "LU solve (device): device LU not yet implemented")
					std::cout << "device: SKIPPED (not yet implemented)\n";
				else
				{
					std::cerr << "device: FAILED: " << error.what() << '\n';
					passed = false;
				}
			}
		}
		return passed ? 0 : 1;
	}
	catch (const std::exception& error)
	{
		std::cerr << error.what() << '\n';
		return 1;
	}
}
