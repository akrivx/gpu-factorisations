#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>

namespace gpu_fact
{

inline void cuda_check(cudaError_t status, const char* operation = "CUDA operation")
{
	if (status != cudaSuccess)
		throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

// Reports failures to stderr and exits with EXIT_FAILURE; suitable for destructors.
inline void cuda_check_noexcept(cudaError_t status,
                                const char* operation = "CUDA operation") noexcept
{
	if (status == cudaSuccess)
		return;
	std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
	std::exit(EXIT_FAILURE);
}

} // namespace gpu_fact
