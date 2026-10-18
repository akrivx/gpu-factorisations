#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>

namespace gpu_fact::detail
{

inline void cuda_check(cudaError_t status, const char* operation, const char* function,
                       const char* file, int line)
{
	if (status != cudaSuccess)
		throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + " in " +
		                         function + ": " + operation + ": " + cudaGetErrorString(status));
}

// Reports failures to stderr and exits with EXIT_FAILURE; suitable for destructors.
inline void cuda_check_noexcept(cudaError_t status, const char* operation, const char* function,
                                const char* file, int line) noexcept
{
	if (status == cudaSuccess)
		return;
	std::fprintf(stderr, "%s:%d in %s: %s: %s\n", file, line, function, operation,
	             cudaGetErrorString(status));
	std::exit(EXIT_FAILURE);
}

} // namespace gpu_fact::detail

// Evaluate the operation once and capture its spelling and the caller's location.
#define GPU_FACT_CUDA_CHECK(operation)                                                             \
	::gpu_fact::detail::cuda_check((operation), #operation, __func__, __FILE__, __LINE__)
#define GPU_FACT_CUDA_CHECK_NOEXCEPT(operation)                                                    \
	::gpu_fact::detail::cuda_check_noexcept((operation), #operation, __func__, __FILE__, __LINE__)
