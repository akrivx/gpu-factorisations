#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <stdexcept>
#include <string>

namespace gpu_fact
{

inline void cublas_check(cublasStatus_t status, const char* operation = "cuBLAS operation")
{
	if (status != CUBLAS_STATUS_SUCCESS)
		throw std::runtime_error(std::string(operation) + ": cuBLAS status " +
		                         std::to_string(static_cast<int>(status)));
}

// Reports failures to stderr and exits with EXIT_FAILURE; suitable for destructors.
inline void cublas_check_noexcept(cublasStatus_t status,
                                  const char* operation = "cuBLAS operation") noexcept
{
	if (status == CUBLAS_STATUS_SUCCESS)
		return;
	std::fprintf(stderr, "%s: cuBLAS status %d\n", operation, static_cast<int>(status));
	std::exit(EXIT_FAILURE);
}

} // namespace gpu_fact
