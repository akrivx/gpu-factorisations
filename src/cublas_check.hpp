#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <stdexcept>
#include <string>

namespace gpu_fact::detail
{

inline void cublas_check(cublasStatus_t status, const char* operation, const char* function,
                         const char* file, int line)
{
	if (status != CUBLAS_STATUS_SUCCESS)
		throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + " in " +
		                         function + ": " + operation + ": cuBLAS status " +
		                         std::to_string(static_cast<int>(status)));
}

// Reports failures to stderr and exits with EXIT_FAILURE; suitable for destructors.
inline void cublas_check_noexcept(cublasStatus_t status, const char* operation,
                                  const char* function, const char* file, int line) noexcept
{
	if (status == CUBLAS_STATUS_SUCCESS)
		return;
	std::fprintf(stderr, "%s:%d in %s: %s: cuBLAS status %d\n", file, line, function, operation,
	             static_cast<int>(status));
	std::exit(EXIT_FAILURE);
}

} // namespace gpu_fact::detail

#define GPU_FACT_CUBLAS_CHECK(operation)                                                           \
	::gpu_fact::detail::cublas_check((operation), #operation, __func__, __FILE__, __LINE__)
#define GPU_FACT_CUBLAS_CHECK_NOEXCEPT(operation)                                                  \
	::gpu_fact::detail::cublas_check_noexcept((operation), #operation, __func__, __FILE__, __LINE__)
