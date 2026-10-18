#include "blas_handle.hpp"
#include "device_buffer.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

using gpu_fact::BlasHandle;
using Buffer = gpu_fact::DeviceBuffer<int>;
static_assert(!std::is_copy_constructible_v<Buffer> && !std::is_copy_assignable_v<Buffer>);
static_assert(std::is_nothrow_move_constructible_v<Buffer> &&
              std::is_nothrow_move_assignable_v<Buffer>);
static_assert(!std::is_copy_constructible_v<BlasHandle> && !std::is_copy_assignable_v<BlasHandle>);
static_assert(std::is_nothrow_move_constructible_v<BlasHandle> &&
              std::is_nothrow_move_assignable_v<BlasHandle>);

void require(bool condition)
{
	if (!condition)
		throw std::runtime_error("CUDA utility check failed");
}

int main(int argc, char** argv)
{
	try
	{
		if (argc == 2 && std::string(argv[1]) == "--cuda-failure")
		{
			GPU_FACT_CUDA_CHECK_NOEXCEPT(cudaErrorInvalidValue);
			return 0; // The failure check must exit before reaching this line.
		}
		if (argc == 2 && std::string(argv[1]) == "--cublas-failure")
		{
			GPU_FACT_CUBLAS_CHECK_NOEXCEPT(CUBLAS_STATUS_INVALID_VALUE);
			return 0;
		}
		const bool cpu_only = argc == 2 && std::string(argv[1]) == "--cpu-only";
		if (argc > 1 && !cpu_only)
			throw std::invalid_argument("Usage: cuda_utilities_test [--cpu-only]");
		GPU_FACT_CUDA_CHECK(cudaSuccess);
		GPU_FACT_CUBLAS_CHECK(CUBLAS_STATUS_SUCCESS);
		GPU_FACT_CUDA_CHECK_NOEXCEPT(cudaSuccess);
		GPU_FACT_CUBLAS_CHECK_NOEXCEPT(CUBLAS_STATUS_SUCCESS);
		bool cuda_threw = false, blas_threw = false;
		int failure_line = 0;
		try
		{
			failure_line = __LINE__ + 1;
			GPU_FACT_CUDA_CHECK(cudaErrorInvalidValue);
		}
		catch (const std::runtime_error& error)
		{
			const std::string message = error.what();
			cuda_threw = message.find("cudaErrorInvalidValue") != std::string::npos &&
			             message.find(std::string(__FILE__) + ":" + std::to_string(failure_line)) !=
			                 std::string::npos &&
			             message.find("in main:") != std::string::npos;
		}
		try
		{
			failure_line = __LINE__ + 1;
			GPU_FACT_CUBLAS_CHECK(CUBLAS_STATUS_INVALID_VALUE);
		}
		catch (const std::runtime_error& error)
		{
			const std::string message = error.what();
			blas_threw = message.find("CUBLAS_STATUS_INVALID_VALUE") != std::string::npos &&
			             message.find(std::string(__FILE__) + ":" + std::to_string(failure_line)) !=
			                 std::string::npos &&
			             message.find("in main:") != std::string::npos;
		}
		require(cuda_threw && blas_threw);
		int evaluations = 0;
		GPU_FACT_CUDA_CHECK((++evaluations, cudaSuccess));
		GPU_FACT_CUDA_CHECK_NOEXCEPT((++evaluations, cudaSuccess));
		GPU_FACT_CUBLAS_CHECK((++evaluations, CUBLAS_STATUS_SUCCESS));
		GPU_FACT_CUBLAS_CHECK_NOEXCEPT((++evaluations, CUBLAS_STATUS_SUCCESS));
		require(evaluations == 4);
		Buffer empty(0);
		Buffer moved_empty(std::move(empty));
		require(!empty.data() && !moved_empty.data() && moved_empty.size() == 0);
		bool overflow_threw = false;
		try
		{
			Buffer oversized((std::numeric_limits<std::size_t>::max)());
		}
		catch (const std::length_error&)
		{
			overflow_threw = true;
		}
		require(overflow_threw);
		if (cpu_only)
			return 0;

		Buffer source(1);
		const int value = 42;
		GPU_FACT_CUDA_CHECK(
		    cudaMemcpy(source.data(), &value, sizeof(value), cudaMemcpyHostToDevice));
		auto* allocation = source.data();
		Buffer moved(std::move(source));
		require(!source.data() && source.size() == 0 && moved.data() == allocation &&
		        moved.size() == 1);
		Buffer destination(2);
		destination = std::move(moved);
		require(!moved.data() && moved.size() == 0 && destination.data() == allocation);
		auto& same_buffer = destination;
		destination = std::move(same_buffer);
		int result = 0;
		GPU_FACT_CUDA_CHECK(
		    cudaMemcpy(&result, destination.data(), sizeof(result), cudaMemcpyDeviceToHost));
		require(result == value && destination.size() == 1);
		destination = std::move(moved_empty);
		require(!destination.data() && destination.size() == 0);
		destination.reset();

		BlasHandle handle;
		auto raw_handle = handle.get();
		BlasHandle moved_handle(std::move(handle));
		require(!handle.get() && moved_handle.get() == raw_handle);
		BlasHandle target;
		target = std::move(moved_handle);
		require(!moved_handle.get() && target.get() == raw_handle);
		auto& same_handle = target;
		target = std::move(same_handle);
		cublasPointerMode_t mode;
		GPU_FACT_CUBLAS_CHECK(cublasGetPointerMode(target.get(), &mode));
		target = std::move(moved_handle);
		require(!target.get());
		target.reset();
		return 0;
	}
	catch (const std::exception& error)
	{
		std::cerr << error.what() << '\n';
		return 1;
	}
}
