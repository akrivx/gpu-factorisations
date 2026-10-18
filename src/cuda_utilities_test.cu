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
			gpu_fact::cuda_check_noexcept(cudaErrorInvalidValue, "expected CUDA failure");
			return 0; // The failure check must exit before reaching this line.
		}
		if (argc == 2 && std::string(argv[1]) == "--cublas-failure")
		{
			gpu_fact::cublas_check_noexcept(CUBLAS_STATUS_INVALID_VALUE, "expected cuBLAS failure");
			return 0;
		}
		const bool cpu_only = argc == 2 && std::string(argv[1]) == "--cpu-only";
		if (argc > 1 && !cpu_only)
			throw std::invalid_argument("Usage: cuda_utilities_test [--cpu-only]");
		gpu_fact::cuda_check(cudaSuccess);
		gpu_fact::cublas_check(CUBLAS_STATUS_SUCCESS);
		gpu_fact::cuda_check_noexcept(cudaSuccess);
		gpu_fact::cublas_check_noexcept(CUBLAS_STATUS_SUCCESS);
		bool cuda_threw = false, blas_threw = false;
		try
		{
			gpu_fact::cuda_check(cudaErrorInvalidValue, "test CUDA");
		}
		catch (const std::runtime_error& error)
		{
			cuda_threw = std::string(error.what()).find("test CUDA") != std::string::npos;
		}
		try
		{
			gpu_fact::cublas_check(CUBLAS_STATUS_INVALID_VALUE, "test BLAS");
		}
		catch (const std::runtime_error& error)
		{
			blas_threw = std::string(error.what()).find("test BLAS") != std::string::npos;
		}
		require(cuda_threw && blas_threw);
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
		gpu_fact::cuda_check(
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
		gpu_fact::cuda_check(
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
		gpu_fact::cublas_check(cublasGetPointerMode(target.get(), &mode));
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
