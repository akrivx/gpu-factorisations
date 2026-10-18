#pragma once

#include "cublas_check.hpp"
#include <utility>

namespace gpu_fact
{

// Owns a cuBLAS handle on the current CUDA device. A moved-from handle is null.
// Destroy or replace the handle while its CUDA device/context remains valid.
class BlasHandle
{
  public:
	BlasHandle()
	{
		cublas_check(cublasCreate(&handle_), "cublasCreate");
	}
	~BlasHandle() noexcept
	{
		reset();
	}
	BlasHandle(const BlasHandle&) = delete;
	BlasHandle& operator=(const BlasHandle&) = delete;
	BlasHandle(BlasHandle&& other) noexcept : handle_(std::exchange(other.handle_, nullptr))
	{
	}
	BlasHandle& operator=(BlasHandle&& other) noexcept
	{
		if (this != &other)
		{
			reset();
			handle_ = std::exchange(other.handle_, nullptr);
		}
		return *this;
	}
	cublasHandle_t get() const noexcept
	{
		return handle_;
	}
	void reset() noexcept
	{
		if (handle_)
			cublas_check_noexcept(cublasDestroy(std::exchange(handle_, nullptr)), "cublasDestroy");
	}

  private:
	cublasHandle_t handle_ = nullptr;
};

} // namespace gpu_fact
