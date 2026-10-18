#pragma once

#include "cuda_check.hpp"
#include <cstddef>
#include <limits>
#include <utility>

namespace gpu_fact
{

// Owns n elements on the current CUDA device. Empty and moved-from buffers hold nullptr.
// Destroy or replace the buffer while its CUDA device/context remains valid.
template <typename T = double> class DeviceBuffer
{
  public:
	DeviceBuffer() noexcept = default;
	explicit DeviceBuffer(std::size_t n)
	{
		if (n > (std::numeric_limits<std::size_t>::max)() / sizeof(T))
			throw std::length_error("DeviceBuffer: allocation size overflow");
		if (n != 0)
		{
			void* allocation = nullptr;
			cuda_check(cudaMalloc(&allocation, n * sizeof(T)), "cudaMalloc");
			data_ = static_cast<T*>(allocation);
			size_ = n;
		}
	}
	~DeviceBuffer() noexcept
	{
		reset();
	}
	DeviceBuffer(const DeviceBuffer&) = delete;
	DeviceBuffer& operator=(const DeviceBuffer&) = delete;
	DeviceBuffer(DeviceBuffer&& other) noexcept
	    : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0))
	{
	}
	DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
	{
		if (this != &other)
		{
			reset();
			data_ = std::exchange(other.data_, nullptr);
			size_ = std::exchange(other.size_, 0);
		}
		return *this;
	}
	T* data() noexcept
	{
		return data_;
	}
	const T* data() const noexcept
	{
		return data_;
	}
	std::size_t size() const noexcept
	{
		return size_;
	}
	void reset() noexcept
	{
		if (data_)
			cuda_check_noexcept(cudaFree(std::exchange(data_, nullptr)), "cudaFree");
		size_ = 0;
	}

  private:
	T* data_ = nullptr;
	std::size_t size_ = 0;
};

} // namespace gpu_fact
