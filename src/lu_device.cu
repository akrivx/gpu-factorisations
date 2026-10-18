#include "lu.hpp"

#include "blas_handle.hpp"
#include "cuda_check.hpp"
#include "device_buffer.hpp"
#include "validate_shape.hpp"

#include <stdexcept>

namespace
{

constexpr int BLOCK_SIZE = 256; // 8 warps per thread block
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;

static_assert(BLOCK_SIZE % WARP_SIZE == 0, "BLOCK_SIZE must be a multiple of WARP_SIZE");

__device__ __forceinline__ void warp_argmax(double& value, int& row)
{
	constexpr unsigned mask = 0xffffffff;
	const int lane = threadIdx.x % warpSize;
	for (int offset = warpSize / 2; offset > 0; offset /= 2)
	{
		const double other_value = __shfl_down_sync(mask, value, offset);
		const int other_row = __shfl_down_sync(mask, row, offset);
		if (lane + offset < warpSize &&
		    (other_value > value || (other_value == value && other_row < row)))
		{
			value = other_value;
			row = other_row;
		}
	}
}

__global__ void find_pivot_kernel(gpu_fact::MatrixView<double> a, int k, int* pivot)
{
	__shared__ double warp_values[WARPS_PER_BLOCK];
	__shared__ int warp_rows[WARPS_PER_BLOCK];

	const int tid = threadIdx.x;
	const int warp = tid / warpSize;
	const int lane = tid % warpSize;

	double best_value = -1.0;
	int best_row = k;

	for (int i = k + tid; i < a.rows; i += BLOCK_SIZE)
	{
		const double value = fabs(a(i, k));
		const int row = i;
		if (value > best_value || (value == best_value && row < best_row))
		{
			best_value = value;
			best_row = row;
		}
	}

	warp_argmax(best_value, best_row);

	if (lane == 0)
	{
		warp_values[warp] = best_value;
		warp_rows[warp] = best_row;
	}
	__syncthreads();

	if (warp == 0)
	{
		best_value = lane < WARPS_PER_BLOCK ? warp_values[lane] : -1.0;
		best_row = lane < WARPS_PER_BLOCK ? warp_rows[lane] : k;
		warp_argmax(best_value, best_row);
		if (lane == 0)
			*pivot = best_row + 1; // Store 1-based index
	}
}

void find_pivot(gpu_fact::MatrixView<double> a, int k, int* pivot)
{
	find_pivot_kernel<<<1, BLOCK_SIZE>>>(a, k, pivot);
	GPU_FACT_CUDA_CHECK(cudaGetLastError());
}

__global__ void swap_rows_kernel(gpu_fact::MatrixView<double> a, int k, const int* pivot)
{
	const int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < a.cols)
	{
		const int row1 = k;
		const int row2 = *pivot - 1; // Convert to 0-based index
		const double temp = a(row1, col);
		a(row1, col) = a(row2, col);
		a(row2, col) = temp;
	}
}

void swap_rows(gpu_fact::MatrixView<double> a, int k, const int* pivot)
{
	const int blocks = (a.cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
	swap_rows_kernel<<<blocks, BLOCK_SIZE>>>(a, k, pivot);
	GPU_FACT_CUDA_CHECK(cudaGetLastError());
}

__global__ void compute_multipliers_kernel(gpu_fact::MatrixView<double> a, int k,
                                           int* zero_pivot_flag)
{
	const int row = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
	if (row < a.rows)
	{
		if (a(k, k) == 0.0)
			atomicExch(zero_pivot_flag, 1);
		else
			a(row, k) /= a(k, k);
	}
}

void compute_multipliers(gpu_fact::MatrixView<double> a, int k, int* zero_pivot_flag)
{
	const int remaining = static_cast<int>(a.rows) - k - 1;
	if (remaining <= 0)
		return;
	const int blocks = (remaining + BLOCK_SIZE - 1) / BLOCK_SIZE;
	compute_multipliers_kernel<<<blocks, BLOCK_SIZE>>>(a, k, zero_pivot_flag);
	GPU_FACT_CUDA_CHECK(cudaGetLastError());
}

__global__ void update_trailing_submatrix_kernel(gpu_fact::MatrixView<double> a, int k)
{
	const int row = k + 1 + blockIdx.y * blockDim.y + threadIdx.y;
	const int col = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;

	if (row < a.rows && col < a.cols)
		a(row, col) -= a(row, k) * a(k, col);
}

void update_trailing_submatrix(gpu_fact::MatrixView<double> a, int k)
{
	const int remaining_rows = static_cast<int>(a.rows) - k - 1;
	const int remaining_cols = static_cast<int>(a.cols) - k - 1;
	if (remaining_rows <= 0 || remaining_cols <= 0)
		return;

	dim3 blockDim(16, 16);
	dim3 gridDim((remaining_cols + blockDim.x - 1) / blockDim.x,
	             (remaining_rows + blockDim.y - 1) / blockDim.y);

	update_trailing_submatrix_kernel<<<gridDim, blockDim>>>(a, k);
	GPU_FACT_CUDA_CHECK(cudaGetLastError());
}

__global__ void finish_factorisation_kernel(gpu_fact::MatrixView<double> a,
                                            gpu_fact::MatrixView<int> pivots, int* zero_pivot_flag)
{
	const auto last = a.rows - 1;
	pivots(last, 0) = static_cast<int>(a.rows);
	if (a(last, last) == 0.0)
		*zero_pivot_flag = 1;
}

void factorise_matrix(gpu_fact::MatrixView<double> a, gpu_fact::MatrixView<int> pivots,
                      int* zero_pivot_flag)
{
	const int n = static_cast<int>(a.rows);
	for (int k = 0; k + 1 < n; ++k)
	{
		find_pivot(a, k, &pivots(k, 0));
		swap_rows(a, k, &pivots(k, 0));
		compute_multipliers(a, k, zero_pivot_flag);
		update_trailing_submatrix(a, k);
	}
	finish_factorisation_kernel<<<1, 1>>>(a, pivots, zero_pivot_flag);
	GPU_FACT_CUDA_CHECK(cudaGetLastError());
}

__global__ void permute_rhs_kernel(gpu_fact::MatrixView<const int> pivots,
                                   gpu_fact::MatrixView<double> b)
{
	const std::size_t col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col >= b.cols)
		return;
	// Pivot entries are an ordered swap history, not a final permutation vector.
	for (std::size_t i = 0; i < b.rows; ++i)
	{
		const int pivot = pivots(i, 0) - 1;
		if (pivot != static_cast<int>(i))
		{
			const double value = b(i, col);
			b(i, col) = b(pivot, col);
			b(pivot, col) = value;
		}
	}
}

void permute_rhs(gpu_fact::MatrixView<const int> pivots, gpu_fact::MatrixView<double> b)
{
	const auto blocks = (b.cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
	permute_rhs_kernel<<<blocks, BLOCK_SIZE>>>(pivots, b);
	GPU_FACT_CUDA_CHECK(cudaGetLastError());
}

// Row-major LU appears as (LU)^T to cuBLAS: L is stored in its upper triangle
// and U in its lower triangle. OP_T recovers the original triangular systems.
void solve_triangular(cublasHandle_t handle, gpu_fact::MatrixView<const double> a,
                      gpu_fact::MatrixView<double> b)
{
	const int n = static_cast<int>(a.rows);
	const int lda = static_cast<int>(a.row_stride);
	const int incx = static_cast<int>(b.row_stride);
	for (std::size_t col = 0; col < b.cols; ++col)
	{
		GPU_FACT_CUBLAS_CHECK(cublasDtrsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
		                                  CUBLAS_DIAG_UNIT, n, a.data, lda, b.data + col, incx));
		GPU_FACT_CUBLAS_CHECK(cublasDtrsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
		                                  CUBLAS_DIAG_NON_UNIT, n, a.data, lda, b.data + col,
		                                  incx));
	}
}

} // namespace

namespace gpu_fact
{

void lu_solve_device(MatrixView<double> a, MatrixView<int> pivots, MatrixView<double> b)
{
	validate_lu_shapes(a, pivots, b);
	gpu_fact::DeviceBuffer<int> zero_pivot_flag(1);
	GPU_FACT_CUDA_CHECK(cudaMemset(zero_pivot_flag.data(), 0, sizeof(int)));
	factorise_matrix(a, pivots, zero_pivot_flag.data());
	int singular = 0;
	// The blocking copy also waits for factorisation and exposes execution errors.
	GPU_FACT_CUDA_CHECK(
	    cudaMemcpy(&singular, zero_pivot_flag.data(), sizeof(singular), cudaMemcpyDeviceToHost));
	if (singular)
		throw std::runtime_error("LU solve (device): singular matrix");
	permute_rhs(pivots.as_const(), b);
	gpu_fact::BlasHandle blas;
	solve_triangular(blas.get(), a.as_const(), b);
	GPU_FACT_CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace gpu_fact
