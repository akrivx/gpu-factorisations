# GPU factorisations

C++17/CUDA experiments with a CPU reference LU solve and cuBLAS primitives.

## Formatting

Install clang-format 14 or newer and put it on `PATH`, then configure CMake.
The repository's `.clang-format` applies an LLVM-based style with tab indentation
(width 4), Allman braces (opening braces on new lines), and a 100-column limit
to C++ and CUDA sources.

```sh
cmake --build --preset default --target format
cmake --build --preset default --target format-check
```

`format` updates files in `include/` and `src/`; `format-check` reports violations
without changing files and returns a failure for use in CI. These targets are
optional and are available when CMake finds clang-format. Set
`-DCLANG_FORMAT_EXECUTABLE=/path/to/clang-format` to select a specific executable.
Editors with clang-format integration can also use `.clang-format` directly.

## Build

Install a CUDA Toolkit, a compatible C++ compiler, CMake 3.26+, Git, and Ninja.
On Windows use an **x64 Native Tools Command Prompt for VS 2022** (or the
corresponding developer PowerShell). CUDA must support your host compiler.
On Linux the reference LAPACK build also needs a Fortran compiler such as gfortran.

```sh
cmake --preset default
cmake --build --preset default
ctest --preset default
./build/lu_example
```

On Windows run `build\lu_example.exe`. The first configure downloads a pinned
vcpkg checkout and builds LAPACKE, reference LAPACK, and BLAS automatically. A
local vcpkg overlay enables LAPACKE in the pinned reference LAPACK port. On Windows,
vcpkg also provisions its Fortran build toolchain. This needs network access and
can take several minutes. Subsequent builds reuse the dependency cache.
An existing vcpkg installation can be supplied using
`-DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake`.

The CUDA Toolkit (including cuBLAS) and GPU driver are system prerequisites;
CMake discovers and links them, rather than installing the SDK or driver.
`-DCUDAToolkit_ROOT=...` selects a toolkit installation.

CUDA targets the build machine's GPU by default. Set
`-DCMAKE_CUDA_ARCHITECTURES=86` (or a semicolon-separated list) when building
for another machine or without a visible GPU.

## Experiments

Internal CUDA utilities live in `src/` under namespace `gpu_fact`:

- `GPU_FACT_CUDA_CHECK(expression)` and `GPU_FACT_CUBLAS_CHECK(expression)` throw on failure.
  Each expression is evaluated once; errors include its spelling and the caller's
  function, file, and line number.
- Their `_NOEXCEPT` variants return normally on success; on failure they report to
  stderr and exit with `EXIT_FAILURE`, including when called from destructors.
- `DeviceBuffer<T>` owns device storage, exposes `data()` and `size()`, and supports an empty state.
- `BlasHandle` owns a cuBLAS handle and exposes `get()`.

Both owners are move-only with non-throwing moves and cleanup. Moved-from owners
are empty; move assignment releases the destination's previous resource. `reset()`
releases ownership and reports cleanup failures through the non-throwing checks.
Keep the owning CUDA device/context valid during cleanup.

`cuda_utilities_test --cpu-only` checks status handling and empty buffers;
`cuda_utilities_test` also exercises buffer and handle moves on the GPU. GPU tests
are registered with CTest when `GPU_FACTORISATIONS_TEST_GPU=ON`.

`include/lu.hpp` declares the LU methods in namespace `gpu_fact`.
`gpu_fact::lu_solve_reference(a, pivots, b)` accepts non-owning `MatrixView`
values from `include/matrix_view.hpp`. A view contains `data`, `rows`, `cols`,
and `row_stride` (in elements), with indexing `data[i*row_stride+j]`.
A must be nonempty and square; B has the same row count and one or more RHS
columns. Pivots are an `n*1` integer view with `row_stride=1`. A and B may have
padded rows. The solve overwrites A with LU factors, B with solutions, and
pivots with 1-based LAPACK row indices.

`src/validate_shape.hpp` provides the common `validate_lu_shapes` helper used by
the reference and host entry points. It rejects incompatible shapes, null
pointers, invalid strides, and dimensions/strides exceeding LP64 limits.
Callers must supply sufficiently large, non-overlapping buffers in the correct
address space. The device API uses views pointing to device storage.

The reference uses LAPACKE's `LAPACKE_dgesv` with `LAPACK_ROW_MAJOR`; its internal
layout conversion is part of the reference runtime. Save input copies for
residual checks. Singular matrices raise exceptions. `lu_host` and `lu_device`
are stubs that validate their views and then throw a "not yet implemented" error.

`lu_example` solves a system requiring row pivoting and checks the known solution
and singular-matrix handling. It then evaluates `Ax` with cuBLAS, subtracts `b`
with a CUDA kernel, and checks the residual norm. The LU factorisation itself
runs on the CPU; this is a reference for future GPU implementations.

CTest runs the CPU checks by default (`lu_example --cpu-only`). Configure with
`-DGPU_FACTORISATIONS_TEST_GPU=ON` to register the GPU residual check too.
Add new `.cu` executables and link `factorisation_dependencies` to reuse LAPACK,
the CUDA runtime, and cuBLAS. Use larger matrices and suitable relative residuals
for numerical comparisons; this tiny example is a smoke test, not a benchmark.
