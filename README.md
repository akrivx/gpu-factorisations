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
residual checks. Singular matrices raise exceptions. The host and device solvers
perform pivoted LU factorisation, ordered RHS row swaps, and forward/backward
substitution. GPU triangular solves use two cuBLAS `cublasDtrsv` calls per RHS
(unit lower, then non-unit upper). Transpose operations and reversed triangle
selection account for row-major LU storage; vector increments account for RHS row
strides. Factorisation and RHS permutation still use custom CUDA kernels.
The host solver currently operates on the first RHS only; the device solver
supports multiple RHS columns and padded row strides.

`lu_example.cpp` generates a random, nonsingular system (default seed 42),
with one RHS and a known solution. It solves fresh copies using the reference,
host, and device methods. Completed solves print numerical errors: absolute and
relative infinity-norm solution errors, `||Ax-b||inf`, and the scaled residual
`||Ax-b||inf / (||A||inf*||x||inf + ||b||inf)`. Host and device solutions are
compared against the reference; the reference is
checked against the generated solution. The tolerance (default `1e-10`) controls
the exit status using the relative solution error and scaled residual; raw absolute
errors are reported for inspection. Successful rows show numbers rather than PASS/FAIL.
Solver errors or comparison failures return a nonzero exit code.
Device inputs and outputs are copied between host and device storage.

```sh
./build/lu_example --size 256
./build/lu_example --size 256 --seed 123 --tolerance 1e-12 --cpu-only
```

The default size is 128. `--size` accepts integers from 1 to `INT_MAX`, subject
to available memory. The normal run requires a working CUDA device;
`--cpu-only` skips the device solver.
Seeds are unsigned 32-bit integers (including zero); tolerance must be finite and
positive. The seed and tolerance are printed in each run. The same seed reproduces
the inputs with the same C++ standard library; random distributions and floating-point
results can differ between toolchains and platforms.

CTest runs deterministic LU validation cases in `lu_reference_test`, random CPU
comparisons, and utility checks by default. Configure with
`-DGPU_FACTORISATIONS_TEST_GPU=ON` to register GPU comparison and utility tests too.
Add new `.cu` executables and link `factorisation_dependencies` to reuse LAPACK,
the CUDA runtime, and cuBLAS. Use larger matrices and suitable relative residuals
for numerical comparisons; this tiny example is a smoke test, not a benchmark.
