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

`lu_reference` provides `factorisations::lu_solve(n, nrhs, a, b)` using reference
LAPACKE's `LAPACKE_dgesv` (LU with partial pivoting followed by triangular solves).
Matrices, right-hand sides, and solutions use double precision and contiguous
row-major storage: `A[i*n+j]`, `B[i*nrhs+j]`, and `X[i*nrhs+j]`.
The wrapper passes `LAPACK_ROW_MAJOR`; LAPACKE handles LAPACK's column-major
layout internally. Input copies and LAPACKE's layout conversion are part of the
reference solve and should be accounted for when timing.
The cuBLAS GEMV example uses a transpose operation to interpret the row-major
matrix correctly without a device-side layout conversion.
Inputs are preserved; invalid dimensions and singular matrices
raise exceptions. The dependency uses 32-bit LAPACK integers (LP64).

`lu_example` solves a system requiring row pivoting and checks the known solution
and singular-matrix handling. It then evaluates `Ax` with cuBLAS, subtracts `b`
with a CUDA kernel, and checks the residual norm. The LU factorisation itself
runs on the CPU; this is a reference for future GPU implementations.

CTest runs the CPU checks by default (`lu_example --cpu-only`). Configure with
`-DGPU_FACTORISATIONS_TEST_GPU=ON` to register the GPU residual check too.
Add new `.cu` executables and link `factorisation_dependencies` to reuse LAPACK,
the CUDA runtime, and cuBLAS. Use larger matrices and suitable relative residuals
for numerical comparisons; this tiny example is a smoke test, not a benchmark.
