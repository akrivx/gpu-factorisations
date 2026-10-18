cmake_minimum_required(VERSION 3.26)
include(cmake/Dependencies.cmake)

# native is convenient for experiments on the build machine. Override for deployment.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures")
endif()
project(gpu_factorisations VERSION 0.1.0 LANGUAGES CXX CUDA)
include(cmake/Formatting.cmake)

find_package(CUDAToolkit REQUIRED)
find_package(lapacke CONFIG REQUIRED)

add_library(lu_reference src/lu_reference.cpp)
target_include_directories(lu_reference PUBLIC include)
target_compile_features(lu_reference PUBLIC cxx_std_17)
target_link_libraries(lu_reference PRIVATE lapacke)

# Link this target from new CUDA experiments to get the common dependencies.
add_library(factorisation_dependencies INTERFACE)
target_link_libraries(factorisation_dependencies INTERFACE
  lu_reference CUDA::cudart CUDA::cublas)

add_executable(lu_example src/lu_example.cu)
target_link_libraries(lu_example PRIVATE factorisation_dependencies)
set_target_properties(lu_example PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)

include(CTest)
if(BUILD_TESTING)
  add_test(NAME lu_cpu_reference COMMAND lu_example --cpu-only)
  option(GPU_FACTORISATIONS_TEST_GPU "Register tests requiring an NVIDIA GPU" OFF)
  if(GPU_FACTORISATIONS_TEST_GPU)
    add_test(NAME lu_gpu_residual COMMAND lu_example)
  endif()
endif()
