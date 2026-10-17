# Run before project(): the vcpkg toolchain installs manifest dependencies.
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  include(FetchContent)
  FetchContent_Declare(vcpkg
    GIT_REPOSITORY https://github.com/microsoft/vcpkg.git
    GIT_TAG bf19794383e8fb95eba5aa553235bc8ec2271688
    SOURCE_DIR "${CMAKE_BINARY_DIR}/_deps/vcpkg-src"
  )
  FetchContent_GetProperties(vcpkg)
  if(NOT vcpkg_POPULATED)
    FetchContent_Populate(vcpkg)
  endif()
  set(CMAKE_TOOLCHAIN_FILE "${vcpkg_SOURCE_DIR}/scripts/buildsystems/vcpkg.cmake"
      CACHE FILEPATH "vcpkg toolchain")
endif()
