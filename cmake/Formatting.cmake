find_program(CLANG_FORMAT_EXECUTABLE NAMES clang-format)
if(CLANG_FORMAT_EXECUTABLE)
  # Only project sources: never traverse build trees or downloaded dependencies.
  file(GLOB_RECURSE formatting_sources CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/include/*.h"
    "${PROJECT_SOURCE_DIR}/include/*.hpp"
    "${PROJECT_SOURCE_DIR}/include/*.cuh"
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
    "${PROJECT_SOURCE_DIR}/src/*.cu"
    "${PROJECT_SOURCE_DIR}/src/*.h"
    "${PROJECT_SOURCE_DIR}/src/*.hpp"
    "${PROJECT_SOURCE_DIR}/src/*.cuh"
  )
  add_custom_target(format
    COMMAND "${CLANG_FORMAT_EXECUTABLE}" --style=file -i ${formatting_sources}
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    COMMENT "Formatting C++ and CUDA sources"
    VERBATIM
  )
  add_custom_target(format-check
    COMMAND "${CLANG_FORMAT_EXECUTABLE}" --style=file --dry-run --Werror ${formatting_sources}
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    COMMENT "Checking C++ and CUDA formatting"
    VERBATIM
  )
else()
  message(STATUS "clang-format not found; format and format-check targets are unavailable")
endif()
