# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.20)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("mnist_digit_recognition")
include(FetchContent)

# Fetch GoogleTest
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/main.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Fetch Abseil library
FetchContent_Declare(
  abseil
  URL https://github.com/abseil/abseil-cpp/archive/refs/tags/20240116.2.zip
)
set(ABSL_PROPAGATE_CXX_STD ON)
FetchContent_GetProperties(abseil)
if(NOT abseil_POPULATED)
  FetchContent_Populate(abseil)
  # EXCLUDE_FROM_ALL prevents building targets not required by the executable
  add_subdirectory(${abseil_SOURCE_DIR} ${abseil_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

enable_testing()

# Include sub-projects.
add_subdirectory ("src")
add_subdirectory ("tests")
add_subdirectory ("libs/mnist_loader")
add_subdirectory ("libs/neural")
