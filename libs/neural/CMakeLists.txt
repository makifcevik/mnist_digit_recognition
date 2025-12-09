cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(neural_lib INTERFACE)

target_include_directories(neural_lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(neural_lib 
  INTERFACE
	absl::base
	absl::status
	absl::statusor
	absl::check
    absl::strings)