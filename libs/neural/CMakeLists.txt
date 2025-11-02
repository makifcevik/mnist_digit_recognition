cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(neural_lib 
"src/neural_layer.cc" 
"include/dataset.h" 
"include/matrix.h" 
"include/linear_layer.h" 
"include/neural_network.h" 
"include/linear_layer-inl.h")

target_include_directories(neural_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(neural_lib 
  PUBLIC
	absl::base
	absl::status
	absl::check
    absl::strings)