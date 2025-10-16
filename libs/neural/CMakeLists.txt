cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(neural "src/neural_layer.cc" "include/dataset.h" "include/matrix.h")
