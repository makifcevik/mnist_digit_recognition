cmake_minimum_required(VERSION 3.15)

set(CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(neural "src/neural_layer.cc")
