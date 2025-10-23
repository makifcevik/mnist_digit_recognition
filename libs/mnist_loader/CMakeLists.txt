cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(mnist_loader_lib "src/mnist_loader.cc")

target_include_directories(mnist_loader_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(mnist_loader_lib PUBLIC neural_lib)