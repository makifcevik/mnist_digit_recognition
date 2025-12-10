cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(mnist_digit_recognition_tests 
"mnist_loader_test.cc" 
"matrix_test.cc")

target_link_libraries(mnist_digit_recognition_tests 
PRIVATE 
mnist_loader_lib 
neural_lib 
GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(mnist_digit_recognition_tests)