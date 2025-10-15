cmake_minimum_required(VERSION 3.15)

set(CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED ON)

add_executable(mnist_digit_recognition_tests "mnist_loader_test.cc" "neural_layer_test.cc")

target_link_libraries(mnist_digit_recognition_tests PRIVATE mnist_loader neural gtest_main)

include(GoogleTest)
gtest_discover_tests(mnist_digit_recognition_tests)