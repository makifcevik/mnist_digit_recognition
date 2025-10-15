# CMakeList.txt : CMake project for mnist_digit_recognition, include source and define
# project specific logic here.
#

# Add source to this project's executable.
add_executable (mnist_digit_recognition "mnist_digit_recognition.cpp" "mnist_digit_recognition.h")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET mnist_digit_recognition PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
