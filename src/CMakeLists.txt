# CMakeList.txt : CMake project for mnist_digit_recognition, include source and define
# project specific logic here.
#

# Add source to this project's executable.
add_executable (mnist_digit_recognition "main.cc")

if (CMAKE_VERSION VERSION_GREATER 3.15)
  set_property(TARGET mnist_digit_recognition PROPERTY CXX_STANDARD 17)
endif()

target_link_libraries(mnist_digit_recognition PRIVATE mnist_loader neural)

# TODO: Add tests and install targets if needed.
