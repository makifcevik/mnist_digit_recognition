# CMakeList.txt : CMake project for mnist_digit_recognition, include source and define
# project specific logic here.
#

# Add source to this project's executable.
add_executable (mnist_digit_recognition 
"main.cc")

if (CMAKE_VERSION VERSION_GREATER 3.20)
  set_property(TARGET mnist_digit_recognition PROPERTY CXX_STANDARD 20)
endif()

target_link_libraries(mnist_digit_recognition 
PRIVATE 
mnist_loader_lib 
neural_lib)

# copy for executable target
add_custom_command(TARGET mnist_digit_recognition POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${CMAKE_SOURCE_DIR}/data" "$<TARGET_FILE_DIR:mnist_digit_recognition>/data")

add_custom_command(TARGET mnist_digit_recognition POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${CMAKE_SOURCE_DIR}/models" "$<TARGET_FILE_DIR:mnist_digit_recognition>/models")

# TODO: Add tests and install targets if needed.
