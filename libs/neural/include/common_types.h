// MIT License

#ifndef MNIST_DIGIT_RECOGNITION_LIBS_NEURAL_COMMON_TYPES_H_
#define MNIST_DIGIT_RECOGNITION_LIBS_NEURAL_COMMON_TYPES_H_

#include <cstdint>

// Contains common data types for template classes
enum class DataType : uint32_t {
	kUnknown = 0,
	kFloat = 1,
	kDouble = 2,
	kInt32 = 3,
};

#endif  // MNIST_DIGIT_RECOGNITION_LIBS_NEURAL_COMMON_TYPES_H_
