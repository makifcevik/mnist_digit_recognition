// MIT License

#ifndef MNIST_DIGIT_RECOGNITION_LIBS_NEURAL_TYPE_TRAITS_H_
#define MNIST_DIGIT_RECOGNITION_LIBS_NEURAL_TYPE_TRAITS_H_

#include "common_types.h"

// Base case: Unknown
template <typename T>
struct TypeToEnum {
  static constexpr DataType value = DataType::kUnknown;
};

template <>
struct TypeToEnum<float> {
  static constexpr DataType value = DataType::kFloat;
};

template <>
struct TypeToEnum<double> {
  static constexpr DataType value = DataType::kDouble;
};

template <>
struct TypeToEnum<int32_t> {
  static constexpr DataType value = DataType::kInt32;
};

#endif  // MNIST_DIGIT_RECOGNITION_LIBS_NEURAL_TYPE_TRAITS_H_
