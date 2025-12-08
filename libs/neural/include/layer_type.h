// MIT License

#ifndef MNIST_DIGIT_RECOGNITION_LIBS_NEURAL_NEURAL_LAYER_TYPE_H_
#define MNIST_DIGIT_RECOGNITION_LIBS_NEURAL_NEURAL_LAYER_TYPE_H_

#include <cstdint>

// Provides layer types for neural network
enum class LayerType : uint32_t {
  kUnknown = 0,
  kLinear = 1,
  kReLU = 2
};

#endif  // MNIST_DIGIT_RECOGNITION_LIBS_NEURAL_NEURAL_LAYER_TYPE_H_