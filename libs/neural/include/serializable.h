// MIT License

#ifndef MNIST_DIGIT_RECOGNITION_LIBS_NEURAL_NEURAL_SERIALIZABLE_H_
#define MNIST_DIGIT_RECOGNITION_LIBS_NEURAL_NEURAL_SERIALIZABLE_H_

#include <iostream>

#include <absl/status/status.h>

// Provides a serialization interface
class Serializable {
 public:
  virtual ~Serializable() = default;
  virtual absl::Status Serialize(std::ostream& out) const = 0;
  virtual absl::Status Deserialize(std::istream& in) = 0;
};

#endif  // MNIST_DIGIT_RECOGNITION_LIBS_NEURAL_NEURAL_SERIALIZABLE_H_