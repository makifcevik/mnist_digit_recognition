// MIT License

#ifndef MNIST_DIGIT_RECOGNITION_LIBS_NEURAL_NEURAL_LAYER_H_
#define MNIST_DIGIT_RECOGNITION_LIBS_NEURAL_NEURAL_LAYER_H_

#include <concepts>

#include "matrix.h"

template <typename T>
concept Float = std::floating_point<T>;

// Interface for neural network layers.
// Each layer must implement Forward, Backward, and UpdateWeights methods.
// Templated on floating-point type Ty for numerical computations.
template <Float Ty>
class NeuralLayer {
 public:
  using MatrixType = Matrix<Ty>;

  virtual ~NeuralLayer() = default;

  virtual MatrixType Forward(const MatrixType& input) = 0;
  virtual MatrixType Backward(const MatrixType& grad_output) = 0;
  virtual void UpdateWeights(Ty learning_rate) = 0;
};

#endif  // MNIST_DIGIT_RECOGNITION_LIBS_NEURAL_NEURAL_LAYER_H_
