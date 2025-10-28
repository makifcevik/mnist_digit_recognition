// MIT License

#ifndef MNIST_DIGIT_RECOGNITION_LIBS_NEURAL_NEURAL_LAYER_H_
#define MNIST_DIGIT_RECOGNITION_LIBS_NEURAL_NEURAL_LAYER_H_

#include <concepts>

#include "matrix.h"

// Interface for neural network layers.
// Each layer must implement Forward, Backward, and UpdateWeights methods.
// Templated on floating-point type Ty for numerical computations.
template <std::floating_point Fp>
class NeuralLayer {
 public:
  using MatrixType = Matrix<Fp>;

  virtual ~NeuralLayer() = default;

  virtual MatrixType Forward(const MatrixType& input) = 0;
  virtual MatrixType Backward(const MatrixType& grad_output) = 0;
  // Update the layer's weights based on the internal learning rate and gradients.
  virtual void UpdateWeights() = 0;
};

#endif  // MNIST_DIGIT_RECOGNITION_LIBS_NEURAL_NEURAL_LAYER_H_
