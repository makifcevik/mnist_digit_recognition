// MIT License

#ifndef MNIST_DIGIT_RECOGNITION_LIBS_NEURAL_RELU_LAYER_H_
#define MNIST_DIGIT_RECOGNITION_LIBS_NEURAL_RELU_LAYER_H_

#include "matrix.h"
#include "neural_layer.h"

// Defines a ReLU activation layer
// Applies the ReLU function: f(x) = max(0, x) element-wise
template <std::floating_point Fp>
class ReLULayer : public NeuralLayer<Fp> {
 public:
  ReLULayer() = default;

  Matrix<Fp> Forward(const Matrix<Fp>& input) override;
  Matrix<Fp> Backward(const Matrix<Fp>& grad_output) override;
  void UpdateWeights() override {}  // No parameters to update in ReLU layer

 private:
  Matrix<Fp> input_cache_{0, 0};
};

#include "relu_layer-inl.h"

#endif  // MNIST_DIGIT_RECOGNITION_LIBS_NEURAL_RELU_LAYER_H_