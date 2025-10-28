// MIT License

#ifndef MNIST_DIGIT_RECOGNITION_LIBS_NEURAL_LINEAR_LAYER_H_
#define MNIST_DIGIT_RECOGNITION_LIBS_NEURAL_LINEAR_LAYER_H_

#include <cstdint>

#include <concepts>

#include "matrix.h"
#include "neural_layer.h"


// Defines a linear network layer
template <std::floating_point Fp>
class LinearLayer : public NeuralLayer<Fp> {
 public:
  explicit LinearLayer(uint32_t input_size, uint32_t output_size,
                       Fp learning_rate, uint32_t seed);
  ~LinearLayer() override = default;

  // Disallow copying to prevent expensive memory duplication.
  LinearLayer(const LinearLayer&) = delete;
  LinearLayer& operator=(const LinearLayer&) = delete;
  // Allow moving.
  LinearLayer(LinearLayer&&) noexcept = default;
  LinearLayer& operator=(LinearLayer&&) noexcept = default;

  Matrix<Fp> Forward(const Matrix<Fp>& input) override;
  Matrix<Fp> Backward(const Matrix<Fp>& grad_output) override;
  void UpdateWeights() override;

 private:
  Matrix<Fp> weights_;
  Matrix<Fp> biases_;
  Fp learning_rate_;
};

#endif  // MNIST_DIGIT_RECOGNITION_LIBS_NEURAL_LINEAR_LAYER_H_
