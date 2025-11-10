// MIT License

// Implementation of LinearLayer methods.
// Intended only to be included by linear_layer.h.

#include <cmath>
#include <random>

template <std::floating_point Fp>
LinearLayer<Fp>::LinearLayer(uint32_t input_size, uint32_t output_size,
                             Fp learning_rate, uint32_t seed)
    : weights_(input_size, output_size),
      biases_(1, output_size),  // Biases initialized to zero
      learning_rate_(learning_rate),
      grad_weights_(input_size,
                    output_size),    // Initialized to zero
      grad_biases_(1, output_size),  // Initialized to zero
      input_cache_(0, 0)             // Empty cache
{
  // Xaiver initialization
  Fp limit = std::sqrt(Fp(6) / Fp(input_size + output_size));

  // Initialize weights with random values in [-limit, limit]
  weights_ = Matrix<Fp>::Random(input_size, output_size, -limit, limit, seed);
}

template <std::floating_point Fp>
Matrix<Fp> LinearLayer<Fp>::Forward(const Matrix<Fp>& input) {
  // Linear transformation: output = input * weights + biases
  Matrix<Fp> output = input * weights_;
  output += biases_.BroadcastRows(output.Rows());
  input_cache_ = input;  // Cache input for backpropagation
  return output;
}

template <std::floating_point Fp>
Matrix<Fp> LinearLayer<Fp>::Backward(const Matrix<Fp>& grad_output) {
  // Compute gradients
  grad_weights_ = input_cache_.GetTranspose() * grad_output;
  grad_biases_ = grad_output.CollapseRows();

  // Gradient to propagate to previous layer
  Matrix<Fp> grad_input = grad_output * weights_.GetTranspose();
  return grad_input;
}

template <std::floating_point Fp>
void LinearLayer<Fp>::UpdateWeights() {
  // Update weights and biases using gradient descent
  weights_ -= grad_weights_ * learning_rate_;
  biases_ -= grad_biases_ * learning_rate_;
}
