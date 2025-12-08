// MIT License

// Implementation of ReLULayer methods.
// Intended only to be included by relu_layer.h.

template <std::floating_point Fp>
LayerType ReLULayer<Fp>::Type() const {
  return LayerType::kReLU;
}

template <std::floating_point Fp>
Matrix<Fp> ReLULayer<Fp>::Forward(const Matrix<Fp>& input) {
  // Apply ReLU: output = max(0, input) element-wise
  Matrix<Fp> output = input;
  for (size_t r = 0; r < input.Rows(); ++r) {
    for (size_t c = 0; c < input.Cols(); ++c) {
      output(r, c) = std::max(Fp(0), input(r, c));
    }
  }
  input_cache_ = input;  // Cache input for backpropagation
  return output;
}

template <std::floating_point Fp>
Matrix<Fp> ReLULayer<Fp>::Backward(const Matrix<Fp>& grad_output) {
  // Gradient of ReLU: grad_input = grad_output * (input > 0)
  // Gradient is passed only for positive inputs
  Matrix<Fp> grad_input(grad_output.Rows(), grad_output.Cols());
  for (size_t r = 0; r < input_cache_.Rows(); ++r) {
    for (size_t c = 0; c < input_cache_.Cols(); ++c) {
      grad_input(r, c) = input_cache_(r, c) > Fp(0) ? grad_output(r, c) : Fp(0);
    }
  }
  return grad_input;
}