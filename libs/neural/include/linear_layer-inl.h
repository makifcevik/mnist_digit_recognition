// MIT License

// Implementation of LinearLayer methods.
// Intended only to be included by linear_layer.h.

#include <cmath>
#include <random>

template <std::floating_point Fp>
LinearLayer<Fp>::LinearLayer()
    : weights_(),
      biases_(),
      learning_rate_(Fp(0)),
      grad_weights_(),
      grad_biases_(),
      input_cache_() {}

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
LayerType LinearLayer<Fp>::Type() const {
  return LayerType::kLinear;
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

// Serialization
template <std::floating_point Fp>
absl::Status LinearLayer<Fp>::Serialize(std::ostream& out) const {
  // Handle type Fp
  DataType type_id = TypeToEnum<Fp>::value;
  out.write(reinterpret_cast<const char*>(&type_id), sizeof(type_id));

  // Save learning rate
  out.write(reinterpret_cast<const char*>(&learning_rate_),
            sizeof(learning_rate_));

  // Save weights
  absl::Status status = weights_.Serialize(out);
  if (!status.ok())
    return status;

  // Save biases (returns absl::Status)
  return biases_.Serialize(out);
}
template <std::floating_point Fp>
absl::Status LinearLayer<Fp>::Deserialize(std::istream& in) {
  // Handle type Fp
  DataType stored_type;
  in.read(reinterpret_cast<char*>(&stored_type), sizeof(stored_type));
  if (in.fail())
    return absl::DataLossError("Failed to read type data of linear layer");
  DataType expected_type = TypeToEnum<Fp>::value;

  if (expected_type != stored_type)
    return absl::InvalidArgumentError(
        absl::StrCat("Type mismatch! File containing type ID: ",
                     static_cast<uint32_t>(stored_type),
                     " but linear layer expects type ID: ",
                     static_cast<uint32_t>(expected_type)));

  // Read learning rate
  in.read(reinterpret_cast<char*>(&learning_rate_), sizeof(learning_rate_));
  if (in.fail())
    return absl::DataLossError("Failed to read learning rate");

  // Read weights
  absl::Status status = weights_.Deserialize(in);
  if (!status.ok())
    return status;

  // Read biases (returns absl::Status)
  status = biases_.Deserialize(in);
  if (!status.ok())
    return status;

  // --- CRITICAL
  // Since we used the default constructor, grad_weights_ is 0x0.
  // We must resize it to match the newly loaded weights_, otherwise
  // the first training step after loading will crash.
  grad_weights_.Resize(weights_.Rows(), weights_.Cols(), 0);
  grad_biases_.Resize(biases_.Rows(), biases_.Cols(), 0);
  
  return absl::OkStatus();
}

template <std::floating_point Fp>
Fp LinearLayer<Fp>::GetLearningRate() const {
  return learning_rate_;
}
template <std::floating_point Fp>
void LinearLayer<Fp>::SetLearningRate(Fp lr) {
  learning_rate_ = lr;
}
