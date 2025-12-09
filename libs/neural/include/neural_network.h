// MIT License

#ifndef MNIST_DIGIT_RECOGNITION_LIBS_NEURAL_NEURAL_NETWORK_H_
#define MNIST_DIGIT_RECOGNITION_LIBS_NEURAL_NEURAL_NETWORK_H_

#include <concepts>
#include <memory>
#include <vector>

#include <absl/status/status.h>

#include "matrix.h"
#include "neural_layer.h"
#include "layer_type.h"

// Network class managing multiple neural layers (NeuralLayer<Fp> instances).
template <std::floating_point Fp>
class NeuralNetwork {
 public:
  using MatType = typename NeuralLayer<Fp>::MatrixType;
  using NetworkLayer = NeuralLayer<Fp>;
  using Layers = std::vector<std::unique_ptr<NeuralLayer<Fp>>>;

  // Default constructor and destructor
  NeuralNetwork() = default;
  ~NeuralNetwork() = default;

  // Delete copy
  NeuralNetwork(const NeuralNetwork&) = delete;
  NeuralNetwork& operator=(const NeuralNetwork&) = delete;

  // Allow move
  NeuralNetwork(NeuralNetwork&&) noexcept = default;
  NeuralNetwork& operator=(NeuralNetwork&&) noexcept = default;

  void AddLayer(std::unique_ptr<NetworkLayer> layer);
  MatType Forward(const MatType& input);
  MatType Backward(const MatType& grad_output);
  void UpdateWeights();
  void Train(const MatType& raw_train_data, const MatType& raw_train_labels,
             const MatType& raw_test_data, const MatType& raw_test_labels,
             uint32_t epochs, uint32_t batch_size);
  float EvaluateAccuracy(const MatType& data, const MatType& labels);

  // Serialization
  // NOTE: This method is intended to be used by the `ModelSerializer` class.
  // It does not write the file header (Magic Number). To save a fully valid
  // model file, use `ModelSerializer::Save`.
  //
  // To load a network, use `ModelSerializer::Load`, as this class cannot
  // self-deserialize due to layer factory dependencies.
  absl::Status Serialize(std::ostream& out) const;

  const Layers& GetLayers() const;
  void Clear();

 private:
  std::vector<std::unique_ptr<NetworkLayer>> layers_;
};

#include "neural_network-inl.h"

#endif  // MNIST_DIGIT_RECOGNITION_LIBS_NEURAL_NEURAL_NETWORK_H_