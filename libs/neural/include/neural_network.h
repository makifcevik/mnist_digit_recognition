// MIT License

#ifndef MNIST_DIGIT_RECOGNITION_LIBS_NEURAL_NEURAL_NETWORK_H_
#define MNIST_DIGIT_RECOGNITION_LIBS_NEURAL_NEURAL_NETWORK_H_

#include <memory>
#include <vector>
#include <concepts>

#include "matrix.h"
#include "neural_layer.h"

// Network class managing multiple neural layers (NeuralLayer<Fp> instances).
template <std::floating_point Fp>
class NeuralNetwork {
 public:
  using MatType = NeuralLayer<Fp>::MatrixType;
  using NetworkLayer = NeuralLayer<Fp>;

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
  void Train(const MatType& rawData, const MatType& rawLabels, uint32_t epochs);

 private:
  std::vector<std::unique_ptr<NeuralLayer>> layers_;
};

#endif  // MNIST_DIGIT_RECOGNITION_LIBS_NEURAL_NEURAL_NETWORK_H_