// MIT License

// This file contains the implementation of the NeuralNetwork class template.
// It is meant to be included only inside neural_network.h.

#include <absl/log/check.h>

#include "loss.h"

template <std::floating_point Fp>
void NeuralNetwork<Fp>::AddLayer(std::unique_ptr<NeuralLayer<Fp>> layer) {
  layers_.emplace_back(std::move(layer));
}

template <std::floating_point Fp>
typename NeuralNetwork<Fp>::MatType NeuralNetwork<Fp>::Forward(
    const MatType& input) {
  MatType output = input;
  for (const auto& layer : layers_) {
    output = layer->Forward(output);
  }
  return output;
}

template <std::floating_point Fp>
typename NeuralNetwork<Fp>::MatType NeuralNetwork<Fp>::Backward(
    const MatType& grad_output) {
  MatType grad = grad_output;
  for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
    grad = (*it)->Backward(grad);
  }
  return grad;
}

template <std::floating_point Fp>
void NeuralNetwork<Fp>::UpdateWeights() {
  for (const auto& layer : layers_) {
    layer->UpdateWeights();
  }
}

template <std::floating_point Fp>
void NeuralNetwork<Fp>::Train(const MatType& rawData, const MatType& rawLabels,
                              uint32_t epochs) {
  for (uint32_t epoch = 0; epoch < epochs; ++epoch) {
    MatType predictions = Forward(rawData);

    MatType loss_grad = Loss::SoftmaxCrossEntropyGradient<Fp>(predictions, rawLabels);

    Backward(loss_grad);
    UpdateWeights();
  }
}