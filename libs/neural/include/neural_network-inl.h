// MIT License

// This file contains the implementation of the NeuralNetwork class template.
// It is meant to be included only inside neural_network.h.

#include <absl/log/check.h>
#include <absl/log/log.h>

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
                              uint32_t epochs, uint32_t batch_size) {
  CHECK(rawData.Rows() == rawLabels.Rows())
      << "Number of samples in data and labels must be the same.";

  const size_t kNumSamples = rawData.Rows();
  const size_t kNumBatches =
      (kNumSamples + batch_size - 1) / batch_size;  // Ceiling division

  const size_t num_features = rawData.Cols();
  const size_t num_classes = rawLabels.Cols();

  for (uint32_t epoch = 0; epoch < epochs; ++epoch) {
    Fp epoch_loss = Fp(0);

    // Iterate over each batch
    for (size_t batch_idx = 0; batch_idx < kNumBatches; ++batch_idx) {
      // Determine batch start and end indices
      size_t start_idx = batch_idx * batch_size;
      size_t end_idx = std::min(start_idx + batch_size, kNumSamples);
      size_t current_batch_size = end_idx - start_idx;

      // Create batch matrices
      MatType rawDataBatch(current_batch_size, num_features);
      MatType rawLabelsBatch(current_batch_size, num_classes);

      // Copy data into the batch matrices
      for (size_t i = 0; i < current_batch_size; ++i) {
        size_t data_idx = start_idx + i;
        for (size_t j = 0; j < num_features; ++j) {
          rawDataBatch(i, j) = rawData(data_idx, j);
        }
        for (size_t j = 0; j < num_classes; ++j) {
          rawLabelsBatch(i, j) = rawLabels(data_idx, j);
        }
      }

      // Forward pass
      MatType predictions = Forward(rawDataBatch);

      // Compute loss and its gradient
      Fp loss = Loss::SoftmaxCrossEntropy<Fp>(predictions, rawLabelsBatch);
      MatType loss_grad =
          Loss::SoftmaxCrossEntropyGradient<Fp>(predictions, rawLabelsBatch);
      epoch_loss += loss;

      // Backward pass and weight update
      Backward(loss_grad);
      UpdateWeights();

      // Log progress every N batches
      if (batch_idx % 100 == 0) {
        LOG(INFO) << "Epoch [" << epoch + 1 << "/" << epochs << "], Batch ["
                  << batch_idx + 1 << "/" << kNumBatches << "], Loss: " << loss;
      }
    }  // End of batch loop

    float accuracy = EvaluateAccuracy(rawData, rawLabels);
    LOG(INFO) << "Epoch [" << epoch + 1 << "/" << epochs
              << "] completed. \nAverage Loss: " << (epoch_loss / kNumBatches)
              << "\nAccuracy: " << accuracy * 100.0f;
  }
}

// Evaluate the accuracy of the network on the provided dataset.
template <std::floating_point Fp>
float NeuralNetwork<Fp>::EvaluateAccuracy(const MatType& data,
                                          const MatType& labels) {
  // Get predictions from the network
  MatType predictions = Loss::Softmax(Forward(data));
  size_t correct = 0;
  size_t samples = predictions.Rows();
  // Compare predicted labels with true labels for each sample
  for (size_t i = 0; i < samples; ++i) {
    size_t predicted_label = predictions.ArgMaxRow(i);
    size_t true_label = labels.ArgMaxRow(i);
    if (predicted_label == true_label) {
      ++correct;
    }
  }
  return static_cast<float>(correct) / static_cast<float>(samples);
}
