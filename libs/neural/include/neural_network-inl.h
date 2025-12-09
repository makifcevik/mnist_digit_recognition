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
void NeuralNetwork<Fp>::Train(const MatType& raw_train_data,
                              const MatType& raw_train_labels,
                              const MatType& raw_test_data,
                              const MatType& raw_test_labels, uint32_t epochs,
                              uint32_t batch_size, EpochCallback on_epoch_end) {
  CHECK(raw_train_data.Rows() == raw_train_labels.Rows())
      << "Number of samples in data and labels must be the same.";

  const size_t kNumSamples = raw_train_data.Rows();
  const size_t kNumBatches =
      (kNumSamples + batch_size - 1) / batch_size;  // Ceiling division

  const size_t num_features = raw_train_data.Cols();
  const size_t num_classes = raw_train_labels.Cols();

  for (uint32_t epoch = 0; epoch < epochs; ++epoch) {
    Fp epoch_loss = Fp(0);

    // Shuffle data and labels at the start of each epoch
    MatType shuffled_data = raw_train_data.ShuffleRows(epoch + 42);
    MatType shuffled_labels = raw_train_labels.ShuffleRows(epoch + 42);

    // Iterate over each batch
    for (size_t batch_idx = 0; batch_idx < kNumBatches; ++batch_idx) {
      // Determine batch start and end indices
      size_t start_idx = batch_idx * batch_size;
      size_t end_idx = std::min(start_idx + batch_size, kNumSamples);
      size_t current_batch_size = end_idx - start_idx;

      // Create batch matrices
      MatType data_batch(current_batch_size, num_features);
      MatType labels_batch(current_batch_size, num_classes);

      // Copy data into the batch matrices
      for (size_t i = 0; i < current_batch_size; ++i) {
        size_t data_idx = start_idx + i;
        for (size_t j = 0; j < num_features; ++j) {
          data_batch(i, j) = shuffled_data(data_idx, j);
        }
        for (size_t j = 0; j < num_classes; ++j) {
          labels_batch(i, j) = shuffled_labels(data_idx, j);
        }
      }

      // Forward pass
      MatType predictions = Forward(data_batch);

      // Compute loss and its gradient
      Fp loss = Loss::SoftmaxCrossEntropy<Fp>(predictions, labels_batch);
      MatType loss_grad =
          Loss::SoftmaxCrossEntropyGradient<Fp>(predictions, labels_batch);
      epoch_loss += loss;

      // Backward pass and weight update
      (void)Backward(loss_grad);
      UpdateWeights();

      // Log progress every N batches
      if (batch_idx % 500 == 0) {
        LOG(INFO) << "Epoch [" << epoch + 1 << "/" << epochs << "], Batch ["
                  << batch_idx + 1 << "/" << kNumBatches << "], Loss: " << loss;
      }
    }  // End of batch loop

    float train_accuracy = EvaluateAccuracy(raw_train_data, raw_train_labels);
    float test_accuracy = EvaluateAccuracy(raw_test_data, raw_test_labels);

    LOG(INFO) << "Epoch [" << epoch + 1 << "/" << epochs << "] completed."
              << "\nAverage Loss: " << (epoch_loss / kNumBatches)
              << "\nTraining Accuracy: " << train_accuracy * 100.0f << "%"
              << "\nTesting Accuracy: " << test_accuracy * 100.0f << "%";

    // If the user provided a callback, fire it
    // on_epoch_end defaults to nullptr if the user didn't provide anything
    if (on_epoch_end)
        on_epoch_end(epoch, test_accuracy);

  } // End of epoch loop
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

// Serialization
template <std::floating_point Fp>
absl::Status NeuralNetwork<Fp>::Serialize(std::ostream& out) const {
  // Handle Type Fp
  DataType data_type = TypeToEnum<Fp>::value;
  out.write(reinterpret_cast<const char*>(&data_type), sizeof(data_type));

  // Number of layers
  uint32_t num_layers = static_cast<uint32_t>(layers_.size());
  out.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));

  // Layers
  for (const auto& layer : layers_) {
    // Layer ID
    LayerType layer_type = layer->Type();
    out.write(reinterpret_cast<const char*>(&layer_type), sizeof(layer_type));

    // Serialize layer
    absl::Status status = layer->Serialize(out);
    if (!status.ok())
      return status;
  }
  return absl::OkStatus();
}

template <std::floating_point Fp>
const NeuralNetwork<Fp>::Layers& NeuralNetwork<Fp>::GetLayers() const {
  return layers_;
}

template <std::floating_point Fp>
void NeuralNetwork<Fp>::Clear() {
  layers_.clear();
}
