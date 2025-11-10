// MIT License

#include <cstdint>

#include <iostream>
#include <chrono>

#include "linear_layer.h"
#include "loss.h"
#include "matrix.h"
#include "mnist_loader.h"
#include "neural_network.h"

static void RunExperiment() {
  using Fp = float;
  const size_t kNumClasses = 10;
  const float kNormalizationFactor = 1.0f / 255.0f;
  const int kNumEpochs = 1;

  MNISTLoader loader;

  // Load MNIST data
  auto [raw_train_img_vec, raw_train_label_vec] = loader.Load(
      "data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");

  // Convert dataset to Matrix format
  Matrix<uint8_t> raw_train_images(raw_train_img_vec,
                                   raw_train_img_vec.size() / 784, 784);
  Matrix<uint8_t> raw_train_labels(raw_train_label_vec,
                                   raw_train_label_vec.size(), 1);

  Matrix<Fp> x_train = raw_train_images.ToFloat(kNormalizationFactor);
  Matrix<Fp> y_train =
      Matrix<Fp>::OneHotEncode(raw_train_labels.ToFloat(), kNumClasses);

  // Create neural network
  NeuralNetwork<Fp> network;
  network.AddLayer(std::make_unique<LinearLayer<Fp>>(784, 128, 0.01f, 42));
  network.AddLayer(std::make_unique<LinearLayer<Fp>>(128, 10, 0.01f, 43));

  // Train the network
  std::cout << "Training started..." << std::endl;
  std::chrono::steady_clock::time_point start_time =
      std::chrono::steady_clock::now();
  network.Train(x_train, y_train, kNumEpochs);
  std::chrono::steady_clock::time_point end_time =
      std::chrono::steady_clock::now();
  std::cout << "Training completed." << std::endl;
  std::cout << "Time taken for " << kNumEpochs << " epoch(s): "
            << std::chrono::duration_cast<std::chrono::seconds>(
                   end_time - start_time)
                   .count()
            << " seconds." << std::endl;
}

int main() {
  RunExperiment();
  return 0;
}
