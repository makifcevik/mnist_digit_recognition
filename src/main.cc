// MIT License

#include <cstdint>

#include <chrono>
#include <iostream>

#include "linear_layer.h"
#include "loss.h"
#include "matrix.h"
#include "mnist_loader.h"
#include "model_serializer.h"
#include "neural_network.h"
#include "relu_layer.h"

static void RunExperiment();

int main() {
  RunExperiment();
  return 0;
}

static void RunExperiment() {
  using Fp = float;
  const size_t kNumClasses = 10;
  const float kNormalizationFactor = 1.0f / 255.0f;
  const uint32_t kNumEpochs = 30;
  const uint32_t kBatchSize = 24;

  MNISTLoader loader;

  // Load MNIST data
  auto [raw_train_img_vec, raw_train_label_vec] = loader.Load(
      "data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
  auto [raw_test_img_vec, raw_test_label_vec] =
      loader.Load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte");

  // Convert dataset to Matrix format
  /*Matrix<uint8_t> raw_train_images(raw_train_img_vec,
                                   raw_train_img_vec.size() / 784, 784);
  Matrix<uint8_t> raw_train_labels(raw_train_label_vec,
                                   raw_train_label_vec.size(), 1);*/

  Matrix<uint8_t> raw_test_images(raw_test_img_vec,
                                  raw_test_img_vec.size() / 784, 784);
  Matrix<uint8_t> raw_test_labels(raw_test_label_vec, raw_test_label_vec.size(),
                                  1);

  /*Matrix<Fp> x_train = raw_train_images.ToFloat(kNormalizationFactor);
  Matrix<Fp> y_train =
      Matrix<Fp>::OneHotEncode(raw_train_labels.ToFloat(), kNumClasses);*/

  Matrix<Fp> x_test = raw_test_images.ToFloat(kNormalizationFactor);
  Matrix<Fp> y_test =
      Matrix<Fp>::OneHotEncode(raw_test_labels.ToFloat(), kNumClasses);

  // Create neural network
  NeuralNetwork<Fp> network;
  const std::string kModelPath = "models/best_mnist_model.bin";
  float best_accuracy = 0.0f;

  auto save_policy = [&network, &kModelPath, &best_accuracy](
                         uint32_t epoch, float current_accuracy) {
    if (current_accuracy > best_accuracy) {
      best_accuracy = current_accuracy;
      absl::Status status = ModelSerializer::Save(network, kModelPath);
      if (status.ok()) {
        std::cout << "  [Checkpoint] New best model saved! Accuracy: "
                  << (best_accuracy * 100.0f) << "%" << std::endl;
      } else {
        std::cerr << "  [Error] Failed to save model: " << status.message()
                  << std::endl;
      }
    }
  };

  auto result = ModelSerializer::Load<Fp>(kModelPath);
  if (!result.ok())
    abort();
  network = std::move(*result);
  /*network.AddLayer(std::make_unique<LinearLayer<Fp>>(784, 256, 0.01f, 42));
  network.AddLayer(std::make_unique<ReLULayer<Fp>>());
  network.AddLayer(std::make_unique<LinearLayer<Fp>>(256, 256, 0.01f, 43));
  network.AddLayer(std::make_unique<ReLULayer<Fp>>());
  network.AddLayer(std::make_unique<LinearLayer<Fp>>(256, 10, 0.01f, 44));*/

  /*float initial_training_accuracy = network.EvaluateAccuracy(x_train, y_train);
  std::cout << "Initial Training Accuracy: "
            << initial_training_accuracy * 100.0f << "%" << std::endl;

  float initial_testing_accuracy = network.EvaluateAccuracy(x_test, y_test);
  std::cout << "Initial Testing Accuracy: " << initial_testing_accuracy * 100.0f
            << "%" << std::endl;*/

  //// Train the network
  //std::cout << "Training started..." << std::endl;
  //std::chrono::steady_clock::time_point start_time =
  //    std::chrono::steady_clock::now();

  //network.Train(x_train, y_train, x_test, y_test, kNumEpochs, kBatchSize, save_policy);

  //std::chrono::steady_clock::time_point end_time =
  //    std::chrono::steady_clock::now();
  //std::cout << "Training completed." << std::endl;
  //std::cout << "Time taken for " << kNumEpochs << " epoch(s): "
  //          << std::chrono::duration_cast<std::chrono::seconds>(end_time -
  //                                                              start_time)
  //                 .count()
  //          << " seconds." << std::endl;

  float test_acc = network.EvaluateAccuracy(x_test, y_test);
  std::cout << "\nTesting accuracy of loaded model [" << kModelPath
            << "]: " << test_acc * 100.0f << "\n";
}
