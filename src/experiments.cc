// MIT License

#include "experiments.h"

#include <chrono>
#include <filesystem>
#include <iostream>

#include "linear_layer.h"
#include "matrix.h"
#include "mnist_loader.h"
#include "model_serializer.h"
#include "neural_network.h"
#include "relu_layer.h"

void RunTrainingMode(const ExperimentConfig& config) {
  using Fp = float;
  std::cout << "[EXPERIMENT] - TRAINING\n";

  // Load data
  std::cout << "[1/4] - Loading train & test dataset...\n";
  MNISTLoader loader;
  auto [raw_train_images_vec, raw_train_labels_vec] =
      loader.Load(config.train_images_path, config.train_labels_path);
  auto [raw_test_images_vec, raw_test_labels_vec] =
      loader.Load(config.test_images_path, config.test_labels_path);

  // Prepare data
  Matrix<uint8_t> raw_train_images(raw_train_images_vec,
                                   raw_train_images_vec.size() / 784, 784);
  Matrix<uint8_t> raw_train_labels(raw_train_labels_vec,
                                   raw_train_labels_vec.size(), 1);

  Matrix<uint8_t> raw_test_images(raw_test_images_vec,
                                  raw_test_images_vec.size() / 784, 784);
  Matrix<uint8_t> raw_test_labels(raw_test_labels_vec,
                                  raw_test_labels_vec.size(), 1);

  Matrix<Fp> x_train = raw_train_images.ToFloat(config.normalization_factor);
  Matrix<Fp> y_train =
      Matrix<Fp>::OneHotEncode(raw_train_labels.ToFloat(), config.num_classes);

  Matrix<Fp> x_test = raw_test_images.ToFloat(config.normalization_factor);
  Matrix<Fp> y_test =
      Matrix<Fp>::OneHotEncode(raw_test_labels.ToFloat(), config.num_classes);

  // Create network
  std::cout << "[2/4] - Constructing Network...\n";
  NeuralNetwork<Fp> network;
  network.AddLayer(std::make_unique<LinearLayer<Fp>>(784, 256, 0.01f, 42));
  network.AddLayer(std::make_unique<ReLULayer<Fp>>());
  network.AddLayer(std::make_unique<LinearLayer<Fp>>(256, 256, 0.01f, 43));
  network.AddLayer(std::make_unique<ReLULayer<Fp>>());
  network.AddLayer(std::make_unique<LinearLayer<Fp>>(256, 10, 0.01f, 44));

  // Save policy
  float best_accuracy = 0.0f;

  auto save_policy = [&network, &config, &best_accuracy](
                         uint32_t epoch, float current_accuracy) {
    if (current_accuracy > best_accuracy) {
      best_accuracy = current_accuracy;
      absl::Status status = ModelSerializer::Save(
          network, (config.model_path / config.model_name).string());
      if (status.ok()) {
        std::cout << "  [Checkpoint] New best model saved! Accuracy: "
                  << (best_accuracy * 100.0f) << "%" << std::endl;
      } else {
        std::cerr << "  [Error] Failed to save model: " << status.message()
                  << std::endl;
      }
    }
  };

  // Train
  std::cout << "[3/4] - Starting Training (" << config.epochs
            << " epochs)...\n";
  auto start = std::chrono::steady_clock::now();

  network.Train(x_train, y_train, x_test, y_test, config.epochs,
                config.batch_size, save_policy);

  auto end = std::chrono::steady_clock::now();
  auto seconds =
      std::chrono::duration_cast<std::chrono::seconds>(end - start).count();

  std::cout << "[4/4] - Training Complete in " << seconds << "s.\n";
  std::cout << "Best Accuracy Reached: " << (best_accuracy * 100.0f) << "%\n";
}

void RunInferenceMode(const ExperimentConfig& config) {
  using Fp = float;
  std::cout << "[EXPERIMENT] - INFERENCE\n";

  // Load data (only test data)
  std::cout << "[1/3] - Loading test dataset...\n";
  MNISTLoader loader;
  auto [raw_test_images_vec, raw_test_labels_vec] =
      loader.Load(config.test_images_path, config.test_labels_path);

  // Prepare data
  Matrix<uint8_t> raw_test_images(raw_test_images_vec,
                                  raw_test_images_vec.size() / 784, 784);
  Matrix<uint8_t> raw_test_labels(raw_test_labels_vec,
                                  raw_test_labels_vec.size(), 1);

  Matrix<Fp> x_test = raw_test_images.ToFloat(config.normalization_factor);
  Matrix<Fp> y_test =
      Matrix<Fp>::OneHotEncode(raw_test_labels.ToFloat(), config.num_classes);

  // Load the BEST model
  std::cout << "[2/3] - Loading Model from "
            << (config.model_path / config.model_name).string() << "...\n";
  auto result = ModelSerializer::Load<Fp>(
      (config.model_path / config.model_name).string());
  if (!result.ok()) {
    std::cerr << "[Fatal] - Failed to load model: " << result.status().message()
              << "\n";
    return;
  }
  NeuralNetwork network = std::move(*result);

  // Evaluate
  std::cout << "[3/3] - Evaluating...\n";
  float accuracy = network.EvaluateAccuracy(x_test, y_test);

  std::cout << "------------------------------------------\n";
  std::cout << "FINAL TEST ACCURACY: " << (accuracy * 100.0f) << "%\n";
  std::cout << "------------------------------------------\n";
}
