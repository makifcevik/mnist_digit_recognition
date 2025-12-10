// MIT License

#ifndef MNIST_DIGIT_RECOGNITION_SRC_EXPERIMENT_CONFIG_H_
#define MNIST_DIGIT_RECOGNITION_SRC_EXPERIMENT_CONFIG_H_

#include <cstdint>
#include <filesystem>
#include <string>

// Contains configuration values ragarding experiment.
// Default values (values in this file) can be overriden in other files.
struct ExperimentConfig {

// Use the CMake injected path.
// If (for some reason) it is missing, fallback to "."
#ifdef PROJECT_ROOT_PATH
  std::filesystem::path base_path = PROJECT_ROOT_PATH;
#else
  std::string base_path = ".";
#endif

  // Model name
  std::string model_name = "new_model.bin";
  // Models path
  std::filesystem::path model_path = (base_path / "models");

  // Train dataset path
  std::string train_images_path =
      (base_path / "data" / "train-images.idx3-ubyte").string();
  std::string train_labels_path =
      (base_path / "data" / "train-labels.idx1-ubyte").string();
  // Test dataset path
  std::string test_images_path =
      (base_path / "data" / "t10k-images.idx3-ubyte").string();
  std::string test_labels_path =
      (base_path / "data" / "t10k-labels.idx1-ubyte").string();
  

  // Hyperparameters
  uint32_t epochs = 3;
  uint32_t batch_size = 24;
  float learning_rate = 0.01f;

  // Data processing
  float normalization_factor = 1.0f / 255.0f;
  size_t num_classes = 10;
};

#endif  // MNIST_DIGIT_RECOGNITION_SRC_EXPERIMENT_CONFIG_H_