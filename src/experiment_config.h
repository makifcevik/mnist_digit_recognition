// MIT License

#ifndef MNIST_DIGIT_RECOGNITION_SRC_EXPERIMENT_CONFIG_H_
#define MNIST_DIGIT_RECOGNITION_SRC_EXPERIMENT_CONFIG_H_

#include <cstdint>
#include <filesystem>
#include <string>

// Helper function to handle both hypen (Offical) and dot (Kaggle) file naming conventions.
inline std::string resolve_mnist_path(const std::filesystem::path& base,
                                      const std::filesystem::path& hypen_name,
                                      const std::filesystem::path& dot_name) {
  std::filesystem::path hypen_path = base / "data" / hypen_name;
  std::filesystem::path dot_path = base / "data" / dot_name;
  // Prefer dot if it exists
  if (std::filesystem::exists(dot_path))
    return dot_path.string();
  else
    return hypen_path.string();
}

// Contains configuration values ragarding experiment.
// Default values (values in this file) can be overriden in other files.
struct ExperimentConfig {

// Use the CMake injected path.
// If (for some reason) it is missing, fallback to "."
#ifdef PROJECT_ROOT_PATH
  std::filesystem::path base_path = PROJECT_ROOT_PATH;
#else
  std::filesystem::path base_path = ".";
#endif

  // Model name
  std::string model_name = "new_model.bin";
  // Models path
  std::filesystem::path model_path = (base_path / "models");

  // Train dataset path
  std::string train_images_path = resolve_mnist_path(
      base_path, "train-images-idx3-ubyte", "train-images.idx3-ubyte");
  std::string train_labels_path = resolve_mnist_path(
      base_path, "train-labels-idx1-ubyte", "train-labels.idx1-ubyte");

  // Test dataset path
  std::string test_images_path = resolve_mnist_path(
      base_path, "t10k-images-idx3-ubyte", "t10k-images.idx3-ubyte");
  std::string test_labels_path = resolve_mnist_path(
      base_path, "t10k-labels-idx1-ubyte", "t10k-labels.idx1-ubyte");

  // Hyperparameters
  uint32_t epochs = 3;
  uint32_t batch_size = 24;
  float learning_rate = 0.01f;

  // Data processing
  float normalization_factor = 1.0f / 255.0f;
  size_t num_classes = 10;
};

#endif  // MNIST_DIGIT_RECOGNITION_SRC_EXPERIMENT_CONFIG_H_