// MIT License

#ifndef MNIST_DIGIT_RECOGNITION_LIBS_MNIST_LOADER_MNIST_LOADER_H_
#define MNIST_DIGIT_RECOGNITION_LIBS_MNIST_LOADER_MNIST_LOADER_H_

#include <fstream>
#include <string>
#include <vector>

#include "dataset.h"

// MNISTLoader class to load the MNIST dataset from binary files.
// It handles endianness and reads image and label data into a Dataset structure.
// Usage: Use the Load method with paths to the image and label files.
class MNISTLoader {
 public:
  // Default constructor sets the system endianness.
  MNISTLoader();
  // Loads the MNIST dataset from the specified image and label file paths.
  // Returns a Dataset structure containing the images and labels.
  Dataset Load(const std::string& image_file_path,
               const std::string& label_file_path) const;

 private:
  int32_t ReverseInt(int32_t i) const;
  int32_t ReadInt32(std::ifstream& file) const;
  void SetSystemEndianness() noexcept;
  std::vector<uint8_t> ReadImages(const std::string& path) const;
  std::vector<uint8_t> ReadLabels(const std::string& path) const;

  enum class Endianness { kLittle, kBig };
  Endianness system_endianness_;
  static constexpr int32_t kLabelMagicNumber = 0x00000801;  // 2049
  static constexpr int32_t kImageMagicNumber = 0x00000803;  // 2051
};

#endif  // MNIST_DIGIT_RECOGNITION_LIBS_MNIST_LOADER_MNIST_LOADER_H_
