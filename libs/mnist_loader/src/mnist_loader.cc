// MIT License

#include "mnist_loader.h"

#include <absl/log/check.h>
#include <absl/log/log.h>

MNISTLoader::MNISTLoader() {
  SetSystemEndianness();
}

int32_t MNISTLoader::ReverseInt(int32_t i) const {
  uint8_t byte_1 = i & 0xFF;
  uint8_t byte_2 = (i >> 8) & 0xFF;
  uint8_t byte_3 = (i >> 16) & 0xFF;
  uint8_t byte_4 = (i >> 24) & 0xFF;

  return (byte_1 << 24) | (byte_2 << 16) | (byte_3 << 8) | byte_4;
}

int32_t MNISTLoader::ReadInt32(std::ifstream& file) const {
  int32_t num;
  file.read(reinterpret_cast<char*>(&num), sizeof(num));
  if (system_endianness_ == Endianness::kLittle)
    return ReverseInt(num);
  return num;
}

std::vector<uint8_t> MNISTLoader::ReadImages(const std::string& path) const {
  std::ifstream file(path, std::ios::binary);
  CHECK(file.is_open()) << "Could not open file: " << path;
  LOG(INFO) << "Reading images from: " << path;

  // ReadInt32() already checks for endianness and reverses if necessary
  CHECK(ReadInt32(file) == kImageMagicNumber)
      << "Invalid magic number in image file: " << path;

  const int32_t num_images = ReadInt32(file);
  const int32_t num_rows = ReadInt32(file);
  const int32_t num_cols = ReadInt32(file);

  CHECK(num_images > 0) << "Invalid number of images in file: " << path;

  // MNIST images are always 28x28 pixels
  CHECK(num_rows == 28 && num_cols == 28)
      << "Unexpected image dimensions in file: " << path;

  // Read image data
  std::vector<uint8_t> images(num_images * num_rows * num_cols);
  file.read(reinterpret_cast<char*>(images.data()), images.size());
  LOG(INFO) << "Completed reading images from: " << path;
  return images;
}

std::vector<uint8_t> MNISTLoader::ReadLabels(const std::string& path) const {
  std::ifstream file(path, std::ios::binary);
  CHECK(file.is_open()) << "Could not open file: " << path;
  LOG(INFO) << "Reading labels from: " << path;

  // ReadInt32() already checks for endianness and reverses if necessary
  CHECK(ReadInt32(file) == kLabelMagicNumber)
      << "Invalid magic number in label file: " << path;

  // Read label data
  const int32_t num_labels = ReadInt32(file);
  std::vector<uint8_t> labels(num_labels);
  file.read(reinterpret_cast<char*>(labels.data()), labels.size());
  LOG(INFO) << "Completed reading labels from: " << path;
  return labels;
}

Dataset MNISTLoader::Load(const std::string& image_file_path,
                          const std::string& label_file_path) const {
  Dataset dataset;
  dataset.data = ReadImages(image_file_path);
  dataset.labels = ReadLabels(label_file_path);
  LOG(INFO) << "MNIST dataset loaded successfully.";
  return dataset;
}

void MNISTLoader::SetSystemEndianness() noexcept {
  uint16_t test = 0x0001;
  uint8_t* byte = reinterpret_cast<uint8_t*>(&test);
  if (byte[0] == 0)
    system_endianness_ = Endianness::kBig;
  else
    system_endianness_ = Endianness::kLittle;
}