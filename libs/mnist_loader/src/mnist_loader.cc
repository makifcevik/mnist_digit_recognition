// MIT License

#include "mnist_loader.h"

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
  if (!file.is_open()) {
    // Handle error: could not open file
  }

  // ReadInt32() already checks for endianness and reverses if necessary
  if (ReadInt32(file) != kImageMagicNumber) {
    // Handle error: invalid magic number
  }

  const int32_t num_images = ReadInt32(file);
  const int32_t num_rows = ReadInt32(file);
  const int32_t num_cols = ReadInt32(file);

  // MNIST images are always 28x28 pixels
  if (num_rows != 28 || num_cols != 28) {
    // Handle error: unexpected image dimensions
  }

  // Read image data
  std::vector<uint8_t> images(num_images * num_rows * num_cols);
  file.read(reinterpret_cast<char*>(images.data()), images.size());
  return images;
}

std::vector<uint8_t> MNISTLoader::ReadLabels(const std::string& path) const {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    // Handle error: could not open file
  }

  // ReadInt32() already checks for endianness and reverses if necessary
  if (ReadInt32(file) != kLabelMagicNumber) {
    // Handle error: invalid magic number
  }

  // Read label data
  const int32_t num_labels = ReadInt32(file);
  std::vector<uint8_t> labels(num_labels);
  file.read(reinterpret_cast<char*>(labels.data()), labels.size());
  return labels;
}

Dataset MNISTLoader::Load(const std::string& image_file_path,
                          const std::string& label_file_path) const {
  Dataset dataset;
  dataset.data = ReadImages(image_file_path);
  dataset.labels = ReadLabels(label_file_path);
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