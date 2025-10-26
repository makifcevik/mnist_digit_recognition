#include "mnist_loader.h"

#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "dataset.h"

namespace {
// Helper function to write a 32-bit integer in big-endian format,
// Which is what the MNIST files use.
void WriteBigEndianInt32(std::ofstream& file, int32_t value) {
  uint8_t bytes[4];
  bytes[0] = (value >> 24) & 0xFF;
  bytes[1] = (value >> 16) & 0xFF;
  bytes[2] = (value >> 8) & 0xFF;
  bytes[3] = value & 0xFF;
  file.write(reinterpret_cast<char*>(bytes), 4);
}
}  // namespace

class MNISTLoaderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Get temporary directory
    test_dir_ = testing::TempDir();
    fake_images_path_ = test_dir_ + "/fake_images.idx3-ubyte";
    fake_labels_path_ = test_dir_ + "/fake_labels.idx1-ubyte";

    // Define fake data
    // Image one is 784 pixels of 0xAA
    // Image two is 784 pixels of 0xCC
    constexpr int32_t kImageSize = 28 * 28;
    expected_images_.resize(2 * kImageSize);
    std::fill(expected_images_.begin(), expected_images_.begin() + kImageSize,
              0xAA);
    std::fill(expected_images_.begin() + kImageSize, expected_images_.end(),
              0xCC);
    // Labels
    expected_labels_ = {1, 3};

    // Create fake image file
    std::ofstream image_file(fake_images_path_, std::ios::binary);
    ASSERT_TRUE(image_file.is_open());
    WriteBigEndianInt32(image_file, 0x00000803);  // Image magic number
    WriteBigEndianInt32(image_file, 2);           // Number of images
    WriteBigEndianInt32(image_file, 28);          // Number of rows
    WriteBigEndianInt32(image_file, 28);          // Number of cols
    image_file.write(reinterpret_cast<char*>(expected_images_.data()),
                     expected_images_.size());
    image_file.close();

    // Create fake label file
    std::ofstream label_file(fake_labels_path_, std::ios::binary);
    ASSERT_TRUE(label_file.is_open());
    WriteBigEndianInt32(label_file, 0x00000801);  // Label magic number
    WriteBigEndianInt32(label_file, 2);           // Number of labels
    label_file.write(reinterpret_cast<char*>(expected_labels_.data()),
                     expected_labels_.size());
    label_file.close();
  }

  std::string test_dir_;
  std::string fake_images_path_;
  std::string fake_labels_path_;
  std::vector<uint8_t> expected_images_;
  std::vector<uint8_t> expected_labels_;
};

TEST_F(MNISTLoaderTest, Load_SuccessfullyLoadsValidFiles) {
  // Arrange
  MNISTLoader loader;
  // Act
  Dataset dataset{loader.Load(fake_images_path_, fake_labels_path_)};
  // Assert
  EXPECT_EQ(dataset.data, expected_images_);
  EXPECT_EQ(dataset.labels, expected_labels_);
}

TEST_F(MNISTLoaderTest, Load_FailsOnInvalidImagePath) {
  // Arrange
  MNISTLoader loader;
  auto bad_path = test_dir_ + "/non_existent_file";
  // Act & Assert
  EXPECT_DEATH(loader.Load(bad_path, fake_labels_path_), "Could not open file");
}

TEST_F(MNISTLoaderTest, Load_FailsOnInvalidLabelPath) {
  // Arrange
  MNISTLoader loader;
  auto bad_path = test_dir_ + "/non_existent_file";
  // Act & Assert
  EXPECT_DEATH(loader.Load(fake_images_path_, bad_path), "Could not open file");
}

TEST_F(MNISTLoaderTest, Load_FailsOnInvalidImageMagicNumber) {
  // Arrange
  MNISTLoader loader;
  std::ofstream bad_image_file(fake_images_path_,
                               std::ios::in | std::ios::out | std::ios::binary);
  bad_image_file.seekp(0, std::ios::beg);  // Only overwrite the magic number
  WriteBigEndianInt32(
      bad_image_file,
      0x12345678);  // The correct magic number was: 0x00'00'08'03
  bad_image_file.close();
  // Act & Assert
  EXPECT_DEATH(loader.Load(fake_images_path_, fake_labels_path_),
               "Invalid magic number in image file");
}

TEST_F(MNISTLoaderTest, Load_FailsOnInvalidLabelMagicNumber) {
  // Arrange
  MNISTLoader loader;
  std::ofstream bad_label_file(fake_labels_path_,
                               std::ios::in | std::ios::out | std::ios::binary);
  bad_label_file.seekp(0, std::ios::beg);  // Only overwrite the magic number
  WriteBigEndianInt32(
      bad_label_file,
      0x87654321);  // The correct magic number was: 0x00'00'08'01
  bad_label_file.close();
  // Act & Assert
  EXPECT_DEATH(loader.Load(fake_images_path_, fake_labels_path_),
               "Invalid magic number in label file");
}

TEST_F(MNISTLoaderTest, Load_FailsOnInvalidNumberOfImages) {
  // Arrange
  MNISTLoader loader;
  std::ofstream bad_image_file(fake_images_path_,
                               std::ios::in | std::ios::out | std::ios::binary);
  bad_image_file.seekp(4, std::ios::beg);   // Overwrite number of images
  WriteBigEndianInt32(bad_image_file, -1);  // Invalid number of images
  bad_image_file.close();
  // Act & Assert
  EXPECT_DEATH(loader.Load(fake_images_path_, fake_labels_path_),
               "Invalid number of images in file");
}

TEST_F(MNISTLoaderTest, Load_FailsOnInvalidImageDimensions) {
  // Arrange
  MNISTLoader loader;
  std::ofstream bad_image_file(fake_images_path_,
                               std::ios::in | std::ios::out | std::ios::binary);
  bad_image_file.seekp(8, std::ios::beg);  // Overwrite number of rows
  WriteBigEndianInt32(bad_image_file,
                      30);  // Invalid number of rows (should be 28)
  WriteBigEndianInt32(bad_image_file,
                      15);  // Invalid number of cols (should be 28)
  bad_image_file.close();
  // Act & Assert
  EXPECT_DEATH(loader.Load(fake_images_path_, fake_labels_path_),
               "Unexpected image dimensions in file");
}

TEST_F(MNISTLoaderTest, Load_FailsOnMismatchedImageAndLabelCount) {
  // Arrange
  MNISTLoader loader;
  std::ofstream bad_label_file(fake_labels_path_,
                               std::ios::in | std::ios::out | std::ios::binary);
  bad_label_file.seekp(4, std::ios::beg);
  WriteBigEndianInt32(bad_label_file, 3);  // Should be 2
  bad_label_file.close();
  // Act & Assert
  EXPECT_DEATH(loader.Load(fake_images_path_, fake_labels_path_),
               "Number of images and labels do not match");
}
