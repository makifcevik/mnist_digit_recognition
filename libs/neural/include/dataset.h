// MIT License

#ifndef MNIST_DIGIT_RECOGNITION_LIBS_NEURAL_DATASET_H_
#define MNIST_DIGIT_RECOGNITION_LIBS_NEURAL_DATASET_H_

#include <vector>

// Dataset structure to hold data and corresponding labels
// data is a flat vector representing multiple samples
// data: vector of input data (e.g., images)
// labels: vector of corresponding labels (e.g., digit classes)

struct Dataset {
  std::vector<uint8_t> data;
  std::vector<uint8_t> labels;
};

#endif  // !MNIST_DIGIT_RECOGNITION_LIBS_NEURAL_DATASET_H_
