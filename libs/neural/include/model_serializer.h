// MIT License

#ifndef MNIST_DIGIT_RECOGNITION_LIBS_NEURAL_MODEL_SERIALIZER_H_
#define MNIST_DIGIT_RECOGNITION_LIBS_NEURAL_MODEL_SERIALIZER_H_

#include <cstdint>

#include <fstream>
#include <string>

#include <absl/status/status.h>
#include <absl/status/statusor.h>

#include "neural_network.h"

class ModelSerializer {
 public:
  template <typename Fp>
  static absl::Status Save(const NeuralNetwork<Fp>& net,
                           const std::string& file_path);

  template <typename Fp>
  static absl::StatusOr<NeuralNetwork<Fp>> Load(const std::string& file_path);

 private:
  // Magic Number: "MNST" in ASCII
  static constexpr uint32_t kMagicNumber = 0x4D4E5354;

};  // namespace ModelSerializer

#include "model_serializer-inl.h"

#endif  // MNIST_DIGIT_RECOGNITION_LIBS_NEURAL_MODEL_SERIALIZER_H_
