// MIT License

// This file contains the implementation of the ModelSerializer class template.
// It is meant to be included only inside model_serializer.h.

#include <memory>

#include <absl/strings/str_cat.h>
#include <absl/status/status.h>
#include <absl/status/statusor.h>

#include "common_types.h"
#include "layer_type.h"
#include "type_traits.h"
#include "linear_layer.h"
#include "relu_layer.h"

template <typename Fp>
absl::Status ModelSerializer::Save(const NeuralNetwork<Fp>& net,
                                   const std::string& file_path) {
  std::ofstream out(file_path, std::ios::binary);
  if (!out.is_open())
    return absl::UnavailableError(
        absl::StrCat("Could not open file: ", file_path));

  // Write the magic number
  out.write(reinterpret_cast<const char*>(&kMagicNumber), sizeof(kMagicNumber));

  // NeuralNetwork<Fp>::Serialize(out) returns absl::Status
  return net.Serialize(out);
}

template <typename Fp>
absl::StatusOr<NeuralNetwork<Fp>> ModelSerializer::Load(
    const std::string& file_path) {
  std::ifstream in(file_path, std::ios::binary);
  if (!in.is_open())
    return absl::UnavailableError(
        absl::StrCat("Could not open file: ", file_path));

  // Read the magic number
  uint32_t magic;
  in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
  if (magic != kMagicNumber)
    return absl::InvalidArgumentError("File is not a valid model.");

  // Read the data type
  DataType data_type;
  DataType expected_data_type = TypeToEnum<Fp>::value;
  in.read(reinterpret_cast<char*>(&data_type), sizeof(data_type));
  if (data_type != expected_data_type)
    return absl::InvalidArgumentError("Model data mismatch.");

  // Create the network
  NeuralNetwork<Fp> network;

  // Read the layer count
  uint32_t num_layers;
  in.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));

  // Reconstruct layers
  for (uint32_t i = 0; i < num_layers; ++i) {
    LayerType layer_type;
    in.read(reinterpret_cast<char*>(&layer_type), sizeof(layer_type));

    // Factory logic
    std::unique_ptr<NeuralLayer<Fp>> layer;
    switch (layer_type) {
      case LayerType::kLinear:
        layer = std::make_unique<LinearLayer<Fp>>();
        break;
      case LayerType::kReLU:
        layer = std::make_unique<ReLULayer<Fp>>();
        break;
      default:
        return absl::UnimplementedError(
            "Encountered unknown layer type id in file");
    }

    // Deserialize the specific layer
    absl::Status status = layer->Deserialize(in);
    if (!status.ok())
      return status;

    // Add to network
    network.AddLayer(std::move(layer));
  }

  return network;
}