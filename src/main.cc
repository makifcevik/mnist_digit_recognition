// MIT License

#include <iostream>
#include <string>

#include "experiment_config.h"
#include "experiments.h"

int main() {
  // Setup Configuration: Defaults are defined in `experiment.h`
  ExperimentConfig config;

  // You can override defaults
  // e.g. config.epochs = 10;
  config.model_name = "best_mnist_model.bin";  // Already trained model

  // Select the mode to run
  // "train" trains a new model from scratch with the config values
  // "test" loads the pretrained model and evaluates it on the test dataset
  std::string mode = "test";

  if (mode == "train") {
    RunTrainingMode(config);
  } else if (mode == "test") {
    RunInferenceMode(config);
  } else {
    std::cerr << "Unknown mode: " << mode
              << "\nAvailable modes are: train | test\n";
    return 1;
  }

  return 0;
}