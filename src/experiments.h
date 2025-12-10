// MIT License

#ifndef MNIST_DIGIT_RECOGNITION_SRC_EXPERIMENTS_H
#define MNIST_DIGIT_RECOGNITION_SRC_EXPERIMENTS_H

#include "experiment_config.h"

// Trains a new model from scratch and saves the best version
// depending on the testing accuracy.
void RunTrainingMode(const ExperimentConfig& config);

// Loads an existing model and evaluates it on the test set.
void RunInferenceMode(const ExperimentConfig& config);

#endif  // MNIST_DIGIT_RECOGNITION_SRC_EXPERIMENTS_H
