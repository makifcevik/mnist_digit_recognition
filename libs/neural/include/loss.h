// MIT License

#ifndef MNIST_DIGIT_RECOGNITON_LIBS_NEURAL_LOSS_H_
#define MNIST_DIGIT_RECOGNITON_LIBS_NEURAL_LOSS_H_

#include <cmath>

#include "matrix.h"

namespace Loss {

// Turns one row of logits into probabilities using the softmax function.
template <std::floating_point Fp>
Matrix<Fp> Softmax(const Matrix<Fp>& logits) {
  // Create a copy of logits to store probabilities
  Matrix<Fp> probabilities = logits;
  // Row by row softmax computation
  for (size_t r = 0; r < logits.Rows(); ++r) {
    // Compute sum of exponentials for the current row (denominator)
    Fp sum_of_exponentials = Fp(0);
    for (size_t c = 0; c < logits.Cols(); ++c) {
      sum_of_exponentials += std::exp(logits(r, c));
    }
    // Compute probabilities for the current row
    for (size_t c = 0; c < logits.Cols(); ++c) {
      probabilities(r, c) = std::exp(logits(r, c)) / sum_of_exponentials;
    }
  }
  return probabilities;
}

// Computes the softmax cross-entropy loss between logits and true labels.
template <std::floating_point Fp>
Fp SoftmaxCrossEntropy(const Matrix<Fp>& logits,
                       const Matrix<Fp>& true_labels) {
  // Get the probabilities via softmax
  Matrix<Fp> probabilities = Softmax(logits);

  // Compute the cross-entropy loss
  Fp total_loss = Fp(0);
  for (size_t r = 0; r < logits.Rows(); ++r) {
    for (size_t c = 0; c < logits.Cols(); ++c) {
      // Find the correct class and accumulate loss
      if (true_labels(r, c) == Fp(1)) {
        total_loss -= std::log(probabilities(r, c) + 1e-15);  // Avoid log(0)
        break;
      }
    }
  }
  return total_loss / static_cast<Fp>(logits.Rows());  // Average loss
}

// Computes the gradient of the softmax cross-entropy loss w.r.t. logits.
template <std::floating_point Fp>
Matrix<Fp> SoftmaxCrossEntropyGradient(const Matrix<Fp>& logits,
                                       const Matrix<Fp>& true_labels) {
  // Get the probabilities via softmax
  Matrix<Fp> probabilities = Softmax(logits);

  // Gradient is probabilities - true_labels (Copy probabilities to avoid modifying input)
  Matrix<Fp> gradient = probabilities;

  for (size_t r = 0; r < logits.Rows(); ++r) {
    for (size_t c = 0; c < logits.Cols(); ++c) {
      gradient(r, c) -= true_labels(r, c);  // Subtract true label
      gradient(r, c) /= static_cast<Fp>(logits.Rows());  // Average
    }
  }

  return gradient;
}

}  // namespace Loss

#endif  // MNIST_DIGIT_RECOGNITON_LIBS_NEURAL_LOSS_H_