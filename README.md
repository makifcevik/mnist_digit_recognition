# MNIST Digit Recognition — From-Scratch Neural Network in C++20

![Language](https://img.shields.io/badge/language-C%2B%2B20-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)

A fully-custom, end-to-end neural network implementation in Modern C++ (C++20) following **Google C++ Standards**. This project features a handwritten matrix engine, multithreaded operations, backpropagation from scratch, and a full training pipeline capable of learning the MNIST handwritten digit dataset.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Tech Stack & Engineering](#tech-stack--engineering)
- [Performance Optimizations](#performance-optimizations)
- [Getting Started](#getting-started)
- [Dataset](#dataset)
- [Lessons Learned](#lessons-learned)
- [License](#license)

## Overview

This project is a complete neural network engine built entirely from scratch without any machine learning frameworks (TensorFlow, PyTorch) or external linear algebra libraries (Eigen, BLAS). 

Everything, from memory management and matrix multiplication to Softmax and Backpropagation, is manually implemented. The goal was to bridge the gap between theoretical deep learning concepts and high-performance C++ systems engineering.

## Key Features

### 1. Custom Linear Algebra Engine
* **Efficient Storage:** Row-major 1D contiguous memory allocation for cache locality.
* **Arithmetic:** Full support for scalar, vector, and matrix operations including Broadcasting.
* **Concurrency:** Multithreaded matrix multiplication using `std::jthread`.
* **Manipulation:** Optimized row shuffling, slicing, and transposition.

### 2. Neural Network Architecture
* **Layers:** Fully connected linear (Dense) layers.
* **Activation:** ReLU (Rectified Linear Unit) for hidden layers.
* **Output:** Softmax activation for probability distribution.
* **Loss Function:** Categorical Cross-Entropy with full gradient implementation.

### 3. Training Pipeline
* **Mini-batch Gradient Descent:** Implemented with epoch-based shuffling.
* **Metrics:** Real-time logging of Training vs. Testing accuracy/loss.
* **Inference:** Fast forward-pass evaluation for testing.

## Tech Stack & Engineering

This project adheres to **Google C++ Style** principles and Modern C++ practices:

* **C++20:** Utilizes modern features (concepts, auto, etc.).
* **Abseil (absl):** Used for robust error handling (Check()) and logging, strictly avoiding C++ exceptions.
* **Google Test (GTest):** Comprehensive unit testing for the matrix engine and network components.
* **CMake:** Professional build system handling dependencies and cross-platform compilation.

## Performance Optimizations

1.  **Multithreaded Matrix Multiplication:** The engine analyzes matrix dimensions and dynamically dispatches threads to parallelize dot products, significantly reducing training time on large matrices.
    
2.  **Cache-Friendly Memory Layout:** Data is stored in flattened 1D arrays. During multiplication, the right-hand matrix is transposed to access memory sequentially, minimizing cache misses.

3.  **Batching:** Training is performed in batches rather than single-item updates.

## Getting Started

### Prerequisites
* CMake (3.20+)
* C++ Compiler supporting C++20 (GCC, Clang, or MSVC)

### Build
```bash
git clone [https://github.com/makifcevik/mnist_digit_recognition.git](https://github.com/makifcevik/mnist_digit_recognition.git)
cd mnist_digit_recognition

mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

### Run
To train the model:
```bash
./mnist_digit_recognition
```

## Dataset

The project uses MNIST, consisting of 60,000 training images and 10,000 test images.

- Input: 28x28 grayscale images (flattened to 784 features).
- Output: 10 classes (Digits 0-9).

Note: The dataset loader expects the standard binary (`.idx1` & `.idx3`) format (e.g from [Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)).

## Lessons Learned

- Math: Implementing a matrix class and a neural network from scratch clarified the math behind neural networks.
- Machine Learning: Building a neural network without external libraries deepened my understanding of Machine Learning. 
- Concurrency Cost: I learned that threading isn't "free." Managing thread overhead vs. workload size was crucial for actual speedups.
- Memory Matters: Cache locality optimizations (transposition before multiplication) resulted in an 8x performance gain, highlighting the importance of hardware-aware programming.
- Tooling: Integrating GTest and Abseil taught me how to structure a project for maintainability, not just functionality.

## License

Distributed under the **MIT License**.
