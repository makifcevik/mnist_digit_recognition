// MIT License

#ifndef MNIST_DIGIT_RECOGNITION_LIBS_NEURAL_MATRIX_H
#define MNIST_DIGIT_RECOGNITION_LIBS_NEURAL_MATRIX_H

#include <concepts>
#include <vector>

template <typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

template <Numeric T>
class Matrix {
 public:
  // Constructors
  explicit Matrix(const std::vector<T>& data, size_t rows, size_t cols);
  explicit Matrix(size_t rows, size_t cols);

  // Destructor
  ~Matrix() = default;

  // Copy
  Matrix(const Matrix& other);
  Matrix& operator=(const Matrix& other);

  // Move
  Matrix(Matrix&& other) noexcept;
  Matrix& operator=(Matrix&& other) noexcept;

  // Static Random Matrix Generator
  static Matrix& Random(size_t rows, size_t cols, T min = T(0), T max = T(1),
                        unsigned int seed = 0);

  Matrix& GetTranspose() const;

  // Element Access
  T& operator()(size_t row, size_t col);
  const T& operator()(size_t row, size_t col) const;
  T& At(size_t row, size_t col);
  const T& At(size_t row, size_t col) const;

  // Basic Arithmetic Operations
  Matrix operator*(const Matrix& other) const;
  Matrix operator/(const Matrix& other) const;
  Matrix operator+(const Matrix& other) const;
  Matrix operator-(const Matrix& other) const;

  Matrix operator*=(const Matrix& other);
  Matrix operator/=(const Matrix& other);
  Matrix operator+=(const Matrix& other);
  Matrix operator-=(const Matrix& other);

  // Scalar Operations
  Matrix operator*(T scalar) const;
  Matrix operator/(T scalar) const;
  Matrix operator+(T scalar) const;
  Matrix operator-(T scalar) const;

  Matrix operator*=(T scalar);
  Matrix operator/=(T scalar);
  Matrix operator+=(T scalar);
  Matrix operator-=(T scalar);

  // Utility Functions
  std::vector<T>& ToVector() const noexcept;
  size_t Rows() const noexcept;
  size_t Cols() const noexcept;
  //void Print() const;

 private:
  std::vector<T> data_;
  size_t rows_;
  size_t cols_;
};

#endif  // MNIST_DIGIT_RECOGNITION_LIBS_NEURAL_MATRIX_H
