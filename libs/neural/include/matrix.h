// MIT License

// Defines a template Matrix class for 2D data storage.
// This class manages memory and provides basic element access.
// Multithreaded matrix multiplication is supported.

#ifndef MNIST_DIGIT_RECOGNITION_LIBS_NEURAL_MATRIX_H_
#define MNIST_DIGIT_RECOGNITION_LIBS_NEURAL_MATRIX_H_

#include <concepts>
#include <vector>

template <typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

template <Numeric T>
class [[nodiscard]] Matrix {
 public:
  // Constructors
  Matrix() = delete;
  explicit Matrix(const std::vector<T>& data, size_t rows, size_t cols);
  explicit Matrix(size_t rows, size_t cols);

  // Destructor
  ~Matrix() = default;

  // Copy
  Matrix(const Matrix& other) = default;
  Matrix& operator=(const Matrix& other) = default;

  // Move
  Matrix(Matrix&& other) noexcept = default;
  Matrix& operator=(Matrix&& other) noexcept = default;

  // Static Random Matrix Generator
  static Matrix Random(size_t rows, size_t cols, T min, T max, uint32_t seed);

  Matrix GetTranspose() const;

  // Collapse Functions
  // Sums over rows or columns, returning a single-row or single-column matrix
  Matrix CollapseRows() const;  // Returns a row matrix
  Matrix CollapseCols() const;  // Returns a column matrix

  // Element Access
  T& operator()(size_t row, size_t col) noexcept;
  const T& operator()(size_t row, size_t col) const noexcept;

  T& At(size_t row, size_t col);
  const T& At(size_t row, size_t col) const;

  // Basic Arithmetic Operations
  // Operator Overloading for mathematical operations is prohibited in Google style,
  // but since this is a pure mathematical construct, we will make an exception here.
  Matrix operator*(
      const Matrix& other) const;  // Uses concurrent multiplication
  Matrix operator+(const Matrix& other) const;
  Matrix operator-(const Matrix& other) const;

  Matrix& operator*=(const Matrix& other);  // Uses concurrent multiplication
  Matrix& operator+=(const Matrix& other);
  Matrix& operator-=(const Matrix& other);

  // Scalar Operations
  Matrix& operator*=(T scalar);
  Matrix& operator/=(T scalar);
  Matrix& operator+=(T scalar);
  Matrix& operator-=(T scalar);

  // Utility Functions
  const std::vector<T>& ToVector() const noexcept;
  std::vector<T>& ToVector() noexcept;
  Matrix<float> ToFloat(float scale = 1.0f) const;
  Matrix<double> ToDouble(double scale = 1.0) const;

  // Broadcasts the matrix along rows to match new_rows
  Matrix<T> BroadcastRows(size_t new_rows) const;

  Matrix<T> ShuffleRows(uint32_t seed) const;

  // Returns the index of the maximum or maximum element in the specified row / col
  size_t ArgMaxRow(size_t row) const;
  size_t ArgMinRow(size_t row) const;
  size_t ArgMaxCol(size_t col) const;
  size_t ArgMinCol(size_t col) const;

  // Returns the value of the maximum or minimum element in the specified row / col
  T MaxInRow(size_t row) const;
  T MinInRow(size_t row) const;
  T MaxInCol(size_t col) const;
  T MinInCol(size_t col) const;

  // One-Hot Encoding
  static Matrix<T> OneHotEncode(const Matrix<T>& labels, size_t num_classes);

  size_t Rows() const noexcept;
  size_t Cols() const noexcept;
  //void Print() const;

 private:
  // Internal single-threaded matrix multiplication
  Matrix SingleThreadedMatMul(const Matrix& other) const;

  std::vector<T> data_;
  size_t rows_;
  size_t cols_;
};

// Non-member Scalar Operations
// Operator Overloading for mathematical operations is prohibited in Google style,
// but since this is a pure mathematical construct, we will make an exception here.
template <Numeric T>
inline Matrix<T> operator*(Matrix<T> lhs, T rhs);
template <Numeric T>
inline Matrix<T> operator*(T lhs, Matrix<T> rhs);

template <Numeric T>
inline Matrix<T> operator/(Matrix<T> lhs, T rhs);
template <Numeric T>
inline Matrix<T> operator/(T lhs, Matrix<T> rhs);

template <Numeric T>
inline Matrix<T> operator+(Matrix<T> lhs, T rhs);
template <Numeric T>
inline Matrix<T> operator+(T lhs, Matrix<T> rhs);

template <Numeric T>
inline Matrix<T> operator-(Matrix<T> lhs, T rhs);
template <Numeric T>
inline Matrix<T> operator-(T lhs, Matrix<T> rhs);

#include "matrix-inl.h"

#endif  // MNIST_DIGIT_RECOGNITION_LIBS_NEURAL_MATRIX_H_
