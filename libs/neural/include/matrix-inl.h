// MIT License

// This file contains the implementation of the Matrix class template.
// It is meant to be included only inside matrix.h.

#include <random>

#include <absl/log/check.h>

// Constructor from data vector
template <Numeric T>
Matrix<T>::Matrix(const std::vector<T>& data, size_t rows, size_t cols)
    : data_(data), rows_(rows), cols_(cols) {
  CHECK(data.size() == rows * cols && "Data size must match matrix dimensions");
}

// Constructor with specified dimensions and default initialization
template <Numeric T>
Matrix<T>::Matrix(size_t rows, size_t cols)
    : data_(rows * cols, T(0)), rows_(rows), cols_(cols) {}

// Static function
template <Numeric T>
Matrix<T> Matrix<T>::Random(size_t rows, size_t cols, T min, T max,
                            uint32_t seed) {
  std::vector<T> data(rows * cols);
  std::mt19937 gen(seed);
  if constexpr (std::is_integral_v<T>) {
    std::uniform_int_distribution<T> dist(min, max);
    for (auto& val : data)
      val = dist(gen);
  } else {
    std::uniform_real_distribution<T> dist(min, max);
    for (auto& val : data)
      val = dist(gen);
  }
  return Matrix<T>(data, rows, cols);
}

template <Numeric T>
Matrix<T> Matrix<T>::GetTranspose() const {
  Matrix<T> transposed(cols_, rows_);
  for (size_t r = 0; r < rows_; ++r) {
    for (size_t c = 0; c < cols_; ++c) {
      transposed(c, r) = (*this)(r, c);
    }
  }
  return transposed;
}

template <Numeric T>
T& Matrix<T>::operator()(size_t row, size_t col) noexcept {
  return data_[row * cols_ + col];
}

template <Numeric T>
const T& Matrix<T>::operator()(size_t row, size_t col) const noexcept {
  return data_[row * cols_ + col];
}

template <Numeric T>
T& Matrix<T>::At(size_t row, size_t col) {
  CHECK(row < rows_ && col < cols_ && "Index out of bounds.");
  return data_[row * cols_ + col];
}

template <Numeric T>
const T& Matrix<T>::At(size_t row, size_t col) const {
  CHECK(row < rows_ && col < cols_ && "Index out of bounds.");
  return data_[row * cols_ + col];
}

template <Numeric T>
Matrix<T> Matrix<T>::operator*(const Matrix<T>& other) const {
  CHECK(cols_ == other.rows_ &&
        "Matrix dimensions must match for multiplication");
  Matrix<T> result(rows_, other.cols_);
  for (size_t r = 0; r < rows_; ++r) {
    for (size_t c = 0; c < other.cols_; ++c) {
      T sum = T(0);
      for (size_t k = 0; k < cols_; ++k) {
        sum += (*this)(r, k) * other(k, c);
      }
      result(r, c) = sum;
    }
  }
  return result;
}

template <Numeric T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& other) const {
  CHECK(rows_ == other.rows_ && cols_ == other.cols_ &&
        "Matrix dimensions must match for addition.");
  Matrix<T> result(rows_, cols_);
  for (size_t i = 0; i < rows_ * cols_; ++i) {
    result.data_[i] = data_[i] + other.data_[i];
  }
  return result;
}

template <Numeric T>
Matrix<T> Matrix<T>::operator-(const Matrix<T>& other) const {
  CHECK(rows_ == other.rows_ && cols_ == other.cols_ &&
        "Matrix dimensions must match for subtraction.");
  Matrix<T> result(rows_, cols_);
  for (size_t i = 0; i < rows_ * cols_; ++i) {
    result.data_[i] = data_[i] - other.data_[i];
  }
  return result;
}

template <Numeric T>
Matrix<T>& Matrix<T>::operator*=(const Matrix& other) {
  return *this = (*this) * other;
}

template <Numeric T>
Matrix<T>& Matrix<T>::operator+=(const Matrix& other) {
  return *this = (*this) + other;
}

template <Numeric T>
Matrix<T>& Matrix<T>::operator*=(T scalar) {
  for (auto& val : data_) {
    val *= scalar;
  }
  return *this;
}

template <Numeric T>
Matrix<T>& Matrix<T>::operator/=(T scalar) {
  for (auto& val : data_) {
    CHECK(scalar != T(0) && "Division by zero is invalid.");
    val /= scalar;
  }
  return *this;
}

template <Numeric T>
Matrix<T>& Matrix<T>::operator+=(T scalar) {
  for (auto& val : data_) {
    val += scalar;
  }
  return *this;
}

template <Numeric T>
Matrix<T>& Matrix<T>::operator-=(T scalar) {
  for (auto& val : data_) {
    val -= scalar;
  }
  return *this;
}

template <Numeric T>
const std::vector<T>& Matrix<T>::ToVector() const noexcept {
  return data_;
}

template <Numeric T>
std::vector<T>& Matrix<T>::ToVector() noexcept {
  return data_;
}

template <Numeric T>
size_t Matrix<T>::Rows() const noexcept {
  return rows_;
}

template <Numeric T>
size_t Matrix<T>::Cols() const noexcept {
  return cols_;
}

// ---------
// Non-member Scalar Operations
// ---------
template <Numeric T>
inline Matrix<T> operator*(Matrix<T> lhs, T rhs) {
  return lhs *= rhs;
}

template <Numeric T>
inline Matrix<T> operator*(T lhs, Matrix<T> rhs) {
  return rhs *= lhs;
}

template <Numeric T>
inline Matrix<T> operator+(Matrix<T> lhs, T rhs) {
  return lhs += rhs;
}

template <Numeric T>
inline Matrix<T> operator+(T lhs, Matrix<T> rhs) {
  return rhs += lhs;
}

template <Numeric T>
inline Matrix<T> operator-(Matrix<T> lhs, T rhs) {
  return lhs -= rhs;
}

template <Numeric T>
inline Matrix<T> operator-(T lhs, Matrix<T> rhs) {
  for (size_t r = 0; r < rhs.Rows(); ++r) {
    for (size_t c = 0; c < rhs.Cols(); ++c) {
      rhs(r, c) = lhs - rhs(r, c);
    }
  }
  return rhs;
}

template <Numeric T>
inline Matrix<T> operator/(Matrix<T> lhs, T rhs) {
  CHECK(rhs != T(0) && "Division by zero is invalid.");
  return lhs /= rhs;
}

template <Numeric T>
inline Matrix<T> operator/(T lhs, Matrix<T> rhs) {
  for (size_t r = 0; r < rhs.Rows(); ++r) {
    for (size_t c = 0; c < rhs.Cols(); ++c) {
      CHECK(rhs(r, c) != T(0) && "Division by zero is invalid.");
      rhs(r, c) = lhs / rhs(r, c);
    }
  }
  return rhs;
}
