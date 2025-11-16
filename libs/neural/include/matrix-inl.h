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

// Collapse Functions
// Sums over rows or columns, returning a single-row or single-column matrix

// CollapseRows: returns a row matrix (1 x cols)
template <Numeric T>
Matrix<T> Matrix<T>::CollapseRows() const {
  CHECK(rows_ > 0 && "Cannot collapse rows of an empty matrix.");
  Matrix<T> result(1, cols_);
  for (size_t c = 0; c < cols_; ++c) {
    T sum = T(0);
    for (size_t r = 0; r < rows_; ++r) {
      sum += (*this)(r, c);
    }
    result(0, c) = sum;
  }
  return result;
}

// CollapseCols: returns a column matrix (rows x 1)
template <Numeric T>
Matrix<T> Matrix<T>::CollapseCols() const {
  CHECK(cols_ > 0 && "Cannot collapse columns of an empty matrix.");
  Matrix<T> result(rows_, 1);
  for (size_t r = 0; r < rows_; ++r) {
    T sum = T(0);
    for (size_t c = 0; c < cols_; ++c) {
      sum += (*this)(r, c);
    }
    result(r, 0) = sum;
  }
  return result;
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
Matrix<T>& Matrix<T>::operator-=(const Matrix& other) {
  return *this = (*this) - other;
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
Matrix<float> Matrix<T>::ToFloat(float scale) const {
  Matrix<float> result(rows_, cols_);
  for (size_t r = 0; r < rows_; ++r) {
    for (size_t c = 0; c < cols_; ++c) {
      result(r, c) = static_cast<float>((*this)(r, c)) * scale;
    }
  }
  return result;
}

template <Numeric T>
Matrix<double> Matrix<T>::ToDouble(double scale) const {
  Matrix<double> result(rows_, cols_);
  for (size_t r = 0; r < rows_; ++r) {
    for (size_t c = 0; c < cols_; ++c) {
      result(r, c) = static_cast<double>((*this)(r, c)) * scale;
    }
  }
  return result;
}

template <Numeric T>
Matrix<T> Matrix<T>::BroadcastRows(size_t new_rows) const {
  CHECK(new_rows >= rows_ &&
        "New row count must be greater than or equal to current rows.");
  Matrix<T> result(new_rows, cols_);
  for (size_t r = 0; r < new_rows; ++r) {
    for (size_t c = 0; c < cols_; ++c) {
      result(r, c) = (*this)(r % rows_, c);
    }
  }
  return result;
}

template <Numeric T>
Matrix<T> Matrix<T>::ShuffleRows(uint32_t seed) const {
  Matrix<T> shuffled(*this);
  std::vector<size_t> indices(rows_);
  for (size_t i = 0; i < rows_; ++i) {
    indices[i] = i;
  }
  std::mt19937 gen(seed);
  std::shuffle(indices.begin(), indices.end(), gen);
  Matrix<T> result(rows_, cols_);
  for (size_t r = 0; r < rows_; ++r) {
    for (size_t c = 0; c < cols_; ++c) {
      result(r, c) = shuffled(indices[r], c);
    }
  }
  return result;
}

template <Numeric T>
size_t Matrix<T>::ArgMaxRow(size_t row) const {
  CHECK(row < rows_ && "Row index out of bounds.");
  size_t max_index = 0;
  T max_value = (*this)(row, 0);
  for (size_t c = 1; c < cols_; ++c) {
    if ((*this)(row, c) > max_value) {
      max_value = (*this)(row, c);
      max_index = c;
    }
  }
  return max_index;
}

template <Numeric T>
size_t Matrix<T>::ArgMinRow(size_t row) const {
  CHECK(row < rows_ && "Row index out of bounds.");
  size_t min_index = 0;
  T min_value = (*this)(row, 0);
  for (size_t c = 1; c < cols_; ++c) {
    if ((*this)(row, c) < min_value) {
      min_value = (*this)(row, c);
      min_index = c;
    }
  }
  return min_index;
}

template <Numeric T>
size_t Matrix<T>::ArgMaxCol(size_t col) const {
  CHECK(col < cols_ && "Column index out of bounds.");
  size_t max_index = 0;
  T max_value = (*this)(0, col);
  for (size_t r = 1; r < rows_; ++r) {
    if ((*this)(r, col) > max_value) {
      max_value = (*this)(r, col);
      max_index = r;
    }
  }
  return max_index;
}

template <Numeric T>
size_t Matrix<T>::ArgMinCol(size_t col) const {
  CHECK(col < cols_ && "Column index out of bounds.");
  size_t min_index = 0;
  T min_value = (*this)(0, col);
  for (size_t r = 1; r < rows_; ++r) {
    if ((*this)(r, col) < min_value) {
      min_value = (*this)(r, col);
      min_index = r;
    }
  }
  return min_index;
}

template <Numeric T>
T Matrix<T>::MaxInRow(size_t row) const {
  CHECK(row < rows_ && "Row index out of bounds.");
  T max_value = (*this)(row, 0);
  for (size_t c = 1; c < cols_; ++c) {
    if ((*this)(row, c) > max_value) {
      max_value = (*this)(row, c);
    }
  }
  return max_value;
}

template <Numeric T>
T Matrix<T>::MinInRow(size_t row) const {
  CHECK(row < rows_ && "Row index out of bounds.");
  T min_value = (*this)(row, 0);
  for (size_t c = 1; c < cols_; ++c) {
    if ((*this)(row, c) < min_value) {
      min_value = (*this)(row, c);
    }
  }
  return min_value;
}

template <Numeric T>
T Matrix<T>::MaxInCol(size_t col) const {
  CHECK(col < cols_ && "Column index out of bounds.");
  T max_value = (*this)(0, col);
  for (size_t r = 1; r < rows_; ++r) {
    if ((*this)(r, col) > max_value) {
      max_value = (*this)(r, col);
    }
  }
  return max_value;
}

template <Numeric T>
T Matrix<T>::MinInCol(size_t col) const {
  CHECK(col < cols_ && "Column index out of bounds.");
  T min_value = (*this)(0, col);
  for (size_t r = 1; r < rows_; ++r) {
    if ((*this)(r, col) < min_value) {
      min_value = (*this)(r, col);
    }
  }
  return min_value;
}

// Static One-Hot Encoding Function
template <Numeric T>
Matrix<T> Matrix<T>::OneHotEncode(const Matrix<T>& labels, size_t num_classes) {
  Matrix<T> one_hot(labels.Rows(), num_classes);
  for (size_t r = 0; r < labels.Rows(); ++r) {
    size_t label = static_cast<size_t>(labels(r, 0));
    CHECK(label < num_classes && "Label out of bounds.");
    one_hot(r, label) = T(1);
  }
  return one_hot;
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
