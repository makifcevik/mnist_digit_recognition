// MIT License

#include "matrix.h"

#include <gtest/gtest.h>

TEST(Rows, ReturnsMatrixRows) {
  // Arrange
  Matrix<int> mat(4, 5);
  // Act
  size_t cols = mat.Rows();
  // Assert
  EXPECT_EQ(cols, 4);
}

TEST(Cols, ReturnsMatrixCols) {
  // Arrange
  Matrix<int> mat(4, 5);
  // Act
  size_t cols = mat.Cols();
  // Assert
  EXPECT_EQ(cols, 5);
}

TEST(OperatorParentheses, AccessesTheCorrectElement) {
  // Arrange
  Matrix<int> mat({1, 2, 3, 4, 5, 6}, 3, 2);
  // Act and Assert
  EXPECT_EQ(mat(1, 1), 4);
  EXPECT_EQ(mat(2, 0), 5);
  EXPECT_EQ(mat(0, 0), 1);
  EXPECT_EQ(mat(2, 1), 6);
}

TEST(At, AccessesTheCorrectElement) {
  // Arrange
  Matrix<double> mat({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, 3, 2);
  // Act and Assert
  EXPECT_EQ(mat.At(1, 1), 4.0);
  EXPECT_EQ(mat.At(2, 0), 5.0);
  EXPECT_EQ(mat.At(0, 0), 1.0);
  EXPECT_EQ(mat.At(2, 1), 6.0);
}

TEST(MatrixConstructor, DefaultConstructorWorks) {
  // Arrange & Act
  Matrix<double> mat(3, 4);
  // Assert
  ASSERT_EQ(mat.Rows(), 3);
  ASSERT_EQ(mat.Cols(), 4);
  for (size_t r = 0; r < mat.Rows(); ++r) {
    for (size_t c = 0; c < mat.Cols(); ++c) {
      EXPECT_EQ(mat(r, c), 0.0);
    }
  }
}

TEST(MatrixConstructor, DataConstructorWorks) {
  // Arrange
  std::vector<float> data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  // Act
  Matrix<float> mat(data, 2, 3);
  // Assert
  ASSERT_EQ(mat.Rows(), 2);
  ASSERT_EQ(mat.Cols(), 3);
  for (size_t r = 0; r < mat.Rows(); ++r) {
    for (size_t c = 0; c < mat.Cols(); ++c) {
      EXPECT_EQ(mat(r, c), data[r * mat.Cols() + c]);
    }
  }
}

TEST(Random, ReturnsMatrixWithCorrectDimensions) {
  // Arrange
  size_t rows = 4;
  size_t cols = 5;
  // Act
  Matrix<int> mat = Matrix<int>::Random(rows, cols, 0, 10, 42);
  // Assert
  EXPECT_EQ(mat.Rows(), rows);
  EXPECT_EQ(mat.Cols(), cols);
}

TEST(GetTranspose, ReturnsTransposedMatrix) {
  // Arrange
  Matrix<int> mat({1, 2, 3, 4, 5, 6}, 2, 3);
  // Act
  Matrix<int> transposed = mat.GetTranspose();
  // Assert
  ASSERT_EQ(transposed.Rows(), 3);
  ASSERT_EQ(transposed.Cols(), 2);
  EXPECT_EQ(transposed(0, 0), 1);
  EXPECT_EQ(transposed(0, 1), 4);
  EXPECT_EQ(transposed(1, 0), 2);
  EXPECT_EQ(transposed(1, 1), 5);
  EXPECT_EQ(transposed(2, 0), 3);
  EXPECT_EQ(transposed(2, 1), 6);
}

TEST(ToVector, ReturnsCorrectDataVector) {
  // Arrange
  std::vector<int> data = {1, 2, 3, 4, 5, 6};
  Matrix<int> mat(data, 2, 3);
  // Act
  const std::vector<int>& vec = mat.ToVector();
  // Assert
  EXPECT_EQ(vec, data);
}

TEST(ToVector, NonConstVersionReturnsCorrectDataVector) {
  // Arrange
  std::vector<int> data = {1, 2, 3, 4, 5, 6};
  Matrix<int> mat(data, 2, 3);
  // Act
  std::vector<int>& vec = mat.ToVector();
  // Assert
  EXPECT_EQ(vec, data);
}

TEST(OperatorMultiplyByMatrix, MultipliesTwoMatrices) {
  // Arrange
  Matrix<int> matA({1, 2, 3, 4}, 2, 2);
  Matrix<int> matB({5, 6, 7, 8}, 2, 2);
  // Act
  Matrix<int> result = matA * matB;
  // Assert
  EXPECT_EQ(result(0, 0), 19);
  EXPECT_EQ(result(0, 1), 22);
  EXPECT_EQ(result(1, 0), 43);
  EXPECT_EQ(result(1, 1), 50);
}

TEST(OperatorAddMatrix, AddsTwoMatrices) {
  // Arrange
  Matrix<int> matA({1, 2, 3, 4}, 2, 2);
  Matrix<int> matB({5, 6, 7, 8}, 2, 2);
  // Act
  Matrix<int> result = matA + matB;
  // Assert
  EXPECT_EQ(result(0, 0), 6);
  EXPECT_EQ(result(0, 1), 8);
  EXPECT_EQ(result(1, 0), 10);
  EXPECT_EQ(result(1, 1), 12);
}

TEST(OperatorSubstractMatrix, SubstractsTwoMatrices) {
  // Arrange
  Matrix<int> matA({5, 6, 7, 8}, 2, 2);
  Matrix<int> matB({1, 2, 3, 4}, 2, 2);
  // Act
  Matrix<int> result = matA - matB;
  // Assert
  EXPECT_EQ(result(0, 0), 4);
  EXPECT_EQ(result(0, 1), 4);
  EXPECT_EQ(result(1, 0), 4);
  EXPECT_EQ(result(1, 1), 4);
}

TEST(OperatorMultiplyAssignScalar, MultipliesEachElementByScalarInPlace) {
  // Arrange
  Matrix<int> mat({1, 2, 3, 4}, 2, 2);
  int scalar = 3;
  // Act
  mat *= scalar;
  // Assert
  EXPECT_EQ(mat(0, 0), 3);
  EXPECT_EQ(mat(0, 1), 6);
  EXPECT_EQ(mat(1, 0), 9);
  EXPECT_EQ(mat(1, 1), 12);
}

TEST(OperatorDivideAssignScalar, DividesEachElementByScalarInPlace) {
  // Arrange
  Matrix<int> mat({5, 10, 15, 20}, 2, 2);
  int scalar = 5;
  // Act
  mat /= scalar;
  // Assert
  EXPECT_EQ(mat(0, 0), 1);
  EXPECT_EQ(mat(0, 1), 2);
  EXPECT_EQ(mat(1, 0), 3);
  EXPECT_EQ(mat(1, 1), 4);
}

TEST(OperatorAddAssignScalar, AddsScalarToEachElementInPlace) {
  // Arrange
  Matrix<int> mat({1, 2, 3, 4}, 2, 2);
  int scalar = 5;
  // Act
  mat += scalar;
  // Assert
  EXPECT_EQ(mat(0, 0), 6);
  EXPECT_EQ(mat(0, 1), 7);
  EXPECT_EQ(mat(1, 0), 8);
  EXPECT_EQ(mat(1, 1), 9);
}

TEST(OperatorSubstractAssignScalar, SubstractsScalarFromEachElementInPlace) {
  // Arrange
  Matrix<float> mat({1.0f, 2.0f, 3.0f, 4.0f}, 2, 2);
  float scalar = 2.0f;
  // Act
  mat -= scalar;
  // Assert
  EXPECT_EQ(mat(0, 0), -1.0f);
  EXPECT_EQ(mat(0, 1), 0.0f);
  EXPECT_EQ(mat(1, 0), 1.0f);
  EXPECT_EQ(mat(1, 1), 2.0f);
}

// ----------
// Non-member operations
// ----------
TEST(OperatorMultiplyMatrixByScalar, MultipliesEachElementByScalar) {
  // Arrange
  Matrix<int> mat({1, 2, 3, 4}, 2, 2);
  int scalar = 3;
  // Act
  Matrix<int> result = mat * scalar;
  // Assert
  EXPECT_EQ(result(0, 0), 3);
  EXPECT_EQ(result(0, 1), 6);
  EXPECT_EQ(result(1, 0), 9);
  EXPECT_EQ(result(1, 1), 12);
}

TEST(OperatorDivideMatrixByScalar, DividesEachElementByScalar) {
  // Arrange
  Matrix<int> mat({5, 10, 15, 20}, 2, 2);
  int scalar = 5;
  // Act
  Matrix<int> result = mat / scalar;
  // Assert
  EXPECT_EQ(result(0, 0), 1);
  EXPECT_EQ(result(0, 1), 2);
  EXPECT_EQ(result(1, 0), 3);
  EXPECT_EQ(result(1, 1), 4);
}

TEST(OperatorAddMatrixToScalar, AddsScalarToEachElement) {
  // Arrange
  Matrix<int> mat({1, 2, 3, 4}, 2, 2);
  int scalar = 5;
  // Act
  Matrix<int> result = mat + scalar;
  // Assert
  EXPECT_EQ(result(0, 0), 6);
  EXPECT_EQ(result(0, 1), 7);
  EXPECT_EQ(result(1, 0), 8);
  EXPECT_EQ(result(1, 1), 9);
}

TEST(OperatorSubstractMatrixFromScalar, SubstractsScalarFromEachElement) {
  // Arrange
  Matrix<float> mat({1.0f, 2.0f, 3.0f, 4.0f}, 2, 2);
  float scalar = 2.0f;
  // Act
  Matrix<float> result = mat - scalar;
  // Assert
  EXPECT_EQ(result(0, 0), -1.0f);
  EXPECT_EQ(result(0, 1), 0.0f);
  EXPECT_EQ(result(1, 0), 1.0f);
  EXPECT_EQ(result(1, 1), 2.0f);
}

TEST(OperatorMultiplyScalarByMatrix, MultipliesScalarWithEachElement) {
  // Arrange
  Matrix<int> mat({1, 2, 3, 4}, 2, 2);
  int scalar = 3;
  // Act
  Matrix<int> result = scalar * mat;
  // Assert
  EXPECT_EQ(result(0, 0), 3);
  EXPECT_EQ(result(0, 1), 6);
  EXPECT_EQ(result(1, 0), 9);
  EXPECT_EQ(result(1, 1), 12);
}

TEST(OperatorDivideScalarByMatrix, DividesScalarByEachElement) {
  // Arrange
  Matrix<int> mat({1, 5, 10, 20}, 2, 2);
  int scalar = 20;
  // Act
  Matrix<int> result = scalar / mat;
  // Assert
  EXPECT_EQ(result(0, 0), 20);
  EXPECT_EQ(result(0, 1), 4);
  EXPECT_EQ(result(1, 0), 2);
  EXPECT_EQ(result(1, 1), 1);
}

TEST(OperatorAddScalarToMatrix, AddsScalarToEachElement) {
  // Arrange
  Matrix<int> mat({1, 2, 3, 4}, 2, 2);
  int scalar = 5;
  // Act
  Matrix<int> result = scalar + mat;
  // Assert
  EXPECT_EQ(result(0, 0), 6);
  EXPECT_EQ(result(0, 1), 7);
  EXPECT_EQ(result(1, 0), 8);
  EXPECT_EQ(result(1, 1), 9);
}

TEST(OperatorSubstractScalarFromMatrix, SubstractsEachElementByScalar) {
  // Arrange
  Matrix<float> mat({1.0f, 2.0f, 3.0f, 4.0f}, 2, 2);
  float scalar = 2.0f;
  // Act
  Matrix<float> result = scalar - mat;
  // Assert
  EXPECT_EQ(result(0, 0), 1.0f);
  EXPECT_EQ(result(0, 1), 0.0f);
  EXPECT_EQ(result(1, 0), -1.0f);
  EXPECT_EQ(result(1, 1), -2.0f);
}