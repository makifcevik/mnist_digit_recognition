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

TEST(CollapseRows, ReturnsSummedUpRowMatrix) {
  // Arrange
  std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  Matrix<int> mat(data, 3, 3);
  // Act
  auto result =
      mat.CollapseRows();  // result should be a row matrix: {12, 15, 18}
  // Assert
  EXPECT_EQ(1, result.Rows());
  EXPECT_EQ(3, result.Cols());
  EXPECT_EQ(12, result(0, 0));
  EXPECT_EQ(15, result(0, 1));
  EXPECT_EQ(18, result(0, 2));
}

TEST(CollapseCols, ReturnsSummedUpColMatrix) {
  // Arrange
  std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  Matrix<int> mat(data, 3, 3);
  // Act
  auto result =
      mat.CollapseCols();  // result should be a col matrix: {6, 15, 24}
  // Assert
  EXPECT_EQ(3, result.Rows());
  EXPECT_EQ(1, result.Cols());
  EXPECT_EQ(6, result(0, 0));
  EXPECT_EQ(15, result(1, 0));
  EXPECT_EQ(24, result(2, 0));
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

TEST(ToFloat, ConvertsMatrixToFloat) {
  // Arrange
  Matrix<int> mat({1, 2, 3, 4}, 2, 2);
  // Act
  Matrix<float> floatMat = mat.ToFloat();
  // Assert
  EXPECT_EQ(floatMat.Rows(), 2);
  EXPECT_EQ(floatMat.Cols(), 2);
  EXPECT_FLOAT_EQ(floatMat(0, 0), 1.0f);
  EXPECT_FLOAT_EQ(floatMat(0, 1), 2.0f);
  EXPECT_FLOAT_EQ(floatMat(1, 0), 3.0f);
  EXPECT_FLOAT_EQ(floatMat(1, 1), 4.0f);
}

TEST(ToDouble, ConvertsMatrixToDouble) {
  // Arrange
  Matrix<int> mat({1, 2, 3, 4}, 2, 2);
  // Act
  Matrix<double> doubleMat = mat.ToDouble();
  // Assert
  EXPECT_EQ(doubleMat.Rows(), 2);
  EXPECT_EQ(doubleMat.Cols(), 2);
  EXPECT_DOUBLE_EQ(doubleMat(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(doubleMat(0, 1), 2.0);
  EXPECT_DOUBLE_EQ(doubleMat(1, 0), 3.0);
  EXPECT_DOUBLE_EQ(doubleMat(1, 1), 4.0);
}

TEST(BroadcastRows, BroadcastsAlongRowsToMatchNewRows) {
  // Arrange
  Matrix<int> mat({1, 2, 3, 4}, 2, 2);
  size_t new_rows = 4;
  // Act
  Matrix<int> broadcasted = mat.BroadcastRows(new_rows);
  // Assert
  EXPECT_EQ(broadcasted.Rows(), new_rows);
  EXPECT_EQ(broadcasted.Cols(), 2);
  for (size_t r = 0; r < new_rows; ++r) {
    EXPECT_EQ(broadcasted(r, 0), mat(r % 2, 0));
    EXPECT_EQ(broadcasted(r, 1), mat(r % 2, 1));
  }
}

TEST(ShuffleRows, CorrectlyShufflesRows) {
  // Arrange
  Matrix<int> mat({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 5, 2);
  // Act
  Matrix<int> shuffled = mat.ShuffleRows(42);
  // Assert
  EXPECT_EQ(mat.Rows(), shuffled.Rows());
  EXPECT_EQ(mat.Cols(), shuffled.Cols());
  bool is_identical = true;
  for (int i = 0; i < mat.Rows(); ++i) {
    if (mat(i, 0) != shuffled(i, 0)) {  // Checking across the first col
      is_identical = false;
      break;
    }
  }
  EXPECT_FALSE(is_identical);
}

TEST(ArgMaxRow, ReturnsTheIndexOfTheMaxValueInSpecifiedRow) {
  // Arrange
  Matrix<int> mat({1, 2, 3, 4, 6, 5, 9, 0, 7}, 3, 3);
  // Act
  int first = mat.ArgMaxRow(0);
  int second = mat.ArgMaxRow(1);
  int third = mat.ArgMaxRow(2);
  // Assert
  EXPECT_EQ(first, 2);
  EXPECT_EQ(second, 1);
  EXPECT_EQ(third, 0);
}

TEST(ArgMaxCol, ReturnsTheIndexOfTheMaxValueInSpecifiedCol) {
  // Arrange
  Matrix<int> mat({1, 2, 9, 4, 6, 5, 8, 0, 7}, 3, 3);
  // Act
  int first = mat.ArgMaxCol(0);
  int second = mat.ArgMaxCol(1);
  int third = mat.ArgMaxCol(2);
  // Assert
  EXPECT_EQ(first, 2);
  EXPECT_EQ(second, 1);
  EXPECT_EQ(third, 0);
}

TEST(ArgMinRow, ReturnsTheIndexOfTheMinValueInSpecifiedRow) {
  // Arrange
  Matrix<int> mat({1, 2, 3, 5, 6, 4, 9, 0, 7}, 3, 3);
  // Act
  int first = mat.ArgMinRow(0);
  int second = mat.ArgMinRow(1);
  int third = mat.ArgMinRow(2);
  // Assert
  EXPECT_EQ(first, 0);
  EXPECT_EQ(second, 2);
  EXPECT_EQ(third, 1);
}

TEST(ArgMinCol, ReturnsTheIndexOfTheMinValueInSpecifiedCol) {
  // Arrange
  Matrix<int> mat({1, 2, 7, 5, 6, 2, 9, 0, 3}, 3, 3);
  // Act
  int first = mat.ArgMinCol(0);
  int second = mat.ArgMinCol(1);
  int third = mat.ArgMinCol(2);
  // Assert
  EXPECT_EQ(first, 0);
  EXPECT_EQ(second, 2);
  EXPECT_EQ(third, 1);
}

TEST(MaxInRow, ReturnsTheMaxValueInSpecifiedRow) {
  // Arrange
  Matrix<int> mat({1, 2, 7, 5, 6, 2, 9, 0, 3}, 3, 3);
  // Act
  int first = mat.MaxInRow(0);
  int second = mat.MaxInRow(1);
  int third = mat.MaxInRow(2);
  // Assert
  EXPECT_EQ(first, 7);
  EXPECT_EQ(second, 6);
  EXPECT_EQ(third, 9);
}

TEST(MaxInCol, ReturnsTheMaxValueInSpecifiedCol) {
  // Arrange
  Matrix<int> mat({1, 2, 7, 5, 6, 2, 9, 0, 3}, 3, 3);
  // Act
  int first = mat.MaxInCol(0);
  int second = mat.MaxInCol(1);
  int third = mat.MaxInCol(2);
  // Assert
  EXPECT_EQ(first, 9);
  EXPECT_EQ(second, 6);
  EXPECT_EQ(third, 7);
}

TEST(MinInRow, ReturnsTheMinValueInSpecifiedRow) {
  // Arrange
  Matrix<int> mat({1, 2, 7, 5, 6, 2, 9, 0, 3}, 3, 3);
  // Act
  int first = mat.MinInRow(0);
  int second = mat.MinInRow(1);
  int third = mat.MinInRow(2);
  // Assert
  EXPECT_EQ(first, 1);
  EXPECT_EQ(second, 2);
  EXPECT_EQ(third, 0);
}

TEST(MinInCol, ReturnsTheMinValueInSpecifiedCol) {
  // Arrange
  Matrix<int> mat({1, 2, 7, 5, 6, 2, 9, 0, 3}, 3, 3);
  // Act
  int first = mat.MinInCol(0);
  int second = mat.MinInCol(1);
  int third = mat.MinInCol(2);
  // Assert
  EXPECT_EQ(first, 1);
  EXPECT_EQ(second, 0);
  EXPECT_EQ(third, 2);
}

TEST(OneHotEncode, CorrectlyOneHotEncodesLabels) {
  // Arrange
  Matrix<int> labels({0, 2, 1, 2}, 4, 1);
  size_t num_classes = 3;
  // Act
  Matrix<int> one_hot = Matrix<int>::OneHotEncode(labels, num_classes);
  // Assert
  EXPECT_EQ(one_hot.Rows(), 4);
  EXPECT_EQ(one_hot.Cols(), num_classes);
  EXPECT_EQ(one_hot(0, 0), 1);
  EXPECT_EQ(one_hot(0, 1), 0);
  EXPECT_EQ(one_hot(0, 2), 0);
  EXPECT_EQ(one_hot(1, 0), 0);
  EXPECT_EQ(one_hot(1, 1), 0);
  EXPECT_EQ(one_hot(1, 2), 1);
  EXPECT_EQ(one_hot(2, 0), 0);
  EXPECT_EQ(one_hot(2, 1), 1);
  EXPECT_EQ(one_hot(2, 2), 0);
  EXPECT_EQ(one_hot(3, 0), 0);
  EXPECT_EQ(one_hot(3, 1), 0);
  EXPECT_EQ(one_hot(3, 2), 1);
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

TEST(OperatorMultiplyByMatrix, ForcesConcurrencyOnWorkload) {
  // Arrange
  Matrix<int> matA({1, 2, 3, 4}, 2, 2);
  Matrix<int> matB({5, 6, 7, 8}, 2, 2);

  // Save the original value
  size_t original_threshold = Matrix<int>::kMinWorkPerThread;

  // Set threshold to 1 to FORCE the threading logic to trigger
  Matrix<int>::kMinWorkPerThread = 1;

  // Act
  Matrix<int> result = matA * matB;

  // Restore the value so other tests aren't affected
  Matrix<int>::kMinWorkPerThread = original_threshold;

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