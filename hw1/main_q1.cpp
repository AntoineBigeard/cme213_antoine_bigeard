#include <cassert>
#include <cstring>
#include <iostream>
#include <sstream>

#include "gtest/gtest.h"
#include "matrix.hpp"

TEST(testMatrix, sampleTest)
{
  ASSERT_EQ(1000, 1000)
      << "This does not fail, hence this message is not printed.";
  EXPECT_EQ(2000, 2000)
      << "This does not fail, hence this message is not printed.";
  // If uncommented, the following line will make this test fail.
  // EXPECT_EQ(2000, 3000) << "This expect statement fails, and this message
  // will be printed.";
}

/*
TODO:

For both the MatrixDiagonal and the MatrixSymmetric classes, do the following:

Write at least the following tests to get full credit here:
1. Declare an empty matrix with the default constructor for MatrixSymmetric.
Assert that the NormL0 and size functions return appropriate values for these.
2. Using the second constructor that takes size as argument, create a matrix of
size zero. Repeat the assertions from (1).
3. Provide a negative argument to the second constructor and assert that the
constructor throws an exception.
4. Create and initialize a matrix of some size, and verify that the NormL0
function returns the correct value.
5. Create a matrix, initialize some or all of its elements, then retrieve and
check that they are what you initialized them to.
6. Create a matrix of some size. Make an out-of-bounds access into it and check
that an exception is thrown.
7. Test the stream operator using std::stringstream and using the "<<" operator.

*/

// 1

TEST(testMatrix, MatrixDiagonal)
{
  // 1
  def_matrix MatrixDiagonal;
  EXPECT_EQ(def_matrix.size(), 0) << "Matrix size should be 0 for empty matrix";
  EXPECT_EQ(def_matrix.NormL0(), 0) << "Matrix normL0 should be 0 for empty matrix";

  // 2
  matrix_0 MatrixDiagonal(0);
  EXPECT_EQ(matrix_0.size(), 0) << "Matrix size should be 0 for matrix with size 0";
  EXPECT_EQ(matrix_0.NormL0(), 0) << "Matrix normL0 should be 0 for matrix with size 0";

  // 3
  matrix_neg MatrixDiagonal(-1);
  EXPECT_THROW(matrix_neg, std::invalid_argument);

  matrix_normal MatrixDiagonal(10);
  matrix_normal(1, 1) = 1;
  matrix_normal(3, 3) = 1;
  matrix_normal(2, 2) = 1;

  // 4
  EXPECT_EQ(matrix_normal.NormL0(), 3);

  // 5
  EXPECT_EQ(matrix_normal(1, 1), 1);
  EXPECT_EQ(matrix_normal(2, 2), 1);
  EXPECT_EQ(matrix_normal(3, 3), 1);

  // 6
  EXPECT_THROW(matrix_normal(10, 1), std::out_of_range);
  EXPECT_THROW(matrix_normal(0, -1), std::out_of_range);

  // 7
  std::stringstream s;
  s << matrix_normal;
}

TEST(testMatrix, MatrixSymmetic)
{
  // 1
  def_matrix MatrixSymmetric;
  EXPECT_EQ(def_matrix.size(), 0) << "Matrix size should be 0 for empty matrix";
  EXPECT_EQ(def_matrix.NormL0(), 0) << "Matrix normL0 should be 0 for empty matrix";

  // 2
  matrix_0 MatrixSymmetric(0);
  EXPECT_EQ(matrix_0.size(), 0) << "Matrix size should be 0 for matrix with size 0";
  EXPECT_EQ(matrix_0.NormL0(), 0) << "Matrix normL0 should be 0 for matrix with size 0";

  // 3
  matrix_neg MatrixSymmetric(-1);
  EXPECT_THROW(matrix_neg, std::invalid_argument);

  matrix_normal MatrixSymmetric(10);
  matrix_normal(1, 2) = 1;
  matrix_normal(1, 3) = 1;
  matrix_normal(2, 2) = 1;

  // 4
  EXPECT_EQ(matrix_normal.NormL0(), 5);

  // 5
  EXPECT_EQ(matrix_normal(1, 2), 1);
  EXPECT_EQ(matrix_normal(1, 3), 1);
  EXPECT_EQ(matrix_normal(2, 1), 1);
  EXPECT_EQ(matrix_normal(3, 1), 1);
  EXPECT_EQ(matrix_normal(2, 2), 1);

  // 6
  EXPECT_THROW(matrix_normal(10, 1), std::out_of_range);
  EXPECT_THROW(matrix_normal(0, -1), std::out_of_range);

  // 7
  std::stringstream s;
  s << matrix_normal;
}
