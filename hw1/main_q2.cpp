#include <cassert>
#include <cstring>
#include <iostream>
#include <sstream>

#include "gtest/gtest.h"
#include "matrix_rect.hpp"

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
Test your implementation by writing tests that cover most scenarios of 2D matrix
broadcasting. Say you are testing the result C = A * B, test with:
1. A of shape (m != 1, n != 1), B of shape (m != 1, n != 1)
2. A of shape (1, n != 1), B of shape (m != 1, n != 1)
3. A of shape (m != 1, n != 1), B of shape (m != 1, 1)
4. A of shape (1, 1), B of shape (m != 1, n != 1)
Please test any more cases that you can think of.
*/

// 1
TEST(testMatrix, testDot1)
{
  Matrix2D<int> A(2, 3);
  Matrix2D<int> B(2, 3);

  A(0, 0) = 1;
  A(0, 1) = 2;
  A(0, 2) = 3;
  A(1, 0) = 4;
  A(1, 1) = 5;
  A(1, 2) = 6;

  B(0, 0) = 2;
  B(0, 1) = 2;
  B(0, 2) = 1;
  B(1, 0) = 4;
  B(1, 1) = 3;
  B(1, 2) = 0;

  Matrix2D<int> C = A.dot(B);

  EXPECT_EQ(C(0, 0), 2)<< "Result should be 2 for this element";
  EXPECT_EQ(C(0, 1), 4)<< "Result should be 4 for this element";
  EXPECT_EQ(C(0, 2), 3)<< "Result should be 3 for this element";
  EXPECT_EQ(C(1, 0), 16)<< "Result should be 16 for this element";
  EXPECT_EQ(C(1, 1), 15)<< "Result should be 15 for this element";
  EXPECT_EQ(C(1, 2), 0)<< "Result should be 0 for this element";
}

// 2
TEST(testMatrix, testDot2)
{
  Matrix2D<int> A(1, 2);
  Matrix2D<int> B(2, 2);

  A(0, 0) = 1;
  A(0, 1) = 2;

  B(0, 0) = 2;
  B(0, 1) = 2;
  B(1, 0) = 4;
  B(1, 1) = 3;

  Matrix2D<int> C = A.dot(B);

  EXPECT_EQ(C(0, 0), 2) << "Result should be 2 for this element";
  EXPECT_EQ(C(0, 1), 4) << "Result should be 4 for this element";
  EXPECT_EQ(C(1, 0), 4) << "Result should be 4 for this element";
  EXPECT_EQ(C(1, 1), 6) << "Result should be 6 for this element";
}

// 3
TEST(testMatrix, testDot3)
{
  Matrix2D<int> A(2, 2);
  Matrix2D<int> B(2, 1);

  B(0, 0) = 1;
  B(1, 0) = 2;

  A(0, 0) = 2;
  A(0, 1) = 2;
  A(1, 0) = 4;
  A(1, 1) = 3;

  Matrix2D<int> C = A.dot(B);

  EXPECT_EQ(C(0, 0), 2)<< "Result should be 2 for this element";
  EXPECT_EQ(C(0, 1), 2)<< "Result should be 2 for this element";
  EXPECT_EQ(C(1, 0), 8)<< "Result should be 8 for this element";
  EXPECT_EQ(C(1, 1), 6)<< "Result should be 6 for this element";
}

// 4
TEST(testMatrix, testDot4)
{
  Matrix2D<int> A(1, 1);
  Matrix2D<int> B(2, 2);

  A(0, 0) = 2;

  B(0, 0) = 2;
  B(0, 1) = 2;
  B(1, 0) = 4;
  B(1, 1) = 3;

  Matrix2D<int> C = A.dot(B);

  EXPECT_EQ(C(0, 0), 4)<< "Result should be 4 for this element";
  EXPECT_EQ(C(0, 1), 4)<< "Result should be 4 for this element";
  EXPECT_EQ(C(1, 0), 8)<< "Result should be 8 for this element";
  EXPECT_EQ(C(1, 1), 6)<< "Result should be 6 for this element";
}