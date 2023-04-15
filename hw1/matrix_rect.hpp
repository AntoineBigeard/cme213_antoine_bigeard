#ifndef MATRIX_RECT
#define MATRIX_RECT

#include <algorithm>
#include <iomanip>
#include <numeric>
#include <ostream>
#include <vector>

template <typename T>
class Matrix2D;

template <typename T>
bool Broadcastable(Matrix2D<T> &A, Matrix2D<T> &B)
{
  int n_rows_A = A.size_rows();
  int n_cols_A = A.size_cols();
  int n_rows_B = B.size_rows();
  int n_cols_B = B.size_cols();
  if ((n_rows_A == n_rows_B && n_cols_A == n_cols_B) ||
      (n_rows_A == 1 && n_cols_A == 1) ||
      (n_rows_B == 1 && n_cols_B == 1) ||
      (n_rows_A == 1 && n_cols_A == n_cols_B) ||
      (n_rows_B == 1 && n_cols_A == n_cols_B) ||
      (n_cols_A == 1 && n_rows_A == n_rows_B) ||
      (n_cols_B == 1 && n_rows_A == n_rows_B))
  {
    return true;
  }

  else
    return false;
}

template <typename T>
class Matrix2D
{
private:
  // The size of the matrix is (n_rows, n_cols)
  unsigned int n_rows;
  unsigned int n_cols;

  // Vector storing the data in row major order. Element (i,j) for 0 <= i <
  // n_rows and 0 <= j < n_cols is stored at data[i * n_cols + j].
  std::vector<T> data_;

public:
  // Empty matrix
  Matrix2D()
  {
    n_rows = 0;
    n_cols = 0;
  }

  // Constructor takes argument (m,n) = matrix dimension.
  Matrix2D(const int m, const int n)
  {
    n_rows = m;
    n_cols = n;
    data_.resize(n * m);
  }

  unsigned int size_rows() const { return n_rows; }
  unsigned int size_cols() const { return n_cols; }

  // Returns reference to matrix element (i, j).
  T &operator()(int i, int j)
  {
    return data_[i * n_cols + j];
  }

  void Print(std::ostream &ostream)
  {
    for (int i = 0; i < n_rows; i++)
    {
      for (int j = 0; j < n_cols; j++)
      {
        ostream << data_[i * n_cols + j] << " ";
      }
      ostream << std::endl;
    }
  }

  Matrix2D<T> dot(Matrix2D<T> &mat)
  {
    if (Broadcastable<T>(*this, mat))
    {
      int out_rows = std::max(this.n_rows, mat.size_rows());
      int out_cols = std::max(this.n_cols, mat.size_cols());

      Matrix2D<T> ret(out_rows, out_cols);

      for (int i = 0; i < n_rows; ++i)
      {
        for (int j = 0; j < n_cols; ++j)
        {
          ret(i, j) = (*this)(i, j) * mat(i % mat.size_rows(), j % mat.size_cols());
        }
      }

      return ret;
    }
    else
    {
      throw std::invalid_argument("Incompatible shapes of the two matrices.");
    }
  }

  template <typename U>
  friend std::ostream &operator<<(std::ostream &stream, Matrix2D<U> &m);
};

template <typename T>
std::ostream &operator<<(std::ostream &stream, Matrix2D<T> &m)
{
  m.Print(stream);
  return stream;
}

#endif /* MATRIX_RECT */
