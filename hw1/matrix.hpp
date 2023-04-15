#ifndef _MATRIX_HPP
#define _MATRIX_HPP

#include <ostream>
#include <vector>

/*
This is the pure abstract base class specifying general set of functions for a
square matrix.

Concrete classes for specific types of matrices, like MatrixSymmetric, should
implement these functions.
*/
template <typename T>
class Matrix
{
  // Returns reference to matrix element (i, j).
  virtual T &operator()(int i, int j) = 0;
  // Number of non-zero elements in matrix.
  virtual unsigned NormL0() const = 0;
  // Enables printing all matrix elements using the overloaded << operator
  virtual void Print(std::ostream &ostream) = 0;

  template <typename U>
  friend std::ostream &operator<<(std::ostream &stream, Matrix<U> &m);
};

template <typename T>
std::ostream &operator<<(std::ostream &stream, Matrix<T> &m)
{
  m.Print(stream);
  return stream;
}

/* MatrixDiagonal Class is a subclass of the Matrix class */
template <typename T>
class MatrixDiagonal : public Matrix<T>
{
private:
  // Matrix Dimension. Equals the number of columns and the number of rows.
  unsigned int n_;

  // Elements of the matrix. You get to choose how to organize the matrix
  // elements into this vector.
  std::vector<T> data_;

public:
  MatrixDiagonal() { n_ = 0; }

  MatrixDiagonal(const int n)
  {
    if (n < 0)
    {
      throw std::invalid_argument("MatrixDiagonal can't have negative size.");
    }
    else
    {
      n_ = n;
      data_.resize(n);
    }
  }

  unsigned int size() const { return n_; }

  T &operator()(int i, int j) override
  {
    if ((unsigned)i >= n_ || (unsigned)j >= n_ || (unsigned)i < 0 || (unsigned)j < 0)
      throw std::out_of_range("Index out of range for matrix of size n_");
    else if (i == j)
      return data_[i];
    else
      throw std::out_of_range("Index out of range for a diagonal matrix.");
  }

  unsigned NormL0() const override
  {
    unsigned count = 0;
    for (unsigned i = 0; i < n_; i++)
    {
      if (data_[i] != 0)
        count++;
    }
    return count;
  }

  void Print(std::ostream &ostream) override
  {
    for (unsigned i = 0; i < n_; i++)
    {
      for (unsigned j = 0; j < n_; j++)
      {
        if (i == j)
          ostream << data_[i] << " ";
        else
          ostream << 0 << " ";
      }
      ostream << std::endl;
    }
  }
};

/* MatrixSymmetric Class is a subclass of the Matrix class */
template <typename T>
class MatrixSymmetric : public Matrix<T>
{
private:
  // Matrix Dimension. Equals the number of columns and the number of rows.
  unsigned int n_;
  // Elements of the matrix. You get to choose how to organize the matrix
  // elements into this vector.
  std::vector<T> data_;

public:
  MatrixSymmetric() { n_ = 0; }

  MatrixSymmetric(const int n)
  {
    if (n < 0)
    {
      throw std::invalid_argument("MatrixSymmetric can't have negative size.");
    }
    else
    {
      n_ = n;
      data_.resize(n * (n + 1) / 2);
    }
  }

  unsigned int size() const
  {
    return n_;
  }

  T &operator()(int i, int j) override
  {
    if ((unsigned)i >= n_ || (unsigned)j >= n_ || (unsigned)i < 0 || (unsigned)j < 0)
      throw std::out_of_range("Index out of range for matrix of size n_");
    else if (i >= j)
      return data_[j + i * (i + 1) / 2];
    else
      return data_[i + j * (j + 1) / 2];
  }

  unsigned NormL0() const override
  {
    unsigned count = 0;
    for (unsigned i = 0; i < n_; i++)
    {
      for (unsigned j = 0; j <= i; j++)
      {
        if (data_[j + i * (i + 1) / 2] != 0)
          {if (i==j) count ++;
          else count += 2;}
      }
    }
    return count;
  }

  void Print(std::ostream &ostream) override
  {
    for (unsigned i = 0; i < n_; i++)
    {
      for (unsigned j = 0; j < n_; j++)
      {
        if (i >= j)
        {
          ostream << data_[j + i * (i + 1) / 2] << " ";
        }
        else
        {
          ostream << data_[i + j * (j + 1) / 2] << " ";
        }
      }
      ostream << std::endl;
    }
  }
};

#endif /* MATRIX_HPP */