# Homework 1

## Problem 1

For MatrixDiagonal, we just use a vector of size n to save data.

For MatrixSymmetric, we use a vector of size n*(n+1)/2. Value (i,j), with i >= j, is stored in index (j + i*(i+1)/2).

## Problem 2, 3, 4
It is pretty straigth forward in the code.

## Problem 5
### Qa
Just use std::transform to compute the result.

### Qb
I used std::all_of with the appropriate lambda function.

### Qc
I used std::sort with a special comparator that allows to put the odd numbers first.

### Qd
I used std::list::sort and a special comparator to do the job in "one" line.