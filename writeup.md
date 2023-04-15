# Homework 1

## Problem 1

For MatrixDiagonal, we just use a vector of size n to save data.

For MatrixSymmetric, we use a vector of size n*(n+1)/2. Value (i,j), with i >= j, is stored in index (j + i*(i+1)/2).

## Problem 2
Use np.broadcast rules to check if it is broadcastable.
To do the work efficiently, use euclidian rest with the operator % to access the elements of both the matrices properly.

## Problem 3
To build the subclass specific matrices, overwrite the repr().

## Problem 4
Don't forget to throw an error in case the range is not valid.
Then just use std::distance to compute the number of "hops" between the two iterators.

## Problem 5
### Qa
Just use std::transform to compute the result.

### Qb
I used std::all_of with the appropriate lambda function.

### Qc
I used std::sort with a special comparator that allows to put the odd numbers first.

### Qd
I used std::list::sort and a special comparator to do the job in "one" line.