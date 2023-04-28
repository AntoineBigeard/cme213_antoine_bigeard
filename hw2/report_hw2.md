# CME 213 Homework 2 - Antoine Bigeard

## Problem 1

Thanks to parallel computing for the sum, the algorithm is between 10 and 100 times faster: around 0.1 secs for serial sum, and 0.004 secs for parallel sum!

## Problem 2

### Question 1

I just had to add a condition on the index when I am going through all the blockHistograms vector. In certain cases (when the num of blocks does
not divide the number of keys for instance); numBlocks*numBuckets is larger than keys.size(). In that case, I have to break out the loop at some point.
Apart from that it is straightforward with openMP.

### Question 2, 3, 4

Just have to go through all the elements and apply the appropriate transformations to build the histograms etc..

### Question 5
I created an additional localOffSets vector to be able to populate the new keys vector properly, using the constant vector of the ExScan.

### Question 6

Just call the functions properly.

About the results: we see that as we increase number of blocks and threads, it reduces the time complexity. However,
of course it does not change anything when you just increase the number of threads and not the blocks because there is nothing to put in parallel in that case.
Also, it reaches its best performance for 8 blocks and 8 threads. It is due to the fact that on icme-gpu.stanford.edu, the max number of threads is 8.
Therefore after 8 it starts to do serial operations again, therefore slowing down the program.

Given a number of blocks, the timing is quite sensitive to the number of threads, being able to divide it by 4. It is less sensitive when you increase
the number of blocks for a given number of threads: it can improve up to twice the time (approximately).