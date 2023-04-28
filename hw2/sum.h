#include <vector>

std::vector<uint> serialSum(const std::vector<uint> &v)
{
    std::vector<uint> sums(2);
    // TODO
    for (unsigned i = 0; i < v.size(); i++)
    {
        if (v[i] % 2 == 0)
        {
            sums[0] += v[i];
        }
        else
        {
            sums[1] += v[i];
        }
    }
    return sums;
}

std::vector<uint> parallelSum(const std::vector<uint> &v)
{
    std::vector<uint> sums(2);
    // TODO
    unsigned even = 0;
    unsigned odd = 0;

#pragma omp parallel for reduction(+ \
                                   : even, odd)
    for (unsigned i = 0; i < v.size(); i++)
    {
        if (v[i] % 2 == 0)
        {
            even += v[i];
        }
        else
        {
            odd += v[i];
        }
    }
    sums[0] = even;
    sums[1] = odd;
    return sums;
}