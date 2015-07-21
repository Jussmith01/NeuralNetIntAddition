#ifndef BINARY_CONV_C
#define BINARY_CONV_C

#include <vector>
#include <bitset>
#include <limits.h>
#include <math.h>
#include <cstring>
#include <iostream>

/*----------------------------------------
  ***************************************
  |  Produce a Binary Vector of floats  |
  ***************************************
Input a vector of floating points and return
vector of their bit representation.
------------------------------------------*/
std::vector<double> ProduceBinaryVector(int value)
{
    std::vector<double> bitvec;

    const int S = sizeof(int) * CHAR_BIT;

    bitvec.resize(S);

    std::bitset<S> bits(value);
    for (int j=0; j<S; ++j)
        bitvec[j]=(double)bits[j];

    return bitvec;
};

/*----------------------------------------
  ***************************************
  |  Produce a Binary Vector of Ints    |
  ***************************************
Input a vector of floating points and return
vector of their bit representation.
------------------------------------------*/
int ProduceIntegerFromBinary(std::vector<double>  &bitvec)
{
    const int S = sizeof(int) * CHAR_BIT;

    std::bitset<S> bits;
    for (int j=0; j<S; ++j)
    {
        //std::cout << bitvec[j] << " = " <<  << std::endl;
        bits[j]=int(round(bitvec[j]));
    }

    return bits.to_ulong();
};


#endif
