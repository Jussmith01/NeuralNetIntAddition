/*----------------------------------------------
        Written by Justin Smith ~August 2015
        E-Mail Jussmith48@gmail.com
        Copyright the Roitberg research group
        Chemistry Department
        University of Florida
        Gainesville FL.
------------------------------------------------*/
#ifndef TOOLS_HPP
#define TOOLS_HPP

#include <vector>
#include <bitset>
#include <limits.h>
#include <math.h>
#include <cstring>
#include <iostream>

namespace tools {

/*--------Calculate Prime Factors----------

Obtains and returns a std::vector of the
primefactors of the argument int n.

--------------------------------------*/
inline std::vector<int> primeFactors(int n) {
    std::vector<int> pfrtn;

    // Print the number of 2s that divide n
    while (n%2 == 0) {
        pfrtn.push_back(2);
        n = n/2;
    }

    // n must be odd at this point.  So we can skip one element (Note i = i +2)
    for (int i = 3; i <= sqrt(n); i = i+2) {
        // While i divides n, print i and divide n
        while (n%i == 0) {
            pfrtn.push_back(i);
            n = n/i;
        }
    }

    // This condition is to handle the case whien n is a prime number
    // greater than 2
    if (n > 2)
        pfrtn.push_back(n);

    return pfrtn;
};


}

#endif
