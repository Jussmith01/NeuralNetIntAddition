/*----------------------------------------------
        Written by Justin Smith ~August 2015
        E-Mail Jussmith48@gmail.com
        Copyright the Roitberg research group
        Chemistry Department
        University of Florida
        Gainesville FL.
------------------------------------------------*/

#ifndef CUDA_DEVICE_TOOLS_CU
#define CUDA_DEVICE_TOOLS_CU

#include <cuda.h>

#include "../tools/tools.hpp"

namespace devt {

/*------CUDA Hadamard Product---------

--------------------------------------*/
extern cudaError_t cuHadamardProduct(const float *d_data1,float *d_data2,const unsigned int N,const int MAX_THREADS=1024);

};

#endif
