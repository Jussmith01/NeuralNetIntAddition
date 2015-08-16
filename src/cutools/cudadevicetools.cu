/*----------------------------------------------
        Written by Justin Smith ~August 2015
        E-Mail Jussmith48@gmail.com
        Copyright the Roitberg research group
        Chemistry Department
        University of Florida
        Gainesville FL.
------------------------------------------------*/
#include <cmath>

#include "cudadevicetools.cuh"

__constant__ unsigned int dc_N[1];

// ****************************************** //
// *****FLOATS DEVICE SIDE KERNEL PROGRAM**** //
// ****************************************** //
__global__ void cudaHadamardProductKernel(const float *data1,float *data2) {
    /* Define Index */
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    /* Calculate and save results */
    if (i<dc_N[0])
        data2[i] = data1[i]*data2[i];
};

// ****************************************** //
// ***********KERNEL LAUNCHER PROGRAM******** //
// ****************************************** //
cudaError_t devt::cuHadamardProduct(const float *d_data1,float *d_data2,const unsigned int N,const int MAX_THREADS) {
    int BLOCKS = std::ceil(N/float(MAX_THREADS));
    int THREADS = MAX_THREADS;

    /* Copy Size to Constant */
    cudaMemcpyToSymbol(dc_N,&N,sizeof(int));

    /* Launch Integration Kernel */
    cudaHadamardProductKernel<<<BLOCKS,THREADS>>>(d_data1,d_data2);

    /* Function Implicit Sync */
    cudaDeviceSynchronize();

    cudaError_t _err = cudaPeekAtLastError();

    return _err;
};
