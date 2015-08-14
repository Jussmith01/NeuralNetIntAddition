#ifndef CUBLAS_HOST_TOOLS_CU
#define CUBLAS_HOST_TOOLS_CU

#include "cudahosttools.cuh"
#include <cuda.h>
#include <cublas_v2.h>

namespace fpn {

/*------Mean Squared Error Cost Function--------

Calculates cost C as:

C = 0.5 * sum((xi-yi)^2)

where xi is expected output and yi is actual.

------------------------------------------------*/
void meanSquaredErrorCostFunction(cublasHandle_t &cublasHandle,int Nd,int No,float* srcData, float* dstData) {

    float alpha = float(-1.0);

    /* Loop over all data sets */
    //for (int i=0; i<Nc; ++i) {
        /* Copy the biases into dstData at a specific index */
        //cudaThrowHandler( cudaMemcpy(dstData+i*dim_y,bias_d,dim_y*sizeof(float),cudaMemcpyDeviceToDevice) );

    printCudaData(No*Nd,srcData,"EXPECTED in dstData: ");
    printCudaData(No*Nd,dstData,"ACTUAL in dstData: ");

        /* SUBTRACTION */
    cublasThrowHandler( cublasSaxpy(cublasHandle,No*Nd,&alpha,dstData,1,srcData,1) );
    //}

    printCudaData(No*Nd,dstData,"DIFFERENCE dstData: ");
};

};

#endif
