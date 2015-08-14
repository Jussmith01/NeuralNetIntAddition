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
inline void meanSquaredErrorCostFunction(cublasHandle_t &cublasHandle,int Nd,int No,float* srcData, float* dstData) {

    /* Loop over all data sets */
    //for (int i=0; i<Nc; ++i) {
        /* Copy the biases into dstData at a specific index */
        //cudaThrowHandler( cudaMemcpy(dstData+i*dim_y,bias_d,dim_y*sizeof(float),cudaMemcpyDeviceToDevice) );

    printCudaData(No,srcData,"EXPECTED in dstData: ");
    printCudaData(No,dstData,"ACTUAL in dstData: ");

    /* SUBTRACTION */
    float alpha = float(-1.0);
    cublasThrowHandler( cublasSaxpy(cublasHandle,No,&alpha,dstData,1,srcData,1) );
    printCudaData(No,srcData,"DIFFERENCE srcData: ");

    /* SQUARING */
    alpha = 1.0;
    float beta = float(0.0);
    cublasThrowHandler( cublasSgemv(cublasHandle, CUBLAS_OP_T,
                                    1, No,
                                    &alpha,
                                    srcData, 1,
                                    srcData, 1,
                                    &beta,
                                    dstData, 1) );

    printCudaData(No,dstData,"SQUARE dstData: ");
};

};

#endif
