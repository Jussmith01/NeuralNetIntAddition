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
inline void meanSquaredErrorCostFunction(cublasHandle_t &cublasHandle,int Nd,int No,const float* srcData, float* dstData) {
    /* SUBTRACTION */
    float alpha = float(-1.0);
    cublasThrowHandler( cublasSaxpy(cublasHandle,No,&alpha,srcData,1,dstData,1) );

    /* DOT PRODUCT THE ENTIRE SET */
    float TotalCost;
    cublasThrowHandler( cublasSdot(cublasHandle,No,dstData,1,dstData,1,&TotalCost) );

    /* AVERAGE */
    std::cout << "   Avg. COST: " << (0.5*TotalCost)/Nd << std::endl;
};

};

#endif
