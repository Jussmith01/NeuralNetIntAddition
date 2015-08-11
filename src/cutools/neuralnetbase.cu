// STD Lib Headers
#include <iostream>
#include <vector>
#include <sstream>
#include <string>

// CUDA Headers
#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>

#include "cudaerrorhandling.cuh"
#include "cudadevicestools.cuh"

#include "neuralnetbase.cuh"

void cuNeuralNetworkbase::m_setupCudaDevice()
{
    cudaErrorHandler(cudaGetDeviceCount(&numdevice));
    devprops.resize(numdevice);

    /*GET CUDA DEVICE PROPERTIES*/
    for (int i=0; i<int(devprops.size()); ++i)
        cudaErrorHandler(cudaGetDeviceProperties(&devprops[i],i));

    printDevProps(devprops);

    cudaErrorHandler(cudaSetDevice(0));
};

void cuNeuralNetworkbase::m_createHandles()
{
    std::cout << "Creating cuDNN Handles!" << "\n";
    cudnnErrorHandler(cudnnCreate(&cudnnHandle));
    cudnnErrorHandler(cudnnCreateTensorDescriptor(&srcTensorDesc));
    cudnnErrorHandler(cudnnCreateTensorDescriptor(&dstTensorDesc));
    std::cout << " Running cuDNN version: " << cudnnGetVersion() << "\n\n";

    std::cout << "Creating cuBLAS Handles!" << "\n";
    cublasErrorHandler( cublasCreate(&cublasHandle) );
    int version;
    cublasGetVersion(cublasHandle,&version);
    std::cout << " Running cuBLAS version: " << version << "\n\n";
};

void cuNeuralNetworkbase::m_destroyHandles()
{
    std::cout << "\nDestroying cuDNN Handles!" << "\n";
    cudnnErrorHandler(cudnnDestroyTensorDescriptor(dstTensorDesc));
    cudnnErrorHandler(cudnnDestroyTensorDescriptor(srcTensorDesc));
    cudnnErrorHandler(cudnnDestroy(cudnnHandle));

    std::cout << "Destroying cuBLAS Handles!" << "\n";
    cublasErrorHandler( cublasDestroy(cublasHandle) );
};

/*void cuNeuralNetworkbase::fullyConnectedForward(
                           int& n, int& c, int& h, int& w,
                           float* srcData, float** dstData,
                           float* weight_d,float* bias_d)
{
    if (n != 1) {
        FatalError("Not Implemented");
    }
    int dim_x = c*h*w;
    int dim_y = ip.outputs;
    resize(dim_y, dstData);

    float alpha = float(1), beta = float(1);



    // place bias into dstData
    checkCudaErrors( cudaMemcpy(*dstData, bias_d, dim_y*sizeof(float), cudaMemcpyDeviceToDevice) );

    checkCudaErrors( cublasSgemv(cublasHandle, CUBLAS_OP_T,
                                 dim_x, dim_y,
                                 &alpha,
                                 weight_d, dim_x,
                                 srcData, 1,
                                 &beta,
                                 *dstData, 1) );

    h = 1;
    w = 1;
    c = dim_y;
}*/

/*void cuNeuralNetworkbase::activationForward(int n, int c, int h, int w, float* srcData, float** dstData)
{
    cudnnErrorHandler( cudnnSetTensor4dDescriptor(srcTensorDesc,
                       CUDNN_TENSOR_NCHW,
                       CUDNN_DATA_FLOAT,
                       n, c,
                       h,
                       w) );
    cudnnErrorHandler( cudnnSetTensor4dDescriptor(dstTensorDesc,
                       CUDNN_TENSOR_NCHW,
                       CUDNN_DATA_FLOAT,
                       n, c,
                       h,
                       w) );
    float alpha = 1.0f;
    float beta  = 0.0f;
    cudnnErrorHandler( cudnnActivationForward(cudnnHandle,
                       CUDNN_ACTIVATION_RELU,
                       &alpha,
                       srcTensorDesc,
                       srcData,
                       &beta,
                       dstTensorDesc,
                       *dstData) );
};*/
