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

void cuNeuralNetworkbase::setupCudaDevice()
{
    cudaErrorHandler(cudaGetDeviceCount(&numdevice));
    devprops.resize(numdevice);

    /*GET CUDA DEVICE PROPERTIES*/
    for (int i=0; i<int(devprops.size()); ++i)
        cudaErrorHandler(cudaGetDeviceProperties(&devprops[i],i));

    printDevProps(devprops);

    cudaErrorHandler(cudaSetDevice(0));
};

void cuNeuralNetworkbase::createHandles()
{
    std::cout << "Running cuDNN version: " << cudnnGetVersion() << "\n";
    std::cout << "Creating cuDNN Handles!" << "\n\n";
    cudnnErrorHandler(cudnnCreate(&cudnnHandle));
    cudnnErrorHandler(cudnnCreateTensorDescriptor(&srcTensorDesc));
    cudnnErrorHandler(cudnnCreateTensorDescriptor(&dstTensorDesc));
};

void cuNeuralNetworkbase::destroyHandles()
{
    std::cout << "Destroying cuDNN Handles!" << "\n\n";
    cudnnErrorHandler(cudnnDestroyTensorDescriptor(dstTensorDesc));
    cudnnErrorHandler(cudnnDestroyTensorDescriptor(srcTensorDesc));
    cudnnErrorHandler(cudnnDestroy(cudnnHandle));
};

void cuNeuralNetworkbase::fullyConnectedForward(const Layer_t& ip,
                           int& n, int& c, int& h, int& w,
                           value_type* srcData, value_type** dstData)
{
    if (n != 1) {
        FatalError("Not Implemented");
    }
    int dim_x = c*h*w;
    int dim_y = ip.outputs;
    resize(dim_y, dstData);

    value_type alpha = value_type(1), beta = value_type(1);
    // place bias into dstData
    checkCudaErrors( cudaMemcpy(*dstData, ip.bias_d, dim_y*sizeof(value_type), cudaMemcpyDeviceToDevice) );

    checkCudaErrors( cublasSgemv(cublasHandle, CUBLAS_OP_T,
                                 dim_x, dim_y,
                                 &alpha,
                                 ip.data_d, dim_x,
                                 srcData, 1,
                                 &beta,
                                 *dstData, 1) );

    h = 1;
    w = 1;
    c = dim_y;
}

void cuNeuralNetworkbase::activationForward(int n, int c, int h, int w, float* srcData, float** dstData)
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
};
