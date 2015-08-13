// STD Lib Headers
#include <iostream>
#include <vector>
#include <sstream>
#include <string>
#include <regex>
#include <unistd.h>

// CUDA Headers
#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>

#include "../errorhandling.h"

#include "cudnnlayer_t.cuh"

// Declare Layer ID Counter
int fpn::ReLUlayer_t::IDindex = 0;

/*--------Resize a CUDA Container-------


--------------------------------------*/
void fpn::ReLUlayer_t::m_resize(int size, float **data) {
    if (*data != NULL) {
        cudaThrowHandler( cudaFree(*data) );
    }
    cudaThrowHandler( cudaMalloc(data, size*sizeof(float)) );
}

/*-----Load Layer Data to Device------


--------------------------------------*/
void fpn::ReLUlayer_t::m_loadDataToDevice() {
    int nw = weight_h.size();
    int nb =   bias_h.size();

    if (nw == 0 || nb == 0)
        fpnThrowHandler(std::string("Weights and/or biases cannot be empty."));

    cudnnThrowHandler(cudnnCreateTensorDescriptor(&srcTensorDesc));
    cudnnThrowHandler(cudnnCreateTensorDescriptor(&dstTensorDesc));

    /* Allocate Weights and Bias on Device */
    if (!dataLoad) {
        cudaThrowHandler(cudaMalloc((void**)&weight_d,nw*sizeof(float)));
        cudaThrowHandler(cudaMalloc((void**)&bias_d  ,nb*sizeof(float)));
    }

    /* Allocate Cost Derivatives on Device */
    if (trainer) {
        cudaThrowHandler(cudaMalloc((void**)&dCdw_d,nw*sizeof(float)));
        cudaThrowHandler(cudaMalloc((void**)&dCdb_d,nb*sizeof(float)));
        cudaThrowHandler(cudaMalloc((void**)&Z_d,nb*sizeof(float)));
    }


    /* Copy Data */
    cudaThrowHandler(cudaMemcpy(weight_d,&weight_h[0] ,nw*sizeof(float),cudaMemcpyHostToDevice));
    cudaThrowHandler(cudaMemcpy(bias_d  ,&bias_h[0]   ,nb*sizeof(float),cudaMemcpyHostToDevice));

    dataLoad=true;
};

/*--------Clear Data on Device--------


--------------------------------------*/
void fpn::ReLUlayer_t::m_clearDataOnDevice() {
    weight_h.clear();
    bias_h.clear  ();

    cudaThrowHandler(cudaDeviceSynchronize());

    cudnnThrowHandler(cudnnDestroyTensorDescriptor(dstTensorDesc));
    cudnnThrowHandler(cudnnDestroyTensorDescriptor(srcTensorDesc));

    if (dataLoad) {
        cudaThrowHandler(cudaFree(weight_d));
        cudaThrowHandler(cudaFree(bias_d  ));
    }

    if (trainer) {
        cudaThrowHandler(cudaFree(dCdw_d));
        cudaThrowHandler(cudaFree(dCdb_d));
        cudaThrowHandler(cudaFree(Z_d));
    }

    dataLoad=false;
};

/*-----Retrieve Data from Device------


--------------------------------------*/
void fpn::ReLUlayer_t::m_retriveDataFromDevice() {
    int nw = weight_h.size();
    int nb =   bias_h.size();

    if (dataLoad) {
        cudaThrowHandler(cudaMemcpy(&weight_h[0],weight_d,nw*sizeof(float),cudaMemcpyDeviceToHost));
        cudaThrowHandler(cudaMemcpy(&bias_h[0]  ,bias_d  ,nb*sizeof(float),cudaMemcpyDeviceToHost));
    }
};

/*------Fully Connected Forward--------

Fully connected forward is called to
calculate the z values, which are used
to calculate the activations.

--------------------------------------*/
void fpn::ReLUlayer_t::fullyConnectedForward(int c,float* srcData, float** dstData) {
    if (n != 1) {
        fpnThrowHandler(std::string("Not Implemented"));
    }

    // c = data points
    // w = weights
    // b = biases
    // n = feature maps - NOT USED! Always 1;

    int dim_x = w*b;
    int dim_y = b;
    m_resize(dim_y, dstData);

    float alpha = float(1), beta = float(1);


    int ils = w/b; // Input layer size

    // place bias into dstData
    for (int i=0; i<c; ++i) {
        cudaThrowHandler( cudaMemcpy(dstData[c*b], bias_d, dim_y*sizeof(float), cudaMemcpyDeviceToDevice) );

        cublasThrowHandler( cublasSgemv(*cublasHandle, CUBLAS_OP_T,
                                        dim_x, dim_y,
                                        &alpha,
                                        weight_d, dim_x,
                                        &srcData[c*ils], 1,
                                        &beta,
                                        dstData[c*b], 1) );
    }
}

/*void cuNeuralNetworkbase::activationForward(int n, int c, int h, int w, float* srcData, float** dstData) {
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
