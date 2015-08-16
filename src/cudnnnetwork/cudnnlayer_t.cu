/*----------------------------------------------
        Written by Justin Smith ~August 2015
        E-Mail Jussmith48@gmail.com
        Copyright the Roitberg research group
        Chemistry Department
        University of Florida
        Gainesville FL.
------------------------------------------------*/
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

#include "../cutools/cudahosttools.cuh"
#include "cudnnlayer_t.cuh"

// Declare Layer ID Counter
int fpn::ReLUlayer_t::IDindex = 0;

/*-----Load Layer Data to Device------

Allocate device space and load data.

--------------------------------------*/
void fpn::ReLUlayer_t::m_loadDataToDevice() {
    w = weight_h.size();
    b =   bias_h.size();

    if ( w == 0 || b == 0 )
        fpnThrowHandler(std::string("Weights and/or biases cannot be empty."));

    cudnnThrowHandler(cudnnCreateTensorDescriptor(&srcTensorDesc));
    cudnnThrowHandler(cudnnCreateTensorDescriptor(&dstTensorDesc));

    /* Allocate Weights and Bias on Device */
    if (!dataLoad) {
        cudaThrowHandler(cudaMalloc((void**)&weight_d,w*sizeof(float)));
        cudaThrowHandler(cudaMalloc((void**)&bias_d  ,b*sizeof(float)));
    }

    /* Allocate Cost Derivatives and Z storage on Device, if training */
    if (trainer) {
        cudaThrowHandler(cudaMalloc((void**)&dCdw_d,w*sizeof(float)));
        cudaThrowHandler(cudaMalloc((void**)&dCdb_d,b*sizeof(float)));
        cudaThrowHandler(cudaMalloc((void**)&Z_d,   b*sizeof(float)));
    }


    /* Copy Data */
    cudaThrowHandler(cudaMemcpy(weight_d,&weight_h[0] ,w*sizeof(float),cudaMemcpyHostToDevice));
    cudaThrowHandler(cudaMemcpy(bias_d  ,&bias_h[0]   ,b*sizeof(float),cudaMemcpyHostToDevice));

    dataLoad=true;
};

/*--------Clear Data on Device--------

Cleanup the device storage.

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

Get weights and biases from the device.

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

    int dim_x = w/b;
    int dim_y = b;
    cu_resize(dim_y*c,dstData);

    float alpha = float(1), beta = float(1);

    /* Copy Biases into the Dest set */
    cu_MemcpySmalltoLargeD2D(c,dim_y,bias_d,dstData);

    cudaDeviceSynchronize();

    /* Feed forward all data via gemm */
    cublasThrowHandler( cublasSgemm(*cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                                    dim_y, c, dim_x,
                                    &alpha,
                                    weight_d, dim_x,
                                    srcData, dim_x,
                                    &beta,
                                    (*dstData),dim_y) );

    //printMatCudaData(dim_y,c,*dstData,"MULT dstData: ");
};

void fpn::ReLUlayer_t::activationForward(int c,float* srcData, float** dstData) {
    cu_resize(b*c,dstData);

    //printCudaData(b*c,srcData,"ACT1 srcData: ");

    cudnnThrowHandler( cudnnSetTensor4dDescriptor(srcTensorDesc,
                       CUDNN_TENSOR_NCHW,
                       CUDNN_DATA_FLOAT,
                       1, n,
                       b,
                       c) );
    cudnnThrowHandler( cudnnSetTensor4dDescriptor(dstTensorDesc,
                       CUDNN_TENSOR_NCHW,
                       CUDNN_DATA_FLOAT,
                       1, n,
                       b,
                       c) );
    float alpha = 1.0f;
    float beta  = 0.0f;
    cudnnThrowHandler( cudnnActivationForward(*cudnnHandle,
                       CUDNN_ACTIVATION_RELU,
                       &alpha,
                       srcTensorDesc,
                       srcData,
                       &beta,
                       dstTensorDesc,
                       *dstData) );

    //printCudaData(b*c,dstData,"ACT2 dstData: ");
};
