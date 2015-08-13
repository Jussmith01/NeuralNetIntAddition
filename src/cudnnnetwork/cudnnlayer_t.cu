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

/*-----Load Layer Data to Device------


--------------------------------------*/
void fpn::ReLUlayer_t::m_loadDataToDevice() {
    int nw = weight_h.size();
    int nb =   bias_h.size();

    if (nw == 0 || nb == 0)
        fpnThrowHandler(std::string("Weights and/or biases cannot be empty."));

    /* Allocate Device */
    if (!dataLoad)
    {
        cudaThrowHandler(cudaMalloc((void**)&weight_d,nw*sizeof(float)));
        cudaThrowHandler(cudaMalloc((void**)&bias_d  ,nb*sizeof(float)));
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

     if (dataLoad)
     {
        cudaThrowHandler(cudaFree(weight_d));
        cudaThrowHandler(cudaFree(bias_d  ));
     }

     dataLoad=false;
};

/*-----Retrieve Data from Device------


--------------------------------------*/
void fpn::ReLUlayer_t::m_retriveDataFromDevice() {
    int nw = weight_h.size();
    int nb =   bias_h.size();

    if (dataLoad)
    {
        cudaThrowHandler(cudaMemcpy(&weight_h[0],weight_d,nw*sizeof(float),cudaMemcpyDeviceToHost));
        cudaThrowHandler(cudaMemcpy(&bias_h[0]  ,bias_d  ,nb*sizeof(float),cudaMemcpyDeviceToHost));
    }
};

