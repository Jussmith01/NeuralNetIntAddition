// STD Lib Headers
#include <iostream>
#include <vector>
#include <sstream>
#include <string>

// CUDA Headers
#include <cuda.h>
#include <cudnn.h>

#include "cudaerrorhandling.cuh"
#include "cudadevicestools.cuh"

#include "neuralnetbase.cuh"

void cuNeuralnetbase::setupCudaDevice()
{
    cudaErrorHandler(cudaGetDeviceCount(&numdevice));
    devprops.resize(numdevice);

    /*GET CUDA DEVICE PROPERTIES*/
    for (int i=0;i<int(devprops.size());++i)
        cudaErrorHandler(cudaGetDeviceProperties(&devprops[i],i));

    printDevProps(devprops);

    cudaErrorHandler(cudaSetDevice(1));
};
