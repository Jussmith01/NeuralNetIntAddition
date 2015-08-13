// STD Lib Headers
#include <iostream>
#include <vector>
#include <sstream>
#include <string>
#include <cstring>
#include <regex>

// CUDA Headers
#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>

#include "cudaerrorhandling.cuh"
#include "cudahosttools.cuh"
#include "curandhosttools.cuh"

#include "neuralnetbase.cuh"

/*--------Setup CUDA Devices----------

Obtains number of devices, and device
properties.

--------------------------------------*/
void fpn::cuNeuralNetworkbase::m_setupCudaDevice() {
    cudaThrowHandler(cudaGetDeviceCount(&numdevice));
    devprops.resize(numdevice);

    /*GET CUDA DEVICE PROPERTIES*/
    for (int i=0; i<int(devprops.size()); ++i)
        cudaThrowHandler(cudaGetDeviceProperties(&devprops[i],i));

    printDevProps(devprops);

    cudaThrowHandler(cudaSetDevice(1));
};

/*-------Setup CUDA Libraries---------

Creates cuDNN handles/descriptors and
cuBLAS handles for use by the child
classes.

--------------------------------------*/
void fpn::cuNeuralNetworkbase::m_createHandles() {
    std::cout << "Creating cuDNN Handles!" << "\n";
    cudnnThrowHandler(cudnnCreate(&cudnnHandle));
    cudnnThrowHandler(cudnnCreateTensorDescriptor(&srcTensorDesc));
    cudnnThrowHandler(cudnnCreateTensorDescriptor(&dstTensorDesc));
    std::cout << " Running cuDNN version: " << cudnnGetVersion() << "\n\n";

    std::cout << "Creating cuBLAS Handles!" << "\n";
    cublasThrowHandler( cublasCreate(&cublasHandle) );
    int version;
    cublasGetVersion(cublasHandle,&version);
    std::cout << " Running cuBLAS version: " << version << "\n\n";
};

/*-------Destroy CUDA Libraries-------

Destroys cuDNN handles/descriptors and
cuBLAS handles that were defined in:

fpn::cuNeuralNetworkbase::m_createHandles()

--------------------------------------*/
void fpn::cuNeuralNetworkbase::m_destroyHandles() {
    std::cout << "Destroying cuDNN Handles!" << "\n";
    cudnnThrowHandler(cudnnDestroyTensorDescriptor(dstTensorDesc));
    cudnnThrowHandler(cudnnDestroyTensorDescriptor(srcTensorDesc));
    cudnnThrowHandler(cudnnDestroy(cudnnHandle));

    std::cout << "Destroying cuBLAS Handles!" << "\n";
    cublasThrowHandler( cublasDestroy(cublasHandle) );
};

/*-------Create Neural Network-------

--------------------------------------*/
void fpn::cuNeuralNetworkbase::m_createNetwork(const std::string templateString) {
    std::regex pattern_nntformat("^([0-9]{1,8}:){1,64}[0-9]{1,8}$"); // Ensure proper network template formatting
    if (!std::regex_search(templateString,pattern_nntformat))
        {fpnThrowHandler(std::string("The network creation template syntax is incorrect."));}

    std::cout << "Creating a Neural Network from template " << std::endl;
    std::vector<unsigned int> netarch(m_parseNetworkTemplate(templateString));

    inlayersize = netarch.front();
    std::cout << " Neural net architecture requested: \n  Input layer size=" << netarch.front() << " followed by layers of size ";
    unsigned int Nw = 0; int Nb = 0;
    std::vector<unsigned int>::iterator it;
    for (it=netarch.begin()+1;it!=netarch.end();++it)
    {
        std::cout << *it << " ";
        Nw += *(it-1) * *it;
        Nb += *it;
    }

    wbdataSize = ((Nw + Nb) * sizeof(float)) / float(1024*1024);
    std::cout << "\n  Num. Weights: " << Nw << " -- Num. Biases: " << Nb << " required" << std::endl;
    std::cout << "  Network Device Memory Cost: " << wbdataSize << "MB" << std::endl;

    std::cout << "  Generating random weights and biases w/ cuRAND! " << std::endl;
    std::vector<float> rn;
    fpn::curandGenRandomFloats(rn,Nw+Nb);

    std::cout << "\n Building Neural Network Layers: " << std::endl;
    unsigned long long int idx=0;
    for (it=netarch.begin()+1;it!=netarch.end();++it)
    {
        Nw = *(it-1) * *it;
        Nb = *it;
        std::cout << "   Layer " << it-netarch.begin()-1  << " w/ " << Nw << " weights and " << Nb << " bias.\n";

        std::vector<float> weight(Nw);
        std::vector<float> bias(Nb);

        std::memcpy(&weight[0],&rn[idx   ],Nw*sizeof(float));
        std::memcpy(&bias  [0],&rn[idx+Nw],Nb*sizeof(float));

        idx += Nw+Nb;

        // Locally construct the class and emplace it on the layers vector
        layers.emplace_back(weight,bias);
        // Load the data to the devices, must call clearDevice() to reset device data.
        layers.back().loadDevice();
    }

    //std::cout << " Neural net architecture requested: " << std::endl;

};

/*-------Create Neural Network-------

--------------------------------------*/
std::vector<unsigned int> fpn::cuNeuralNetworkbase::m_parseNetworkTemplate(const std::string templateString) {
    std::vector<unsigned int> netarch;
    std::string wks(templateString);

    while (wks.find_first_of(":")!=std::string::npos)
    {
        size_t pos = wks.find_first_of(":");
        netarch.push_back(atoi(wks.substr(0,pos).c_str()));
        wks = wks.substr(pos+1);
    }

    netarch.push_back(atoi(wks.c_str()));

    return netarch;
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
