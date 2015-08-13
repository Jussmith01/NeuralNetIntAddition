// STD Lib Headers
#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <string>
#include <cstring>
#include <regex>
#include <signal.h>

// CUDA Headers
#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>

#include "../errorhandling.h"
#include "../tools/csvreader.hpp"

#include "../cutools/cudahosttools.cuh"
#include "../cutools/curandhosttools.cuh"

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

    cudaThrowHandler(cudaSetDevice(0));
};

/*-------Setup CUDA Libraries---------

Creates cuDNN handles/descriptors and
cuBLAS handles for use by the child
classes.

--------------------------------------*/
void fpn::cuNeuralNetworkbase::m_createHandles() {
    std::cout << "Creating cuDNN Handle!" << "\n";
    cudnnThrowHandler(cudnnCreate(&cudnnHandle));
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
    std::cout << "Destroying cuDNN Handle!" << "\n";
    cudnnThrowHandler(cudnnDestroy(cudnnHandle));
    std::cout << "Destroying cuBLAS Handles!" << "\n";
    cublasThrowHandler( cublasDestroy(cublasHandle) );
};

/*-------Create Neural Network-------

--------------------------------------*/
void fpn::cuNeuralNetworkbase::m_createNetwork(const std::string templateString) {
    std::regex pattern_nntformat("^([0-9]{1,8}:){1,64}[0-9]{1,8}$"); // Ensure proper network template formatting
    if (!std::regex_search(templateString,pattern_nntformat)) {
        fpnThrowHandler(std::string("The network creation template syntax is incorrect."));
    }

    std::cout << "Creating a Neural Network from template " << std::endl;
    std::vector<unsigned int> netarch(m_parseNetworkTemplate(templateString));

    inlayersize = netarch.front();
    std::cout << " Neural net architecture requested: \n  Input layer size=" << netarch.front() << " followed by layers of size ";
    unsigned int Nw = 0;
    int Nb = 0;
    std::vector<unsigned int>::iterator it;
    for (it=netarch.begin()+1; it!=netarch.end(); ++it) {
        std::cout << *it << " ";
        Nw += *(it-1) **it;
        Nb += *it;
    }

    wbdataSize = ((Nw + Nb) * sizeof(float)) / float(1024*1024);
    if (trainer) {wbdataSize*=2.0;wbdataSize+=((Nb*sizeof(float))/float(1024*1024));} // Memory req doubled for training (derivatives)

    std::cout << "\n  Num. Weights: " << Nw << " -- Num. Biases: " << Nb << " required" << std::endl;
    std::cout << "  Network Device Memory Cost: " << wbdataSize << "MB" << std::endl;

    std::cout << "  Generating random weights and biases w/ cuRAND! " << std::endl;
    std::vector<float> rn;
    fpn::curandGenRandomFloats(rn,Nw+Nb);

    std::cout << "\n Building Neural Network Layers: " << std::endl;
    unsigned long long int idx=0;
    for (it=netarch.begin()+1; it!=netarch.end(); ++it) {
        Nw = *(it-1) **it;
        Nb = *it;
        std::cout << "   Layer " << it-netarch.begin()-1  << " w/ " << Nw << " weights and " << Nb << " bias.\n";

        std::vector<float> weight(Nw);
        std::vector<float> bias(Nb);

        std::memcpy(&weight[0],&rn[idx]   ,Nw*sizeof(float));
        std::memcpy(&bias[0]  ,&rn[idx+Nw],Nb*sizeof(float));

        idx += Nw+Nb;

        // Locally construct the class and emplace it on the layers vector
        layers.emplace_back(weight,bias,&cudnnHandle,&cublasHandle,trainer);
        // Load the data to the devices, must call clearDevice() to reset device data.
        layers.back().loadToDevice();
    }
};

/*-------Parse Network Template-------

Parse a string formatted as:
16:32:48:2
into a vector of uints. This is used
to define the neural network archit-
ecture.

--------------------------------------*/
std::vector<unsigned int> fpn::cuNeuralNetworkbase::m_parseNetworkTemplate(const std::string templateString) {
    std::vector<unsigned int> netarch;
    std::string wks(templateString);

    while (wks.find_first_of(":")!=std::string::npos) {
        size_t pos = wks.find_first_of(":");
        netarch.push_back(atoi(wks.substr(0,pos).c_str()));
        wks = wks.substr(pos+1);
    }

    netarch.push_back(atoi(wks.c_str()));

    return netarch;
};

/*---------Save Network Data----------

Load the network from the GPU and save
it to a file named in the argument
fname.

--------------------------------------*/
void fpn::cuNeuralNetworkbase::m_saveNetwork(const std::string &fname) {

    if (!layers.empty()) {
        std::cout << "\nSaving network data!" << std::endl;
        std::ofstream dataFile (fname);
        if (!dataFile) {
            std::stringstream _error;
            _error << "Error creating file: " << fname;
            fpnThrowHandler(_error.str());
        }

        for (auto l : layers) {
            l.loadFromDevice();
            dataFile << "$STARTLAYER\n";

            dataFile << "weights=";
            for (auto w : l.weightAccess())
                dataFile << w << ",";

            dataFile << "\n";

            dataFile << "biases=";
            for (auto b : l.biasAccess())
                dataFile << b << ",";

            dataFile << "\n";
        }

        dataFile.close();
    } else {
        std::cout << "\nCannot save data! Layers not loaded." << std::endl;
    }
};

/*---------Load Network Data----------

Load the network from the GPU and save
it to a file named in the argument
fname.

--------------------------------------*/
void fpn::cuNeuralNetworkbase::m_loadNetwork(const std::string &fname) {
    std::regex pattern_nnffile(".*\\.nnf$"); // Ensure only .nnf (Neural Network Format) files are given
    if (!std::regex_search(fname,pattern_nnffile)) {
        fpnThrowHandler(std::string("Only .nnf files can be used to construct the cuNeuralNetworkbase class"));
    }

    std::cout << "Loading the Neural Network data from file: " << fname << std::endl;

    std::string line;
    std::ifstream dataFile (fname.c_str());

    if (!dataFile) {
        std::stringstream _error;
        _error << "Error opening file: " << fname;
        fpnThrowHandler(_error.str());
    }

    int expline=-1;
    bool SAVE=false;
    if (dataFile.is_open()) {

        std::vector<float> weight_v,bias_v;
        while ( getline (dataFile,line) ) {
            if (expline==1) {
                std::string bias_s(line.substr(line.find_first_of("=")+1));
                csvreader(bias_s,bias_v);
                expline=-1;
                SAVE=true;
            }

            if (expline==0) {
                std::string weight_s(line.substr(line.find_first_of("=")+1));
                csvreader(weight_s,weight_v);
                expline=1;
            }

            if (line.find("$STARTLAYER")!=std::string::npos)
                expline=0;

            if (SAVE) {
                layers.emplace_back(weight_v,bias_v,&cudnnHandle,&cublasHandle,trainer);
                layers.back().loadToDevice();

                bias_v.clear();
                weight_v.clear();

                SAVE=false;
            }
        }
        dataFile.close();
    } else {
        std::cout << "NOT OPEN!" << std::endl;
    }
};
