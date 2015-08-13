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

#include "neuralnettrainer.cuh"


/*-----Load the training data---------


--------------------------------------*/
void fpn::cuNeuralNetworkTrainer::m_loadTrainingData() {
    std::ifstream infile(inparams.fname.c_str());

    inputData.reserve (inparams.tss*(inparams.idEnd-inparams.idBeg));
    expectData.reserve(inparams.tss*(inparams.edEnd-inparams.edBeg));

    for (int i=0;i<inparams.tss;++i)
    {
        std::string line;
        getline(infile,line);
        pcsvreader(line,inputData ,inparams.idBeg,inparams.idEnd,1.0,0.0);
        pcsvreader(line,expectData,inparams.edBeg,inparams.edEnd,1.0,0.0);
    }

    infile.close();
};

/*--------Set Data on Device----------


--------------------------------------*/
void fpn::cuNeuralNetworkTrainer::m_setDataOnDevice() {
    int ns = inputData.size ();
    int nd = expectData.size();

    if (ns == 0 || nd == 0)
        fpnThrowHandler(std::string("Input data sets cannot be empty!"));

    std::cout << "Allocating space and loading training data to the device...\n";

    /* Allocate Device Data */
    cudaThrowHandler(cudaMalloc((void**)&srcData_d,ns*sizeof(float)));
    cudaThrowHandler(cudaMalloc((void**)&cmpData_d,nd*sizeof(float)));

    /* Copy Data */
    cudaThrowHandler(cudaMemcpy(srcData_d,&inputData[0] ,ns*sizeof(float),cudaMemcpyHostToDevice));
    cudaThrowHandler(cudaMemcpy(cmpData_d,&expectData[0],nd*sizeof(float),cudaMemcpyHostToDevice));
};

/*------Clear Data From Device--------


--------------------------------------*/
void fpn::cuNeuralNetworkTrainer::m_clearDataFromDevice() {
        cudaThrowHandler(cudaFree(srcData_d));
        cudaThrowHandler(cudaFree(cmpData_d));
};
