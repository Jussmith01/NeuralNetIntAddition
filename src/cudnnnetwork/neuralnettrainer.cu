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

#include "neuralnettrainer.cuh"


/*-----Load the training data---------


--------------------------------------*/
void fpn::cuNeuralNetworkTrainer::m_loadTrainingData(csvdataStructure &dfname) {
    std::fstream infile(dfname.fname.c_str(), )

    infile.close();
};

/*--------Set Data on Device----------


--------------------------------------*/
void fpn::cuNeuralNetworkTrainer::m_setDataOnDevice() {

};

/*------Clear Data From Device--------


--------------------------------------*/
void fpn::cuNeuralNetworkTrainer::m_clearDataFromDevice() {

};
