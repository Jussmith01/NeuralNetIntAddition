/*----------------------------------------------
        Written by Justin Smith ~August 2015
        E-Mail Jussmith48@gmail.com
        Copyright the Roitberg research group
        Chemistry Department
        University of Florida
        Gainesville FL.
------------------------------------------------*/
/* Std. Lib. Includes */
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <regex>
#include <signal.h>

/* CUDA Includes */
#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>

/* Error Handling */
#include "errorhandling.h"

/* Tools */
#include "tools/micro_timer.h"

/* Cuda Neural Network Tools */
#include "cudnnnetwork/neuralnetbase.cuh"
#include "cudnnnetwork/neuralnettrainer.cuh"

inline void sigsegvHandler (int param) {
    signalHandler(std::string("A Segmentation violation has been detected. Attempting to cleanup..."));
}

inline void terminationHandler (int param) {
    signalHandler(std::string("A termination signal has been detected. Attempting to cleanup..."));
}

int main(int argc, char *argv[]) {
    if (argv[1]==NULL)
    {
        std::cout << "Error: Missing arguments!" << std::endl;
        exit(1);
    }

    /* Setup special signaling */
    signal(SIGSEGV,sigsegvHandler);
    signal(SIGINT,terminationHandler);
    signal(SIGTERM,terminationHandler);

    using namespace fpn;

    std::cout << "GNU Version: " << __GNUC__ << "." << __GNUC_MINOR__ << std::endl;

    try {

    inDataStructure dataStruct("trainingData.dat",3,14,15,23,30000);
    //csvdataStructure dataStruct("trainingData.dat",3,4,5,7,3);
    cuNeuralNetworkTrainer nnt(dataStruct,std::string(argv[1]),FPN_CREATE_AND_TRAIN);
    //cuNeuralNetworkTrainer nnt("networkData.nnf",FPN_LOAD_AND_TRAIN);
    nnt.trainNetwork();
    nnt.saveNetwork();
    nnt.clearNetwork();

    } catch (std::string _caught) {

    /* NOTE! This will only work if used within main!

    It's purpose is to catch any throws defined within
    errorhandling.h header file. Doing things in this
    order allows the program to shutdown and save the
    network (as long as the the heap is uncorrupted as
    can occur in a segfault) and release the device and
    memory normally. This ensures that any internal
    problems will result in the network being stored
    for future use. No string catched should be added

    */
    cudaFatalError(_caught);

    }

    signal(SIGSEGV,SIG_DFL);
    signal(SIGINT,SIG_DFL);
    signal(SIGTERM,SIG_DFL);

    return 0;
};
