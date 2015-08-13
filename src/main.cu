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
//#include "tools/binaryconversion.hpp"

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

    //cuNeuralNetworkTrainer nnt("networkData.nnf",FPN_LOAD_AND_TRAIN);
    csvdataStructure dataStruct("trainingData.dat",3,8,9,13);
    cuNeuralNetworkTrainer nnt(dataStruct,std::string(argv[1]),FPN_CREATE_AND_TRAIN);
    nnt.trainNetwork();
    nnt.testNetwork();
    nnt.saveNetwork();
    nnt.clearNetwork();

    } catch (std::string _caught) {

    cudaFatalError(_caught);

    }

    signal(SIGSEGV,SIG_DFL);
    signal(SIGINT,SIG_DFL);
    signal(SIGTERM,SIG_DFL);
    //nnb.ActivationTest();

    /*double eta = atof(argv[1]);
    int tss = atoi(argv[2]);
    int ess = atoi(argv[3]);
    int hls = atoi(argv[4]);
    int nnd = atoi(argv[5]);
    double con = atof(argv[6]);

    std::cout << "eta: " << eta << " tss: " << tss  << " ess: " << ess << " hls: " << hls << " nnd: " << nnd << " con: " << con << std::endl;


    std::vector<double> input;
    std::vector<double> desired;
    std::vector<double> output;

    RandomInt irandgen; // Train with a set of 10000
    std::vector<int> irand(tss);
    irandgen.FillVector(irand,-1000000,1000000);
    irandgen.Clear();

    NeuralNetwork nn(32,hls,32,nnd,eta);

    int ep=0;
    double avgcost = 100.0;

    microTimer mt;

    while (avgcost>con)
    {
        mt.start_point();

        std::random_shuffle(irand.begin(),irand.end());

        for (auto&& i : irand)
        //for (int i=0;i<(int)irand.size()/2;++i)
        {
            //Begin Neural Network Computation
            input = ProduceBinaryVector(i);
            desired = ProduceBinaryVector(i+1);

            nn.NewTrainingData(input,desired);
            nn.ComputeLayers();
            nn.ComputeDerivatives();
            nn.ResetForNewTrainingData();
        }

        std::cout << " Epoch " << ep << " - ";
        avgcost = nn.CompleteTrainingSet();

        ++ep;

        mt.end_point();
        std::cout << mt.get_generic_print_string(" ") << std::endl;
        mt.reset();
    }

    irand.clear();

    std::cout << " |------Testing Set-------|\n";
    RandomInt itestrandgen; // Train with a set of 10000
    std::vector<int> irandtest(ess);
    itestrandgen.FillVector(irandtest,-1000000,1000000);
    itestrandgen.Clear();

    int correct = 0;
    for (int i=0;i<ess;++i)
    {
        input = ProduceBinaryVector(irandtest[i]);
        desired = ProduceBinaryVector(irandtest[i]+1);

        nn.NewTrainingData(input,desired);
        nn.ComputeLayers();

        nn.GetOutput(output);
        int value = ProduceIntegerFromBinary(output);

        if (value == irandtest[i]+1)
            ++correct;
        else
            std::cout << " Miss! " << value << " != " << irandtest[i]+1 << std::endl;
    }

    std::cout << "Accuracy:" << correct/double(ess) << std::endl;*/

    /*std::cout << "Input:  ";
    for (auto&& op : input)
        std::cout << " " << round(op);
    std::cout << std::endl;

    std::cout << "Desire: ";
    for (auto&& op : desired)
        std::cout << " " << round(op);
    std::cout << std::endl;

    std::cout << "Output: ";
    for (auto&& op : output)
        std::cout << " " << round(op);
    std::cout << std::endl;

    std::cout << "Output: ";
    for (auto&& op : output)
        std::cout << std::setprecision(5) << " " << op;
    std::cout << std::endl;

    nn.Clear();*/

    return 0;
};
