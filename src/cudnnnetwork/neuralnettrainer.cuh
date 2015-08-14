#ifndef NNTRAINER_CU
#define NNTRAINER_CU

#include <unistd.h>

#include "cudnnlayer_t.cuh"
#include "neuralnetbase.cuh"

namespace fpn { // Force Prediction Network

struct csvdataStructure {
    std::string fname;
    unsigned int idBeg,idEnd;
    unsigned int edBeg,edEnd;
    unsigned int tss; //training set size

    csvdataStructure(std::string fname,unsigned int idBeg,unsigned int idEnd,
                                       unsigned int edBeg,unsigned int edEnd,
                                       unsigned int tss) :
        fname(fname),idBeg(idBeg),idEnd(idEnd),edBeg(edBeg),edEnd(edEnd),tss(tss)
    {};
};

//________________________________________________________________________//
//      *************************************************************     //
//                        NeuralNet Trainer Class
//                 Carries out training of the base network
//      *************************************************************     //
class cuNeuralNetworkTrainer : public cuNeuralNetworkbase {
    /* Input Data */
    csvdataStructure inparams;

    /* Host Data Storage */
    std::vector<float> inputData;
    std::vector<float> expectData;

    /* Device Data Pointers */
    float* srcData_d;
    float* cmpData_d;

    /* Private Functions */
    void m_loadTrainingData();
    void m_setDataOnDevice();
    void m_clearDataFromDevice();

public:

    cuNeuralNetworkTrainer(csvdataStructure datastruct,std::string init,Initializers type) :
            cuNeuralNetworkbase(init,type),inparams(datastruct) {
        std::cout << "Building Training Class and Loading Training Data!\n";
        m_loadTrainingData();
        m_setDataOnDevice();
    }

    void trainNetwork() {
        std::cout << " Training Network\n";
        feedForward(inputData.size(),srcData_d,expectData.size(),cmpData_d,inparams.tss);
    };

    void testNetwork() {
        std::cout << " Testing Network\n";
    };

    void saveNetwork() {
        std::cout << " Saving Network\n";
    };

    void clearNetwork() {
        std::cout << " Clearing Network\n";
        inputData.clear();
        expectData.clear();
        m_clearDataFromDevice();
    };

    ~cuNeuralNetworkTrainer() {};
};

};
#endif
