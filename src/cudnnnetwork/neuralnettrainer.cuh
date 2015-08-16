/*----------------------------------------------
        Written by Justin Smith ~August 2015
        E-Mail Jussmith48@gmail.com
        Copyright the Roitberg research group
        Chemistry Department
        University of Florida
        Gainesville FL.
------------------------------------------------*/

#ifndef NNTRAINER_CU
#define NNTRAINER_CU

#include <unistd.h>

#include "cudnnlayer_t.cuh"
#include "neuralnetbase.cuh"

namespace fpn { // Force Prediction Network

/*--------Input Data Structure----------



--------------------------------------*/
struct inDataStructure {
    std::string fname;
    unsigned int idBeg,idEnd;
    unsigned int edBeg,edEnd;
    unsigned int tss; //training set size

    inDataStructure(std::string fname,unsigned int idBeg,unsigned int idEnd,
                                       unsigned int edBeg,unsigned int edEnd,
                                       unsigned int tss) :
        fname(fname),idBeg(idBeg),idEnd(idEnd),edBeg(edBeg),edEnd(edEnd),tss(tss)
    {};
};

/*----------------------NeuralNet Trainer Class-----------------------

    This class inherits the cuNeuralNetworkbase class and public
    functionality. This class only constructs the cuNeuralNetworkbase
    class to carry out training of a network, be it CREATE and TRAIN
    or LOAD and TRAIN. The sister class to this one
    cuNeuralNetworkCompare is specifically for testing already trained
    classes, and also inherits the cuNeuralNetworkbase class and public
    functionality.

    The constuctor is initialized with the following arguments:
    1) indataStructure datastruct
        This input data structure contains needed parameters for the
        class's operations.
    2) std::string init
        This is the initializer string. Basically either a filename
        or a network template string, based on the initializer type
        parameter.
    3) Initializers type
        There are three current enums to initialize the class.
        FPN_CREATE_AND_TRAIN : Create a network, Train the network
        FPN_LOAD_AND_TEST : Load a network, test the network
        FPN_LOAD_AND_TRAIN : Load a network, train the network

        Written by Justin Smith ~August 2015
        E-Mail Jussmith48@gmail.com
---------------------------------------------------------------------*/
class cuNeuralNetworkTrainer : public cuNeuralNetworkbase {
    /* Input Data */
    inDataStructure inparams;

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
    /* Constructor */
    cuNeuralNetworkTrainer(inDataStructure datastruct,std::string init,Initializers type) :
            cuNeuralNetworkbase(init,type),inparams(datastruct) {
        std::cout << "Building Training Class and Loading Training Data!\n";
        m_loadTrainingData();
        m_setDataOnDevice();
    }

    /* Train the network */
    void trainNetwork() {
        std::cout << " Training Network\n";
        feedForwardTrainer(inputData.size(),srcData_d,expectData.size(),cmpData_d,inparams.tss);
    };

    /* Save the network (Currently in base class destructor, may keep it there.) */
    void saveNetwork() {
        std::cout << " Saving Network\n";
    };

    /* Clear all data from host and device */
    void clearNetwork() {
        std::cout << " Clearing Network\n";
        inputData.clear();
        expectData.clear();
        m_clearDataFromDevice();
    };

    /* Destructor */
    ~cuNeuralNetworkTrainer() {};
};

};
#endif
