/*----------------------------------------------
        Written by Justin Smith ~August 2015
        E-Mail Jussmith48@gmail.com
        Copyright the Roitberg research group
        Chemistry Department
        University of Florida
        Gainesville FL.
------------------------------------------------*/
#ifndef NNBASE_CU
#define NNBASE_CU

#include "../cutools/cublashosttools.cuh"
#include "cudnnlayer_t.cuh"

namespace fpn { // Force Prediction Network

/*--------cuNeuralNetworkbase Initializers----------

These initializers have different effects when used
to construct the cuNeuralNetworkbase class.

CREATE means that a network will be randomly generated
LOAD means that a network will be loaded from a file
TRAIN allocate memory for storing derivatives so that
      the networks cost can be minimized in training
TEST Only allocate what is necessary to feed forward,
     in other words no space is allocated for training
     the network, only feeding forward.

----------------------------------------------------*/
enum  Initializers {
    FPN_CREATE_AND_TRAIN,
    FPN_LOAD_AND_TEST,
    FPN_LOAD_AND_TRAIN
};

/*-----------------------NeuralNet Base Class-----------------------

      Class holds all required functionality for creating or loading
      a deep neural network and training or testing on that network.
      The class does this by either randomly initializing a network
      or loading a previously defined network. Then defined a network
      or layer types which handle their own data individually. This
      class then works with the layer types to carryout different
      feed forward and back propagation algoithms to minimize the
      cost of the neural network.

      NOTE: Currently only ReLU layers are used in the layer type by
      the name of ReLUlayer_t. I want to possibly change this later
      to work with a general layer_t type rather than something so
      specific.

      The constuctor is initialized with the following arguments:
      1) const std::string networknfo
          This passes in either a file to load or a network template
          string. This is decided based on the initializer defined
          below.

      2) enum Initializers setup
          There are three current enums to initialize the class.
          FPN_CREATE_AND_TRAIN : Create a network, Train the network
          FPN_LOAD_AND_TEST : Load a network, test the network
          FPN_LOAD_AND_TRAIN : Load a network, train the network

        Written by Justin Smith ~August 2015
        E-Mail Jussmith48@gmail.com
---------------------------------------------------------------------*/
class cuNeuralNetworkbase {

    // cuda Information
    int numdevice; // Number of devices detected on the system
    std::vector<cudaDeviceProp> devprops; // Holds each devices properties

    // cuDNN Handles
    cudnnHandle_t cudnnHandle;

    // cuBLAS Handle
    cublasHandle_t cublasHandle;

    // Network Layers
    bool trainer; // Determines whether or not to expect training
    int inlayersize;
    long long int wbdataSize; // Weights and Bias total data size
    std::vector<ReLUlayer_t> layers;

    /*-------------Class Private functions----------------*/
    /*  Fuction for creating needed cuda handlers   */
    void m_createHandles(void);

    /*  Fuction for destroying needed cuda handlers   */
    void m_destroyHandles(void);

    /* Get system device information and setup devices */
    void m_setupCudaDevice(void);

    /*  Network Creator (Currently randomly initializes a networks weights and biases) */
    void m_createNetwork(const std::string file);

    /*  Network Save */
    void m_saveNetwork(const std::string &file);

    /*  Network Load */
    void m_loadNetwork(const std::string &file);

    /*  Builds the individual layers from input weights and biases */
    void m_buildLayers(std::vector<float> data_wh,std::vector<float> data_bh);

    /* Reads a network template fule */
    std::vector<unsigned int> m_parseNetworkTemplate(const std::string templateString);

    /*-------Primary Constructor--------*/
    cuNeuralNetworkbase () {
        // Get cuda device information and setup device
        m_setupCudaDevice();

        // Create cudnn Handle
        m_createHandles();
    };

public:

    /*   Network Creation/Load Constructor     */
    cuNeuralNetworkbase(const std::string networknfo,enum Initializers setup) : cuNeuralNetworkbase() {
        switch (setup) {
        /* This case creates a new network from a network template string stored in
        network nfo and sets it up */
        case FPN_CREATE_AND_TRAIN: {
                trainer=true; // Always expect training on data upon creation
                m_createNetwork(networknfo);
                break;
            };

        /* This case loads a network from a saved network file and sets it up */
        case FPN_LOAD_AND_TEST: {
            trainer=false; // Load network and optimize data
            m_loadNetwork(networknfo);
            break;
        };

        /* This case loads a network from a saved network file and sets it up */
        case FPN_LOAD_AND_TRAIN: {
            trainer=true; // Load network and train on data
            m_loadNetwork(networknfo);
            break;
        };
        }
    }

    /*     Destructor      */
    ~cuNeuralNetworkbase () {
        m_saveNetwork("networkData.nnf");

        std::cout << "Cleaning up the Neural Network Base class!" << std::endl;

        // Clear Layers
        while (!layers.empty()) {
            layers.back().clearDevice();
            layers.pop_back();
        }

        // Clear device properties
        devprops.clear();

        // Destroy all cuda Handle
        m_destroyHandles();
    };

    /*-------------Class Public functions----------------*/

    /*   Feed Forward Training Function    */
    void feedForwardTrainer(int Ns,const float *srcData,int No,const float *cmpData,int Nd);

    /*   Feed Forward Comparing Function    */
    void feedForwardCompare(int Ns,const float *srcData,int No,const float *cmpData,int Nd);

    void backPropagate(float *srcData,float **dstData) {};
};

};
#endif
