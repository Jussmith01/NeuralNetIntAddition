#ifndef CUDNNLAYERS_T_CU
#define CUDNNLAYERS_T_CU

namespace fpn
{
//________________________________________________________________________//
//      *************************************************************     //
//                           ReLU Layer Type
//                Hold data pointer and functionality for
//              the cuDNN interface to simulate a ReLU network
//      *************************************************************     //
class ReLUlayer_t {
    /* Layer Indexing Data */
    static int IDindex;
    int ID;

    /* Host Vectors */
    std::vector<float> weight_h,bias_h;

    // cuDNN Handles
    cudnnHandle_t *cudnnHandle;

    // cuBLAS Handle
    cublasHandle_t *cublasHandle;

    /* Device Pointers */
    bool dataLoad; // Is the network data set?
    float *weight_d,*bias_d; // Weights and biases
    float *dCdw_d,*dCdb_d; // Derivatives
    float *Z_d; // Z for calculating errors

    /* Data Descriptors */
    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
    bool idataSet; // Is the input data set?
    int n,w,b; // Data Descriptors, n defaults to 1, no feature map, w = number of weights, b = number of biases
    bool trainer;

    /* Member Functions */
    void m_loadDataToDevice ();
    void m_clearDataOnDevice ();
    void m_retriveDataFromDevice ();

public:
    ReLUlayer_t (const std::vector<float> &weight,const std::vector<float> &bias,
                 cudnnHandle_t *cudnnHandle,cublasHandle_t *cublasHandle,bool traintype) :
        weight_h(weight),bias_h(bias),dataLoad(false),idataSet(false),n(1),
        cudnnHandle(cudnnHandle),cublasHandle(cublasHandle),trainer(traintype)
    {
        ID=IDindex;
        ++IDindex;
    };

    /*--------------PUBLIC FUNCTIONS---------------*/
    void loadToDevice() {
        std::cout << "   Loading Layer " << ID << " Data to Device..." << std::endl;
        m_loadDataToDevice();
    }

    void loadFromDevice() {
        std::cout << "   Loading Layer " << ID << " Data from Device..." << std::endl;
        m_retriveDataFromDevice();
    }

    void clearDevice() {
        std::cout << "  Clearing Layer " << ID << " from Device..." << std::endl;
        m_clearDataOnDevice();
    }

    /*----------Data Marching Functions------------*/
    void fullyConnectedForward(int c,float* srcData, float* dstData);
    void activationForward(int c,float* srcData, float* dstData);
    void meanSquaredErrorCostFunction(int c,float* srcData, float* dstData);

    /*------------PUBLIC MEMBER ACCESS-------------*/
    const std::vector<float>& weightAccess() {
        return weight_h;
    }

    const std::vector<float>& biasAccess() {
        return bias_h;
    }

    ~ReLUlayer_t () {}
};

};

#endif
