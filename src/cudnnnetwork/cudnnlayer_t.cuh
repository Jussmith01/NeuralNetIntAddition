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

    /* Device Pointers */
    bool dataLoad;
    float *weight_d,*bias_d;

    /* Member Functions */
    void m_loadDataToDevice ();
    void m_clearDataOnDevice ();
    void m_retriveDataFromDevice ();

public:
    ReLUlayer_t (const std::vector<float> &weight,const std::vector<float> &bias) :
        weight_h(weight),bias_h(bias),dataLoad(false) // Send layer data to layer type
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
