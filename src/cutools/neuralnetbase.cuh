#ifndef NNBASE_CU
#define NNBASE_CU

enum  Initializers {
    NNB_CREATE,
    NNB_LOAD
};

//________________________________________________________________________//
//      *************************************************************     //
//                        NeuralNet Base Class
//                  Holds timer variables and class functions
//      *************************************************************     //
class cuNeuralNetworkbase {

    // cuda Information
    int numdevice; // Number of devices detected on the system
    std::vector<cudaDeviceProp> devprops; // Holds each devices properties

    // Get system device information and setup devices
    void m_setupCudaDevice(void);

    // cuDNN Handles
    cudnnHandle_t cudnnHandle;
    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;

    // cuBLAS Handle
    cublasHandle_t cublasHandle;

    /*---------------Class functions--------------------*/
    /*  Fuction for creating needed cuda handlers   */
    void m_createHandles(void);

    /*  Fuction for destroying needed cuda handlers   */
    void m_destroyHandles(void);

    /*  Network Creator (Currently randomly initializes a networks weights and biases) */
    void m_createNetwork(std::string file);

    /*  Network Save */
    void m_saveNetwork(std::string file);

    /*  Network Load */
    void m_loadNetwork(std::string file);

    /*  Builds the individual layers from input weights and biases */
    void m_buildLayers(std::vector<float> data_wh,std::vector<float> data_bh);

    //void activationForward(int n, int c, int h, int w, float* srcData, float** dstData);
    //void fullyConnectedForward(int& n, int& c, int& h, int& w,float* srcData, float** dstData);

    /*     Base Constructor     */
    cuNeuralNetworkbase ()
    {
        // Get cuda device information and setup device
        m_setupCudaDevice();

        // Create cudnn Handle
        m_createHandles();
    };

public:

    /*     Load Constructor     */
    cuNeuralNetworkbase(std::string networkfile,enum Initializers setup) :
        cuNeuralNetworkbase()
    {
        switch (setup)
        {
            /* This case creates a new network from a network template file and sets it up */
            case NNB_CREATE: {
                std::cout << "Creating a Neural Network from template file: " << networkfile << std::endl;
                //m_createNetwork(networkfile);
                break;
            }

            /* This case creates loads a network from a saved network file and sets it up */
            case NNB_LOAD: {
                std::cout << "Loading the Neural Network from file: " << networkfile << std::endl;
                //m_loadNetwork(networkfile);
                break;
            }
        }
    }

    /*     Destructor      */
    ~cuNeuralNetworkbase ()
    {
        // Clear device properties
        devprops.clear();

        // Destroy all cuda Handle
        m_destroyHandles();
    };

    /*     Activation Test      */
    /*void ActivationTest ()
    {
        int n,c,h,w,IMAGE_H,IMAGE_W;
        n=2; // Number of structures in data set
        c=1; // Feature maps, no use I can thing of for our purposes
        IMAGE_H=3; // Number of vectors per structure
        IMAGE_W=3; // Number of components per vector
        float *srcData = NULL, *dstData = NULL; // Pointers to device data
        //float *weight_d = NULL, *dstData = NULL; // Pointers to device data
        float imgData_h[2*IMAGE_H*IMAGE_W]; // "image" or structure data

        // Copy data into image
        std::cout << "Starting vector:" << std::endl;
        for (int k = 0; k < 1; k++) { //
            for (int i = 0; i < IMAGE_H; i++) { // Num Vectors
                for (int j = 0; j < IMAGE_W; j++) { // Num Components
                    int idx = (IMAGE_W*i + j)+k*IMAGE_W*IMAGE_H;
                    imgData_h[idx] = -1 / float(2);
                    std::cout << imgData_h[idx] << " ";
                }
                std::cout << std::endl;
            }
        }

        std::cout << "Performing forward propagation...\n";
        cudaErrorHandler( cudaMalloc(&srcData, IMAGE_H*IMAGE_W*sizeof(float)) ); // Allocate space on GPU
        cudaErrorHandler( cudaMalloc(&dstData, IMAGE_H*IMAGE_W*sizeof(float)) ); // Allocate space on GPU
        cudaErrorHandler( cudaMemcpy(srcData, imgData_h,IMAGE_H*IMAGE_W*sizeof(float),cudaMemcpyHostToDevice) ); // Move data

        n = 1;
        c = 1;
        h = IMAGE_H;
        w = IMAGE_W;
        int N = n*c*h*w;

        //fullyConnectedForward(n,c,h,w,srcData,&dstData,);
        activationForward(n, c, h, w, srcData,&dstData);

        float result[N];
        cudaErrorHandler( cudaMemcpy(result, dstData, N*sizeof(float), cudaMemcpyDeviceToHost) );

        std::cout << "Resulting vector:" << std::endl;
        // Plot to console and normalize image to be in range [0,1]
        for (int k = 0; k < 1; k++) { //
            for (int i = 0; i < IMAGE_H; i++) {
                for (int j = 0; j < IMAGE_W; j++) {
                    int idx = (IMAGE_W*i + j)+k*IMAGE_W*IMAGE_H;
                    std::cout << result[idx] << " ";
                }
                std::cout << std::endl;
            }
        }

        cudaErrorHandler( cudaFree(srcData) );
        cudaErrorHandler( cudaFree(dstData) );
    };*/
};

#endif
