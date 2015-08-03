#ifndef NNBASE_CU
#define NNBASE_CU

//________________________________________________________________________//
//      *************************************************************     //
//                      CUDA and cuDNN Error Handling
//      *************************************************************     //

inline void Fatal(std::string errstr)
{
    std::cerr << errstr << " in location -- \n" << __FILE__ << ":" << __LINE__ << std::endl;
    cudaDeviceReset();
    exit(1);
};

inline void cudaErrorHandler(cudaError_t _errchk)
{
    if (_errchk != cudaSuccess)
    {
        std::stringstream _error;
        _error << "CUDA Error: \"" << cudaGetErrorString(_errchk) << "\"";
        Fatal(_error.str());
    }
};

inline void cudnnErrorHandler(cudnnStatus_t _errchk)
{
    if (_errchk != CUDNN_STATUS_SUCCESS)
    {
        std::stringstream _error;
        _error << "cuDNN Error: \"" << cudnnGetErrorString(_errchk) << "\"";
        Fatal(_error.str());
    }
};

//________________________________________________________________________//
//      *************************************************************     //
//                        NeuralNet Base Class
//                  Holds timer variables and class functions
//      *************************************************************     //
class cuNeuralnetbase
{
    int numdevice;
	int devices;
	std::vector<cudaDeviceProp> devprops;

	cudnnHandle_t handle;

	void setupCudaDevice(void);

public:
    /*     Constructor      */
	cuNeuralnetbase ()
	{
		setupCudaDevice();

		// Create cudnn Handle
		cudnnErrorHandler(cudnnCreate(&handle));
    };

    ~cuNeuralnetbase ()
    {
        devprops.clear();

		// Destroy cudnn Handle
		cudnnErrorHandler(cudnnDestroy(handle));
    };
};

#endif

