#ifndef NNBASE_CU
#define NNBASE_CU

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
