#ifndef NNBASE_CU
#define NNBASE_CU

//________________________________________________________________________//
//      *************************************************************     //
//                        NeuralNet Base Class
//                  Holds timer variables and class functions
//      *************************************************************     //
class cuNeuralNetworkbase
{
    // cuda Information
    int numdevice;
	int devices;
	std::vector<cudaDeviceProp> devprops;

    void setupCudaDevice(void);

    // cuDNN Handles
	cudnnHandle_t cudnnHandle;
	cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;


    /*  Fuction for creating needed cuda handlers   */
    void createHandles(void);

    /*  Fuction for destroying needed cuda handlers   */
    void destroyHandles(void);

    void activationForward(int n, int c, int h, int w, float* srcData, float** dstData);

public:
    /*     Constructor      */
	cuNeuralNetworkbase ()
	{
        // Get cuda device information and setup device
		setupCudaDevice();

        // Set data formatting
        // CUDNN_TENSOR_NCHW;
        // CUDNN_DATA_FLOAT;

		// Create cudnn Handle
        createHandles();

    };

    /*     Destructor      */
    ~cuNeuralNetworkbase ()
    {
        // Clear device properties
        devprops.clear();

		// Destroy cudnn Handle
        destroyHandles();
    };

    /*     Activation Test      */
	void ActivationTest ()
	{
        int n,c,h,w,IMAGE_H,IMAGE_W;
        IMAGE_H=IMAGE_W=3;
        float *srcData = NULL, *dstData = NULL;
        float imgData_h[IMAGE_H*IMAGE_W];

        // Plot to console and normalize image to be in range [0,1]
        std::cout << "Starting vector:" << std::endl;
        for (int i = 0; i < IMAGE_H; i++)
        {
            for (int j = 0; j < IMAGE_W; j++)
            {
                int idx = IMAGE_W*i + j;
                imgData_h[idx] = -1 / float(2);
                std::cout << imgData_h[idx] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "Performing forward propagation...\n";

        cudaErrorHandler( cudaMalloc(&srcData, IMAGE_H*IMAGE_W*sizeof(float)) );
        cudaErrorHandler( cudaMalloc(&dstData, IMAGE_H*IMAGE_W*sizeof(float)) );
        cudaErrorHandler( cudaMemcpy(srcData, imgData_h,IMAGE_H*IMAGE_W*sizeof(float),cudaMemcpyHostToDevice) );

        n = c = 1; h = IMAGE_H; w = IMAGE_W;
        int N = n*c*h*w;

        activationForward(n, c, h, w, srcData, &dstData);

        float result[N];
        cudaErrorHandler( cudaMemcpy(result, dstData, N*sizeof(float), cudaMemcpyDeviceToHost) );

        std::cout << "Resulting vector:" << std::endl;
        // Plot to console and normalize image to be in range [0,1]
        for (int i = 0; i < IMAGE_H; i++)
        {
            for (int j = 0; j < IMAGE_W; j++)
            {
                int idx = IMAGE_W*i + j;
                std::cout << result[idx] << " ";
            }
            std::cout << std::endl;
        }

        cudaErrorHandler( cudaFree(srcData) );
        cudaErrorHandler( cudaFree(dstData) );
    };
};

#endif
