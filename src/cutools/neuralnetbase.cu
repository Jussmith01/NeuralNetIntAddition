#include <iostream>
#include <vector>
#include <sstream>
#include <string>

#include <cuda.h>
#include <cudnn.h>

#include "neuralnetbase.cuh"

void cuNeuralnetbase::setupCudaDevice()
{
    cudaErrorHandler(cudaGetDeviceCount(&numdevice));
    devprops.resize(numdevice);

    /*GET CUDA DEVICE PROPERTIES*/
    for (int i=0;i<int(devprops.size());++i)
    {
        cudaErrorHandler(cudaGetDeviceProperties(&devprops[i],i));
        //cudaErrorHandler(cudaGetLastError());
    }

    std::cout << "|--------------CUDA Device Information--------------|\n";
    std::cout << "CUDA Devices Detected: " << numdevice << "\n\n";

    for (int i=0;i<int(devprops.size());++i)
    {
        std::cout << " Device (" << i << "): " << devprops[i].name << std::endl;
        std::cout << "  Streaming Multiprocessors: " << devprops[i].multiProcessorCount << "\n";
        std::cout << "  Device Clock Rate: " << devprops[i].clockRate/float(1024*1024) << "GHz\n";
        std::cout << "  Total Global Memory: " << devprops[i].totalGlobalMem/float(1024*1024) << "MB\n";
        std::cout << "  Memory Clock Rate: " << devprops[i].memoryClockRate/float(1024*1024) << "GHz\n";
        std::cout << "  Memory Bus Width: " << devprops[i].memoryBusWidth << "bit\n";
        std::cout << "  Total Constant Memory: " << devprops[i].totalConstMem/float(1024) << "KB\n";
        std::cout << "  Shared Memory per Block: " << devprops[i].sharedMemPerBlock/float(1024) << "KB\n";
        std::cout << "  Maximum Threads per Block: " << devprops[i].maxThreadsPerBlock << "\n";
        std::cout << "  Maximum Threads dim Block: [" << devprops[i].maxThreadsDim[0] << ","
                                                      << devprops[i].maxThreadsDim[1] << ","
                                                      << devprops[i].maxThreadsDim[2] << "]\n";
        std::cout << "  Maximum Block per Grid: [" << devprops[i].maxGridSize[0] << ","
                                                   << devprops[i].maxGridSize[1] << ","
                                                   << devprops[i].maxGridSize[2] << "]\n";
        std::cout << "  Registers per Block: " << devprops[i].regsPerBlock << "\n";

        std::cout << std::endl;
    }

    std::cout << "|---------------------------------------------------|\n\n";

    cudaErrorHandler(cudaSetDevice(1));
};
