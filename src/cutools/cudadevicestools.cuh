#ifndef CUDA_DEVICE_TOOLS_CU
#define CUDA_DEVICE_TOOLS_CU

inline void printDevProps(std::vector<cudaDeviceProp> &devprops)
{
    std::cout << "|--------------CUDA Device Information--------------|\n";
    std::cout << "CUDA Devices Detected: " << devprops.size() << "\n\n";

    std::vector<cudaDeviceProp>::iterator dev;
    for (dev=devprops.begin();dev!=devprops.end();dev++)
    {
        std::cout << " Device (" << dev - devprops.begin() << "): " << (*dev).name << std::endl;
        std::cout << "  Streaming Multiprocessors: " << (*dev).multiProcessorCount << "\n";
        std::cout << "  Device Clock Rate: " << (*dev).clockRate/float(1024*1024) << "GHz\n";
        std::cout << "  Total Global Memory: " << (*dev).totalGlobalMem/float(1024*1024) << "MB\n";
        std::cout << "  Memory Clock Rate: " << (*dev).memoryClockRate/float(1024*1024) << "GHz\n";
        std::cout << "  Memory Bus Width: " << (*dev).memoryBusWidth << "bit\n";
        std::cout << "  Total Constant Memory: " << (*dev).totalConstMem/float(1024) << "KB\n";
        std::cout << "  Shared Memory per Block: " << (*dev).sharedMemPerBlock/float(1024) << "KB\n";
        std::cout << "  Maximum Threads per Block: " << (*dev).maxThreadsPerBlock << "\n";
        std::cout << "  Maximum Threads dim Block: [" << (*dev).maxThreadsDim[0] << ","
                                                      << (*dev).maxThreadsDim[1] << ","
                                                      << (*dev).maxThreadsDim[2] << "]\n";
        std::cout << "  Maximum Block per Grid: [" << (*dev).maxGridSize[0] << ","
                                                  << (*dev).maxGridSize[1] << ","
                                                  << (*dev).maxGridSize[2] << "]\n";
        std::cout << "  Registers per Block: " << (*dev).regsPerBlock << "\n";

        std::cout << std::endl;
    }

    std::cout << "|---------------------------------------------------|\n\n";
};

#endif
