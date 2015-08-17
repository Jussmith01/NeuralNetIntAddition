/*----------------------------------------------
        Written by Justin Smith ~August 2015
        E-Mail Jussmith48@gmail.com
        Copyright the Roitberg research group
        Chemistry Department
        University of Florida
        Gainesville FL.
------------------------------------------------*/

#ifndef CUDA_HOST_TOOLS_CU
#define CUDA_HOST_TOOLS_CU

#include <cuda.h>

#include "../tools/tools.hpp"

namespace fpn {

/*----Print CUDA Device Properties-----

--------------------------------------*/
inline void printDevProps(std::vector<cudaDeviceProp> &devprops) {
    std::cout << "|--------------CUDA Device Information--------------|\n";
    std::cout << "CUDA Devices Detected: " << devprops.size() << "\n\n";

    std::vector<cudaDeviceProp>::iterator dev;
    for (dev=devprops.begin(); dev!=devprops.end(); dev++) {
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

// FOR TESTING PURPOSES!!!!!!!
inline void printCudaData(int size,float *data,std::string message) {
    std::vector<float> test(size);
    cudaThrowHandler( cudaMemcpy(&test[0],data,size*sizeof(float),cudaMemcpyDeviceToHost) );
    std::cout << message << "\n";
    for (auto i : test) {
        std::cout << i << " ";
    }
    std::cout << "\n";
};

// FOR TESTING PURPOSES!!!!!!!
inline void printMatCudaData(int row,int col,float *data,std::string message) {
    std::vector<float> test(row*col);
    cudaThrowHandler( cudaMemcpy(&test[0],data,row*col*sizeof(float),cudaMemcpyDeviceToHost) );
    std::cout << message << "\n";
    for (int i=0; i<row; ++i) {
        for (int j=0; j<col; ++j) {
            std::cout << test[i+j*row] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
};

/*--------Resize a CUDA Container-------

This function resizes a CUDA container
for use with different sized data
structures.

--------------------------------------*/
inline void cu_resize(int size, float **data) {
    if (data != NULL) {
        cudaThrowHandler( cudaFree(*data) );
    }
    cudaThrowHandler( cudaMalloc((void**)data, size*sizeof(float)) );
};

/*--------Resize a CUDA Container-------

Copy a smaller container many times into
a larger container. Ex.

 Small array = 3,1,5

 Produces a

 Large array = 3,1,5,3,1,5,3,1,5

Nd = Smaller data size to be moved
Nl = Number of rows to fill.

Total size of new array must be Nd*Nl

This fucntion uses prime facorization
to reduce the number of memory calls
required to copy one small array to
a much larger array many times.

--------------------------------------*/
inline void cu_MemcpySmalltoLargeD2D(int Nl,int Ns,const float *src,float **data) {
    /* Fill the first column of data to seed the rest */
    cudaThrowHandler( cudaMemcpyAsync(*data,src,Ns*sizeof(float),cudaMemcpyDeviceToDevice) );

    /* Determine how many to fill initialially */
    std::vector<int> primes(tools::primeFactors(Nl));

    /* sync to ensure previous memcpyasync is complete*/
    cudaDeviceSynchronize();

    /* Use primes to chuck copying of data - More efficient than a bunch of small calls. */
    size_t cIdx=1;
    for (auto p : primes) {
        for (int i=1; i<p; ++i) {
            cudaThrowHandler( cudaMemcpy((*data)+i*cIdx*Ns,*data,cIdx*Ns*sizeof(float),cudaMemcpyDeviceToDevice) );
        }
        cIdx *= p;
    }

};

inline void cu_MemcpySmalltoLargeD2D(int Nl,int Ns,const float *src,float *data) {
    /* Fill the first column of data to seed the rest */
    cudaThrowHandler( cudaMemcpyAsync(data,src,Ns*sizeof(float),cudaMemcpyDeviceToDevice) );

    /* Determine how many to fill initialially */
    std::vector<int> primes(tools::primeFactors(Nl));

    /* sync to ensure previous memcpyasync is complete*/
    cudaDeviceSynchronize();

    /* Use primes to chuck copying of data - More efficient than a bunch of small calls. */
    size_t cIdx=1;
    for (auto p : primes) {
        for (int i=1; i<p; ++i) {
            cudaThrowHandler( cudaMemcpy(data+i*cIdx*Ns,data,cIdx*Ns*sizeof(float),cudaMemcpyDeviceToDevice) );
        }
        cIdx *= p;
    }

};

};

#endif
