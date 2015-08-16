/*----------------------------------------------
        Written by Justin Smith ~August 2015
        E-Mail Jussmith48@gmail.com
        Copyright the Roitberg research group
        Chemistry Department
        University of Florida
        Gainesville FL.
------------------------------------------------*/
#ifndef CURAND_HOST_TOOLS_CU
#define CURAND_HOST_TOOLS_CU

#include <ctime>
#include <climits>
#include <curand.h>

#define curandThrowHandler(_errchk)                                                             \
{                                                                                               \
    if (_errchk != CURAND_STATUS_SUCCESS)                                                       \
    {                                                                                           \
        std::cerr <<  "ERROR: cuRAND throw detected!" << std::endl;                             \
        std::stringstream _error;                                                               \
        _error << "cuRAND Error -- Error Code: \"" << _errchk << "\"\n";                        \
        _error << " in location -- " << __FILE__ << ":" << __LINE__ << std::endl;    \
        _error << " in function -- " << __FUNCTION__ << "()" <<  std::endl;                     \
        throw _error.str();                                                                     \
    }                                                                                           \
};

namespace fpn
{

/*-----Generate Random Uniform Floats-------

Uses cuRAND Host API to generate the random
floats to initialize the network.

--------------------------------------------*/
void curandGenRandomFloats (std::vector<float> &hostData,const unsigned int n) {
    curandGenerator_t generator;
    float *devData;

    if (n >= UINT_MAX)
    {
        std::stringstream _error;
        _error << "\nDon't you think " << UINT_MAX << " is a lot of random numbers? I do too.\n";
        std::cout << _error.str() << std::endl;
        fpnThrowHandler(std::string("Network size is too large!"));
    }

    /* Allocate Host */
    hostData.resize(n);

    /* Allocate Device */
    cudaThrowHandler(cudaMalloc((void**)&devData, n * sizeof(float)));

    /* Create Generator */
    curandThrowHandler(curandCreateGenerator(&generator,CURAND_RNG_PSEUDO_DEFAULT));

    /* Seed Generator */
    curandThrowHandler(curandSetPseudoRandomGeneratorSeed(generator,clock()));

    /* Generate Random Numbers */
    curandThrowHandler(curandGenerateUniform(generator,devData,n));

    /* Get data from device */
    cudaThrowHandler(cudaMemcpy(&hostData[0],devData,n*sizeof(float),cudaMemcpyDeviceToHost));

    /* Cleanup Device */
    curandThrowHandler(curandDestroyGenerator(generator));
    cudaThrowHandler(cudaFree(devData));
};

};
#endif
