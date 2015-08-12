#ifndef CURAND_HOST_TOOLS_CU
#define CURAND_HOST_TOOLS_CU

#include <ctime>
#include <curand.h>

#define curandThrowHandler(_errchk)                                                             \
{                                                                                               \
    if (_errchk != CURAND_STATUS_SUCCESS)                                                       \
    {                                                                                           \
        std::cerr <<  "ERROR: cuRAND throw detected!" << std::endl;                             \
        std::stringstream _error;                                                               \
        _error << "cuDNN Error -- Error Code: \"" << _errchk << "\"";                           \
        _error << _errchk << " in location -- " << __FILE__ << ":" << __LINE__ << std::endl;    \
        _error << " in function -- " << __FUNCTION__ << "()" <<  std::endl;                     \
        throw _error.str();                                                                     \
    }                                                                                           \
};

namespace fpn
{

void curandGenRandomFloats (std::vector<float> &hostData,const unsigned int n)
{
    curandGenerator_t generator;
    float *devData;

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
