//________________________________________________________________________//
//      *************************************************************     //
//                      CUDA and cuDNN Error Handling
//      *************************************************************     //

#define FatalError(_errstr)                                                                   \
{                                                                                             \
    std::cerr << _errstr << " in location -- \n" << __FILE__ << ":" << __LINE__ << std::endl; \
    std::cerr << " Function -- " << __FUNCTION__ << std::endl;                                \
    cudaDeviceReset();                                                                        \
    exit(EXIT_FAILURE);                                                                       \
};

#define cudaErrorHandler(_errchk)                                                             \
{                                                                                             \
    if (_errchk != cudaSuccess)                                                               \
    {                                                                                         \
        std::stringstream _error;                                                             \
        _error << "CUDA Error " << _errchk << ": \"" << cudaGetErrorString(_errchk) << "\"";  \
        FatalError(_error.str());                                                             \
    }                                                                                         \
};

#define cudnnErrorHandler(_errchk)                                          \
{                                                                           \
    if (_errchk != CUDNN_STATUS_SUCCESS)                                    \
    {                                                                       \
        std::stringstream _error;                                           \
        _error << "cuDNN Error: \"" << cudnnGetErrorString(_errchk) << "\"";\
        FatalError(_error.str());                                           \
    }                                                                       \
};
