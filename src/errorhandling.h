//________________________________________________________________________//
//      *************************************************************     //
//                      Compiler Error Handling
//      *************************************************************     //
/* Check for sufficient compiler version */
#if defined(__GNUC__) || defined(__GNUG__)
    #if !(__GNUC__ >= 4 && __GNUC_MINOR__ >= 9)
        #error "Insufficient GNU Compiler Version -- 4.9 or greater required"
    #endif
#else
    #warning "Currently only GNU compilers are supported and tested, but go ahead if you want to."
#endif

//________________________________________________________________________//
//      *************************************************************     //
//                      CUDA and cuDNN Error Handling
//      *************************************************************     //

#define cudaFatalError(_errstr)                                                               \
{                                                                                             \
    std::cerr <<  _errstr << std::endl;                                                       \
    std::cerr << " Resetting CUDA Device and returning! " << std::endl;                       \
    cudaDeviceReset();                                                                        \
    return(EXIT_FAILURE);                                                                     \
};

#define cudaThrowHandler(_errchk)                                                                  \
{                                                                                                  \
    if (_errchk != cudaSuccess)                                                                    \
    {                                                                                              \
        std::cerr <<  "\nERROR: CUDA throw detected! Attempting to shut down nicely!" << std::endl;\
        std::stringstream _error;                                                                  \
        _error << "CUDA Error -- \"" << cudaGetErrorString(_errchk) << "\"\n";                     \
        _error << _errchk << " in location -- " << __FILE__ << ":" << __LINE__ << std::endl;       \
        _error << " in function -- " << __FUNCTION__ << "()" <<  std::endl;                        \
        std::cerr << _error.str() << std::endl;                                                          \
        throw _error.str();                                                                        \
    }                                                                                              \
};

#define cudnnThrowHandler(_errchk)                                                                  \
{                                                                                                   \
    if (_errchk != CUDNN_STATUS_SUCCESS)                                                            \
    {                                                                                               \
        std::cerr <<  "\nERROR: cuDNN throw detected! Attempting to shut down nicely!" << std::endl;\
        std::stringstream _error;                                                                   \
        _error << "cuDNN Error -- \"" << cudnnGetErrorString(_errchk) << "\"\n";                    \
        _error << " in location -- " << __FILE__ << ":" << __LINE__ << std::endl;                   \
        _error << " in function -- " << __FUNCTION__ << "()" <<  std::endl;                         \
        std::cerr << _error.str() << std::endl;                                                           \
        throw _error.str();                                                                         \
    }                                                                                               \
};

#define cublasThrowHandler(_errchk)                                                                  \
{                                                                                                    \
    if (_errchk != CUBLAS_STATUS_SUCCESS)                                                            \
    {                                                                                                \
        std::cerr <<  "\nERROR: cuBLAS throw detected! Attempting to shut down nicely!" << std::endl;\
        std::stringstream _error;                                                                    \
        _error << "cuBLAS Error -- Error Code: \"" << _errchk << "\"";                               \
        _error << " in location -- " << __FILE__ << ":" << __LINE__ << std::endl;                    \
        _error << " in function -- " << __FUNCTION__ << "()" <<  std::endl;                          \
        std::cerr << _error.str() << std::endl;                                                            \
        throw _error.str();                                                                          \
    }                                                                                                \
};

#define fpnThrowHandler(_errchk)                                                                  \
{                                                                                                 \
    if (!_errchk.empty())                                                                         \
    {                                                                                             \
        std::cerr <<  "\nERROR: FPN throw detected! Attempting to shut down nicely!" << std::endl;\
        std::stringstream _error;                                                                 \
        _error << "\nFPN Error -- \"" << _errchk << "\" \n";                                      \
        _error << " in location -- " << __FILE__ << ":" << __LINE__ << std::endl;                 \
        _error << " in function -- " << __FUNCTION__ << "()" << std::endl;                        \
        std::cerr << _error.str() << std::endl;                                                         \
        throw _error.str();                                                                       \
    }                                                                                             \
};

#define signalHandler(_errchk)                                                                  \
{                                                                                               \
    if (!_errchk.empty())                                                                       \
    {                                                                                           \
        std::stringstream _error;                                                               \
        _error << "\nError -- \"" << _errchk << "\" \n";                                        \
        std::cerr << _error.str();                                                              \
        cudaDeviceReset();                                                                      \
        std::cerr << "Exiting with Failure!\n";                                                 \
        exit(EXIT_FAILURE);                                                                     \
    }                                                                                           \
};
