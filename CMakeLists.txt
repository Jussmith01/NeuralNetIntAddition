cmake_minimum_required(VERSION 2.8)
project(NeuralNetIntAddition)

include_directories("/usr/local/cuda-7.0/include")
link_directories("/usr/local/cuda-7.0/lib64")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lcuda -lcublas -lcudnn -lcurand -std=c++11")
set(CMAKE_CXX_COMPILER "/usr/local/cuda-7.0/bin/nvcc")
set(SRC_FILES src/main.cu src/neuralnet.cpp src/tools/micro_timer.cpp src/cutools/neuralnetbase.cu src/cutools/cudnnlayer_t.cu )

add_executable(NeuralNetIntAddition ${SRC_FILES} )
