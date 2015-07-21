#ifndef NEURAL_NET_LAYER_C
#define NEURAL_NET_LAYER_C

#include <vector>
#include <cstring>
#include <math.h>

#include "tools/random.hpp"

class NeuralNetLayer
{
    std::vector<float> w; // Weights
    std::vector<float> b; // Bias
    std::vector<float> a; // Activations
    std::vector<float> z; // Weighted Input

    int Nn; // Number of neurons in the layer
    int Nw;

public:

    NeuralNetLayer() {};

    NeuralNetLayer(int Nw,int Nn)
    {
        Init(Nw,Nn);
    };

    void Init(int Nw,int Nn)
    {
        this->Nn=Nn;
        this->Nw=Nw;
        w.resize(Nw*Nn);
        b.resize(Nn);
        a.resize(Nn);
        z.resize(Nn);

        memset(&a[0],0,Nn*sizeof(int));

        NormRandomReal randgen;
        randgen.FillVector(w,0.0,1.0);
        randgen.Clear();
        randgen.FillVector(b,0.0,1.0);
        randgen.Clear();
    };

    void ComputeActivation(std::vector<float> &ia) // ia must be of nw size
    {
        for (int i=0;i<Nn;++i)
        {
            for (int j=0;j<Nw;++j)
            {
                z[i]+=w[j+i*Nw]*ia[i];
            }

            z[i]+=b[i];
        }

        SigmoidFuncOnActivation();
    };

    void SigmoidFuncOnActivation()
    {
        for (int i=0;i<Nn;++i)
            a[i]=1.0/(1.0+exp(-z[i]));
    };

    std::vector<float>& GetActivation()
    {
        return a;
    }

    void Clear()
    {
        w.clear();
        b.clear();
        a.clear();
        z.clear();
    };
};

#endif
