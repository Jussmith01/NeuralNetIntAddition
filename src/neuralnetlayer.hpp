#ifndef NEURAL_NET_LAYER_C
#define NEURAL_NET_LAYER_C

#include <vector>
#include <cstring>
#include <math.h>
#include <fstream>

#include "tools/random.hpp"

class NeuralNetLayer
{
    //***************************//
    //        Class Data         //
    //***************************//
    std::vector<double> w; // Weights
    std::vector<double> b; // Bias
    std::vector<double> a; // Activations
    std::vector<double> z; // Weighted Input
    std::vector<double> d; // Error

    std::vector<double> dCdw; // Derivative of cost wrt weights
    std::vector<double> dCdb; // Derivative of cost wrt biases

    long int cntr; // Number of training data set
    float avgCost;

    int Nn; // Number of neurons in the layer
    int Nw; // Number of weights per neuron

    std::ofstream wgraph;
    std::ofstream bgraph;

    //***************************//
    //  Class Private Functions  //
    //***************************//
    // The sigmoid function
    void SigmoidFuncOnActivation()
    {
        for (int i=0; i<Nn; ++i)
                a[i]=1.0/(1.0+exp(-z[i]));
    };

    //The derivative of the sigmoid function
    void SigmoidFuncPrimeOnError()
    {
        for (int i=0; i<Nn; ++i)
                d[i]=exp(z[i])/pow(1.0+exp(z[i]),2);
    };

public:
    //**************************//
    //    Class Constructors    //
    //**************************//
    NeuralNetLayer() {};

    NeuralNetLayer(int Nw,int Nn,std::string graph1)
    {
        Init(Nw,Nn,graph1);
    };

    //***************************//
    // Class Operation Functions //
    //***************************//
    void Init(int Nw,int Nn,std::string graph1)
    {
        this->Nn=Nn;
        this->Nw=Nw;
        w.resize(Nw*Nn);
        b.resize(Nn);
        a.resize(Nn);
        z.resize(Nn);
        d.resize(Nn);

        dCdw.resize(Nw*Nn);
        dCdb.resize(Nn);

        memset(&dCdw[0],0,Nn*Nw*sizeof(float));
        memset(&dCdb[0],0,Nn*sizeof(float));

        NormRandomReal randgen;
        randgen.FillVector(w,0.0,1.0,74322678);
        randgen.Clear();
        randgen.FillVector(b,0.0,1.0,19883278);
        randgen.Clear();

        wgraph.open("wgraph.dat");
        bgraph.open(graph1.c_str());

        cntr=0;
    };

    void ComputeActivation(std::vector<double> &ia) // ia must be of nw size
    {
        for (int i=0; i<Nn; ++i)
        {
            z[i]=0.0;

            for (int j=0; j<Nw; ++j)
                z[i]+=w[j+i*Nw]*ia[j];

            z[i]+=b[i];
        }

        SigmoidFuncOnActivation();
    };

    // This should only be used on the LAST layer/first layer in the backpropagation
    void ComputeInitalError(std::vector<double> &de)
    {
       if (de.size() != d.size())
        {
            std::cout << "Error: de.size != d.size in ComputeInitalError.\n";
        }

        memset(&d[0],0,Nn*sizeof(double));

        SigmoidFuncPrimeOnError();

        for (int i=0; i<Nn; ++i)
            d[i]=(a[i]-de[i])*d[i];
    };

    // This should only be used if the LAST layer/first layer in the backpropagation error is defined
    void ComputeError(NeuralNetLayer &lp1)
    {
        memset(&d[0],0,Nn*sizeof(double));

        SigmoidFuncPrimeOnError();

        for (int i=0; i<Nn; ++i)
        {
            double sum=0;

            for (int j=0; j<lp1.Nn; ++j)
                sum+=lp1.Getw()[j*lp1.Nw+i]*lp1.Getd()[j];

            d[i]=sum*d[i];
        }
    };

    void ComputeDerivatives(std::vector<double> &am1)
    {
        for (int i=0; i<Nn; ++i)
            dCdb[i] += d[i];

        if (int(am1.size()) != Nw)
            std::cout << "Error: am1.size(" << am1.size() << ") != Nw.size(" << Nw << ") -> ComputeDerivatives.\n";

        for (int i=0; i<Nn; ++i)
        {
            for (int j=0; j<Nw; ++j)
                dCdw[j+i*Nw] += am1[j]*d[i];
        }

        ++cntr;
    };

    void EndTrainingSet(double eta)
    {
        double invc = 1.0/double(cntr);

        for (int i=0; i<Nn; ++i)
            b[i] = b[i] - eta * dCdb[i] * invc;

        for (int i=0; i<Nn; ++i)
        {
            for (int j=0; j<Nw; ++j)
                w[j+i*Nw] = w[j+i*Nw] - eta * dCdw[j+i*Nw] * invc;
        }

        double val=0;

        for (int i=0; i<Nn; ++i)
            val += dCdb[i]/double(cntr);

        bgraph << val/double(Nn) << std::endl;

        cntr=0;
        memset(&dCdb[0],0,sizeof(double)*dCdb.size());
        memset(&dCdw[0],0,sizeof(double)*dCdw.size());
    };

    //**************************//
    //  Memory Access Functions //
    //**************************//
    int GetNn()
    {
        return Nn;
    }

    int GetNw()
    {
        return Nw;
    }

    std::vector<double>& GetActivation()
    {
        return a;
    }

    std::vector<double>& Getw()
    {
        return w;
    }

    std::vector<double>& Getd()
    {
        return d;
    }

    std::vector<double>& GetdCdw()
    {
        return dCdw;
    }

    std::vector<double>& GetdCdb()
    {
        return dCdb;
    }

    //**************************//
    //      Class Cleanup       //
    //**************************//
    void Clear()
    {
        w.clear();
        b.clear();
        a.clear();
        z.clear();
        d.clear();

        dCdw.clear();
        dCdb.clear();

        bgraph.close();
    };
};

#endif
