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
        {
            if (fabs(z[i])>150)
            {
                a[i]=0.0;
            }
            else
            {
                a[i]=1.0/(1.0+exp(-z[i]));
            }

            if (a[i]!=a[i])
            {
                std::cout << " NAN DETECTED (SigmoidFunc) a: " << a[i] << std::endl;
                std::cout << " z[i]: " << z[i] << " exp(-z[i]): " << exp(-z[i]) << std::endl;
                exit(1);
            }
        }
    };

    //The derivative of the sigmoid function
    void SigmoidFuncPrimeOnError()
    {
        for (int i=0; i<Nn; ++i)
        {
            if (fabs(z[i])>150)
            {
                d[i]=0.0;
            }
            else
            {
                d[i]=exp(z[i])/pow(1.0+exp(z[i]),2);
            }

            if (d[i]!=d[i])
            {
                std::cout << " NAN DETECTED (SigmoidFuncPrime) d[i]: " << d[i] << std::endl;
                std::cout << " z[i]: " << z[i] << " exp(-z[i]): " << exp(-z[i]) << std::endl;
                exit(1);
            }
        }
    };

    /*double WeightTimesError()
    {
        return ;
    }*/

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
            {
                z[i]+=w[j+i*Nw]*ia[j];

                if (isinf(w[j+i*Nw]*ia[j]))
                {
                  z[i]=1.0e+296;
                }

                if (z[i]!=z[i])
                {
                    std::cout << " NAN DETECTED (ComputeActivation in loop) z: " << z[i] << std::endl;
                    std::cout << " w[j+i*Nw]* ia[i] = (" << j+i*Nw << ") "<< w[j+i*Nw] << " * " << ia[j] << " = " << w[j+i*Nw]*ia[j] << std::endl;
                    exit(1);
                }
            }

            z[i]+=b[i];

            if (z[i]!=z[i])
            {
                std::cout << " NAN DETECTED (ComputeActivation) z: " << z[i] << std::endl;
                std::cout << " b[i]: " << b[i] << std::endl;
                exit(1);
            }
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
        {
            d[i]=(a[i]-de[i])*d[i];
            if (d[i]!=d[i])
            {
                std::cout << " NAN DETECTED (ComputeInitalError) d[i]: " << d[i] << std::endl;
                exit(1);
            }
        }
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
            {
                //std::cout << " iidx: " << i << " of k: " << Nn << " j: " << lp1.Nn << std::endl;
                //std::cout << " jidx: " << j << " of k: " << Nw << " j: " << lp1.Nw << std::endl;

                sum+=lp1.Getw()[j*lp1.Nw+i]*lp1.Getd()[j];


                if (sum != sum  || isinf(sum))
                {
                    std::cout << " NAN DETECTED (ComputeError in loop) sum: " << sum << std::endl;
                    std::cout << " lp1.Getw()[j+i*lp1.Nw]: " << lp1.Getw()[j+i*lp1.Nw] << " lp1.Getd()[i]: " << lp1.Getd()[i] << std::endl;
                    exit(1);
                }
            }

            d[i]=sum*d[i];
            if (d[i]!=d[i] || isinf(d[i]))
            {
                std::cout << " NAN DETECTED (ComputeError) d[i]: " << d[i] << std::endl;
                std::cout << " sum: " << sum << std::endl;
                exit(1);
            }
        }
    };

    void ComputeDerivatives(std::vector<double> &am1)
    {
        //std::cout << "Derivatives! : \n";
        for (int i=0; i<Nn; ++i)
        {
            dCdb[i] += d[i];
            //std::cout << " dCdb[" << i << "]=" << dCdb[i] << "\n";
            /*if (dCdb[i] != dCdb[i])
            {
                std::cout << " NAN DETECTED (ComputeDerivatives) dCdb[i]: " << dCdb[i] << std::endl;
                std::cout << " d[i]: " << d[i] << std::endl;
                exit(1);
            }*/
        }

        if (int(am1.size()) != Nw)
        {
            std::cout << "Error: am1.size(" << am1.size() << ") != Nw.size(" << Nw << ") -> ComputeDerivatives.\n";
        }

        for (int i=0; i<Nn; ++i)
        {
            for (int j=0; j<Nw; ++j)
            {
                dCdw[j+i*Nw] += am1[j]*d[i];
                //std::cout << " dCdw[" << i << "," << j << "]=" << dCdw[j+i*Nw] << "\n";
                /*if (dCdw[j+i*Nw] != dCdw[j+i*Nw])
                {
                    std::cout << " NAN DETECTED (ComputeDerivatives) dCdw[j+i*Nw]: " << dCdw[j+i*Nw] << std::endl;
                    std::cout << " am1[j]: " << am1[j] << " d[i]: " << d[i] << std::endl;
                    exit(1);
                }*/
            }
        }

        ++cntr;
    };

    void EndTrainingSet(double eta)
    {
        //std::cout << "Derivatives! : \n";
        for (int i=0; i<Nn; ++i)
        {
            b[i] = b[i] - eta*dCdb[i]/double(cntr);
            if (b[i] != b[i])
            {
                std::cout << " NAN DETECTED (EndTrainingSet) b[i]: " << b[i] << std::endl;
                std::cout << " eta: " << eta << " dCdb[i]: " << dCdb[i] << " cntr: " << double(cntr) << std::endl;
                exit(1);
            }
        }

        for (int i=0; i<Nn; ++i)
        {
            for (int j=0; j<Nw; ++j)
            {
                double store = w[j+i*Nw];
                w[j+i*Nw] += w[j+i*Nw] - eta * dCdw[j+i*Nw];

                if (w[j+i*Nw] > 1000)
                {
                    w[j+i*Nw]=1000.0;
                }

                if (w[j+i*Nw] < -1000)
                {
                    w[j+i*Nw]=-1000.0;
                }

                //std::cout << " dCdw[" << i << "," << j << "=" << j+i*Nw << "]=" << dCdw[j+i*Nw] << "\n";


                if (w[j+i*Nw] != w[j+i*Nw])
                {
                    std::cout << " NAN DETECTED (EndTrainingSet) w[j+i*Nw]: " << w[j+i*Nw] << std::endl;
                    std::cout << " TEST: " << store - eta * dCdw[j+i*Nw] << std::endl;
                    std::cout << " eta: " << eta << " dCdw[i]: " << dCdw[j+i*Nw] << " w[j+i*Nw]old: " << store << std::endl;
                    exit(1);
                }
            }
        }

        double val=0;
        for (int i=0; i<Nn; ++i)
        {
            val += dCdb[i]/double(cntr);
        }

        bgraph << val/double(Nn) << std::endl;

        cntr=0;
        dCdw.clear();
        dCdb.clear();
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
