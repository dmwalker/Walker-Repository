//
//  Forces.h
//  mcWHAM
//

#ifndef mcWHAM_Forces_h
#define mcWHAM_Forces_h

#endif

#include <cmath>

using namespace std;

class WHAM_DENSITY
{
    long double ** omegas;
    long double ** omegaTranspose;
    long double ** deltaOmegas;
    int * counts;
    int * samples;
    int nstates; // length of counts
    int nsamples; // number of simulations
    
public:
    long double ** DeltaOmegas() {return deltaOmegas;}
    
    void WhamForce( long double ** nD_omegas, int * state_counts, int * sample_sizes, int states, int n_experiments)
    {
        counts=state_counts;
        samples=sample_sizes;
        nstates=states;
        nsamples=n_experiments;
        omegas=new long double * [nsamples];
        omegas=nD_omegas;

        WhamDensity();
        omegaManipulate();
    };
    
    void omegaManipulate()
    {
        omegaTranspose=new long double * [nstates];
        deltaOmegas=new long double * [nstates];
        for (int i=0; i<nstates; i++)
        {
            omegaTranspose[i]=new long double [nsamples];
            deltaOmegas[i]=new long double [nsamples];
            for (int j=0; j<nsamples; j++)
            {
                deltaOmegas[i][j]=omegas[j][i]-omegas[0][i];
            }
        }
    }

};

