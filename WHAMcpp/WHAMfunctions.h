//
//  WHAMfunctions.h
//  mcWHAM
//

#ifndef mcWHAM_WHAMfunctions_h
#define mcWHAM_WHAMfunctions_h

#include "BefeusTools.h"

#endif


class DOWHAM
{
    int points, nstates, nexperiments;
    int * ncounts, * sample_sizes;
    long double * wvec;
    long double ** trajectory;
    long double ** omega_vectors;
    long double ** omega_transpose;
    long double * opt_trajectory;
public:
    int Points();
    long double * OptTrajectory();
    
    void DoWham( long double * omega_vec, int * counts, int * samples, long double ** omegas, int states, int nsamples, int niter, float tol);
    
    long double * WhamStep( long double * trajectory );
    
    long double dot_product( double long * a, double long * b, int size );
    
    void OmegaTranspose();
    
    long double Sum( long double * an_array, int length_of_an_array );
    
    long double * square_diff( long double * array_1, long double * array_2, int length_of_arrays );
};