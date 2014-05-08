#ifndef mcWHAM_distributions_h
#define mcWHAM_distributions_h

#include "Forces.h"

#endif

class WHAM_MAX_PROPOSAL_DENSITY
{
    long double * samples;
    long double * pvector;
    double Msamples;
    int nstates;
public:
    long double * Distributions(){return samples;}
    int nStates(){return nstates;}
    void MaxProposalDensity( long double * trajectory, int states, double diriscale )
    {
        pvector=trajectory;
        Msamples=diriscale;
        nstates=states;
        samples = new long double [nstates];
    };

    long double * Call( long double * w0 )
    {
        // multinomial random number generator goes here!
        return samples;
    };
};


long double ** montecarlo( long double * w0, long double ** omegas, int * counts, int * samples, int nstates, int nexperiments, double diriscale, int niter)
{
    WHAM_DENSITY targetDensity;
    targetDensity.WhamForce(omegas, counts, samples, nstates, nexperiments);
    WHAM_MAX_PROPOSAL_DENSITY proposalDensity;
    proposalDensity.MaxProposalDensity(w0, nstates, diriscale);

    long double ** trajectory = new long double * [niter];
    for (int i=0;i<niter;i++){trajectory[i]=new long double [nstates];for(int j=0;j<nstates;j++){trajectory[i][j]=0;}}
    trajectory[0]=w0;

    cout << "Sampling " << niter << " points..." << endl;
    int i=0;
    int naccept=0;
    while ( i < niter )
    {
        long double * oldx = new long double [nstates];
        long double * newx = new long double [nstates];
        oldx = trajectory[0];
        newx = proposalDensity.Call( oldx );

        exit(1);

    }

    // do monte carlo!

    return trajectory;
}
