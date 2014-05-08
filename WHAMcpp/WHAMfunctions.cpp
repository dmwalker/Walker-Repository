//
//  WHAMfunctions.cpp
//  mcWHAM
//


#include "WHAMfunctions.h"

int DOWHAM::Points() {return points;}
long double * DOWHAM::OptTrajectory() {return opt_trajectory;}
    
void DOWHAM::DoWham( long double * omega_vec, int * counts, int * samples, long double ** omegas, int states, int nsamples, int niter, float tol)
{
    nstates=states;
    nexperiments=nsamples;
    ncounts=counts;
    sample_sizes=samples;
    wvec=omega_vec;
    omega_vectors=omegas;
    OmegaTranspose();
    
    /* First WHAM step */
    trajectory=new long double * [nstates];
    trajectory[0]=new long double [nstates];
    trajectory[0]=omega_vec;
    points=0;
    
    /* WHAM iterations */
    while (points < niter )
    {
        trajectory[points+1]=new long double [nstates];
        trajectory[points+1]=WhamStep(trajectory[points]);
        
        if ( Sum( square_diff(trajectory[points+1], trajectory[points], nstates), nstates ) < tol )
        {
            std::cout << "Converged to " << tol << " at step " << points << std::endl;
            break;
        }
        if ( 0/Sum( trajectory[points+1], nstates ) != 0 )
        {
            std::cout << "nan at " << points << std::endl;
            break;
        }
            
        points++;
    }
    opt_trajectory = new long double [nstates];
    opt_trajectory = trajectory[points];
    
};

long double * DOWHAM::WhamStep( long double * trajectory )
{
    double long * fms=new double long [nexperiments];
    for (int i=0;i<nexperiments;i++)
    {
        fms[i]=sample_sizes[i]/dot_product(trajectory, omega_vectors[i], nstates);
    }
    
    long double * wguess = new double long [nstates];
    for (int i=0;i<nstates;i++)
    {
        wguess[i] = ncounts[i]/dot_product(omega_transpose[i], fms, nexperiments);
    }
    long double wguess_sum = Sum( wguess, nstates );
    for (int i=0;i<nstates;i++){wguess[i]/=wguess_sum;}
    
    return wguess;
};

long double DOWHAM::dot_product( double long * a, double long * b, int size )
{
    double long product=0;
    for (int i=0;i<size;i++)
    {
        product += a[i]*b[i];
    }
    return product;
};

void DOWHAM::OmegaTranspose()
{
    omega_transpose = new long double * [nstates];
    for (int i=0;i<nstates;i++)
    {
        omega_transpose[i] = new long double [nexperiments];
        for (int j=0;j<nexperiments;j++)
        {
            omega_transpose[i][j]=omega_vectors[j][i];
        }
    }
};
    
long double DOWHAM::Sum( long double * an_array, int length_of_an_array )
{
    long double sum=0;
    for (int i=0;i<length_of_an_array;i++)
    {
        sum += an_array[i];
    }
    return sum;
}
    
long double * DOWHAM::square_diff( long double * array_1, long double * array_2, int length_of_arrays )
{
    long double * difference = new long double [length_of_arrays];
    for (int i=0;i<length_of_arrays;i++)
    {
        difference[i]=(array_1[i]-array_2[i])*(array_1[i]-array_2[i]);
    }
    return difference;
}
    