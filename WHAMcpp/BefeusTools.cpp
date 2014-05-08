//
//  BefeusTools.cpp
//  mcWHAM
//


#include "BefeusTools.h"


vec convert( double angle )
{
    vec answer;
    answer.x=cos(angle*M_PI/180);
    answer.y=sin(angle*M_PI/180);
    return answer;
};

double periodicity ( double angle )
{
    vec theta=convert( angle );
    return atan2(theta.y, theta.x)*180/M_PI;
};

double angle_difference(double a, double b)
{
    vec theta1 = convert( a );
    vec theta2 = convert( b );
    return acos( theta1.x*theta2.x + theta1.y*theta2.y )*180/M_PI; //This always returns the absolute value of the difference
};

double remove_periodicity( double angle )
{
    vec Cartesian=convert(angle);
    return atan2(Cartesian.y,Cartesian.x)*180/M_PI;
}

int Experiment::nFrames() {return frame;}
int Experiment::nBins() {return nbins;}
double * Experiment::Coord() {return coordinates;}
int * Experiment::BinCounts() {return bincounts;}
int * Experiment::BinID() {return frame_bins;}

void Experiment::readtrajectory( std::string trajectory, double begin, double end, double bin_sizes, bool bVerbose )
{
    frame=0;
    double * tempcoordinates;
    int size_of_array=100000000; // 100,000,000 frames should be sufficient
    tempcoordinates=new double [size_of_array];
    double time;
    
    std::string line;
    std::ifstream file ( trajectory.c_str() ) ;
    if ( file.is_open() ) {
        while (file.good()) {
            getline( file, line );
            if ( line == "" || line == "\n" ) { break; }
            std::stringstream check_line( line );
            std::string item;
            int members = 0;
            while( check_line >> item)
            {
                members++;
            }
                
            if (members == 1 )
            {
                std::stringstream linestream( line );
                linestream >> tempcoordinates[frame];
                tempcoordinates[frame]=remove_periodicity(tempcoordinates[frame]);
                frame++;
            }
            else if (members == 2 )
            {
                std::stringstream linestream( line );
                linestream >> time >> tempcoordinates[frame];
                tempcoordinates[frame]=remove_periodicity(tempcoordinates[frame]);
                frame++;
            }
            else {std::cout << "Error in xvg file format.  It should either be:\n<angle>\n -or-\n<time> <angle>" << std::endl; exit(1);}
        }
    }
    else if ( !file ) { std::cout << "Error opening " << trajectory << "." << std::endl; exit(1);}
    
    coordinates=new double [frame];
    for ( int n=0; n<frame; n++ ) {
        coordinates[n]=tempcoordinates[n];
    }
    delete [] tempcoordinates;
    histogram(coordinates, begin, end, bin_sizes, bVerbose);
};
    
void Experiment::histogram( double * array, double begin, double end, double bin_size, bool bVerbose )
{
    nbins = (end-begin)/bin_size;
    
    int bin=0;
    bincounts=new int [nbins];
    frame_bins=new int [frame];
    memset(bincounts, 0, sizeof(int));
    
    for (double i=begin; i<end; i+=bin_size)
    {
        bincounts[bin]=0;
        for (int n=0; n<frame; n++)
        {
            if ( array[n] >= begin && array[n] <= end )
            {
                if ( i <= array[n] && array[n] < i+bin_size)
                {
                    bincounts[bin]++;
                    frame_bins[n]=bin;
                }
            }
            else
            {
                std::cout << "WARNING: frame " << n << " has a value of " << array[n] << " which is outside the range of " << begin << " to " << end << std::endl;
                exit(1);
            }
        }
        bin++;
    }
        
    std::cout<<"1-Dimensional bin counts: [ ";
    for (int i=0;i<bin;i++)
    {
        std::cout<<bincounts[i]<<" ";
    }
    std::cout<<"]"<<std::endl;
};

int TorsionExperiment::DoF() {return this_dof;}
int TorsionExperiment::Experiment() {return experiment;}
double TorsionExperiment::Phi0() {return phi0;}
double TorsionExperiment::DeltaPhi() {return deltaphi;}
double TorsionExperiment::FC() {return fc;}
double TorsionExperiment::T() {return t;}
std::string TorsionExperiment::Datafile() {return xvgfile;}
long double * TorsionExperiment::Omega() {return omega;}
long double * TorsionExperiment::Potential() {return potentials;}

void TorsionExperiment::read_filelist( std::string line, int previous_experiment, int previous_dof, double *begins, double *ends, double *bin_sizes, double kb, bool bVerbose)
{
    std::stringstream check_line( line ) ;
    std::string item;
    int values = 0;
    while ( check_line >> item)
    {
        values++;
    }
    if ( values != 6 ) { std::cout << "Error! Input file incorrectly formatted.  Correct format is:\n<experiment number> <xvg path> <biased position> <zero potential +/- range> <force constant> <temperature>" << std::endl; exit(1);}
        
    std::stringstream linestream( line );
        
    linestream >> experiment >> xvgfile >> phi0 >> deltaphi >> fc >> t;
    beta=1/(kb*t);
    fc *= M_PI/180*M_PI/180; //convert from kJ/mol/radian^2 to kJ/mol/degree^2
    std::cout << "Reading data file: " << xvgfile << " from trajectory " << experiment << std::endl;
    std::cout << "Biasing coordinate centered on: " << phi0 << " +/- " << deltaphi << std::endl;
    std::cout << "Force constant = " << fc << " kJ/mol/degree^2 at temperature " << t << " kJ^-1." << std::endl;
        
    if ( experiment == previous_experiment ) { this_dof=previous_dof+1; }
    else { this_dof=0; }
        
    begin=begins[this_dof];
    end=ends[this_dof];
    bin_size=bin_sizes[this_dof];
        
    if ( bin_size == 0 || begin == 0 || end == 0 )
    {
        std::cout << "Error, it appears you did not flag the correct number of --dof" << std::endl;
        exit(1);
    }
        
    trajectory.readtrajectory(xvgfile, begin, end, bin_size, bVerbose);
    precision_error=false;
    assign_omega();
    weight_dihedral_restraint_potentials();
        
    /*
    for (int i=0;i<trajectory.nBins();i++)
    {
    std::cout<<omega[i]<<" ";
        }
        std::cout<<std::endl;
    */
        
    if (precision_error){std::cout << "WARNING!  Numeric precision has been exceeded.  " << std::endl;}
};
    
    
void TorsionExperiment::weight_dihedral_restraint_potentials()
{
    potentials=new long double [trajectory.nFrames()];
    for (int i=0; i<trajectory.nFrames(); i++ )
    {
        potentials[i] = weight_dihedral_restraint_potential(trajectory.Coord()[i]);
    }
};
    
long double TorsionExperiment::weight_dihedral_restraint_potential( double phi )
{
    double diffphi = angle_difference( phi0, phi );
    double ddp;
    if ( diffphi > deltaphi )
    {
        ddp = angle_difference( diffphi, deltaphi );
    }
    else { ddp = 0; }
    //ddp *= M_PI/180; // is this correct? Looks like: K*mol/(kJ*K) * kJ/mol/degree**2 * radian**2 = radian**2/degree**2
        
        /*        // This is directly from the gromacs source code, /src/gmxtools/dihres.c
         double dp;
         dp = phi - phi0;
         double ddp;
         if (fabs(dp) >= deltaphi )
         {
         if (dp >= 180 )
         {
         dp -= 2*180;
         }
         else if (dp < -180)
         {
         dp += 2*180;
         }
         
         if (dp > deltaphi)
         {
         ddp = dp - deltaphi;
         }
         else if (dp < -deltaphi)
         {
         ddp = dp + deltaphi;
         }
         else { ddp = 0; }
         }
         else { ddp =0; }
         */
        /*        // This is lifted straight from Dan's code
         double diffphi, ddp;
         diffphi=phi-phi0;
         if (diffphi >= 180) { diffphi -= 360;}
         else if (diffphi < -180) {diffphi += 360;}
         if (diffphi > deltaphi) { ddp=diffphi-deltaphi;}
         else if (diffphi < -deltaphi) { ddp=diffphi+deltaphi;}
         else { ddp=0; }
         */
        /* My method, Dan's method, and the gromacs method all yield equivalent results for data in range -pi:pi */
        
    return exp(-beta*0.5*fc*ddp*ddp);
};

long double TorsionExperiment::trapezoidal_integration( double a, double b )
{
    long double integrand=0;
    long double width=(b-a)/100;
    for (double i=a;i<=b;i+=width)
    {
        long double functiona=weight_dihedral_restraint_potential(i);
        long double functionb=weight_dihedral_restraint_potential(i+width);
        integrand += 0.5*(width)*(functiona+functionb);
    }
    /*
        long double functiona=weight_dihedral_restraint_potential(a);
        long double functionb=weight_dihedral_restraint_potential(b);
        return 0.5*(b-a)*(functiona+functionb);
    */
    return integrand;
};
    
void TorsionExperiment::assign_omega()
{
    omega = new long double [trajectory.nBins()];
    for (int i=0; i<trajectory.nBins(); i++)
    {
        long double a=i*bin_size + begin;
        long double b=(i+1)*bin_size + begin;
        omega[i]=trapezoidal_integration(a,b);
        if ( !precision_error && omega[i] == 0 ) { precision_error = true; }
    }
};


