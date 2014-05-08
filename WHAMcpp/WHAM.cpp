//
//  main.cpp
//  mcWHAM
//

//#include <iostream>
//#include <fstream>
//#include <sstream>
//#include <cmath>
//#include <cstring>
//#include <cstdlib>
#include "fileoptions.h"


int main(int argc, char * argv[])
{
    /* set defaults */
    const char *    filename; //= "/Users/ritchie/Desktop/MD/Rap1a/cpp_scripts/mcWHAM/mcWHAM/1dof.file";
    const char *    outname;
    char            out[100];
    char            infile[100];
    double          T = 300; // K
    range           begins,ends,bin_sizes;
    double          kb = 0.0083144621; // kJ/(mol*k)
    int             dof = 1;
    bool            bVerbose = false;
    int             niter = 50000;
    double          tol = 1e-9;
    int             firstframe = 0, lastframe = -1;
    
    /* parse through arguments */
    if (cmdOptionExists(argv, argv+argc, "--t")) {
        char * temp = getCmdOption(argv, argv+argc, "--t");
        sscanf(temp, "%lf", &T);
        if ( T < 0 ) { std::cout << "ERROR: Temperature < 0" << std::endl; exit(1); }
    }
    if (cmdOptionExists(argv, argv+argc, "--dof")) {
        char * chardof = getCmdOption(argv, argv+argc, "--dof")  ;
        sscanf(chardof, "%d", &dof);
        if ( dof < 1 ) { std::cout << "ERROR: Fewer than 1 degree of freedom" << std::endl; exit(1); }
    }
    if (cmdOptionExists(argv, argv+argc, "--s")) {
        begins = getMultiCmdOption(argv, argv+argc, "--s");
    }
    else 
    {
        begins.values=new double [dof];
        for (int i=0;i<dof;i++){begins.values[i]=-180;}
        begins.n_values=dof;
    }
    if (cmdOptionExists(argv, argv+argc, "--e")) {
        ends = getMultiCmdOption(argv, argv+argc, "--e");
    }
    else 
    {
        ends.values=new double [dof];
        for (int i=0;i<dof;i++){ends.values[i]=180;}
        ends.n_values=dof;
    }
    if (cmdOptionExists(argv, argv+argc, "--b")) {
        bin_sizes = getMultiCmdOption(argv, argv+argc, "--b");
        for (int i=0 ; i<bin_sizes.n_values ; i++)
        {
            if ( bin_sizes.values[i] <= 0 ) { std::cout << "ERROR: Bin size <= 0" << std::endl; exit(1); }
        }
    }
    else 
    {
        bin_sizes.values=new double [dof];
        for (int i=0;i<dof;i++){bin_sizes.values[i]=5;}
        bin_sizes.n_values=dof;
    }
    if (cmdOptionExists(argv, argv+argc, "--tol")) {
        char * tolerance = getCmdOption(argv, argv+argc, "--tol");
        sscanf(tolerance, "%lf", &tol);
    }
    if (cmdOptionExists(argv, argv+argc, "--iter")) {
        char * iter = getCmdOption(argv, argv+argc, "--iter");
        sscanf(iter, "%d", &niter);
    }
    if (cmdOptionExists(argv, argv+argc, "--v")) {
        bVerbose = true;
    }
    if (cmdOptionExists(argv, argv+argc, "--fb")) {
        char * frame0 = getCmdOption(argv, argv+argc, "--fb");
        sscanf(frame0,"%d",&firstframe);
    }
    if (cmdOptionExists(argv, argv+argc, "--fe")) {
        char * frameN = getCmdOption(argv, argv+argc, "--fe");
        sscanf(frameN,"%d",&lastframe);
    }
    if (cmdOptionExists(argv, argv+argc, "--f")) {
        filename = getCmdOption(argv, argv+argc, "--f");
    }
    else { 
        userinfo(T, begins.values[0], ends.values[0], bin_sizes.values[0], dof, tol, niter, bVerbose);
        exit(1);
    }
    if (cmdOptionExists(argv, argv+argc, "--o")) {
        outname = getCmdOption(argv, argv+argc, "--o");
    }
    else { outname = filename; }
    if (cmdOptionExists(argv, argv+argc , "--h") || cmdOptionExists(argv, argv+argc, "--help") || cmdOptionExists(argv, argv+argc, "-h") || argc == 1) {
        userinfo(T, begins.values[0], ends.values[0], bin_sizes.values[0], dof, tol, niter, bVerbose);
        exit(1);
    }

    else { userinfo(T,begins.values[0],ends.values[0],bin_sizes.values[0],dof,tol,niter,bVerbose); }
    compare_len(dof, "degrees of freedom", begins.n_values, "coordinate starting ranges" );
    compare_len(dof, "degrees of freedom", ends.n_values, "coordinate ending ranges" );
    compare_len(dof, "degrees of freedom", bin_sizes.n_values, "bin sizes" );

    strcpy(infile,filename);
    strcpy(out,outname);
    strcat(out,".mean");
    
    /* This is purely for output information purposes */
    char finalframe[100];
    if (lastframe != -1) { sprintf(finalframe,"%d",lastframe); }
    else { sprintf(finalframe,"end"); }
    
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    std::cout << "Will read " << infile << " from frame " << firstframe << " to " << finalframe << "." << std::endl;
    std::cout << "Temperature = " << T << " K" << std::endl;
    std::cout << "Thermal Energy = " << kb*T << " kJ/mol" << std::endl;
    std::cout << "Biasing coordinate range(s) = ";
    for (int i=0; i<dof; i++) 
    {  
        std::cout << begins.values[i] << " to " << ends.values[i] << "    ";
    }        
    std::cout << "\nBin size(s) = ";
    for (int i=0; i<dof; i++) 
    {
        std::cout << bin_sizes.values[i] << "    ";
    }
    std::cout << "\nDegrees of freedom = " << dof << std::endl;
    std::cout << "Will iterate up to " << niter << " times until converged to " << tol <<"."<< std::endl;
    std::cout << "Will write to " << out << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;

    /* Extract biasing coordinate information from files listed in infile */
    int n_files=get_n_lines(infile);
    TorsionExperiment * experiment = new TorsionExperiment [n_files];
    
    std::string line;
    std::ifstream file (infile);
    int first_experiment=1000;
    int previous_experiment=-1;
    int previous_dof=-1;
    int i=0;
    if ( file.is_open() )
    {
        while ( file.good() )
        {
            getline( file, line) ;
            if ( line == "\n" || line == "" ) { break;}
            std::cout << "Opening " << infile << std::endl;

            experiment[i].read_filelist( line, previous_experiment, previous_dof, begins.values, ends.values, bin_sizes.values, kb, bVerbose );
            previous_experiment=experiment[i].Experiment();
            previous_dof=experiment[i].DoF();
            if ( experiment[i].Experiment() < first_experiment ) { first_experiment=experiment[i].Experiment(); }
            i++;
        }
    }
    else if (!file) {std::cout << "Error opening " << infile << "." << std::endl; exit(1);} 
                
    /* Find out how many total bins */
    int nstates=1;
    for (int d=0; d<dof; d++)
    {
        for (int i=0; i<n_files; i++)
        {
            if ( experiment[i].DoF() == d )
            {
                nstates *= experiment[i].trajectory.nBins();
                break; // the nth DoF for experiment 1 should have the same number of bins as the nth DoF for experiment 2, which should have the same number of bins as experiment 3, which...  so only look at the first experiment worth of DoFs
            }
        }
    }
    
    int * counts = new int [nstates];
    int * samples = new int [n_files/dof];
    long double ** omegas = new long double * [n_files/dof];
    for (int i=0;i<nstates;i++){counts[i]=0;}
    std::cout<<std::endl;
    
    int ** group_traj;
    nD_WHAM_grouping(experiment, dof, n_files, first_experiment, group_traj, bVerbose);

    /* Count the number of times each bin is visited throughout the overall simulation */
    for (int trajectory=0; trajectory<n_files/dof; trajectory++)
    {        
        std::ofstream binfile;
        char binname[100];
        sprintf(binname,"%s.%i.bin",outname,trajectory);
        binfile.open(binname);
        if (bVerbose){std::cout << "Writing bin assignments for experiment " << trajectory << " to " << binname << "." << std::endl;}
        int final_frame_to_read = experiment[group_traj[trajectory][0]].trajectory.nFrames();
        if ( lastframe < final_frame_to_read && lastframe != -1 ) { final_frame_to_read = lastframe+1; }
        for (int frame=firstframe; frame<final_frame_to_read; frame++)
        {
            int this_bin=0;
            int dofprod=1;
            for (int this_dof=0; this_dof<dof ; this_dof++)
            {
                if (this_dof > 0)
                {
                    dofprod *= experiment[group_traj[trajectory][this_dof-1]].trajectory.nBins();
                }
                this_bin += experiment[group_traj[trajectory][this_dof]].trajectory.BinID()[frame]*dofprod;
            }
//            cout << experiment[group_traj[trajectory][0]].trajectory.BinID()[frame] << " " << experiment[group_traj[trajectory][1]].trajectory.BinID()[frame] << " " << this_bin << endl;
            counts[this_bin]++;
            binfile << this_bin << "\n";
        }
        binfile.close();
        
        /* Also, while looking through trajectories, we need to build nD omega array for each
         I could not figure out a good (any) way to do this for n dimensions, so I'm going up to 3... */
        omegas[trajectory]=new long double [nstates];
        
        switch ( dof )
        {
            case 1:
            {
                omegas[trajectory]=experiment[group_traj[trajectory][dof-1]].Omega();
                break;
            }
            case 2:
            {
                int m=0;
                for (int j=0; j<experiment[group_traj[trajectory][dof-1]].trajectory.nBins(); j++)
                {
                    for (int i=0; i<experiment[group_traj[trajectory][dof-2]].trajectory.nBins(); i++)
                    {
                        omegas[trajectory][m]=experiment[group_traj[trajectory][dof-2]].Omega()[i]*experiment[group_traj[trajectory][dof-1]].Omega()[j];
                        m++;
                    }
                }
                break;
            }
            case 3:
            {
                int m=0;
                for (int k=0; k<experiment[group_traj[trajectory][dof-1]].trajectory.nBins(); k++)
                {
                    for (int j=0; j<experiment[group_traj[trajectory][dof-2]].trajectory.nBins(); j++)
                    {
                        for (int i=0; i<experiment[group_traj[trajectory][dof-3]].trajectory.nBins(); i++)
                        {
                            omegas[trajectory][m]=experiment[group_traj[trajectory][dof-1]].Omega()[i]*experiment[group_traj[trajectory][dof-2]].Omega()[j]*experiment[group_traj[trajectory][dof-3]].trajectory.nBins();
                            m++;
                        }
                    }
                }
                break;
            }
            default :
            {
                std::cout << "You have " << dof << " degrees of freedom.  Currently, only up to 3 degrees of freedom can be handled.  Exiting..." << std::endl;
                exit(1);
            }
        }
    }
    
    std::cout << "Built (" << n_files/dof << "x" << nstates << ") omega matrix." << std::endl;
    int nzeros=0;
    std::cout << dof << "-Dimensional State counts: [ ";
    for (int i=0;i<nstates;i++){
        std::cout<<counts[i]<<" ";
        if (counts[i] == 0){nzeros++;}
    }
    std::cout << "]\n" << std::endl;
   
    if (bVerbose) 
    {
        for (int n=0;n<n_files/dof;n++)
        {
            std::cout << "Omega for experiment " << n << ": [ ";
            for (int i=0;i<nstates;i++){
                std::cout<<omegas[n][i]<<" ";
            }
            std::cout << "]\n" << std::endl;
        }
    }
   
    if ( nzeros > 0 )
    {
        float percent = 100.0;
        percent *= nzeros;
        percent /= nstates;
        std::cout << nzeros << " (" << percent << "%) bins unvisited.  Consider additional sampling if this number is large." << std::endl;
    }
    
    std::cout << "Samples: [ ";
    int n=0;
    for (int i=0;i<n_files;i+=dof)
    {
        std::cout<<experiment[i].trajectory.nFrames()<<" ";
        samples[n]=experiment[i].trajectory.nFrames();
        n++;
    }
    std::cout << "]" << std::endl;
    
    // settings and inital conditions 
    long double * w0 = new long double [nstates];
    long double pnstates = 1.0/nstates;
    for (int i=0; i<nstates; i++)
    {
        w0[i]=pnstates;
    }
    
    std::cout << "OPTIMIZING" << std::endl;
    DOWHAM opttrajectory;
    opttrajectory.DoWham(w0, counts, samples, omegas, nstates, n_files/dof, niter, tol);
    std::cout << "Ran " << opttrajectory.Points() << " points of optimization." << "\n[ ";
    for (int i=0;i<nstates;i++){
        std::cout << opttrajectory.OptTrajectory()[i] << " ";
    }
    std::cout << "]" << std::endl;
  
/* Not doing the Monte Carlo part; it seems unneccessary */
    
    /* I don't know if this is the best way to handle the experimental temperatures... */
    double exp_T = 0;
    int total_frames=0;
    for (int i=0; i<n_files/dof; i++)
    {
        total_frames += experiment[group_traj[i][0]].trajectory.nFrames();
        exp_T += experiment[group_traj[i][0]].T()*experiment[group_traj[i][0]].trajectory.nFrames();
    }
    exp_T /= total_frames;
    
    /* Calculate the PMF_bin(i) as -kT ln(w(i)).  Assume for any bin that had '0' visits, the potential is 1000 */
    double long * potential = new double long [nstates];
    for (int i=0; i<nstates; i++)
    {
        if ( opttrajectory.OptTrajectory()[i] == 0 )
        {
            potential[i]=100;
        }
        else 
        {
            potential[i]=-kb*exp_T*log( opttrajectory.OptTrajectory()[i] );
        }
    }

    
    /* Now go back and calculate the probability for the now-appropriate temperature */
    double long * probability = new double long [nstates];
    double long prob_sum=0;
    for (int i=0; i<nstates; i++)
    {
        probability[i]=exp(-1/(kb*T)*potential[i]);
        prob_sum += probability[i];
    }
    std::cout << std::endl;
    for (int i=0; i<nstates; i++)
    {
        probability[i] /= prob_sum;
    }

    /* Make output files */
    /* Start with 1 DoF PMF and probability files.  I've not had much reason to use the PMF file, but the 1 DoF probability file is best used with the .bin files for boltzmann weighting, since they are assigned in 1 dimension, no matter the number of DoF used. */
    std::ofstream prob;
    std::ofstream pmf;
    std::ofstream count;
    char probname[100];
    char pmfname[100];
    char countname[100];
    sprintf(probname, "%s.prob", outname);
    sprintf(pmfname, "%s.mean", outname);
    sprintf(countname,"%s.count", outname);
    prob.open(probname);
    pmf.open(pmfname);
    count.open(countname);
    for (int i=0; i<nstates; i++)
    {
        prob << probability[i] << std::endl;
        pmf << potential[i] << std::endl;
        count << counts[i] << std::endl;
    }
    prob.close();
    pmf.close();
    count.close();
    std::cout << "Potential of mean force written to " << pmfname << "." << std::endl;
    std::cout << "Probabilities written to " << probname << "." << std::endl;

    
    if ( dof == 2 )
    {
        std::ofstream d2prob;
        std::ofstream ncount;
        std::ofstream d2pmf;
        char countname[100];
        sprintf(countname,"%s.2dcount", outname);
        ncount.open(countname);
        char d2name[100];
        sprintf(d2name,"%s.2dprob",outname);
        d2prob.open(d2name);
        char d2pmfname[100];
        sprintf(d2pmfname,"%s.2dmean",outname);
        d2pmf.open(d2pmfname);
        int bin=0;
        for (int j=0; j < experiment[group_traj[0][1]].trajectory.nBins() ; j++)
        {
            for (int i=0; i < experiment[group_traj[0][0]].trajectory.nBins() ; i++)
            {
                d2prob << i*bin_sizes.values[0]-ends.values[0] << " " << j*bin_sizes.values[1]-ends.values[1] << " " << probability[bin] << std::endl;

                ncount << i*bin_sizes.values[0]-ends.values[0] << " " << j*bin_sizes.values[1]-ends.values[1] << " " << counts[bin] << std::endl;
                d2pmf  << i*bin_sizes.values[0]-ends.values[0] << " " << j*bin_sizes.values[1]-ends.values[1] << " " << potential[bin] << std::endl;
                bin++;
            }
            d2prob << std::endl;
            ncount << std::endl;
            d2pmf << std::endl;
        }
        d2prob.close();
        ncount.close();
        d2pmf.close();
    }
    
    else if ( dof == 3 )
    {
        std::ofstream d3prob;
        std::ofstream ncount;
        char countname[100];
        sprintf(countname,"%s.3dcount", outname);
        ncount.open(countname);
        char d3name[100];
        sprintf(d3name,"%s.3dprob",outname);
        d3prob.open(d3name);
        int bin=0;
        for (int k=0; k < experiment[group_traj[0][2]].trajectory.nBins() ; k++)
        {
            for (int j=0; j < experiment[group_traj[0][1]].trajectory.nBins() ; j++)
            {
                for (int i=0; i < experiment[group_traj[0][0]].trajectory.nBins() ; i++)
                {
                    d3prob << j*bin_sizes.values[0]-ends.values[0] << " " << i*bin_sizes.values[1]-ends.values[1] << " " << k*bin_sizes.values[2]-ends.values[2] << " " << probability[bin] << std::endl;
                    ncount << j*bin_sizes.values[0]-ends.values[0] << " " << i*bin_sizes.values[1]-ends.values[1] << " " << k*bin_sizes.values[2]-ends.values[2] << " " << counts[bin] << std::endl;
                    bin++;
                }
            }
            d3prob << std::endl;
            ncount << std::endl;
        }
        d3prob.close();   
    }

    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    std::cout << "Read " << infile << " from frame " << firstframe << " to " << finalframe << "." << std::endl;
    std::cout << "Temperature = " << T << " K" << std::endl;
    std::cout << "Thermal Energy = " << kb*T << " kJ/mol" << std::endl;
    std::cout << "Biasing coordinate range(s) = ";
    for (int i=0; i<dof; i++) 
    {  
        std::cout << begins.values[i] << " to " << ends.values[i] << "    ";
    }        
    std::cout << "\nBin size(s) = ";
    for (int i=0; i<dof; i++) 
    {
        std::cout << bin_sizes.values[i] << "    ";
    }
    std::cout << "\nDegrees of freedom = " << dof << std::endl;
    std::cout << "Converged to " << tol <<"."<< std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    return 0;
}

