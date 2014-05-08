#!/usr/bin/python
#
# David M. Walker
# The University of Texas at Austin
# 11/17/2010
# 
# minimize.py
#
# Script for running optimization for fitting a spectrum with noise
# to a function.
# TODO 11/17/2010:
#   document the initial guessing algorithm
#   add options for using different model functions (need a function class first)
# 
  
import sys

from numpy import loadtxt, savetxt
from optparse import OptionParser
from pylab import savefig, figure, show, plot

from Optimizers import *

def modify_txt_extension( refname, extension ):
    extension = "-opt" + extension
    if refname[-4:] == ".txt":
        newname = refname.replace( ".txt", extension )
    else : newname = refname + extension
    return newname

# set up the option parser
parser = OptionParser()

# decide how to minimize -- for now only one of these is possible upon one invokation
parser.add_option("-n", "--nsteep", dest="nsteep", action="store", type="int", default=0, help="Number of steepest descents steps (default)" )
parser.add_option("-c", "--ncg", dest="ncg", action="store", type="int", default=0, help="Number of conjugate gradients steps" )
parser.add_option("-a", "--nalternate", dest="nalternate", action="store", type="int", default=0, help="Use the steepest descents/conjugate gradients alternator with of this many steps, for <maxiter> number of iterations" )
parser.add_option("-k", "--initial-stepsize", dest="initialStepsize", action="store", type="float", default=0.1, help="Initial step size" )
parser.add_option("--stepsize-reduce", dest="stepsizeReduce", action="store", type="float", default=0.5, help="Factor to reduce the step size after a rejected step, so that the algorithm takes a smaller step the next time around" )
parser.add_option("--stepsize-augment", dest="stepsizeAugment", action="store", type="float", default=1.2, help="Factor to increase the step size after an accepted step, so that the algorithm tries to take a larger step the next time" )

# options for deciding when to quit optimization
parser.add_option("-i", "--maxiter", dest="maxiter", action="store", type="int", default=100, help="Global number of maximum iterations" )
parser.add_option("-t", "--tol", dest="tol", action="store", type="float", default=10**-7, help="Energy tolerance for convergence (default 10**-7)" )
 
# Model and data stuff
# How to guess initial parameters: this must be supplied in a flat file
# If none is supplied we try to make a reasonable guess. (Needs to be documented.)
parser.add_option("-G", "--guess", dest="guessfile", action="store", type="string", default=None, help="User-supplied guesses for function parameters in a flat file" )
parser.add_option("-s", "--spectrum", dest="spectrumfile", action="store", type="string", default=None, help="Spectrum file" )
parser.add_option("-F", "--function", dest="function", action="store", type="string", default=None, help="Function to use (right now: use the marginal p3gaussian, no other option)" )

# general settings
parser.add_option("-H", "--long-help", dest="longhelp", action="store_true", default=False, help="Display long help" )
parser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False, help="Print verbosely" )
parser.add_option("--double", "--doublegaussian", dest="gaussian", action="store_true", default=False, help="Flag's the system for a double Gaussian" )
parser.add_option("--testmode", dest="testmode", action="store_true", default=False, help="Run in test mode; print initial state only" )


# extra output
parser.add_option("--plot", dest="plotbool", action="store_true", default=False, help="Show a plot of the final fit (requires pylab)" )

# parse command line arguments
(options, args) = parser.parse_args()
nsteep = options.nsteep
ncg = options.ncg
nalternate = options.nalternate
initialStepsize = options.initialStepsize
stepsizeReduce = options.stepsizeReduce
stepsizeAugment = options.stepsizeAugment
maxiter = options.maxiter
tol = options.tol
guessfile = options.guessfile
spectrumfile = options.spectrumfile
function = options.function
longhelp = options.longhelp
verbose = options.verbose 
testmode = options.testmode 
plotbool = options.plotbool
gaussian = options.gaussian


# deal with special cases
if verbose :
    print "If you need help, try adding the --help or --long-help options."

if longhelp or len(sys.argv) == 1 or spectrumfile is None :
    scriptname = sys.argv[0]
    print "Long help"
    parser.print_help()
    
    print "This is the long help."
    print "Example for running 1000 steepest descents steps on the spectrum in 'spectrum.txt':"
    print "\t%s -s spectrum.txt -n 1000" % scriptname
    print "This will automatically guess (might not be a good guess) the initial parameters,"
    print "then run steepest descents until 1,000 iterations have completed or the negative"
    print "log probability (the energy) changes by less than the default tolerance (10^-6)."
    print 
    print "Example for running 1000 steepest descents steps followed by 500 conjugate gradients"
    print "steps, with the initial guess in 'guess.txt,' print additional messages (-v), and"
    print "using an initial stepsize for each method of k=0.1"
    print "\t%s -s spectrum.txt -n 1000 -c 500 -k 0.1 --verbose --guess=guess.txt" % scriptname
    print
    print "Example for running no more than 1000 steps of alternating steepest descents/"
    print "conjugate gradients, each 10 at a time:"
    print "\t%s --spectrum=spectrum.txt -a 10 --maxiter=1000" % scriptname
    print "This last way seems to be very successful finding minima quickly."
    print
    print "At minimum, a spectrum file must be supplied. If this is the only argument then a"
    print "steepest descents minimization will be run with default parameters."

    if verbose and spectrumfile :
        print "verbose is SET"
    else :
        sys.exit()

if gaussian:
    print "We are going to assume there are two Gaussians to fit...."
    from SpecialSpectrumModels import *
else:
    print "We are going to assume there isonly one Gaussian to fit...."
    from SpectrumModels import *

# sanity check
if nsteep + ncg > maxiter : 
    print "WARNING: %d steepest descents and %d conjugate gradients is more than the maximum" % ( nsteep, ncg )
    print "specified iterations of %d steps. Adjusting maxiter to %d." % ( maxiter, nsteep + ncg )
    print "Note that this will not run any steep/cg alternation!"
    maxiter = nsteep + ncg

# a special default: if nothing was specified, then run 100 steps of steep
if nsteep == ncg == nalternate == 0: 
    if verbose : print "Doing 100 steps of steepest descents (default run)"
    nsteep = 100
    maxiter = 100

# open the spectrum
try:
    frequencies, signals = loadtxt( spectrumfile ).transpose()
except ValueError:
    print "ERROR: Problem reading %s" % spectrumfile
    print "Perhaps the contents of the file is not numeric?"
    sys.exit()
except IOError :
    print "ERROR: %s not found" % spectrumfile
    sys.exit()
except : raise

# open the initial guesses file
try:
    if guessfile: initialState = loadtxt( guessfile )
    elif verbose: print "Guessing initial state"
except ValueError :
    print "ERROR: Problem reading %s" % guessfile
    print "Perhaps the contents of the file is not numeric?"
    sys.exit()
except IOError :
    print "ERROR: %s not found" % guessfile
    sys.exit()
except : raise

# Scale frequencies and signals to lie on 0,1 since it seems to make the algorithms
# more stable.
# This is probably a bad way to do this since inverting it requires knowledge
# of the model function.
minfreq = frequencies.min()
maxfreq = frequencies.max()
frequencies -= minfreq
frequencies /= maxfreq-minfreq

minsig = signals.min()
maxsig = signals.max()
signals -= minsig
signals /= maxsig - minsig

# Set up the model function
if guessfile:
    model = p3gaussianM( data=(frequencies, signals), guess=initialState, verbose=verbose )
else:
    model = p3gaussianM( data=(frequencies, signals), verbose=verbose )

realfrequencies = frequencies  * (maxfreq-minfreq)+minfreq
realsignals = signals * (maxsig-minsig)+minsig

if verbose:
    if plot:
        fig9 = figure()
        plot(realfrequencies, (signals-model.baseline))
        #plot(realfrequencies, signals)
        show()

if testmode :
    print "*** initial state"
    model.showstate()
    if plotbool:
        model.plot()
        #show()
    sys.exit()

if verbose:
    print
    model.showstate()
    print "MINIMIZING"

results = ""
results += "Command line: "
for elem in sys.argv : results += elem + " "
results += "\n\n"
results += "Settings:\n"
results += "%d steepest descents steps\n" % nsteep
results += "%d conjugate gradients steps\n" % ncg
results += "%d steps steepest descents/%d steps CG for %d iterations\n" % ( nalternate, nalternate, maxiter )
results += "initial step size %e\n" % initialStepsize
results += "stepsizeReduce %e\n" % stepsizeReduce
results += "stepsizeAugment %e\n" % stepsizeAugment
results += "energy tolerance %e\n" % tol
results += "guessfile %s\n" % guessfile
results += "spectrumfile %s\n" % spectrumfile 
results += "function %s\n" % function 
results += "\n"
results += "Initial energy = %e\n" % model.energy

energytraj = []

# Do steepest descents
if nsteep > maxiter > 0 : nsteep = maxiter
maxiter -= nsteep
if nsteep > 0:
    if verbose: print "Doing %d steps of steepest descents" % nsteep
    minimizer = SteepestDescents( function=model, niter=nsteep, k=initialStepsize, incfac=stepsizeAugment, decfac=stepsizeReduce, tol=tol, verbose=verbose)
    energytraj.extend( minimizer.energytraj )
    if verbose: print
results += "Energy after steepest descents = %e\n" % model.energy

# Do conjugate gradients
if ncg > maxiter : ncg = maxiter
maxiter -= ncg 
if ncg > 0:
    if verbose: print "Doing %d steps of conjugate gradients" % ncg
    minimizer = ConjugateGradients( function=model, niter=ncg, k=initialStepsize, incfac=stepsizeAugment, decfac=stepsizeReduce, tol=tol, verbose=verbose )
    energytraj.extend( minimizer.energytraj )
    if verbose: print 
results += "Energy after conjugate gradients = %e\n" % model.energy

# Do the alternate steep/cg; here we do the remaining maxiter steps because nalternate is a little
# different. This way to do this may be A Bad Idea.
if maxiter > 0 and nalternate > 0:
    if verbose: print "Doing alternating %d steps of steep/cg, %d total iterations)" % ( nalternate, maxiter ) 
    minimizer = AlternateSteepCG( function=model, niter=maxiter, nalternate=nalternate, k=initialStepsize, incfac=stepsizeAugment, decfac=stepsizeReduce, tol=tol, verbose=verbose )
    energytraj.extend( minimizer.energytraj )
    print
results += "Final energy = %e\n" % model.energy 

# Optionally display results
if verbose: 
    print "Final result:"
    for parameter in model.parameters : print "% .4e" % parameter,
    print
    print results

# Convert back to frequency units
#normfreq = frequencies
peakcenter = model.position * ( maxfreq-minfreq ) + minfreq
peakwidth = model.width * ( maxfreq-minfreq )
peakfwhm = 2*sqrt( 2*log(2) )*peakwidth

results += "\n"
results += "mean frequency = %4.3f\n" % peakcenter
results += "mean width = %4.3f\n" % peakwidth
results += "fwhm = %4.3f\n" % peakfwhm

if gaussian:
    peakcenter2 = model.position2 * ( maxfreq-minfreq ) + minfreq
    peakwidth2 = model.width2 * ( maxfreq-minfreq )
    peakfwhm2 = 2*sqrt( 2*log(2) )*peakwidth2
    results += "\n"
    results += "Second mean frequency = %4.3f\n" % peakcenter2
    results += "Second mean width = %4.3f\n" % peakwidth2
    results += "Second fwhm = %4.3f\n" % peakfwhm2
    

# Write log file
logfile = modify_txt_extension( spectrumfile, ".log" )
file = open( logfile, "w" )
file.write( results )
file.close()
if verbose: print "Wrote log file to '%s'" % logfile

# Write the best fit parameters to file
outputfile = modify_txt_extension( spectrumfile, ".txt" )
savetxt( outputfile, model.parameters )
if verbose: print "Wrote optimized parameters to '%s'" % outputfile

# Do plotting
if plotbool: model.plot()
fitplotfile = modify_txt_extension( spectrumfile, ".fit.eps" )
figure(0)
model.plot(fitplotfile)
if verbose: print "Wrote data plotted with best fit to '%s'" % fitplotfile

correctedfile = modify_txt_extension( spectrumfile, ".cor.eps" )
fit = model.baseline
correctedsignal = signals - fit
figure(1)

if plot:
    plot(realfrequencies, model.baseline)
    
savefig(correctedfile)    
if verbose: print "Wrote baseline corrected spectrum plot to '%s'" % correctedfile

# Write a corrected spectrum
corrspectrumfile = modify_txt_extension( spectrumfile, ".cor.txt" )
fitdata = array((realfrequencies, correctedsignal)).transpose()
savetxt( corrspectrumfile, fitdata )
if verbose: print "Wrote baseline corrected spectrum to '%s'" % corrspectrumfile

# Generate a plot with the energy trajectory
energytrajfile = modify_txt_extension( spectrumfile, ".etr.eps" )
figure(2)
plot(energytraj)
savefig(energytrajfile)
if verbose: print "Wrote energy trajectory to '%s'" % energytrajfile

# Generate a detailed Gaussian Profile
details = modify_txt_extension( spectrumfile, ".detailed.txt")
signal=zeros(len(frequencies))
signal2=zeros(len(frequencies))
for x in range(len(frequencies)):
    signal[x]= model.parameters[4]*exp(-0.5*((frequencies[x] - model.parameters[5])**2)*(model.parameters[6]**2))
    if gaussian:
        signal2[x]=model.parameters[7]*exp(-0.5*((frequencies[x] - model.parameters[8])**2)*(model.parameters[9]**2))

if gaussian:
    fullSignal=fit+signal+signal2
    detailed = array((realfrequencies, fullSignal, fit, signal, signal2)).transpose()
    savetxt( details, detailed )
    print "\nWARNING: We fit the spectrum to two Gaussians!"
else:
    fullSignal=fit+signal
    detailed = array((realfrequencies, fullSignal, fit, signal)).transpose()
    savetxt( details, detailed )
    print "\nWARNING: We fit the spectrum to only one Gaussian!"

if plot:
    fig = figure()
    plot (realfrequencies, correctedsignal, 'k')
    plot (realfrequencies, signal, 'b')
    if gaussian:
        plot(realfrequencies, signal2, 'r')

    show()
    

