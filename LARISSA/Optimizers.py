# Minimize.py module
# David M. Walker
# The University of Texas at Austin
# 11/17/2010
#
# TODO 
#   make the Minimizer objects iterable
#

from copy import copy
from numpy import dot 
from numpy.linalg import norm

TOLDEFAULT = 10**-7
class Minimizer( object ):
    """Minimize the SpectrumModel function 'function' by using the SpectrumModel.force()
    and SpectrumModel.energy() methods. (Generic model.)"""
    def __init__( self, function, niter, k, incfac=1.2, decfac=0.5, tol=TOLDEFAULT, verbose=False ):
        self.function = function
        self.niter = niter
        self.k = k
        self.incfac = incfac
        self.decfac = decfac
        self.tol = tol
        self.verbose = verbose
        self.energytraj = [ self.function.energy, ]
        self._init_more() # a template function for setting up details of the calculation 
        self._minimize()

    def _init_more( self ): pass 
    def _minimize( self ): pass

class SteepestDescents( Minimizer ):
    """Minimize a SpectrumModel function 'function' using steepest descents and the 
    model function's energy() and force() methods."""
    # same __init__ as the parent class
    # Minimizer.__init__() calls _init_more()
    def _init_more( self ): pass
    def _minimize( self ):
        stepsize = self.k 
        iteration = 0
        energy = self.function.energy
        while iteration < self.niter:
            last_energy = self.function.energy 
            energy_changed = False
            oldparameters = self.function.parameters
            self.function.parameters = self.function.parameters + stepsize * self.function.force
            newenergy = self.function.energy

            if newenergy <= energy:
                energy_changed=True
                iteration += 1
                stepsize *= self.incfac
                energy = newenergy
                self.energytraj.append( energy )
                if self.verbose : 
                    print "i=", iteration,
                    print "f=", 
                    for elem in self.function.force : print "% .4e" % elem, 
                    print "|f|=", norm(self.function.force), 
                    print "E =", newenergy
            else :
                self.function.parameters = oldparameters
                stepsize *= self.decfac

            ediff = last_energy - energy 
            if ediff < self.tol and energy_changed :
                if self.verbose: 
                    print "Energy difference %e < %e" % ( ediff, self.tol )
                break
    
class ConjugateGradients( Minimizer ):
    """Minimize a SpectrumModel function 'function' using conjugate gradients and the 
    model function's energy() and force() methods."""
    # same __init__ as the parent class
    # Minimizer.__init__() calls _init_more()
    def _init_more( self ): pass
    def _minimize( self ):
        # We update by 
        # 1. take a cg step
        #       rnew = rold + k*hnew
        #       hnew = Fnew + gamma*hold
        #       gamma = dot(Fnew,Fnew)/dot(Fold,Fold)
        #   with hold = 0 initially
        # 2. Accept the cg step if it goes down in energy,
        #   otherwise decrease the step size
        stepsize = self.k
        Fold = self.function.force
        hold = 0*Fold # same length vector as the force
        energy = self.function.energy
        iteration = 0
        while iteration < self.niter :
            last_energy = self.function.energy 
            energy_changed = False

            oldparameters = self.function.parameters
            Fnew = self.function.force
            gamma = dot(Fnew,Fnew)/dot(Fold,Fold)
            hnew = Fnew + gamma*hold
            self.function.parameters += stepsize * hnew
            newenergy = self.function.energy

            if newenergy <= energy :
                energy_changed = True
                iteration += 1
                stepsize *= self.incfac
                energy = newenergy 
                self.energytraj.append( energy )
                Fold = Fnew
                hold = hnew
                if self.verbose:
                    print "i=", iteration,
                    print "f=", 
                    for elem in self.function.force : print "% .4e" % elem, 
                    print "|f|=", norm(self.function.force), 
                    print "E =", newenergy
                
            else :
                self.function.parameters = oldparameters
                stepsize *= self.decfac

            ediff = last_energy - energy 
            if ediff < self.tol and energy_changed :
                if self.verbose: 
                    print "Energy difference %e < %e" % ( ediff, self.tol )
                break
            
class AlternateSteepCG( Minimizer ):
    """Minimize the SpectrumModel function 'function' by using the SpectrumModel.force()
    and SpectrumModel.energy() methods. This object will minimize by alternating steepest
    descents and conjugate gradients 'nalternate' times each for a total of 'niter' 
    iterations."""
    # different __init__ from parent class
    def __init__( self, function, niter, nalternate, k, incfac=1.2, decfac=0.5, tol=TOLDEFAULT, verbose=False ):
        self.function = function
        self.proposal = copy(function)
        self.niter = niter
        self.nalternate = nalternate
        self.k = k
        self.incfac = incfac
        self.decfac = decfac
        self.tol = tol
        self.verbose = verbose
        self.energytraj = [ self.function.energy, ]
        self._minimize()

    def _minimize( self ):
        nlefttodo = self.niter
        doSteep = True
        doCG = False
        iteration = 0
        while nlefttodo > 0 :
            last_energy = self.function.energy

            if nlefttodo > 10 :
                nsteps = self.nalternate
            else :
                nsteps = nlefttodo 
            nlefttodo -= nsteps

            if self.verbose: print "Iteration %d," % iteration,
            if doSteep :
                if self.verbose: print nsteps, "steps of steepest descents"
                minimizer = SteepestDescents
            if doCG :
                if self.verbose: print nsteps, "steps of conjugate gradients"
                minimizer = ConjugateGradients

            thisminimizer = minimizer( self.function, nsteps, self.k, self.incfac, self.decfac, 0, self.verbose )
            this_energy = self.function.energy
            self.energytraj.extend( thisminimizer.energytraj )
            ediff = last_energy - this_energy
            if ediff < self.tol:
                if self.verbose: 
                    print "Energy difference %e < %e" % ( ediff, self.tol )
                break

            doSteep, doCG = doCG, doSteep
            iteration += nsteps
