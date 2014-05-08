#! /usr/local/bin/python

from numpy import sin, cos, arccos, dot, pi, random, array, arange, exp, logspace
from os import system
from pylab import *
import sys
from optparse import OptionParser
seed(0) # testing purposes

parser=OptionParser()
parser.add_option("-M", \
                  "--WHAM", \
                  dest="wham", \
                  help="WHAM script path.  Default=../WHAM", \
                  default="../WHAM")
parser.add_option("-t", \
                  "--temperature", \
                  dest="temp", \
                  help="System temperature (K) to run WHAM.  Default=300 K", \
                  default=300)
parser.add_option("-k", \
                  "--force", \
                  dest="force", \
                  help="Harmonic force constant.  Default=70 kJ/mol/radian**2", \
                  default=70)
parser.add_option("-b", \
                  "--binsize", \
                  dest="binsize", \
                  help="WHAM bin size (degrees).  Default=5 degrees",\
                  default=5)
parser.add_option("-w", \
                  "--window", \
                  dest="window", \
                  help="WHAM window spacing.  Default=30 degrees", \
                  default=30)
##parser.add_option("-d", \ # This gives really high PMFs and I don't want to spend the time right now determining whether it's a bug or a real result
##                  "--dphi", \
##                  dest="dphi", \
##                  help="Delta phi; +/- range in which the biasing potential is zero.  Default=0 degrees", \
##                  default=0)
parser.add_option("-f", \
                  "--frames", \
                  dest="frames", \
                  help="Number of frames to run for each biasing window.  Default=10000", \
                  default=10000)
parser.add_option("-l", \
                  "--tol", \
                  dest="tol", \
                  help="Monte Carlo convergance limit.  Default=1e-6", \
                  default=1e-6)
parser.add_option("-i", \
                  "--iter", \
                  dest="iter", \
                  help="Max number of iterations.  Default=1e4" ,\
                  default=1e4)
parser.add_option("-s", \
                  "--step", \
                  dest="step", \
                  help="Known potential bin size.  Default=1 degrees", \
                  default=1)
parser.add_option("-p", \
                  "--potential", \
                  dest="potential", \
                  help="Potential function to use: dihedral -or- rb;  \
dihedral function: k[1+cos(3*phi-180)];  \
rb function: SUM(Cn[cos(phi-180)^n],0,5); C0=9.28 C1=12.16 C2=-13.12 C3=-3.06 C4=26.24 C5=-31.5;  \
Ryckaert-Belleman function C values are from the Gromacs user manual.  Default=rb", \
                  default="rb")
options,args=parser.parse_args()

kb=.0083144621 # kJ/(mol*K)
T=float(options.temp)
beta=1/(kb*T)
fc=float(options.force)
whamstep=float(options.binsize)
##dphi=float(options.dphi)
dphi=0
step=float(options.step)
windows=float(options.window)
frames=int(options.frames)
tol=float(options.tol)
maxitern=float(options.iter)
potential=options.potential
whamscript=options.wham
try :
    file=open(whamscript)
    file.close()
except :
    print "ERROR: Could not find WHAM C++ code at %s.  Exiting" %whamscript
    sys.exit()
    
if potential != "rb" and potential != "dihedral" :
    print "Given potential function: --%s--"%potential
    print 'Potential function MUST be either "dihedral" -or- "rb".  No exceptions!'
    sys.exit()
#seed(0) #testing purposes

print "-------------------------------------------------------------------------"
print "Frames :",frames
print "Potential model:",potential
print "Temperature :",T,"K"
print "WHAM bin size :",whamstep,"degrees"
print "Monte Carlo bin size :",step,"degrees"
print "Umbrella window size :",windows,"degrees"
print "Force constant :",fc,"kJ/mol/radian**2"
print "WHAM C++ analysis code located at :",whamscript
print "-------------------------------------------------------------------------"

def r2d( angle ) :
    return angle*180/pi

def d2r( angle ) :
    return angle*pi/180

def d_angle( phi, phi0 ) :
    rphi=d2r(phi)
    rphi0=d2r(phi0)
    dangle=arccos( sin(rphi)*sin(rphi0) + cos(rphi)*cos(rphi0) )
    return dangle

def Vrb( phi, phi0=0,force=fc ) :
    if potential == "dihedral" :
    # Proper dihedral potential
        V=1+cos(d_angle(3*phi,0))
        return V+biasV(phi,phi0,force)

    elif potential == "rb" :
	# Ryckaert-Bellemans alkane potential
        c=array((9.28,12.16,-13.12,-3.06,26.24,-31.5))
        Vrb=0
        for i in range(len(c)) :
            Vrb += c[i]*cos( d_angle( phi, 180. ) )**i
        return Vrb+biasV(phi,phi0,force)

def sumV(phi0,force=fc) :
    totalV=0
    for angle in arange(-180,180,step) :
        totalV += 0.5*step*(exp(-beta*Vrb(angle,phi0,force))+exp(-beta*Vrb(angle+step,phi0,force)))
    return totalV

def prob(phi,phi0,sumV,force=fc) :
    return 0.5*step*(exp(-beta*Vrb(phi,phi0,force))+exp(-beta*Vrb(phi+step,phi0,force)))/sumV

def biasV(phi,phi0,force) :
    diffphi=d_angle(phi,phi0)*180/pi
    if diffphi > dphi :
        ddp=d_angle(diffphi,dphi)
    else :
        ddp=0
    return .5*force*ddp**2

def getSamples( probabilities, size ) :
    sampleCounts = zeros(shape=(360/step))
    total=1
    itern=0
    while total > tol :
        previoustotal = total
        counts = multinomial(size,probabilities)
        newtotal = sum((probabilities-counts/float(frames))**2)
        if newtotal < total :
            sampleCounts=counts
            total=newtotal
            
        itern +=1
        if itern > maxitern : break;
    if itern > 2 : s="s"
    else : s=""
    if total < tol :
        print "Converged to < %.3e (%.3e) after %i iteration%s" %(tol,total,itern-1,s)
    else :
        print "Did not converge: sampling ended at %.3e after %i iteration%s" %(total,itern-1,s)
    samples = array( tuple( countsToSamples( sampleCounts ) ) )
    shuffle( samples )
    return samples

def countsToSamples( counts ) :
    values=[]
    bin=-180
    for value in counts:
        for i in arange(value) :
            values.append(bin+random()*step)
        bin +=  step
    return values

def writeOutput_1D( centered, sample, filelist, n, tag="" ) :
    filename="%i%s.dat" %(n,tag)
    datfile=open(filename,'w')
    for i in getSamples( sample, frames ) :
        datfile.write("%f\n" %i)
    datfile.close()
    filelist.write("%i %s %f 0 %f %f\n" %(n, filename, centered, fc, T))


# Generate the unweighted, analytic probability distribution in 1 dimension
print "Building 1- and 2-Dimensional potential and probability surfaces.  This may take a while..."
angles=[]
unbiased_prob=[]
unbiased_pmf=[]
totalV=sumV(0,force=0)
d2xs=[]
d2ys=[]    
d2zs=[]
d2pmfs=[]
for r1 in arange(-180,180,step) :
    angles.append(r1)
    prob1=prob(r1,0,totalV,force=0)
    pmf1=Vrb(r1,phi0=0,force=0)
    unbiased_prob.append(prob1)
    unbiased_pmf.append(pmf1)
# Generate 2D analytic probability distribution
    for r2 in arange(-180,180,step) :
        d2xs.append(r1)
        d2ys.append(r2)
        prob2=prob(r2,0,totalV,force=0)
        pmf2=Vrb(r2,phi0=0,force=0)
        d2zs.append( prob1*prob2)
        d2pmfs.append(pmf1+pmf2)

d2xs=array(d2xs)
d2ys=array(d2ys)
d2zs=array(d2zs)
d2pmfs=array(d2pmfs)



# Open the 1D file list
filelist="1D_test.file"
file=open(filelist,'w')

xs=[]
for i in arange(-180,180,step) :
    xs.append(i)
mcsamples=zeros(shape=(len(arange(0,360,windows)),len(arange(-180,180,step))))

j=0
for r in arange(0,360,windows) :
    k=0
    totalV=sumV(r,force=fc)
    for i in arange(-180,180,step) :
        mcsamples[j][k]=prob(i, r, totalV, fc)
        k += 1
    writeOutput_1D(r,mcsamples[j],file,j)
    j += 1
file.close()

system("%s --f 1D_test.file --b %f" %(whamscript,whamstep))

probfile=open("1D_test.file.prob")
xs=[]
ys=[]
i=0
total=0
for line in probfile.readlines() :
    ys.append(float(line.split()[0])*step/whamstep)
    total+=float(line.split()[0])
    xs.append(i*whamstep-180)
    i +=1
meanfile=open("1D_test.file.mean")
meanys=[]
for line in meanfile.readlines() :
    meanys.append(float(line.split()[0]))
zeropoint=min(meanys)
for i in range(len(meanys)) :
    meanys[i] -= zeropoint


ax1 = subplot(111)
matplotlib.rc('xtick',labelsize=10)
matplotlib.rc('ytick',labelsize=10)

plot(angles,unbiased_prob,'c-',label='Analytical Probability Distribution',lw=4)
plot(xs,ys,'m--',lw=3,label='WHAM Probability Distribution')
ax1.set_ylabel('Probability')
legend(fontsize=10,loc='upper left')
ax2 = twinx()
ax2.set_ylabel('PMF (kJ/mol)')

plot(angles,unbiased_pmf,'b-',label='Analytical PMF',lw=5)    
plot(xs,meanys,'r--',lw=4,label='WHAM PMF')
title('Comparison between WHAM and Analytic Solutions \n(%i frames for each %.1f degree window and %.1f degree bins)'%(frames,windows,whamstep))
legend(fontsize=10)

xlim(-180,180)
ylim(0,60)
xticks(arange(-180,181,60))
xlabel('Dihedral Angle')
savefig('1D_comparison.pdf',type='pdf')

d2filelist="2D_test.file"
file=open(d2filelist,'w')

n=0
k=0
junk=open('junk.txt','w')
for r1 in arange(0,360,windows) :
    j=0
    for r2 in arange(0,360,windows) :
        file.write("%i %i.dat %f 0 %f 300\n" %(k,n,r1,fc))
        if r1==r2 : # letting r1=r2 artificially increased the frequency at which bin1=bin2
        # I can either only redo r1=r2 or simply redo all r2...
            writeOutput_1D(r2,mcsamples[j],junk,j,tag="d2")
            file.write("%i %id2.dat %f 0 %f 300\n" %(k,j,r2,fc))
        else :
            file.write("%i %i.dat %f 0 %f 300\n" %(k,j,r2,fc))

        j += 1
        k += 1
    n += 1
file.close()
junk.close()


system("%s --f %s --dof 2 --b %f %f" %(whamscript,d2filelist,whamstep, whamstep))

system("rm *.dat *.file *.bin junk.txt")

# Look at 2D now!
matplotlib.rc('xtick',labelsize=10)
matplotlib.rc('ytick',labelsize=10)
xi=linspace(min(d2xs),max(d2xs))
yi=linspace(min(d2ys),max(d2ys))
zi=griddata(d2xs,d2ys,d2zs,xi,yi)
zpmf=griddata(d2xs,d2ys,d2pmfs,xi,yi)
plt.figure(1)
plt.subplot(322)
plt.title('Analytic 2D Probability Distribution')
maxp=max(d2zs)
minp=min(d2zs)
levels=arange(minp,maxp,maxp/100)
CS=plt.contour(xi,yi,zi,levels)
cbar=colorbar(CS)
xlim(-180,180)
ylim(-180,180)
xticks(arange(-180,181,60))
yticks(arange(-180,181,60))

plt.subplot(321)
plt.title('Analytic 2D PMF')
maxpmf=max(d2pmfs)
minpmf=min(d2pmfs)
levels=arange(minpmf,maxpmf,maxpmf/50)
CS1=plt.contour(xi,yi,zpmf,levels)
cbar1=colorbar(CS1)
xlim(-180,180)
ylim(-180,180)
xticks(arange(-180,181,60))
yticks(arange(-180,181,60))

probfile=open('2D_test.file.2dprob')
X=[]
Y=[]
Z=[]
for line in probfile.readlines():
    if line.split() :
        X.append(float(line.split()[0]))
        Y.append(float(line.split()[1]))
        Z.append(float(line.split()[2])*step*step/whamstep/whamstep)
X=array(X)
Y=array(Y)
Z=array(Z)
xi=linspace(min(X),max(X))
yi=linspace(min(Y),max(Y))
zWHAMi=griddata(X,Y,Z,xi,yi)
plt.subplot(324)
plt.title('WHAM 2D Probability Distribution')
maxWHAMp=max(Z)
minWHAMp=min(Z)
levels=arange(minWHAMp,maxWHAMp,maxp/100)
#levels=arange(minp,maxp,maxp/100)
CS=plt.contour(xi,yi,zWHAMi,levels)
cbar=colorbar(CS)
xlim(-180,180)
ylim(-180,180)
xticks(arange(-180,181,60))
yticks(arange(-180,181,60))

plt.subplot(326)
plt.title('Diff 2D Probability Distribution')
difz=zWHAMi-zi
mindif=min(Z)-min(d2zs)
maxdif=max(Z)-max(d2zs)
levels=arange(mindif,maxdif,maxp/100)
CS=plt.contour(xi,yi,difz,levels)
cbar=colorbar(CS)
xlim(-180,180)
ylim(-180,180)
xticks(arange(-180,181,60))
yticks(arange(-180,181,60))

meanfile=open("2D_test.file.2dmean")
X=[]
Y=[]
Z=[]
for line in meanfile.readlines():
    if line.split() :
        X.append(float(line.split()[0]))
        Y.append(float(line.split()[1]))
        Z.append(float(line.split()[2]))
X=array(X)
Y=array(Y)
Z=array(Z)
xi=linspace(min(X),max(X))
yi=linspace(min(Y),max(Y))
zeropoint=min(Z)
for i in range(len(Z)) :
    Z[i] -= zeropoint
zWHAMi=griddata(X,Y,Z,xi,yi)
plt.subplot(323)
plt.title('WHAM 2D PMF')
maxWHAMpmf=max(Z)
minWHAMpmf=min(Z)
#levels=arange(minWHAMpmf,maxWHAMpmf,maxp/100) # This can be 1000 when a bin is unvisited; don't do this
levels=arange(minWHAMpmf,maxpmf,maxpmf/50)
CS=plt.contour(xi,yi,zWHAMi,levels)
cbar=colorbar(CS)
xlim(-180,180)
ylim(-180,180)
xticks(arange(-180,181,60))
yticks(arange(-180,181,60))

plt.subplot(325)
plt.title('Diff 2D PMF')
dZ=Z-d2pmfs
difz=(zWHAMi-zpmf)
mindif=min(Z)-min(d2pmfs)
maxdif=max(Z)-max(d2pmfs)
levels=arange(mindif,maxdif,maxp/100)
#levels=arange(min((zWHAMi-zi)),max((zWHAMi-zpmf)),max((zWHAMi-zpmf))/100)
CS=plt.contour(xi,yi,difz,levels)
cbar=colorbar(CS)
xlim(-180,180)
ylim(-180,180)
xticks(arange(-180,181,60))
yticks(arange(-180,181,60))
savefig('2D_difference.pdf',format='pdf')
##show()


##
##file=open('2D_test.file.mean')
##xs=[]
##ys=[]
##n=0
##for line in file.readlines():
##    ys.append(float(line.split()[0]))
##    xs.append(n)
##    n += 1
##zp=min(ys)
##for i in range(len(ys)) :
##    ys[i] -= zp
###plot(xs[0:72],d2pmfs[0:72],'b-',lw=3)
##for i in range(0,72) :
##    plot(xs[0:72],ys[i*72:i*72+72],'^',label='%i-%i'%(i*72,i*72+72))
##    plot(xs[0:72],d2pmfs[i*72:i*72+72],'-',label='%i-%i analytic'%(i*72,i*72+72))
###xlim(0,148)
##    ylim(0,max(d2pmfs))
###legend(ncol=10,fontsize=8)
##
##show()



