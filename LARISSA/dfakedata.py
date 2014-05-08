#!/usr/bin/python
#from random import random
from numpy.random import random, normal
from numpy import array, exp, savetxt
from pylab import plot, show

def polyPlusGauss( point ):
    a = 1.0
    b = -0.5
    c = -0.3
    d = -0.2
    A = 1.0
    B = 0.5
    C = 16.0
    
    point2=point*point
    point3=point2*point
    poly = 1- 0.5 * point- 0.3*point2 - 0.2*point3

    diff = point - B
    ndiff = point - 0.7
    ndiff2 = ndiff*ndiff
    diff2 = diff * diff
    C2 = C * C
    gauss = exp( -0.5 * C2 * diff2 )
    gauss2 = exp(-0.5 * (16*16)*ndiff2)

    return poly + gauss+ gauss2

function = polyPlusGauss
xdata=[]
ydata=[]
for i in range(0,120) :
    point = i/120.
    fvalue = function(point)
    datavalue = fvalue + 0.3*random()
    xdata.append(point)
    ydata.append(datavalue)

plot( xdata, ydata )
show()

savetxt( "doublesimulated.txt", array(( xdata, ydata )).transpose() ) 
